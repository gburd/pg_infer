//! Scan implementation for the `infer` access method.
//!
//! Uses a batch-then-iterate strategy:
//! 1. On first `amgettuple`, scan the heap to collect candidate texts + TIDs.
//! 2. Call `backend.rank(candidates, query, limit)` in one batch.
//! 3. Cache sorted `(TID, distance)` pairs in scan state.
//! 4. Subsequent `amgettuple` calls pop from the cache.
//! 5. The executor stops calling once LIMIT is satisfied (amcanorderbyop).
//!
//! ## Scalability Note
//!
//! The current implementation performs a full heap scan on every query.
//! This is acceptable for tables with <100K rows. For larger tables,
//! a two-phase approach is needed:
//! 1. Pre-compute and store embeddings in the index pages (Phase 2 AM work)
//! 2. Use approximate nearest neighbor (HNSW/IVF) for candidate selection
//! 3. Re-rank only top-N candidates with the full model
//!
//! This matches the pattern used by pgvecto.rs and pgvector.

use pgrx::pg_sys;
use pgrx::prelude::*;
use pgrx::IntoDatum;

use crate::am_options;
use crate::backend::RankedCandidate;
use crate::error::PgInferError;
use crate::fn_similar::score_to_distance;
use crate::registry;

/// Scan state stored in `IndexScanDesc->opaque`.
struct InferScanState {
    /// Model name from the index metapage.
    model_name: String,
    /// Query text extracted from the ORDER BY scan key (RHS of `<~>`).
    query: Option<String>,
    /// Pre-computed ranked results: (heap TID, distance).
    results: Vec<(pg_sys::ItemPointerData, f64)>,
    /// Current position in results.
    position: usize,
    /// Whether results have been computed.
    computed: bool,
}

/// Begin an index scan.
///
/// Allocates the scan state and stores it in `scan->opaque`.
///
/// # Safety
///
/// Called by PostgreSQL executor.  Arguments are valid for the scan lifetime.
#[pg_guard]
pub unsafe extern "C-unwind" fn infer_ambeginscan(
    index_rel: pg_sys::Relation,
    nkeys: std::os::raw::c_int,
    norderbys: std::os::raw::c_int,
) -> pg_sys::IndexScanDesc {
    let scan = pg_sys::RelationGetIndexScan(index_rel, nkeys, norderbys);

    // Read model name from the metapage.
    let model_name = am_options::read_model_from_metapage(index_rel)
        .unwrap_or_else(|e| pgrx::error!("INFER scan: {e}"));

    let state = Box::new(InferScanState {
        model_name,
        query: None,
        results: Vec::new(),
        position: 0,
        computed: false,
    });

    (*scan).opaque = Box::into_raw(state) as *mut std::os::raw::c_void;
    scan
}

/// Rescan: capture the query text from ORDER BY scan keys.
///
/// Called when scan parameters change (e.g., parameterized nested loop).
///
/// # Safety
///
/// Called by PostgreSQL executor.
#[pg_guard]
pub unsafe extern "C-unwind" fn infer_amrescan(
    scan: pg_sys::IndexScanDesc,
    _keys: pg_sys::ScanKey,
    _nkeys: std::os::raw::c_int,
    orderbys: pg_sys::ScanKey,
    norderbys: std::os::raw::c_int,
) {
    let state = &mut *((*scan).opaque as *mut InferScanState);

    // Reset scan state.
    state.results.clear();
    state.position = 0;
    state.computed = false;
    state.query = None;

    // Extract query text from the ORDER BY scan key.
    // The key is the RHS of `col <~> 'query text'`.
    if norderbys > 0 && !orderbys.is_null() {
        let key = &*orderbys;
        debug_assert!(!key.sk_argument.is_null(), "scan key argument should not be null");
        if !key.sk_argument.is_null() {
            // The argument is a text datum — use text_to_cstring for safe extraction.
            let text_ptr = key.sk_argument.cast_mut_ptr::<pg_sys::varlena>();
            let detoasted = pg_sys::pg_detoast_datum(text_ptr);
            if !detoasted.is_null() {
                let cstr = pg_sys::text_to_cstring(detoasted as *const pg_sys::text);
                if !cstr.is_null() {
                    let s = std::ffi::CStr::from_ptr(cstr)
                        .to_string_lossy()
                        .into_owned();
                    if !s.is_empty() {
                        state.query = Some(s);
                    }
                    pg_sys::pfree(cstr as *mut std::os::raw::c_void);
                }
                // Free detoasted copy if it differs from original.
                if detoasted != text_ptr {
                    pg_sys::pfree(detoasted as *mut std::os::raw::c_void);
                }
            }
        }
    }

    // Copy order-by keys into scan descriptor for executor.
    if norderbys > 0 && !orderbys.is_null() {
        std::ptr::copy_nonoverlapping(
            orderbys,
            (*scan).orderByData,
            norderbys as usize,
        );
    }
}

/// Get the next tuple from the scan.
///
/// On the first call: scans the heap, batches all candidates through
/// `backend.rank()`, caches sorted results.  Subsequent calls just
/// pop from the cache.
///
/// # Safety
///
/// Called by PostgreSQL executor.
#[pg_guard]
pub unsafe extern "C-unwind" fn infer_amgettuple(
    scan: pg_sys::IndexScanDesc,
    direction: pg_sys::ScanDirection::Type,
) -> bool {
    let _ = direction; // We always return in distance order.
    let state = &mut *((*scan).opaque as *mut InferScanState);

    if !state.computed {
        // First call: compute all results.
        match compute_ranked_results(scan, state) {
            Ok(()) => {}
            Err(e) => crate::error::report(e),
        }
        state.computed = true;
    }

    // Return next result.
    if state.position < state.results.len() {
        let (tid, distance) = state.results[state.position];
        state.position += 1;

        // Set the TID in the scan descriptor.
        (*scan).xs_heaptid = tid;
        (*scan).xs_recheck = false;

        // Set the ORDER BY distance value as a proper float8 Datum.
        if !(*scan).xs_orderbyvals.is_null() {
            let datum_ptr = (*scan).xs_orderbyvals;
            *datum_ptr = f64::into_datum(distance).unwrap_or(pg_sys::Datum::from(0u64));
            let null_ptr = (*scan).xs_orderbynulls;
            *null_ptr = false;
        }

        true
    } else {
        false
    }
}

/// End the index scan and free state.
///
/// # Safety
///
/// Called by PostgreSQL executor.
#[pg_guard]
pub unsafe extern "C-unwind" fn infer_amendscan(scan: pg_sys::IndexScanDesc) {
    if !(*scan).opaque.is_null() {
        let _ = Box::from_raw((*scan).opaque as *mut InferScanState);
        (*scan).opaque = std::ptr::null_mut();
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Compute ranked results by scanning the heap and calling `backend.rank()`.
unsafe fn compute_ranked_results(
    scan: pg_sys::IndexScanDesc,
    state: &mut InferScanState,
) -> Result<(), PgInferError> {
    let query = state.query.as_deref().ok_or_else(|| {
        PgInferError::Internal("infer scan: no query text in ORDER BY key".into())
    })?;

    // Get the heap relation from the scan descriptor.
    let heap_rel = (*scan).heapRelation;
    if heap_rel.is_null() {
        return Err(PgInferError::Internal(
            "infer scan: heap relation not available".into(),
        ));
    }

    // Determine the indexed column (attribute number from the index).
    let index_rel = (*scan).indexRelation;
    let index_natts = (*(*index_rel).rd_index).indnatts as usize;
    if index_natts == 0 {
        return Err(PgInferError::Internal(
            "infer scan: index has no columns".into(),
        ));
    }
    let attr_num = *(*(*index_rel).rd_index).indkey.values.as_ptr();

    // Scan the heap to collect (TID, text_value) pairs.
    let mut candidates: Vec<String> = Vec::new();
    let mut tids: Vec<pg_sys::ItemPointerData> = Vec::new();

    let snapshot = pg_sys::GetActiveSnapshot();
    let table_scan = pg_sys::table_beginscan(heap_rel, snapshot, 0, std::ptr::null_mut());

    // Allocate a reusable slot for the scan.
    let tupdesc = (*heap_rel).rd_att;
    let slot = pg_sys::MakeSingleTupleTableSlot(tupdesc, &pg_sys::TTSOpsBufferHeapTuple);

    loop {
        let got_tuple = pg_sys::table_scan_getnextslot(
            table_scan,
            pg_sys::ScanDirection::ForwardScanDirection,
            slot,
        );
        if !got_tuple {
            break;
        }

        // Materialize the slot to access attributes.
        pg_sys::slot_getallattrs(slot);

        // Extract the text value from the indexed column.
        let col_idx = (attr_num - 1) as usize;
        let natts = (*(*slot).tts_tupleDescriptor).natts as usize;
        if col_idx >= natts {
            continue; // Column index out of range for this tuple.
        }
        let isnull = *(*slot).tts_isnull.add(col_idx);
        if isnull {
            continue; // Skip NULL values.
        }

        let datum = *(*slot).tts_values.add(col_idx);

        // Convert text datum to Rust string via text_to_cstring.
        let text_ptr = pg_sys::pg_detoast_datum(datum.cast_mut_ptr::<pg_sys::varlena>());
        if text_ptr.is_null() {
            continue;
        }

        let cstr = pg_sys::text_to_cstring(text_ptr as *const pg_sys::text);
        if !cstr.is_null() {
            let text = std::ffi::CStr::from_ptr(cstr)
                .to_string_lossy()
                .into_owned();
            if !text.is_empty() {
                tids.push((*slot).tts_tid);
                candidates.push(text);
            }
            pg_sys::pfree(cstr as *mut std::os::raw::c_void);
        }

        // Free detoasted copy if different from original.
        if text_ptr != datum.cast_mut_ptr::<pg_sys::varlena>() {
            pg_sys::pfree(text_ptr as *mut std::os::raw::c_void);
        }
    }

    pg_sys::ExecDropSingleTupleTableSlot(slot);
    pg_sys::table_endscan(table_scan);

    if candidates.is_empty() {
        return Ok(());
    }

    // Score all candidates, let executor LIMIT.
    let limit = candidates.len();

    // Call backend.rank() for batch scoring.
    let ranked: Vec<RankedCandidate> =
        registry::with_backend(&state.model_name, |backend| {
            backend.rank(&candidates, query, limit)
        })?;

    // Convert to (TID, distance) pairs sorted by distance ascending.
    state.results = ranked
        .into_iter()
        .filter_map(|rc| {
            if rc.index < tids.len() {
                let distance = score_to_distance(rc.score);
                Some((tids[rc.index], distance))
            } else {
                None
            }
        })
        .collect();

    // Results are already sorted by score descending from rank(),
    // which means distance ascending (score_to_distance is monotonically
    // decreasing with increasing score).

    Ok(())
}
