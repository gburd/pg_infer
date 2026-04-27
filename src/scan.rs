//! Index scan callbacks for the infer AM.
//!
//! Model indexes (index_kind = 0) do not return heap tuples via scans;
//! queries go through walk/describe/similar_to functions.
//!
//! Column indexes (index_kind = 1) support ORDER BY <col> <~> 'query'
//! queries by performing a brute-force heap scan, computing semantic
//! distances for every row, sorting, and returning TIDs in distance order.

use pgrx::pg_sys;

use crate::error::PgInferError;
use crate::pages;

// ---------------------------------------------------------------------------
// Scan state for column index scans
// ---------------------------------------------------------------------------

/// Per-scan state stored in `IndexScanDesc.opaque`.
struct InferScanState {
    /// Index kind from the metapage (MODEL or COLUMN).
    index_kind: u8,
    /// Model name from the metapage.
    model_name: String,
    /// Query text extracted from the ORDER BY scan key.
    query_text: Option<String>,
    /// Sorted results: (distance, heap TID).
    results: Vec<(f64, pg_sys::ItemPointerData)>,
    /// Current position in the results vector.
    cursor: usize,
    /// Whether the brute-force scan has been performed.
    initialized: bool,
}

// ---------------------------------------------------------------------------
// ambeginscan
// ---------------------------------------------------------------------------

pub unsafe extern "C-unwind" fn infer_ambeginscan(
    index_relation: pg_sys::Relation,
    nkeys: std::ffi::c_int,
    norderbys: std::ffi::c_int,
) -> pg_sys::IndexScanDesc {
    let scan = pg_sys::RelationGetIndexScan(index_relation, nkeys, norderbys);

    // Read the metapage to determine index kind and model name.
    let (index_kind, model_name) = read_metapage_info(index_relation);

    let state = Box::new(InferScanState {
        index_kind,
        model_name,
        query_text: None,
        results: Vec::new(),
        cursor: 0,
        initialized: false,
    });

    (*scan).opaque = Box::into_raw(state) as *mut std::ffi::c_void;
    scan
}

// ---------------------------------------------------------------------------
// amrescan
// ---------------------------------------------------------------------------

pub unsafe extern "C-unwind" fn infer_amrescan(
    scan: pg_sys::IndexScanDesc,
    _keys: pg_sys::ScanKey,
    _nkeys: std::ffi::c_int,
    orderbys: pg_sys::ScanKey,
    norderbys: std::ffi::c_int,
) {
    if (*scan).opaque.is_null() {
        return;
    }
    let state = &mut *((*scan).opaque as *mut InferScanState);

    // Reset scan state for a new scan.
    state.results.clear();
    state.cursor = 0;
    state.initialized = false;
    state.query_text = None;

    // Extract the query text from the ORDER BY scan key.
    // For `ORDER BY col <~> 'query'`, the RHS text is in orderbys[0].sk_argument.
    if norderbys > 0 && !orderbys.is_null() {
        let key = &*orderbys;
        // Check the key is not null.
        if key.sk_flags & pg_sys::SK_ISNULL as i32 == 0 {
            state.query_text = datum_text_to_string(key.sk_argument);
        }

        // Copy the order-by keys into the scan descriptor so PG knows we
        // handle them.
        std::ptr::copy_nonoverlapping(orderbys, (*scan).orderByData, norderbys as usize);
    }
}

// ---------------------------------------------------------------------------
// amgettuple
// ---------------------------------------------------------------------------

pub unsafe extern "C-unwind" fn infer_amgettuple(
    scan: pg_sys::IndexScanDesc,
    direction: pg_sys::ScanDirection::Type,
) -> bool {
    if (*scan).opaque.is_null() {
        return false;
    }
    let state = &mut *((*scan).opaque as *mut InferScanState);

    // Model indexes don't return tuples via scan.
    if state.index_kind == pages::INDEX_KIND_MODEL {
        return false;
    }

    // Only support forward scans.
    if direction != pg_sys::ScanDirection::ForwardScanDirection {
        return false;
    }

    // No query text means no ORDER BY <~> query.
    let query_text = match &state.query_text {
        Some(q) => q.clone(),
        None => return false,
    };

    // On first call, perform the brute-force heap scan.
    if !state.initialized {
        state.initialized = true;

        let model_name = state.model_name.clone();
        match compute_distances(scan, &model_name, &query_text) {
            Ok(results) => {
                state.results = results;
            }
            Err(e) => {
                pgrx::warning!("INFER scan error: {}", e);
                return false;
            }
        }
    }

    // Return the next result.
    if state.cursor < state.results.len() {
        let (distance, tid) = state.results[state.cursor];
        (*scan).xs_heaptid = tid;

        // Set the ORDER BY distance value so the executor knows the sort key.
        if !(*scan).xs_orderbyvals.is_null() {
            *(*scan).xs_orderbyvals = pg_sys::Float8GetDatum(distance);
        }
        if !(*scan).xs_orderbynulls.is_null() {
            *(*scan).xs_orderbynulls = false;
        }
        (*scan).xs_recheckorderby = false;

        state.cursor += 1;
        true
    } else {
        false
    }
}

// ---------------------------------------------------------------------------
// amendscan
// ---------------------------------------------------------------------------

pub unsafe extern "C-unwind" fn infer_amendscan(scan: pg_sys::IndexScanDesc) {
    // Free the scan state.
    if !(*scan).opaque.is_null() {
        let _ = Box::from_raw((*scan).opaque as *mut InferScanState);
        (*scan).opaque = std::ptr::null_mut();
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Read the index kind and model name from the metapage (block 0).
unsafe fn read_metapage_info(rel: pg_sys::Relation) -> (u8, String) {
    let nblocks = pg_sys::RelationGetNumberOfBlocksInFork(
        rel,
        pg_sys::ForkNumber::MAIN_FORKNUM,
    );
    if nblocks == 0 {
        return (pages::INDEX_KIND_MODEL, String::new());
    }

    let buf = pg_sys::ReadBuffer(rel, 0);
    pg_sys::LockBuffer(buf, pg_sys::BUFFER_LOCK_SHARE as i32);
    let page = pg_sys::BufferGetPage(buf);
    let data_ptr = pages::page_get_data(page);
    let meta: pages::InferMetaPage = std::ptr::read_unaligned(data_ptr as *const _);
    pg_sys::UnlockReleaseBuffer(buf);

    let model_name = {
        let nul = meta
            .model_name
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(meta.model_name.len());
        String::from_utf8_lossy(&meta.model_name[..nul]).into_owned()
    };

    (meta.index_kind, model_name)
}

/// Get the indexed column attribute number directly from the index's
/// `rd_index` (Form_pg_index) struct, avoiding SPI.
unsafe fn get_heap_attnum(scan: pg_sys::IndexScanDesc) -> Result<i16, PgInferError> {
    let rd_index = (*(*scan).indexRelation).rd_index;
    if rd_index.is_null() {
        return Err(PgInferError::Internal("rd_index is null".into()));
    }
    Ok(*(*rd_index).indkey.values.as_ptr())
}

/// Convert a text Datum to a Rust String, pfree-ing the intermediate C string.
///
/// Returns `None` if the C string is null or not valid UTF-8.
unsafe fn datum_text_to_string(datum: pg_sys::Datum) -> Option<String> {
    let cstr = pg_sys::text_to_cstring(datum.cast_mut_ptr::<pg_sys::text>());
    if cstr.is_null() {
        return None;
    }
    let result = std::ffi::CStr::from_ptr(cstr)
        .to_str()
        .ok()
        .map(|s| s.to_string());
    pg_sys::pfree(cstr as *mut std::ffi::c_void);
    result
}

/// Perform a brute-force heap scan, computing semantic distance for every
/// visible tuple, and return results sorted by distance ascending.
///
/// The query embedding is pre-computed once and reused for all rows.
unsafe fn compute_distances(
    scan: pg_sys::IndexScanDesc,
    model_name: &str,
    query_text: &str,
) -> Result<Vec<(f64, pg_sys::ItemPointerData)>, PgInferError> {
    let attnum = get_heap_attnum(scan)?;

    // Use heapRelation directly from the scan descriptor — no need to
    // relation_open/close since the executor already holds the lock.
    let heap_rel = (*scan).heapRelation;
    if heap_rel.is_null() {
        return Err(PgInferError::Internal(
            "heapRelation is null — scan not initialized correctly".into(),
        ));
    }

    // Begin a table scan.
    let snap = pg_sys::GetActiveSnapshot();
    let tscan = pg_sys::table_beginscan(heap_rel, snap, 0, std::ptr::null_mut());
    let slot = pg_sys::table_slot_create(heap_rel, std::ptr::null_mut());

    let mut results = Vec::new();

    let scan_result = crate::registry::with_model(model_name, |handle| {
        // Pre-compute the query embedding once.
        let query_embedding = crate::fn_similar::embed_text(handle, query_text)?;

        loop {
            let got = pg_sys::table_scan_getnextslot(
                tscan,
                pg_sys::ScanDirection::ForwardScanDirection,
                slot,
            );
            if !got {
                break;
            }

            // Deform the tuple to access the indexed attribute.
            pg_sys::slot_getsomeattrs_int(slot, attnum as _);

            let idx = (attnum - 1) as usize;
            let is_null = *(*slot).tts_isnull.add(idx);
            if is_null {
                continue;
            }
            let datum = *(*slot).tts_values.add(idx);

            let col_text = match datum_text_to_string(datum) {
                Some(s) => s,
                None => continue,
            };

            // Compute similarity with pre-computed query embedding.
            let score = crate::fn_similar::similar_to_with_embedding(
                handle, &col_text, &query_embedding,
            )
            .unwrap_or(0.0);
            let distance = crate::fn_similar::score_to_distance(score);

            let tid = (*slot).tts_tid;
            results.push((distance, tid));
        }

        Ok(())
    });

    // Clean up the scan resources.
    pg_sys::ExecDropSingleTupleTableSlot(slot);
    pg_sys::table_endscan(tscan);

    scan_result?;

    // Sort by distance ascending (most similar first).
    results.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(results)
}
