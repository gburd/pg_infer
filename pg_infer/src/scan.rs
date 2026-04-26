//! Index scan callbacks for the infer AM.
//!
//! Model indexes (index_kind = 0) do not return heap tuples via scans;
//! queries go through walk/describe/similar_to functions.
//!
//! Column indexes (index_kind = 1) support ORDER BY <col> <~> 'query'
//! queries by performing a brute-force heap scan, computing semantic
//! distances for every row, sorting, and returning TIDs in distance order.

use pgrx::pg_sys;
use pgrx::prelude::*;

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
    pg_sys::RelationIncrementReferenceCount(index_relation);
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
            let cstr = pg_sys::text_to_cstring(key.sk_argument.cast_mut_ptr::<pg_sys::text>());
            if !cstr.is_null() {
                let query = std::ffi::CStr::from_ptr(cstr)
                    .to_str()
                    .unwrap_or("")
                    .to_string();
                pg_sys::pfree(cstr as *mut std::ffi::c_void);
                state.query_text = Some(query);
            }
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

    let index_relation = (*scan).indexRelation;
    pg_sys::RelationDecrementReferenceCount(index_relation);
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

/// Get the heap relation OID and indexed column attribute number for an
/// index, via SPI query on `pg_index`.
unsafe fn get_heap_info(index_oid: pg_sys::Oid) -> Result<(pg_sys::Oid, i16), PgInferError> {
    let query = format!(
        "SELECT indrelid, indkey[0] FROM pg_index WHERE indexrelid = {}",
        index_oid.to_u32()
    );

    Spi::connect(|client| {
        let result = client.select(&query, None, &[])?;
        for row in result {
            let heap_oid: pg_sys::Oid = row
                .get(1)?
                .ok_or(pgrx::spi::SpiError::InvalidPosition)?;
            let attnum: i16 = row
                .get(2)?
                .ok_or(pgrx::spi::SpiError::InvalidPosition)?;
            return Ok((heap_oid, attnum));
        }
        Err(pgrx::spi::SpiError::InvalidPosition)
    })
    .map_err(PgInferError::Spi)
}

/// Perform a brute-force heap scan, computing `infer_distance` for every
/// visible tuple, and return results sorted by distance ascending.
unsafe fn compute_distances(
    scan: pg_sys::IndexScanDesc,
    model_name: &str,
    query_text: &str,
) -> Result<Vec<(f64, pg_sys::ItemPointerData)>, PgInferError> {
    let index_rel = (*scan).indexRelation;
    let index_oid = (*index_rel).rd_id;

    let (heap_oid, attnum) = get_heap_info(index_oid)?;

    // Open the heap relation.
    let heap_rel = pg_sys::relation_open(heap_oid, pg_sys::AccessShareLock as _);

    // Begin a table scan.
    let snap = pg_sys::GetActiveSnapshot();
    let tscan = pg_sys::table_beginscan(heap_rel, snap, 0, std::ptr::null_mut());
    let slot = pg_sys::table_slot_create(heap_rel, std::ptr::null_mut());

    let mut results = Vec::new();

    let scan_result = crate::registry::with_model(model_name, |handle| {
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

            // Convert text datum to a Rust string.
            let cstr = pg_sys::text_to_cstring(datum.cast_mut_ptr::<pg_sys::text>());
            if cstr.is_null() {
                continue;
            }
            let col_text = std::ffi::CStr::from_ptr(cstr).to_str().unwrap_or("");

            // Compute similarity score and convert to distance.
            let score =
                crate::fn_similar::similar_to_impl(handle, col_text, query_text).unwrap_or(0.0);
            let distance = if score > 0.0 { 1.0 / score } else { f64::MAX };

            let tid = (*slot).tts_tid;
            results.push((distance, tid));

            pg_sys::pfree(cstr as *mut std::ffi::c_void);
        }

        Ok(())
    });

    // Clean up the scan resources.
    pg_sys::ExecDropSingleTupleTableSlot(slot);
    pg_sys::table_endscan(tscan);
    pg_sys::relation_close(heap_rel, pg_sys::AccessShareLock as _);

    scan_result?;

    // Sort by distance ascending (most similar first).
    results.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(results)
}
