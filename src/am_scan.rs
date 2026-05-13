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
use crate::am_pages;
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
    /// Index version (1 = full scan, 2 = HNSW).
    index_version: u32,
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

    // Read metapage version to determine scan strategy.
    let index_version = am_pages::read_metapage_version(index_rel)
        .unwrap_or(1);

    // Read model name from the metapage.
    let model_name = am_options::read_model_from_metapage(index_rel)
        .unwrap_or_else(|e| pgrx::error!("INFER scan: {e}"));

    let state = Box::new(InferScanState {
        model_name,
        query: None,
        results: Vec::new(),
        position: 0,
        computed: false,
        index_version,
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
        // Dispatch based on index version.
        let result = if state.index_version >= 2 {
            compute_ranked_results_v2(scan, state)
        } else {
            compute_ranked_results(scan, state)
        };
        match result {
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

/// Compute ranked results using the v2 two-phase strategy:
/// 1. Use HNSW graph for initial ANN candidate selection (O(log N)).
/// 2. Re-rank top candidates with the full model for accuracy.
///
/// Falls back to v1 full-scan if the HNSW data cannot be read.
unsafe fn compute_ranked_results_v2(
    scan: pg_sys::IndexScanDesc,
    state: &mut InferScanState,
) -> Result<(), PgInferError> {
    let query = state.query.as_deref().ok_or_else(|| {
        PgInferError::Internal("infer scan v2: no query text in ORDER BY key".into())
    })?;

    let index_rel = (*scan).indexRelation;

    // Read v2 metapage.
    let meta = match am_pages::read_metapage_v2(index_rel) {
        Ok(m) => m,
        Err(_) => {
            // Fall back to v1 full scan if metapage cannot be read as v2.
            return compute_ranked_results(scan, state);
        }
    };

    // If the HNSW graph hasn't been built (pages_start == pages_end),
    // fall back to full scan.
    if meta.hnsw_pages_start >= meta.hnsw_pages_end || meta.num_embeddings == 0 {
        return compute_ranked_results(scan, state);
    }

    // Read HNSW graph data from index pages.
    let hnsw_data = read_hnsw_pages(index_rel, &meta)?;
    let searcher = crate::hnsw::HnswSearcher::from_bytes(&hnsw_data, meta.hnsw_entry_point)?;

    // Read embedding data from index pages for distance computation.
    let embed_data = read_embedding_pages(index_rel, &meta)?;
    let dim = meta.embedding_dim as usize;
    let entry_size = crate::am_pages::EmbeddingEntryHeader::SIZE + dim;

    // Embed the query text to get a float vector.
    let query_embedding: Vec<f32> = registry::with_backend(&state.model_name, |backend| {
        backend.embed(query)
    })?
    .to_vec();

    // HNSW search: get candidates.
    let ef_search = crate::gucs::am_hnsw_ef_search();
    let oversampling = crate::gucs::rerank_oversampling();
    let num_candidates = ef_search * oversampling;

    let distance_fn = |node_id: u32| -> f32 {
        let offset = node_id as usize * entry_size;
        let header_end = offset + crate::am_pages::EmbeddingEntryHeader::SIZE;
        if header_end + dim > embed_data.len() {
            return f32::MAX;
        }
        let header_bytes = &embed_data[offset..header_end];
        // SAFETY: EmbeddingEntryHeader is repr(C) with known layout.
        let header =
            &*(header_bytes.as_ptr() as *const crate::am_pages::EmbeddingEntryHeader);
        let quantized = &embed_data[header_end..header_end + dim];
        crate::sq8::asymmetric_distance_sq8_squared(&query_embedding, quantized, header.min, header.max)
    };

    let hnsw_results = searcher.search(num_candidates, ef_search, &distance_fn);

    if hnsw_results.is_empty() {
        return Ok(());
    }

    // Extract TIDs and candidate texts for re-ranking.
    let heap_rel = (*scan).heapRelation;
    if heap_rel.is_null() {
        return Err(PgInferError::Internal(
            "infer scan v2: heap relation not available".into(),
        ));
    }

    // Collect TIDs from HNSW results (stored in embedding headers).
    let mut candidate_tids: Vec<pg_sys::ItemPointerData> = Vec::new();
    let mut candidate_texts: Vec<String> = Vec::new();

    let index_natts = (*(*index_rel).rd_index).indnatts as usize;
    if index_natts == 0 {
        return Err(PgInferError::Internal(
            "infer scan v2: index has no columns".into(),
        ));
    }
    let attr_num = *(*(*index_rel).rd_index).indkey.values.as_ptr();

    for result in &hnsw_results {
        let offset = result.id as usize * entry_size;
        let header_end = offset + crate::am_pages::EmbeddingEntryHeader::SIZE;
        if header_end > embed_data.len() {
            continue;
        }
        let header =
            &*(embed_data[offset..header_end].as_ptr() as *const crate::am_pages::EmbeddingEntryHeader);

        let tid = header.tid;
        candidate_tids.push(tid);

        // Fetch the actual text from the heap tuple for re-ranking.
        let text = fetch_text_from_heap(heap_rel, &tid, attr_num);
        candidate_texts.push(text.unwrap_or_default());
    }

    if candidate_texts.is_empty() {
        return Ok(());
    }

    // Re-rank with full model.
    let limit = candidate_texts.len();
    let ranked: Vec<RankedCandidate> =
        registry::with_backend(&state.model_name, |backend| {
            backend.rank(&candidate_texts, query, limit)
        })?;

    // Convert to (TID, distance) pairs.
    state.results = ranked
        .into_iter()
        .filter_map(|rc| {
            if rc.index < candidate_tids.len() {
                let distance = score_to_distance(rc.score);
                Some((candidate_tids[rc.index], distance))
            } else {
                None
            }
        })
        .collect();

    Ok(())
}

/// Read HNSW graph pages into a contiguous byte buffer.
unsafe fn read_hnsw_pages(
    index_rel: pg_sys::Relation,
    meta: &am_pages::InferMetaPageV2,
) -> Result<Vec<u8>, PgInferError> {
    let mut data = Vec::new();
    let content_size = pg_sys::BLCKSZ as usize - std::mem::size_of::<pg_sys::PageHeaderData>();

    for blkno in meta.hnsw_pages_start..meta.hnsw_pages_end {
        let buf = pg_sys::ReadBuffer(index_rel, blkno);
        pg_sys::LockBuffer(buf, pg_sys::BUFFER_LOCK_SHARE as i32);
        let page = pg_sys::BufferGetPage(buf);
        let content = pg_sys::PageGetContents(page) as *const u8;
        let slice = std::slice::from_raw_parts(content, content_size);
        data.extend_from_slice(slice);
        pg_sys::UnlockReleaseBuffer(buf);
    }

    Ok(data)
}

/// Read embedding pages into a contiguous byte buffer.
unsafe fn read_embedding_pages(
    index_rel: pg_sys::Relation,
    meta: &am_pages::InferMetaPageV2,
) -> Result<Vec<u8>, PgInferError> {
    let mut data = Vec::new();
    let content_size = pg_sys::BLCKSZ as usize - std::mem::size_of::<pg_sys::PageHeaderData>();

    for blkno in meta.embed_pages_start..meta.embed_pages_end {
        let buf = pg_sys::ReadBuffer(index_rel, blkno);
        pg_sys::LockBuffer(buf, pg_sys::BUFFER_LOCK_SHARE as i32);
        let page = pg_sys::BufferGetPage(buf);
        let content = pg_sys::PageGetContents(page) as *const u8;
        let slice = std::slice::from_raw_parts(content, content_size);
        data.extend_from_slice(slice);
        pg_sys::UnlockReleaseBuffer(buf);
    }

    Ok(data)
}

/// Fetch text from a heap tuple by TID.
///
/// Returns None if the tuple cannot be read.
unsafe fn fetch_text_from_heap(
    heap_rel: pg_sys::Relation,
    tid: &pg_sys::ItemPointerData,
    attr_num: i16,
) -> Option<String> {
    let tupdesc = (*heap_rel).rd_att;
    let slot = pg_sys::MakeSingleTupleTableSlot(tupdesc, &pg_sys::TTSOpsBufferHeapTuple);

    // Fetch the tuple by TID.
    let snapshot = pg_sys::GetActiveSnapshot();
    let mut tid_copy = *tid;
    let found = pg_sys::table_tuple_fetch_row_version(
        heap_rel,
        &mut tid_copy,
        snapshot,
        slot,
    );

    if !found {
        pg_sys::ExecDropSingleTupleTableSlot(slot);
        return None;
    }

    pg_sys::slot_getallattrs(slot);

    let col_idx = (attr_num - 1) as usize;
    let natts = (*(*slot).tts_tupleDescriptor).natts as usize;
    if col_idx >= natts {
        pg_sys::ExecDropSingleTupleTableSlot(slot);
        return None;
    }

    let isnull = *(*slot).tts_isnull.add(col_idx);
    if isnull {
        pg_sys::ExecDropSingleTupleTableSlot(slot);
        return None;
    }

    let datum = *(*slot).tts_values.add(col_idx);
    let text_ptr = pg_sys::pg_detoast_datum(datum.cast_mut_ptr::<pg_sys::varlena>());
    if text_ptr.is_null() {
        pg_sys::ExecDropSingleTupleTableSlot(slot);
        return None;
    }

    let cstr = pg_sys::text_to_cstring(text_ptr as *const pg_sys::text);
    let result = if !cstr.is_null() {
        let s = std::ffi::CStr::from_ptr(cstr)
            .to_string_lossy()
            .into_owned();
        pg_sys::pfree(cstr as *mut std::os::raw::c_void);
        Some(s)
    } else {
        None
    };

    if text_ptr != datum.cast_mut_ptr::<pg_sys::varlena>() {
        pg_sys::pfree(text_ptr as *mut std::os::raw::c_void);
    }

    pg_sys::ExecDropSingleTupleTableSlot(slot);
    result
}
