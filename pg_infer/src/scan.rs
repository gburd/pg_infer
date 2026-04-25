//! Index scan callbacks for the infer AM.
//!
//! The infer index is not a traditional row-returning index.  Scan support
//! is minimal: it exists so PostgreSQL's infrastructure doesn't reject the
//! AM, but actual queries go through the walk/describe/similar_to functions.

use pgrx::pg_sys;

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
    scan
}

// ---------------------------------------------------------------------------
// amrescan
// ---------------------------------------------------------------------------

pub unsafe extern "C-unwind" fn infer_amrescan(
    scan: pg_sys::IndexScanDesc,
    keys: pg_sys::ScanKey,
    nkeys: std::ffi::c_int,
    orderbys: pg_sys::ScanKey,
    norderbys: std::ffi::c_int,
) {
    let _ = (scan, keys, nkeys, orderbys, norderbys);
    // No-op: scan state is reset but we don't maintain cursor state.
}

// ---------------------------------------------------------------------------
// amgettuple — always returns false (no tuples via index scan)
// ---------------------------------------------------------------------------

pub unsafe extern "C-unwind" fn infer_amgettuple(
    _scan: pg_sys::IndexScanDesc,
    _direction: pg_sys::ScanDirection::Type,
) -> bool {
    // The infer index doesn't return heap tuples via scans.
    // Query functions read pages directly.
    false
}

// ---------------------------------------------------------------------------
// amendscan
// ---------------------------------------------------------------------------

pub unsafe extern "C-unwind" fn infer_amendscan(scan: pg_sys::IndexScanDesc) {
    let index_relation = (*scan).indexRelation;
    pg_sys::RelationDecrementReferenceCount(index_relation);
}
