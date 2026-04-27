//! Index access method registration and callbacks.
//!
//! Implements the `infer` custom index AM so that
//! `CREATE INDEX ... USING infer` works.

use pgrx::pg_sys;
use pgrx::prelude::*;

use crate::build;
use crate::options;
use crate::scan;

/// P_NEW block number — tells ReadBufferExtended to extend the relation.
const P_NEW: pg_sys::BlockNumber = crate::pages::INVALID_BLOCK_NUMBER;

// ---------------------------------------------------------------------------
// AM handler — returns the IndexAmRoutine
// ---------------------------------------------------------------------------

/// The handler function registered via CREATE ACCESS METHOD.
///
/// ```sql
/// CREATE FUNCTION infer_am_handler(internal) RETURNS index_am_handler
///     AS 'MODULE_PATHNAME', 'infer_am_handler' LANGUAGE C;
///
/// CREATE ACCESS METHOD infer TYPE INDEX HANDLER infer_am_handler;
/// ```
#[pg_extern(sql = "
CREATE FUNCTION infer_am_handler(internal) RETURNS index_am_handler
    AS 'MODULE_PATHNAME', 'infer_am_handler_wrapper' LANGUAGE C STRICT;

CREATE ACCESS METHOD infer TYPE INDEX HANDLER infer_am_handler;
")]
fn infer_am_handler(
    _fcinfo: pg_sys::FunctionCallInfo,
) -> pgrx::PgBox<pg_sys::IndexAmRoutine> {
    let amroutine = unsafe {
        let ptr = pg_sys::palloc0(std::mem::size_of::<pg_sys::IndexAmRoutine>())
            as *mut pg_sys::IndexAmRoutine;
        (*ptr).type_ = pg_sys::NodeTag::T_IndexAmRoutine;

        // Capabilities
        (*ptr).amstrategies = 1;
        (*ptr).amsupport = 1;
        (*ptr).amoptsprocnum = 0;
        (*ptr).amcanorder = false;
        (*ptr).amcanorderbyop = true;
        (*ptr).amcanhash = false;
        (*ptr).amconsistentequality = false;
        (*ptr).amconsistentordering = false;
        (*ptr).amcanbackward = false;
        (*ptr).amcanunique = false;
        (*ptr).amcanmulticol = false;
        (*ptr).amoptionalkey = true;
        (*ptr).amsearcharray = false;
        (*ptr).amsearchnulls = false;
        (*ptr).amstorage = false;
        (*ptr).amclusterable = false;
        (*ptr).ampredlocks = false;
        (*ptr).amcanparallel = false;
        (*ptr).amcanbuildparallel = false;
        (*ptr).amcaninclude = false;
        (*ptr).amusemaintenanceworkmem = true;
        (*ptr).amsummarizing = false;
        (*ptr).amkeytype = pg_sys::InvalidOid;

        // Required callbacks
        (*ptr).ambuild = Some(infer_ambuild);
        (*ptr).ambuildempty = Some(infer_ambuildempty);
        (*ptr).amvalidate = Some(infer_amvalidate);
        (*ptr).amoptions = Some(infer_amoptions);
        (*ptr).amcostestimate = Some(infer_amcostestimate);

        // Scan callbacks
        (*ptr).ambeginscan = Some(scan::infer_ambeginscan);
        (*ptr).amrescan = Some(scan::infer_amrescan);
        (*ptr).amgettuple = Some(scan::infer_amgettuple);
        (*ptr).amendscan = Some(scan::infer_amendscan);

        // No-op maintenance callbacks (index is immutable)
        (*ptr).aminsert = None;
        (*ptr).ambulkdelete = Some(infer_ambulkdelete);
        (*ptr).amvacuumcleanup = Some(infer_amvacuumcleanup);

        pgrx::PgBox::from_pg(ptr)
    };

    amroutine
}

// ---------------------------------------------------------------------------
// ambuild — populate the index from a vindex directory
// ---------------------------------------------------------------------------

unsafe extern "C-unwind" fn infer_ambuild(
    heap_relation: pg_sys::Relation,
    index_relation: pg_sys::Relation,
    index_info: *mut pg_sys::IndexInfo,
) -> *mut pg_sys::IndexBuildResult {
    let _ = (heap_relation, index_info);

    // Parse the WITH options to determine index type.
    let source = options::get_source_option(index_relation);
    let model = options::get_model_option(index_relation);

    let build_result = match (source, model) {
        (Some(path), _) => {
            // Model index: full vindex data stored in PG pages.
            build::build_index(index_relation, &path)
        }
        (None, Some(model_name)) => {
            // Column index: references a model, enables <~> scans.
            build::build_column_index(index_relation, &model_name)
        }
        (None, None) => {
            pgrx::error!(
                "INFER: missing required option — use WITH (source = '...') \
                 for model indexes or WITH (model = '...') for column indexes"
            );
        }
    };

    match build_result {
        Ok(r) => {
            let ptr = pg_sys::palloc0(std::mem::size_of::<pg_sys::IndexBuildResult>())
                as *mut pg_sys::IndexBuildResult;
            (*ptr) = r;
            ptr
        }
        Err(e) => {
            pgrx::error!("INFER ambuild failed: {}", e);
        }
    }
}

// ---------------------------------------------------------------------------
// ambuildempty — create an empty index (for REINDEX on unlogged tables)
// ---------------------------------------------------------------------------

unsafe extern "C-unwind" fn infer_ambuildempty(index_relation: pg_sys::Relation) {
    let buf = pg_sys::ReadBufferExtended(
        index_relation,
        pg_sys::ForkNumber::INIT_FORKNUM,
        P_NEW,
        pg_sys::ReadBufferMode::RBM_ZERO_AND_LOCK,
        std::ptr::null_mut(),
    );

    let state = pg_sys::GenericXLogStart(index_relation);
    let page = pg_sys::GenericXLogRegisterBuffer(
        state,
        buf,
        pg_sys::GENERIC_XLOG_FULL_IMAGE as i32,
    );

    crate::pages::init_page(page, crate::pages::PageType::Meta);

    let meta = crate::pages::InferMetaPage {
        magic: crate::pages::INFER_MAGIC,
        format_version: crate::pages::INFER_FORMAT_VERSION,
        model_name: [0u8; 128],
        num_layers: 0,
        hidden_size: 0,
        features_per_layer: 0,
        vocab_size: 0,
        embed_scale: 0.0,
        gate_dtype: 0,
        embed_dtype: 0,
        down_top_k: 0,
        extract_level: 0,
        index_kind: crate::pages::INDEX_KIND_MODEL,
        _pad: [0u8; 2],
        layer_dir_blk: 0,
        gate_start_blk: 0,
        gate_end_blk: 0,
        embed_start_blk: 0,
        embed_end_blk: 0,
        down_start_blk: 0,
        down_end_blk: 0,
        tok_start_blk: 0,
        tok_end_blk: 0,
        max_gate_score: 0.0,
        mean_gate_score: 0.0,
        total_pages: 1,
        source_uri: [0u8; 256],
    };

    crate::pages::write_struct_at(page, 0, &meta);
    pg_sys::GenericXLogFinish(state);
    pg_sys::UnlockReleaseBuffer(buf);
}

// ---------------------------------------------------------------------------
// amvalidate
// ---------------------------------------------------------------------------

unsafe extern "C-unwind" fn infer_amvalidate(_opclass_oid: pg_sys::Oid) -> bool {
    true
}

// ---------------------------------------------------------------------------
// amoptions — parse WITH (source = '...', extract_level = '...')
// ---------------------------------------------------------------------------

unsafe extern "C-unwind" fn infer_amoptions(
    reloptions: pg_sys::Datum,
    validate: bool,
) -> *mut pg_sys::bytea {
    options::infer_amoptions_impl(reloptions, validate)
}

// ---------------------------------------------------------------------------
// amcostestimate
// ---------------------------------------------------------------------------

unsafe extern "C-unwind" fn infer_amcostestimate(
    _root: *mut pg_sys::PlannerInfo,
    path: *mut pg_sys::IndexPath,
    _loop_count: f64,
    index_startup_cost: *mut pg_sys::Cost,
    index_total_cost: *mut pg_sys::Cost,
    index_selectivity: *mut pg_sys::Selectivity,
    index_correlation: *mut f64,
    index_pages: *mut f64,
) {
    // Read heap statistics from the planner's IndexPath to produce
    // realistic cost estimates.  The scan is O(N) with expensive per-row
    // tokenization + embedding + KNN, so we want the planner to prefer
    // sequential scan for large tables unless ORDER BY + LIMIT applies.
    let (heap_rows, heap_pages) = if !path.is_null() {
        let indexinfo = (*path).indexinfo;
        if !indexinfo.is_null() {
            let rel = (*indexinfo).rel;
            if !rel.is_null() {
                ((*rel).rows, (*rel).pages as f64)
            } else {
                (1000.0, 10.0)
            }
        } else {
            (1000.0, 10.0)
        }
    } else {
        (1000.0, 10.0)
    };

    let startup = 100.0; // query embedding computation
    let total = startup + heap_rows * 50.0; // 50x cpu_tuple_cost per row

    *index_startup_cost = startup;
    *index_total_cost = total;
    *index_selectivity = 1.0; // we scan all rows; LIMIT prunes
    *index_correlation = 0.0; // output in distance order, not physical
    *index_pages = heap_pages;
}

// ---------------------------------------------------------------------------
// No-op maintenance callbacks
// ---------------------------------------------------------------------------

unsafe extern "C-unwind" fn infer_ambulkdelete(
    _info: *mut pg_sys::IndexVacuumInfo,
    stats: *mut pg_sys::IndexBulkDeleteResult,
    _callback: pg_sys::IndexBulkDeleteCallback,
    _callback_state: *mut std::ffi::c_void,
) -> *mut pg_sys::IndexBulkDeleteResult {
    if stats.is_null() {
        pg_sys::palloc0(std::mem::size_of::<pg_sys::IndexBulkDeleteResult>())
            as *mut pg_sys::IndexBulkDeleteResult
    } else {
        stats
    }
}

unsafe extern "C-unwind" fn infer_amvacuumcleanup(
    _info: *mut pg_sys::IndexVacuumInfo,
    stats: *mut pg_sys::IndexBulkDeleteResult,
) -> *mut pg_sys::IndexBulkDeleteResult {
    if stats.is_null() {
        pg_sys::palloc0(std::mem::size_of::<pg_sys::IndexBulkDeleteResult>())
            as *mut pg_sys::IndexBulkDeleteResult
    } else {
        stats
    }
}
