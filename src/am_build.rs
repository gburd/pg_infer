//! Index build for the `infer` access method.
//!
//! `ambuild` writes a single metapage containing the model name.
//! No vindex data is copied into PG pages — the index is virtual.

use pgrx::pg_sys;
use pgrx::prelude::*;

use crate::am_options::{InferMetaPage, INFER_META_MAGIC};
use crate::error::PgInferError;

/// Build the infer index: write a single metapage with the model name.
///
/// # Safety
///
/// Called by PostgreSQL during CREATE INDEX.  All pointer arguments
/// are provided by the executor and are valid for the duration of
/// this call.
#[pg_guard]
pub unsafe extern "C-unwind" fn infer_ambuild(
    heap: pg_sys::Relation,
    index: pg_sys::Relation,
    index_info: *mut pg_sys::IndexInfo,
) -> *mut pg_sys::IndexBuildResult {
    let _ = index_info;

    // Determine model name: from reloptions or GUC fallback.
    let model_name = resolve_model_for_build(index)
        .unwrap_or_else(|e| pgrx::error!("INFER ambuild: {e}"));

    // Verify the model is registered (fail fast rather than at scan time).
    verify_model_exists(&model_name)
        .unwrap_or_else(|e| pgrx::error!("INFER ambuild: {e}"));

    // Write metapage to block 0.
    write_metapage(index, &model_name)
        .unwrap_or_else(|e| pgrx::error!("INFER ambuild: {e}"));

    // Count heap tuples for the build result.
    let heap_tuples = (*(*heap).rd_rel).reltuples as f64;

    let result = pg_sys::palloc0(std::mem::size_of::<pg_sys::IndexBuildResult>())
        as *mut pg_sys::IndexBuildResult;
    (*result).heap_tuples = heap_tuples;
    (*result).index_tuples = 0.0; // virtual index stores no tuples
    result
}

/// Build an empty index (for unlogged tables).
///
/// # Safety
///
/// Called by PostgreSQL. Arguments are valid for the duration of this call.
#[pg_guard]
pub unsafe extern "C-unwind" fn infer_ambuildempty(index: pg_sys::Relation) {
    let model_name = resolve_model_for_build(index)
        .unwrap_or_else(|e| pgrx::error!("INFER ambuildempty: {e}"));
    write_metapage(index, &model_name)
        .unwrap_or_else(|e| pgrx::error!("INFER ambuildempty: {e}"));
}

/// Bulk delete: no-op for virtual index.
///
/// # Safety
///
/// Called by PostgreSQL during VACUUM.
#[pg_guard]
pub unsafe extern "C-unwind" fn infer_ambulkdelete(
    info: *mut pg_sys::IndexVacuumInfo,
    stats: *mut pg_sys::IndexBulkDeleteResult,
    _callback: pg_sys::IndexBulkDeleteCallback,
    _callback_state: *mut std::os::raw::c_void,
) -> *mut pg_sys::IndexBulkDeleteResult {
    let _ = info;
    if stats.is_null() {
        pg_sys::palloc0(std::mem::size_of::<pg_sys::IndexBulkDeleteResult>())
            as *mut pg_sys::IndexBulkDeleteResult
    } else {
        stats
    }
}

/// Vacuum cleanup: no-op for virtual index.
///
/// # Safety
///
/// Called by PostgreSQL during VACUUM.
#[pg_guard]
pub unsafe extern "C-unwind" fn infer_amvacuumcleanup(
    info: *mut pg_sys::IndexVacuumInfo,
    stats: *mut pg_sys::IndexBulkDeleteResult,
) -> *mut pg_sys::IndexBulkDeleteResult {
    let _ = info;
    if stats.is_null() {
        pg_sys::palloc0(std::mem::size_of::<pg_sys::IndexBulkDeleteResult>())
            as *mut pg_sys::IndexBulkDeleteResult
    } else {
        stats
    }
}

/// amoptions callback: accept and pass through reloptions bytea.
///
/// PostgreSQL calls this during CREATE INDEX to validate WITH options.
/// We accept everything and return the raw bytea for `rd_options`.
/// Actual parsing happens in `extract_model_from_options`.
///
/// # Safety
///
/// Called by PostgreSQL with valid arguments.
#[pg_guard]
pub unsafe extern "C-unwind" fn infer_amoptions(
    reloptions: pg_sys::Datum,
    validate: bool,
) -> *mut pg_sys::bytea {
    // Use PostgreSQL's default_reloptions which stores the raw bytea.
    // We pass kind=RELOPT_KIND_INDEX and accept the default handling.
    // If validate is true, PostgreSQL will report errors for malformed input.
    let _ = validate;
    // Just pass through the raw reloptions.  If it's NULL, return NULL.
    if reloptions.is_null() {
        return std::ptr::null_mut();
    }
    reloptions.cast_mut_ptr::<pg_sys::bytea>()
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Resolve the model name for index creation.
///
/// Priority: reloptions `WITH (model = '...')` > `infer.default_model` GUC.
unsafe fn resolve_model_for_build(index: pg_sys::Relation) -> Result<String, PgInferError> {
    // Try to read model from reloptions.
    let rd_options = (*index).rd_options;
    if !rd_options.is_null() {
        if let Some(name) = extract_model_from_options(rd_options as *mut std::os::raw::c_void) {
            if !name.is_empty() {
                return Ok(name);
            }
        }
    }

    // Fall back to GUC.
    crate::gucs::default_model().ok_or(PgInferError::NoDefaultModel)
}

/// Extract model name from raw reloptions bytea.
///
/// The reloptions pointer comes from `rd_options` which is populated by
/// our `amoptions` callback.  It's a raw bytea containing "key=value"
/// text pairs.
unsafe fn extract_model_from_options(rd_options: *mut std::os::raw::c_void) -> Option<String> {
    if rd_options.is_null() {
        return None;
    }
    let datum = pg_sys::Datum::from(rd_options as usize);
    crate::am_options::parse_model_from_reloptions(datum)
}

/// Verify the model exists in `infer.models`.
fn verify_model_exists(model_name: &str) -> Result<(), PgInferError> {
    use pgrx::prelude::*;

    let exists = Spi::connect(|client| {
        let mut result = client.select(
            "SELECT EXISTS(SELECT 1 FROM infer.models WHERE model_name = $1)",
            Some(1),
            &[pgrx::datum::DatumWithOid::from(model_name)],
        )?;
        if let Some(row) = result.next() {
            return Ok::<_, pgrx::spi::SpiError>(row.get::<bool>(1)?.unwrap_or(false));
        }
        Ok(false)
    })?;

    if !exists {
        return Err(PgInferError::ModelNotFound {
            name: model_name.to_string(),
        });
    }
    Ok(())
}

/// Write the metapage to block 0 of the index using GenericXLog.
unsafe fn write_metapage(index: pg_sys::Relation, model_name: &str) -> Result<(), PgInferError> {
    let meta = InferMetaPage::new(model_name)?;

    // Get or extend the index to have at least one page.
    let nblocks = pg_sys::RelationGetNumberOfBlocksInFork(
        index,
        pg_sys::ForkNumber::MAIN_FORKNUM,
    );
    let buf = if nblocks == 0 {
        // Extend the relation to create block 0.
        // In PG18, P_NEW is represented as InvalidBlockNumber = 0xFFFFFFFF.
        let buf = pg_sys::ReadBufferExtended(
            index,
            pg_sys::ForkNumber::MAIN_FORKNUM,
            0xFFFF_FFFF_u32, // P_NEW / InvalidBlockNumber
            pg_sys::ReadBufferMode::RBM_NORMAL,
            std::ptr::null_mut(),
        );
        pg_sys::LockBuffer(buf, pg_sys::BUFFER_LOCK_EXCLUSIVE as i32);
        buf
    } else {
        let buf = pg_sys::ReadBuffer(index, 0);
        pg_sys::LockBuffer(buf, pg_sys::BUFFER_LOCK_EXCLUSIVE as i32);
        buf
    };

    // Use GenericXLog for WAL safety.
    let xlog_state = pg_sys::GenericXLogStart(index);
    let page = pg_sys::GenericXLogRegisterBuffer(xlog_state, buf, pg_sys::GENERIC_XLOG_FULL_IMAGE as i32);

    // Initialize the page.
    pg_sys::PageInit(page, pg_sys::BLCKSZ as usize, 0);

    // Write our metapage struct into the page's content area.
    let content = pg_sys::PageGetContents(page) as *mut InferMetaPage;
    std::ptr::write(content, meta);

    // Mark the page as having content of our size.
    // Set pd_lower to include our struct.
    let header = page as *mut pg_sys::PageHeaderData;
    (*header).pd_lower = (std::mem::size_of::<pg_sys::PageHeaderData>()
        + std::mem::size_of::<InferMetaPage>()) as u16;

    pg_sys::GenericXLogFinish(xlog_state);
    pg_sys::UnlockReleaseBuffer(buf);

    Ok(())
}

/// Check if an index has a valid infer metapage.
#[allow(dead_code)]
pub unsafe fn has_valid_metapage(index_rel: pg_sys::Relation) -> bool {
    let nblocks = pg_sys::RelationGetNumberOfBlocksInFork(
        index_rel,
        pg_sys::ForkNumber::MAIN_FORKNUM,
    );
    if nblocks == 0 {
        return false;
    }

    let buf = pg_sys::ReadBuffer(index_rel, 0);
    pg_sys::LockBuffer(buf, pg_sys::BUFFER_LOCK_SHARE as i32);

    let page = pg_sys::BufferGetPage(buf);
    let data = pg_sys::PageGetContents(page) as *const InferMetaPage;
    let valid = (*data).magic == INFER_META_MAGIC;

    pg_sys::UnlockReleaseBuffer(buf);
    valid
}
