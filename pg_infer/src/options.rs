//! Reloptions parsing for `WITH (source = '...', extract_level = '...')`.

use pgrx::pg_sys;

// ---------------------------------------------------------------------------
// Reloptions definition
// ---------------------------------------------------------------------------

/// Our custom reloptions struct stored in `rd_options`.
#[repr(C)]
#[allow(dead_code)]
pub struct InferOptions {
    /// Standard reloptions header (vl_len_ field).
    pub vl_len_: i32,
    /// Offset into this struct where the source string starts.
    pub source_offset: i32,
    /// Offset for extract_level string.
    pub extract_level_offset: i32,
}

// ---------------------------------------------------------------------------
// amoptions implementation
// ---------------------------------------------------------------------------

/// Parse WITH options.  We accept arbitrary options through PostgreSQL's
/// standard reloptions mechanism.  The `source` option is extracted during
/// ambuild via SPI query on pg_class.reloptions.
///
/// # Safety
///
/// Called from the amoptions AM callback.
pub unsafe fn infer_amoptions_impl(
    reloptions: pg_sys::Datum,
    validate: bool,
) -> *mut pg_sys::bytea {
    // For custom index AMs that don't use standard relopt parsing,
    // we can return NULL to accept any options.  The actual parsing
    // happens in get_source_option via SPI.
    let _ = (reloptions, validate);
    std::ptr::null_mut()
}

// ---------------------------------------------------------------------------
// Option extraction
// ---------------------------------------------------------------------------

/// Extract the `source` option from an index relation's reloptions.
///
/// This is called during ambuild to find the vindex path.
///
/// # Safety
///
/// `index_relation` must be a valid open relation.
pub unsafe fn get_source_option(index_relation: pg_sys::Relation) -> Option<String> {
    get_reloption(index_relation, "source")
}

/// Extract the `model` option from an index relation's reloptions.
///
/// This is called during ambuild for column indexes to find the model name.
///
/// # Safety
///
/// `index_relation` must be a valid open relation.
pub unsafe fn get_model_option(index_relation: pg_sys::Relation) -> Option<String> {
    get_reloption(index_relation, "model")
}

/// Generic helper to extract a named option from a relation's reloptions.
///
/// # Safety
///
/// `index_relation` must be a valid open relation.
unsafe fn get_reloption(index_relation: pg_sys::Relation, key: &str) -> Option<String> {
    let rel_oid = (*index_relation).rd_id;
    let prefix = format!("{}=", key);

    let query = format!(
        "SELECT unnest(reloptions) FROM pg_class WHERE oid = {}",
        rel_oid
    );

    let result = pgrx::Spi::connect(|client| {
        let result = client.select(&query, None, &[]);
        match result {
            Ok(table) => {
                let mut value = None;
                for row in table {
                    if let Ok(Some(opt)) = row.get::<String>(1) {
                        if let Some(val) = opt.strip_prefix(&prefix) {
                            value = Some(val.to_string());
                        }
                    }
                }
                Ok::<_, pgrx::spi::SpiError>(value)
            }
            Err(e) => Err(e),
        }
    });

    match result {
        Ok(s) => s,
        Err(_) => None,
    }
}
