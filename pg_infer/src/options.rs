//! Reloptions parsing for `WITH (source = '...', model = '...')`.
//!
//! We validate options manually rather than using PG's `build_reloptions`
//! because the latter crashes (SIGABRT) in PG18 with custom AM kinds in
//! pgrx test contexts.  The manual approach parses the `text[]` datum
//! directly and rejects unrecognized keys.

use pgrx::pg_sys;

/// Known WITH option names for the infer AM.
const KNOWN_OPTIONS: &[&str] = &["source", "model"];

// ---------------------------------------------------------------------------
// Registration (called from _PG_init) — placeholder for future use
// ---------------------------------------------------------------------------

/// Placeholder for reloption registration.
///
/// # Safety
///
/// Must be called exactly once, inside `_PG_init`.
pub unsafe fn register_reloptions() {
    // Manual validation is performed in infer_amoptions_impl instead of
    // using PG's global relopt registration (build_reloptions).
}

// ---------------------------------------------------------------------------
// amoptions implementation
// ---------------------------------------------------------------------------

/// Parse and validate WITH options.
///
/// Checks that all provided options are in the known set (source, model).
/// Rejects unrecognized options with a PG ERROR when `validate` is true.
///
/// Returns null — actual option extraction happens via SPI in
/// `get_source_option` / `get_model_option` during ambuild.
///
/// # Safety
///
/// Called from the amoptions AM callback.
pub unsafe fn infer_amoptions_impl(
    reloptions: pg_sys::Datum,
    validate: bool,
) -> *mut pg_sys::bytea {
    if validate && reloptions.value() != 0 {
        // reloptions is a text[] (Datum pointing to an ArrayType).
        // Use deconstruct_array to iterate the elements.
        let arr = reloptions.cast_mut_ptr::<pg_sys::ArrayType>();
        if !arr.is_null() {
            let mut nelems: i32 = 0;
            let mut elems: *mut pg_sys::Datum = std::ptr::null_mut();
            let mut nulls: *mut bool = std::ptr::null_mut();

            pg_sys::deconstruct_array(
                arr,
                pg_sys::TEXTOID,
                -1,   // typlen for text (varlena)
                false, // typbyval
                pg_sys::TYPALIGN_INT as std::ffi::c_char, // typalign
                &mut elems,
                &mut nulls,
                &mut nelems,
            );

            for i in 0..nelems as usize {
                if !nulls.is_null() && *nulls.add(i) {
                    continue;
                }
                let datum = *elems.add(i);
                let cstr = pg_sys::text_to_cstring(
                    datum.cast_mut_ptr::<pg_sys::text>(),
                );
                if cstr.is_null() {
                    continue;
                }
                let opt_str = std::ffi::CStr::from_ptr(cstr)
                    .to_str()
                    .unwrap_or("");

                // Each element is "key=value"; extract the key.
                let key = opt_str.split('=').next().unwrap_or("");

                if !KNOWN_OPTIONS.contains(&key) {
                    pg_sys::pfree(cstr as *mut std::ffi::c_void);
                    pgrx::error!(
                        "unrecognized parameter \"{}\" for infer index",
                        key
                    );
                }

                pg_sys::pfree(cstr as *mut std::ffi::c_void);
            }
        }
    }

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

    pgrx::log!("get_reloption: looking for key '{}' in relation OID {}", key, rel_oid);

    let query = format!(
        "SELECT unnest(reloptions) FROM pg_class WHERE oid = {}",
        rel_oid
    );

    pgrx::log!("get_reloption: executing SPI query: {}", query);

    let result = pgrx::Spi::connect(|client| {
        pgrx::log!("get_reloption: SPI connected");
        let result = client.select(&query, None, &[]);
        pgrx::log!("get_reloption: SPI select completed");
        match result {
            Ok(table) => {
                pgrx::log!("get_reloption: processing {} rows", table.len());
                let mut value = None;
                for row in table {
                    if let Ok(Some(opt)) = row.get::<String>(1) {
                        pgrx::log!("get_reloption: found option: {}", opt);
                        if let Some(val) = opt.strip_prefix(&prefix) {
                            value = Some(val.to_string());
                        }
                    }
                }
                Ok::<_, pgrx::spi::SpiError>(value)
            }
            Err(e) => {
                pgrx::log!("get_reloption: SPI error: {:?}", e);
                Err(e)
            }
        }
    });

    pgrx::log!("get_reloption: result = {:?}", result);

    match result {
        Ok(s) => s,
        Err(_) => None,
    }
}
