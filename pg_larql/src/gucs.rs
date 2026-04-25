use std::ffi::CString;

use pgrx::{GucContext, GucFlags, GucRegistry, GucSetting};

/// Session-level default model name.  When a query function omits the
/// `model` parameter it falls back to this GUC.
pub static DEFAULT_MODEL: GucSetting<Option<CString>> =
    GucSetting::<Option<CString>>::new(None);

/// Base directory for cached vindex files, relative to `$PGDATA` unless
/// an absolute path is given.
pub static DATA_DIRECTORY: GucSetting<Option<CString>> =
    GucSetting::<Option<CString>>::new(Some(c"larql"));

/// Maximum aggregate RSS (megabytes) for all loaded vindexes in this
/// backend.  Used for LRU eviction decisions.
pub static MAX_MEMORY_MB: GucSetting<i32> = GucSetting::<i32>::new(8192);

/// Whether `larql_create_model` may download from HuggingFace when the
/// source is a model ID or `hf://` URI.
pub static AUTO_DOWNLOAD: GucSetting<bool> = GucSetting::<bool>::new(true);

/// Register all GUC parameters.
///
/// # Safety
///
/// Must be called exactly once, inside `_PG_init`.
pub unsafe fn init() {
    GucRegistry::define_string_guc(
        c"larql.default_model",
        c"Default model name for LARQL query functions.",
        c"When a query function omits the model parameter, this model is used. \
         Set with: SET larql.default_model = 'my_model';",
        &DEFAULT_MODEL,
        GucContext::Userset,
        GucFlags::default(),
    );

    GucRegistry::define_string_guc(
        c"larql.data_directory",
        c"Base directory for cached vindex files.",
        c"Relative to $PGDATA unless an absolute path. Default: 'larql'.",
        &DATA_DIRECTORY,
        GucContext::Sighup,
        GucFlags::default(),
    );

    GucRegistry::define_int_guc(
        c"larql.max_memory",
        c"Maximum memory (MB) for loaded vindexes per backend.",
        c"Used for LRU eviction. Default: 8192 (8 GB).",
        &MAX_MEMORY_MB,
        512,
        65536,
        GucContext::Sighup,
        GucFlags::default(),
    );

    GucRegistry::define_bool_guc(
        c"larql.auto_download",
        c"Allow automatic HuggingFace downloads.",
        c"When true, larql_create_model may download models from HF.",
        &AUTO_DOWNLOAD,
        GucContext::Suset,
        GucFlags::default(),
    );
}

// ---------------------------------------------------------------------------
// Convenience accessors
// ---------------------------------------------------------------------------

/// Return the resolved default model name, or `None` if unset.
pub fn default_model() -> Option<String> {
    DEFAULT_MODEL.get().map(|s| s.to_string_lossy().into_owned())
}

/// Return the configured data directory (never `None` in practice).
#[allow(dead_code)]
pub fn data_directory() -> String {
    DATA_DIRECTORY
        .get()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "larql".to_string())
}
