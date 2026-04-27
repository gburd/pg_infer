use std::ffi::CString;

use pgrx::{GucContext, GucFlags, GucRegistry, GucSetting};

/// Session-level default model name.  When a query function omits the
/// `model` parameter it falls back to this GUC.
pub static DEFAULT_MODEL: GucSetting<Option<CString>> =
    GucSetting::<Option<CString>>::new(None);

/// Base directory for cached vindex files, relative to `$PGDATA` unless
/// an absolute path is given.
pub static DATA_DIRECTORY: GucSetting<Option<CString>> =
    GucSetting::<Option<CString>>::new(Some(c"infer"));

/// Maximum aggregate RSS (megabytes) for all loaded vindexes in this
/// backend.  Used for LRU eviction decisions.
pub static MAX_MEMORY_MB: GucSetting<i32> = GucSetting::<i32>::new(8192);

/// Whether `infer_create_model` may download from HuggingFace when the
/// source is a model ID or `hf://` URI.
pub static AUTO_DOWNLOAD: GucSetting<bool> = GucSetting::<bool>::new(true);

/// Gate score threshold for describe()/implies().
///
/// When set to 0 (the default), an adaptive threshold is computed from the
/// query's actual gate activations: `max_score × 0.1`.  A positive value
/// overrides with a fixed threshold.
pub static GATE_THRESHOLD: GucSetting<f64> = GucSetting::<f64>::new(0.0);

/// Top-K features per layer for describe().
pub static DESCRIBE_TOP_K: GucSetting<i32> = GucSetting::<i32>::new(20);

/// Embedding mode for walk(): "average" or "last".
pub static WALK_EMBED_MODE: GucSetting<Option<CString>> =
    GucSetting::<Option<CString>>::new(Some(c"average"));

/// Register all GUC parameters.
///
/// # Safety
///
/// Must be called exactly once, inside `_PG_init`.
pub unsafe fn init() {
    GucRegistry::define_string_guc(
        c"infer.default_model",
        c"Default model name for infer query functions.",
        c"When a query function omits the model parameter, this model is used. \
         Set with: SET infer.default_model = 'my_model';",
        &DEFAULT_MODEL,
        GucContext::Userset,
        GucFlags::default(),
    );

    GucRegistry::define_string_guc(
        c"infer.data_directory",
        c"Base directory for cached vindex files.",
        c"Relative to $PGDATA unless an absolute path. Default: 'infer'.",
        &DATA_DIRECTORY,
        GucContext::Sighup,
        GucFlags::default(),
    );

    GucRegistry::define_int_guc(
        c"infer.max_memory",
        c"Maximum memory (MB) for loaded vindexes per backend.",
        c"Used for LRU eviction. Default: 8192 (8 GB).",
        &MAX_MEMORY_MB,
        512,
        65536,
        GucContext::Sighup,
        GucFlags::default(),
    );

    GucRegistry::define_bool_guc(
        c"infer.auto_download",
        c"Allow automatic HuggingFace downloads.",
        c"When true, infer_create_model may download models from HF.",
        &AUTO_DOWNLOAD,
        GucContext::Suset,
        GucFlags::default(),
    );

    GucRegistry::define_float_guc(
        c"infer.gate_threshold",
        c"Gate score threshold for describe()/implies().",
        c"0 = adaptive (max_score * 0.1). A positive value is a fixed threshold.",
        &GATE_THRESHOLD,
        0.0,
        1000.0,
        GucContext::Userset,
        GucFlags::default(),
    );

    GucRegistry::define_int_guc(
        c"infer.describe_top_k",
        c"Top-K features per layer for describe().",
        c"Controls how many features per layer are examined. Default: 20.",
        &DESCRIBE_TOP_K,
        1,
        1000,
        GucContext::Userset,
        GucFlags::default(),
    );

    GucRegistry::define_string_guc(
        c"infer.walk_embed_mode",
        c"Embedding mode for walk(): 'average' or 'last'.",
        c"'average' averages all token embeddings; 'last' uses only the last token. Default: 'average'.",
        &WALK_EMBED_MODE,
        GucContext::Userset,
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
pub fn data_directory() -> String {
    DATA_DIRECTORY
        .get()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "infer".to_string())
}

/// Return the configured describe top-K value.
pub fn describe_top_k() -> usize {
    DESCRIBE_TOP_K.get() as usize
}

/// Return true if walk() should use last-token-only embedding.
pub fn walk_embed_mode_is_last() -> bool {
    WALK_EMBED_MODE
        .get()
        .map(|s| s.to_string_lossy() == "last")
        .unwrap_or(false)
}
