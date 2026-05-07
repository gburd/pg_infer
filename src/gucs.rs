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
/// Default 5.0 (matches LARQL).  Set to 0 for adaptive mode
/// (`max_score × 0.1`).  A positive value is a fixed threshold.
pub static GATE_THRESHOLD: GucSetting<f64> = GucSetting::<f64>::new(5.0);

/// Top-K features per layer for describe().
pub static DESCRIBE_TOP_K: GucSetting<i32> = GucSetting::<i32>::new(20);

/// Embedding mode for walk(): "average" or "last".
///
/// "last" matches the LARQL CLI behavior: use only the last token's
/// embedding as the query vector.  This produces stronger, more
/// interpretable activations because transformers build up a
/// representation across tokens — the last position captures the full
/// context.  "average" averages all token embeddings (including any
/// special tokens), which dilutes the signal for longer prompts.
pub static WALK_EMBED_MODE: GucSetting<Option<CString>> =
    GucSetting::<Option<CString>>::new(Some(c"last"));

/// Whether to enable HNSW approximate search for gate_knn queries.
///
/// WARNING: HNSW can cause crashes on large models due to memory pressure.
/// Disabled by default. Use layer sampling (infer.similarity_max_layers)
/// for better performance without memory issues.
pub static USE_HNSW: GucSetting<bool> = GucSetting::<bool>::new(false);

/// HNSW beam width (ef_search). Higher values are more accurate but slower.
pub static HNSW_EF_SEARCH: GucSetting<i32> = GucSetting::<i32>::new(200);

/// Whether to pre-decode f16 gate vectors to f32 on model load.
///
/// WARNING: Warmup decodes all layers to f32 in RAM, which can consume
/// 2-3GB per model. Disabled by default to avoid out-of-memory crashes.
/// Layers are decoded lazily on first use and cached.
pub static WARMUP_ON_LOAD: GucSetting<bool> = GucSetting::<bool>::new(false);

/// Whether to build HNSW indexes during model load.
///
/// When true, all HNSW indexes are built during `infer_create_model()`,
/// making the first similarity query fast. When false (default), HNSW is
/// built lazily on first use (slower first query, faster registration).
///
/// WARNING: Eager build can cause crashes on large models. Use layer
/// sampling (infer.similarity_max_layers) instead for better performance.
pub static BUILD_HNSW_ON_LOAD: GucSetting<bool> = GucSetting::<bool>::new(false);

/// Maximum layers to query in similar_to() for performance.
///
/// When a model has more layers than this value, sample evenly across
/// layers instead of querying all. Set to 0 for no limit (query all
/// layers). Lower values trade some accuracy for speed.
pub static SIMILARITY_MAX_LAYERS: GucSetting<i32> = GucSetting::<i32>::new(0);

/// Whether to use parallel processing for similarity queries.
///
/// When true, similar_to() queries layers in parallel using Rayon.
/// Provides 4-8x speedup on multi-core systems but increases CPU usage.
pub static PARALLEL_SIMILARITY: GucSetting<bool> = GucSetting::<bool>::new(false);

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
        GucContext::Suset,
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
        c"Default 5.0 (matches LARQL). Set to 0 for adaptive (max_score * 0.1).",
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
        c"'last' uses the last token (matches LARQL); 'average' averages all tokens. Default: 'last'.",
        &WALK_EMBED_MODE,
        GucContext::Userset,
        GucFlags::default(),
    );

    GucRegistry::define_bool_guc(
        c"infer.use_hnsw",
        c"Enable HNSW approximate search for gate queries.",
        c"When true, gate_knn uses HNSW index for O(log N) search. Default: true.",
        &USE_HNSW,
        GucContext::Userset,
        GucFlags::default(),
    );

    GucRegistry::define_int_guc(
        c"infer.hnsw_ef_search",
        c"HNSW beam width for approximate search.",
        c"Higher values are more accurate but slower. Default: 200.",
        &HNSW_EF_SEARCH,
        50,
        500,
        GucContext::Userset,
        GucFlags::default(),
    );

    GucRegistry::define_bool_guc(
        c"infer.warmup_on_load",
        c"Pre-decode f16 gate vectors on model load.",
        c"When true, f16 gates are decoded to f32 on first load for faster queries. Default: true.",
        &WARMUP_ON_LOAD,
        GucContext::Userset,
        GucFlags::default(),
    );

    GucRegistry::define_bool_guc(
        c"infer.build_hnsw_on_load",
        c"Build HNSW indexes during model load.",
        c"When true, HNSW is built during registration for fast first query. Default: true.",
        &BUILD_HNSW_ON_LOAD,
        GucContext::Userset,
        GucFlags::default(),
    );

    GucRegistry::define_int_guc(
        c"infer.similarity_max_layers",
        c"Maximum layers to query in similar_to().",
        c"Query at most this many layers (sampled evenly). 0 = all layers. Default: 0.",
        &SIMILARITY_MAX_LAYERS,
        0,
        100,
        GucContext::Userset,
        GucFlags::default(),
    );

    GucRegistry::define_bool_guc(
        c"infer.parallel_similarity",
        c"Use parallel processing for similarity queries.",
        c"When true, query layers in parallel for 4-8x speedup. Default: false.",
        &PARALLEL_SIMILARITY,
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

/// Return true if HNSW approximate search is enabled.
pub fn use_hnsw() -> bool {
    USE_HNSW.get()
}

/// Return the HNSW ef_search beam width.
pub fn hnsw_ef_search() -> usize {
    HNSW_EF_SEARCH.get() as usize
}

/// Return true if warmup-on-load is enabled.
pub fn warmup_on_load() -> bool {
    WARMUP_ON_LOAD.get()
}

/// Return true if HNSW should be built during model load.
pub fn build_hnsw_on_load() -> bool {
    BUILD_HNSW_ON_LOAD.get()
}

/// Return the maximum layers to query in similar_to(). 0 = no limit.
pub fn similarity_max_layers() -> usize {
    SIMILARITY_MAX_LAYERS.get() as usize
}

/// Return true if parallel processing is enabled for similarity queries.
pub fn parallel_similarity() -> bool {
    PARALLEL_SIMILARITY.get()
}
