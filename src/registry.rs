use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use ndarray::{Array1, Array2};
use once_cell::sync::Lazy;
use pgrx::datum::DatumWithOid;
use pgrx::prelude::*;

use infer_vindex::{FeatureMeta, SilentLoadCallbacks, VectorIndex, VindexConfig};

use crate::error::PgInferError;
use crate::gucs;

// ---------------------------------------------------------------------------
// Per-backend (process-local) model handle cache
// ---------------------------------------------------------------------------

/// Everything needed to service walk / describe / similar queries against a
/// single vindex.
pub struct ModelHandle {
    pub embeddings: Array2<f32>,
    pub embed_scale: f32,
    pub tokenizer: infer_vindex::tokenizers::Tokenizer,
    pub config: VindexConfig,
    #[cfg_attr(not(feature = "inference"), allow(dead_code))]
    pub path: PathBuf,
    /// Backing storage for gate vectors and feature metadata.
    backend: ModelBackend,
}

/// Discriminator for the storage backend.
///
/// Phase A retains only the mmap backend.  Phase C will add a `Remote`
/// variant that delegates to a `larql-server` endpoint.
enum ModelBackend {
    /// Mmap-based backend (loaded from a vindex directory).  Zero-copy gate
    /// access, HNSW support, demand paging.
    Mmap { vindex: VectorIndex },
}

// SAFETY: VectorIndex uses mmap with raw pointers.  It is safe to send
// across threads (the mmap region is process-wide), and the PG backend is
// single-threaded.  `HANDLE_CACHE` requires `Send`.
unsafe impl Send for ModelBackend {}
unsafe impl Sync for ModelBackend {}

impl ModelHandle {
    /// Compute gate KNN for a layer, dispatching to the appropriate backend.
    pub fn gate_knn(
        &self,
        layer: usize,
        query: &Array1<f32>,
        top_k: usize,
    ) -> Vec<(usize, f32)> {
        match &self.backend {
            ModelBackend::Mmap { vindex } => vindex.gate_knn(layer, query, top_k),
        }
    }

    /// Number of features indexed at a given layer.
    pub fn num_features(&self, layer: usize) -> usize {
        match &self.backend {
            ModelBackend::Mmap { vindex } => vindex.num_features(layer),
        }
    }

    /// Look up metadata for a single feature, dispatching to the appropriate
    /// backend.
    pub fn feature_meta(&self, layer: usize, feature_idx: usize) -> Option<FeatureMeta> {
        match &self.backend {
            ModelBackend::Mmap { vindex } => vindex.feature_meta(layer, feature_idx),
        }
    }
}

/// Process-local cache of loaded model handles, keyed by model name.
///
/// PostgreSQL forks one backend per connection.  Each backend maintains its
/// own `HashMap`, but the OS kernel shares the underlying mmap pages across
/// all backends that have the same vindex open.
static HANDLE_CACHE: Lazy<Mutex<HashMap<String, Arc<ModelHandle>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

/// Resolve the model name for a query function.
///
/// If `explicit` is `Some`, use that.  Otherwise fall back to the
/// `infer.default_model` GUC.
pub fn resolve_model_name(explicit: Option<&str>) -> Result<String, PgInferError> {
    if let Some(name) = explicit {
        return Ok(name.to_string());
    }
    gucs::default_model().ok_or(PgInferError::NoDefaultModel)
}

/// Obtain a reference to the cached `ModelHandle` for `model_name`.
///
/// On cache miss the handle is loaded via the `infer.models` registry
/// table.  The `Arc` is cloned and the mutex is released before calling
/// `f`, so the closure does not block other model access in the backend.
pub fn with_model<F, R>(model_name: &str, f: F) -> Result<R, PgInferError>
where
    F: FnOnce(&ModelHandle) -> Result<R, PgInferError>,
{
    let handle = {
        let mut cache = HANDLE_CACHE
            .lock()
            .map_err(|e| PgInferError::Internal(format!("handle cache lock poisoned: {}", e)))?;

        if !cache.contains_key(model_name) {
            cache.insert(model_name.to_string(), Arc::new(load_model(model_name)?));
        }

        Arc::clone(
            cache
                .get(model_name)
                .ok_or_else(|| PgInferError::Internal("cache entry missing after insert".into()))?,
        )
    }; // lock released

    f(&handle)
}

/// Evict a model from the process-local cache.
pub fn evict(model_name: &str) {
    if let Ok(mut cache) = HANDLE_CACHE.lock() {
        cache.remove(model_name);
    }
}

// ---------------------------------------------------------------------------
// Model loading
// ---------------------------------------------------------------------------

/// Load a model by name from the `infer.models` registry table.
fn load_model(model_name: &str) -> Result<ModelHandle, PgInferError> {
    let vindex_path: String = Spi::get_one_with_args(
        "SELECT vindex_path FROM infer.models WHERE model_name = $1",
        &[DatumWithOid::from(model_name)],
    )?
    .ok_or_else(|| PgInferError::ModelNotFound {
        name: model_name.to_string(),
    })?;

    load_from_path(Path::new(&vindex_path))
}

/// Load a `ModelHandle` directly from a vindex directory path (mmap).
pub fn load_from_path(path: &Path) -> Result<ModelHandle, PgInferError> {
    let mut callbacks = SilentLoadCallbacks;

    // Load the vindex (gate vectors + metadata, mmap'd).
    let vindex = VectorIndex::load_vindex(path, &mut callbacks)?;

    // Pre-decode f16 → f32 for all layers if warmup is enabled.
    if gucs::warmup_on_load() {
        vindex.warmup();
    }

    // Enable HNSW approximate search if configured.
    if gucs::use_hnsw() {
        vindex.enable_hnsw(gucs::hnsw_ef_search());

        // Eagerly build HNSW indexes for all layers if configured.
        // This moves the HNSW build cost from first query to registration,
        // making all queries fast and predictable.
        if gucs::build_hnsw_on_load() {
            pgrx::log!("Building HNSW indexes for all {} layers...", vindex.num_layers);
            for layer in 0..vindex.num_layers {
                // Trigger HNSW build by calling gate_knn once per layer.
                // The get_or_build_hnsw() path will build and cache the index.
                let dummy_vec = ndarray::Array1::<f32>::zeros(vindex.hidden_size);
                let _ = vindex.gate_knn(layer, &dummy_vec, 1);

                // Log progress for large models
                if layer > 0 && (layer + 1) % 10 == 0 {
                    pgrx::log!("  Built HNSW for {}/{} layers", layer + 1, vindex.num_layers);
                }
            }
            pgrx::log!("HNSW build complete for all {} layers", vindex.num_layers);
        }
    }

    // Load the vindex configuration (layer count, hidden size, etc.)
    let config = infer_vindex::load_vindex_config(path)?;

    // Load token embeddings and the embedding scale factor.
    let (embeddings, embed_scale) = infer_vindex::load_vindex_embeddings(path)?;

    // Load the tokenizer.
    let tokenizer = infer_vindex::load_vindex_tokenizer(path)?;

    Ok(ModelHandle {
        embeddings,
        embed_scale,
        tokenizer,
        config,
        path: path.to_path_buf(),
        backend: ModelBackend::Mmap { vindex },
    })
}
