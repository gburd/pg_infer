use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use ndarray::Array2;
use once_cell::sync::Lazy;
use pgrx::datum::DatumWithOid;
use pgrx::prelude::*;

use infer_vindex::{FeatureMeta, SilentLoadCallbacks, VectorIndex, VindexConfig};

use crate::backend::{mmap::MmapBackend, remote::RemoteBackend, Backend};
use crate::error::PgInferError;
use crate::gucs;

// ---------------------------------------------------------------------------
// Per-backend (process-local) model handle
// ---------------------------------------------------------------------------

/// Everything needed to run gate-KNN / feature-meta queries against a
/// locally mmap'd vindex.
///
/// `MmapBackend` wraps an `Arc<ModelHandle>`; remote-only backends don't
/// hold one.  Everything in this struct is demand-paged from disk.
pub struct ModelHandle {
    pub embeddings: Array2<f32>,
    pub embed_scale: f32,
    pub tokenizer: infer_vindex::tokenizers::Tokenizer,
    pub config: VindexConfig,
    #[cfg_attr(not(feature = "inference"), allow(dead_code))]
    pub path: PathBuf,
    backend: ModelBackend,
}

enum ModelBackend {
    Mmap { vindex: VectorIndex },
}

// SAFETY: VectorIndex uses mmap with raw pointers.  It is safe to send
// across threads (the mmap region is process-wide), and the PG backend is
// single-threaded.  `BACKEND_CACHE` requires `Send`.
unsafe impl Send for ModelBackend {}
unsafe impl Sync for ModelBackend {}

impl ModelHandle {
    pub fn gate_knn(
        &self,
        layer: usize,
        query: &ndarray::Array1<f32>,
        top_k: usize,
    ) -> Vec<(usize, f32)> {
        match &self.backend {
            ModelBackend::Mmap { vindex } => vindex.gate_knn(layer, query, top_k),
        }
    }

    pub fn num_features(&self, layer: usize) -> usize {
        match &self.backend {
            ModelBackend::Mmap { vindex } => vindex.num_features(layer),
        }
    }

    pub fn feature_meta(&self, layer: usize, feature_idx: usize) -> Option<FeatureMeta> {
        match &self.backend {
            ModelBackend::Mmap { vindex } => vindex.feature_meta(layer, feature_idx),
        }
    }
}

// ---------------------------------------------------------------------------
// Backend cache
// ---------------------------------------------------------------------------

/// Process-local cache of loaded backends, keyed by model name.
///
/// One entry per model.  MmapBackend entries share underlying mmap pages
/// across PG backends via the kernel page cache; RemoteBackend entries
/// each own a dedicated HTTP runtime thread.
static BACKEND_CACHE: Lazy<Mutex<HashMap<String, Arc<dyn Backend>>>> =
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

/// Run a closure against the named model's backend.
///
/// On cache miss the backend is loaded according to the model's row in
/// `infer.models` (`backend = 'local'` → mmap, `backend = 'remote'` →
/// `RemoteBackend::connect`).  The `Arc` is cloned and the mutex is
/// released before calling `f`, so the closure does not block other
/// model access in the backend.
pub fn with_backend<F, R>(model_name: &str, f: F) -> Result<R, PgInferError>
where
    F: FnOnce(&dyn Backend) -> Result<R, PgInferError>,
{
    let handle = {
        let mut cache = BACKEND_CACHE
            .lock()
            .map_err(|e| PgInferError::Internal(format!("cache lock poisoned: {}", e)))?;

        if !cache.contains_key(model_name) {
            cache.insert(model_name.to_string(), load_backend(model_name)?);
        }

        Arc::clone(
            cache
                .get(model_name)
                .ok_or_else(|| PgInferError::Internal("cache entry missing after insert".into()))?,
        )
    }; // lock released

    f(&*handle)
}

/// Evict a model from the process-local cache.
pub fn evict(model_name: &str) {
    if let Ok(mut cache) = BACKEND_CACHE.lock() {
        cache.remove(model_name);
    }
}

// ---------------------------------------------------------------------------
// Model loading
// ---------------------------------------------------------------------------

/// Row-subset describing how to instantiate a backend.
struct BackendRow {
    vindex_path: String,
    backend: String,
    server_url: Option<String>,
}

fn fetch_backend_row(model_name: &str) -> Result<BackendRow, PgInferError> {
    let row = Spi::connect(|client| {
        let result = client.select(
            "SELECT vindex_path, backend, server_url \
             FROM infer.models WHERE model_name = $1",
            Some(1),
            &[DatumWithOid::from(model_name)],
        )?;

        for row in result {
            let vindex_path: String = row.get(1)?.unwrap_or_default();
            let backend: String = row.get(2)?.unwrap_or_else(|| "local".to_string());
            let server_url: Option<String> = row.get(3)?;
            return Ok::<_, pgrx::spi::SpiError>(Some(BackendRow {
                vindex_path,
                backend,
                server_url,
            }));
        }
        Ok(None)
    })?;

    row.ok_or_else(|| PgInferError::ModelNotFound {
        name: model_name.to_string(),
    })
}

fn load_backend(model_name: &str) -> Result<Arc<dyn Backend>, PgInferError> {
    let row = fetch_backend_row(model_name)?;

    match row.backend.as_str() {
        "remote" => {
            let url = row
                .server_url
                .or_else(gucs::default_server_url)
                .ok_or_else(|| {
                    PgInferError::Internal(format!(
                        "model '{}' uses backend='remote' but no server_url \
                         is set (neither on the row nor in \
                         infer.default_server_url)",
                        model_name
                    ))
                })?;
            let backend = RemoteBackend::connect(&url, gucs::remote_timeout())?;
            Ok(Arc::new(backend))
        }
        "local" | "" => {
            let handle = Arc::new(load_from_path(Path::new(&row.vindex_path))?);
            Ok(Arc::new(MmapBackend::new(handle)))
        }
        other => Err(PgInferError::Internal(format!(
            "unknown backend '{}' for model '{}' (expected 'local' or 'remote')",
            other, model_name
        ))),
    }
}

/// Load a `ModelHandle` directly from a vindex directory path (mmap).
pub fn load_from_path(path: &Path) -> Result<ModelHandle, PgInferError> {
    let mut callbacks = SilentLoadCallbacks;

    // Load the vindex (gate vectors + metadata, mmap'd).
    let vindex = VectorIndex::load_vindex(path, &mut callbacks)?;

    // Cap the f16 decode cache to prevent unbounded memory growth.
    // Each decoded layer is ~112 MB for a 3B model (14336 × 2048 × 4).
    let cache_cap = gucs::gate_cache_max_layers();
    if cache_cap > 0 {
        vindex.set_gate_cache_max_layers(cache_cap);
    }

    // Pre-decode f16 → f32 for all layers if warmup is enabled.
    if gucs::warmup_on_load() {
        vindex.warmup();
    }

    // Enable HNSW approximate search if configured.
    if gucs::use_hnsw() {
        vindex.enable_hnsw(gucs::hnsw_ef_search());

        if gucs::build_hnsw_on_load() {
            pgrx::log!("Building HNSW indexes for all {} layers...", vindex.num_layers);
            for layer in 0..vindex.num_layers {
                let dummy_vec = ndarray::Array1::<f32>::zeros(vindex.hidden_size);
                let _ = vindex.gate_knn(layer, &dummy_vec, 1);

                if layer > 0 && (layer + 1) % 10 == 0 {
                    pgrx::log!("  Built HNSW for {}/{} layers", layer + 1, vindex.num_layers);
                }
            }
            pgrx::log!("HNSW build complete for all {} layers", vindex.num_layers);
        }
    }

    let config = infer_vindex::load_vindex_config(path)?;
    let (embeddings, embed_scale) = infer_vindex::load_vindex_embeddings(path)?;
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
