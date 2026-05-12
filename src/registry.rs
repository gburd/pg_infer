use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use ndarray::Array2;
use once_cell::sync::Lazy;
use pgrx::datum::DatumWithOid;
use pgrx::prelude::*;

use infer_vindex::{FeatureMeta, SilentLoadCallbacks, VectorIndex, VindexConfig};

use crate::backend::{grid::GridBackend, mmap::MmapBackend, remote::RemoteBackend, Backend};
use crate::error::PgInferError;
use crate::gucs;

// ---------------------------------------------------------------------------
// Observability counters
// ---------------------------------------------------------------------------

static CACHE_HITS: AtomicU64 = AtomicU64::new(0);
static CACHE_MISSES: AtomicU64 = AtomicU64::new(0);
static TOTAL_QUERIES: AtomicU64 = AtomicU64::new(0);

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

    pub fn approx_resident_bytes(&self) -> usize {
        match &self.backend {
            ModelBackend::Mmap { vindex } => vindex.approx_resident_bytes(),
        }
    }
}

// ---------------------------------------------------------------------------
// Backend cache with LRU eviction
// ---------------------------------------------------------------------------

struct CacheEntry {
    backend: Arc<dyn Backend>,
    resident_bytes: usize,
}

/// Process-local cache of loaded backends, keyed by model name.
///
/// One entry per model.  MmapBackend entries share underlying mmap pages
/// across PG backends via the kernel page cache; RemoteBackend entries
/// each own a dedicated HTTP runtime thread.
///
/// Enforces the `infer.max_memory` GUC by tracking approximate resident
/// bytes and evicting least-recently-used entries when over budget.
struct BackendCache {
    entries: HashMap<String, CacheEntry>,
    /// LRU order — front is oldest access, back is newest.
    lru: Vec<String>,
    total_bytes: usize,
}

impl BackendCache {
    fn new() -> Self {
        Self {
            entries: HashMap::new(),
            lru: Vec::new(),
            total_bytes: 0,
        }
    }

    /// Move `name` to the back (most recently used) of the LRU list.
    fn touch(&mut self, name: &str) {
        if let Some(pos) = self.lru.iter().position(|n| n == name) {
            self.lru.remove(pos);
        }
        self.lru.push(name.to_string());
    }

    /// Insert a new backend and evict LRU entries if over budget.
    fn insert(&mut self, name: String, backend: Arc<dyn Backend>) {
        let resident_bytes = backend.approx_resident_bytes();

        // Remove old entry if re-inserting (reload scenario).
        if let Some(old) = self.entries.remove(&name) {
            self.total_bytes = self.total_bytes.saturating_sub(old.resident_bytes);
            self.lru.retain(|n| n != &name);
        }

        self.entries.insert(
            name.clone(),
            CacheEntry {
                backend,
                resident_bytes,
            },
        );
        self.total_bytes += resident_bytes;
        self.lru.push(name.clone());

        self.evict_if_needed(&name);
    }

    /// Evict oldest entries until under budget, skipping `keep`.
    fn evict_if_needed(&mut self, keep: &str) {
        let max_bytes = gucs::MAX_MEMORY_MB.get() as usize * 1024 * 1024;
        if max_bytes == 0 {
            return; // unlimited
        }

        while self.total_bytes > max_bytes && self.lru.len() > 1 {
            // Find the oldest entry that isn't the one we just accessed.
            let victim = match self.lru.iter().find(|n| *n != keep).cloned() {
                Some(v) => v,
                None => break,
            };
            self.remove(&victim);
        }

        // Warn if still over budget (single model exceeds limit).
        if self.total_bytes > max_bytes {
            tracing::warn!(
                model = %keep,
                budget_mb = max_bytes / (1024 * 1024),
                actual_mb = self.total_bytes / (1024 * 1024),
                "model exceeds infer.max_memory budget; consider increasing the limit"
            );
        }
    }

    fn remove(&mut self, name: &str) {
        if let Some(entry) = self.entries.remove(name) {
            self.total_bytes = self.total_bytes.saturating_sub(entry.resident_bytes);
            // Drop the Arc — if this was the last reference, the backend's
            // Drop handler runs (e.g. CancellableClient sends Cmd::Shutdown).
        }
        self.lru.retain(|n| n != name);
    }

    fn get(&self, name: &str) -> Option<&Arc<dyn Backend>> {
        self.entries.get(name).map(|e| &e.backend)
    }

    fn contains_key(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }
}

static BACKEND_CACHE: Lazy<Mutex<BackendCache>> =
    Lazy::new(|| Mutex::new(BackendCache::new()));

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
    let _span = tracing::info_span!("with_backend", model = %model_name).entered();

    TOTAL_QUERIES.fetch_add(1, Ordering::Relaxed);

    let handle = {
        let mut cache = BACKEND_CACHE
            .lock()
            .map_err(|e| PgInferError::Internal(format!("cache lock poisoned: {}", e)))?;

        if !cache.contains_key(model_name) {
            CACHE_MISSES.fetch_add(1, Ordering::Relaxed);
            tracing::debug!("cache miss, loading backend");
            let backend = load_backend(model_name)?;
            cache.insert(model_name.to_string(), backend);
        } else {
            CACHE_HITS.fetch_add(1, Ordering::Relaxed);
            cache.touch(model_name);
        }

        Arc::clone(
            cache
                .get(model_name)
                .ok_or_else(|| PgInferError::Internal("cache entry missing after insert".into()))?,
        )
    }; // lock released

    f(&*handle)
}

/// Return cache statistics: (hits, misses, total_queries, loaded_models, total_bytes).
pub fn cache_stats() -> (u64, u64, u64, usize, usize) {
    let hits = CACHE_HITS.load(Ordering::Relaxed);
    let misses = CACHE_MISSES.load(Ordering::Relaxed);
    let total = TOTAL_QUERIES.load(Ordering::Relaxed);
    let (models, bytes) = BACKEND_CACHE
        .lock()
        .map(|c| (c.entries.len(), c.total_bytes))
        .unwrap_or((0, 0));
    (hits, misses, total, models, bytes)
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
        let mut result = client.select(
            "SELECT vindex_path, backend, server_url \
             FROM infer.models WHERE model_name = $1",
            Some(1),
            &[DatumWithOid::from(model_name)],
        )?;

        if let Some(row) = result.next() {
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
        "grid" => {
            let grid_url = gucs::grid_url().ok_or_else(|| {
                PgInferError::Internal(format!(
                    "model '{}' uses backend='grid' but infer.grid_url is not set",
                    model_name
                ))
            })?;
            let model_id = row
                .server_url
                .unwrap_or_else(|| model_name.to_string());
            let backend = GridBackend::connect(
                &model_id,
                &grid_url,
                gucs::grid_poll_interval(),
                gucs::remote_timeout(),
            )?;
            Ok(Arc::new(backend))
        }
        other => Err(PgInferError::Internal(format!(
            "unknown backend '{}' for model '{}' (expected 'local', 'remote', or 'grid')",
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
