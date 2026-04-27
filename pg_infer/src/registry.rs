use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use ndarray::{Array1, Array2};
use once_cell::sync::Lazy;
use pgrx::datum::DatumWithOid;
use pgrx::prelude::*;

use larql_vindex::{FeatureMeta, SilentLoadCallbacks, VectorIndex, VindexConfig};

use crate::error::PgInferError;
use crate::gucs;
use crate::page_reader::PageBackend;

// ---------------------------------------------------------------------------
// Per-backend (process-local) model handle cache
// ---------------------------------------------------------------------------

/// Everything needed to service walk / describe / similar queries against a
/// single vindex.
pub struct ModelHandle {
    pub embeddings: Array2<f32>,
    pub embed_scale: f32,
    pub tokenizer: larql_vindex::tokenizers::Tokenizer,
    pub config: VindexConfig,
    #[cfg_attr(not(feature = "inference"), allow(dead_code))]
    pub path: PathBuf,
    /// Backend: either mmap files or PG index pages.
    backend: ModelBackend,
}

/// Discriminator for the two storage backends.
enum ModelBackend {
    /// Original mmap-based backend (loaded from a vindex directory).
    Mmap {
        vindex: VectorIndex,
    },
    /// Page-based backend (loaded from PG index pages).
    Pages(PageBackend),
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
            ModelBackend::Pages(pages) => {
                pages.gate_knn(layer, query, top_k, self.config.hidden_size)
            }
        }
    }

    /// Look up metadata for a single feature, dispatching to the appropriate
    /// backend.
    pub fn feature_meta(&self, layer: usize, feature_idx: usize) -> Option<FeatureMeta> {
        match &self.backend {
            ModelBackend::Mmap { vindex } => vindex.feature_meta(layer, feature_idx),
            ModelBackend::Pages(pages) => {
                pages.feature_meta(layer, feature_idx, &self.tokenizer)
            }
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
/// On cache miss the function first checks for a PG index using the `infer`
/// AM with a matching name, then falls back to the `infer.models` table.
///
/// The `Arc` is cloned and the mutex is released before calling `f`, so the
/// closure does not block other model access in the backend.
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

/// Load a model by name.  Checks for a PG index first, then falls back to
/// the `infer.models` registry table (mmap path).
fn load_model(model_name: &str) -> Result<ModelHandle, PgInferError> {
    // 1. Check if there's a PG index using the `infer` AM.
    if let Some(handle) = try_load_from_index(model_name)? {
        return Ok(handle);
    }

    // 2. Fall back to the `infer.models` table (mmap path).
    let vindex_path: String = Spi::get_one_with_args(
        "SELECT vindex_path FROM infer.models WHERE model_name = $1",
        &[DatumWithOid::from(model_name)],
    )?
    .ok_or_else(|| PgInferError::ModelNotFound {
        name: model_name.to_string(),
    })?;

    load_from_path(Path::new(&vindex_path))
}

/// Try to load a model from a PG index with the `infer` AM.
///
/// Returns `Ok(None)` if no such index exists, `Ok(Some(handle))` if found
/// and loaded, or `Err(...)` on load failure.
fn try_load_from_index(model_name: &str) -> Result<Option<ModelHandle>, PgInferError> {
    // Find an index using the infer AM with the given name.
    let maybe_oid: Option<pgrx::pg_sys::Oid> = Spi::get_one_with_args(
        "SELECT c.oid \
         FROM pg_class c \
         JOIN pg_am a ON c.relam = a.oid \
         WHERE a.amname = 'infer' AND c.relname = $1 AND c.relkind = 'i' \
         LIMIT 1",
        &[DatumWithOid::from(model_name)],
    )?;

    let oid = match maybe_oid {
        Some(oid) => oid,
        None => return Ok(None),
    };

    let handle = load_from_index(oid)?;
    Ok(Some(handle))
}

/// Load a `ModelHandle` from a PG index by reading its pages.
///
/// # Safety
///
/// The unsafe block reads PG buffer-managed pages.  This requires being
/// inside a valid transaction context (always true when called from a
/// SQL function).
fn load_from_index(index_oid: pgrx::pg_sys::Oid) -> Result<ModelHandle, PgInferError> {
    let (pages, embeddings, embed_scale, tokenizer) = unsafe {
        PageBackend::load(index_oid)?
    };

    // Construct a minimal VindexConfig from the metapage so that existing
    // query functions can read `handle.config.hidden_size`, etc.
    let meta = &pages.meta;
    let config = minimal_config_from_meta(meta);

    // Source URI from metapage.
    let source_uri = {
        let nul = meta.source_uri.iter().position(|&b| b == 0).unwrap_or(meta.source_uri.len());
        String::from_utf8_lossy(&meta.source_uri[..nul]).into_owned()
    };

    Ok(ModelHandle {
        embeddings,
        embed_scale,
        tokenizer,
        config,
        path: PathBuf::from(source_uri),
        backend: ModelBackend::Pages(pages),
    })
}

/// Load a `ModelHandle` directly from a vindex directory path (mmap).
pub fn load_from_path(path: &Path) -> Result<ModelHandle, PgInferError> {
    let mut callbacks = SilentLoadCallbacks;

    // Load the vindex (gate vectors + metadata, mmap'd).
    let vindex = VectorIndex::load_vindex(path, &mut callbacks)?;

    // Load the vindex configuration (layer count, hidden size, etc.)
    let config = larql_vindex::load_vindex_config(path)?;

    // Load token embeddings and the embedding scale factor.
    let (embeddings, embed_scale) = larql_vindex::load_vindex_embeddings(path)?;

    // Load the tokenizer.
    let tokenizer = larql_vindex::load_vindex_tokenizer(path)?;

    Ok(ModelHandle {
        embeddings,
        embed_scale,
        tokenizer,
        config,
        path: path.to_path_buf(),
        backend: ModelBackend::Mmap { vindex },
    })
}

/// Build a minimal `VindexConfig` from the metapage fields.
///
/// Only the fields actually used by query functions are populated;
/// the rest get sensible defaults.
fn minimal_config_from_meta(meta: &crate::pages::InferMetaPage) -> VindexConfig {
    let model_name = {
        let nul = meta.model_name.iter().position(|&b| b == 0).unwrap_or(meta.model_name.len());
        String::from_utf8_lossy(&meta.model_name[..nul]).into_owned()
    };

    let extract_level = match meta.extract_level {
        1 => larql_vindex::ExtractLevel::Inference,
        2 => larql_vindex::ExtractLevel::All,
        _ => larql_vindex::ExtractLevel::Browse,
    };

    let dtype = match meta.gate_dtype {
        0 => larql_vindex::StorageDtype::F32,
        _ => larql_vindex::StorageDtype::F16,
    };

    VindexConfig {
        version: meta.format_version,
        model: model_name,
        family: String::new(),
        source: None,
        checksums: None,
        num_layers: meta.num_layers as usize,
        hidden_size: meta.hidden_size as usize,
        intermediate_size: 0,
        vocab_size: meta.vocab_size as usize,
        embed_scale: meta.embed_scale,
        extract_level,
        dtype,
        quant: larql_vindex::QuantFormat::None,
        layer_bands: None,
        layers: vec![],
        down_top_k: meta.down_top_k as usize,
        has_model_weights: false,
        model_config: None,
    }
}
