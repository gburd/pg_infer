use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use ndarray::Array2;
use once_cell::sync::Lazy;
use pgrx::datum::DatumWithOid;
use pgrx::prelude::*;

use larql_vindex::{SilentLoadCallbacks, VectorIndex, VindexConfig};

use crate::error::PgLarqlError;
use crate::gucs;

// ---------------------------------------------------------------------------
// Per-backend (process-local) model handle cache
// ---------------------------------------------------------------------------

/// Everything needed to service walk / describe / similar queries against a
/// single vindex.
pub struct ModelHandle {
    pub vindex: VectorIndex,
    pub embeddings: Array2<f32>,
    pub embed_scale: f32,
    pub tokenizer: larql_vindex::tokenizers::Tokenizer,
    pub config: VindexConfig,
    #[cfg_attr(not(feature = "inference"), allow(dead_code))]
    pub path: PathBuf,
}

/// Process-local cache of loaded model handles, keyed by model name.
///
/// PostgreSQL forks one backend per connection.  Each backend maintains its
/// own `HashMap`, but the OS kernel shares the underlying mmap pages across
/// all backends that have the same vindex open.
static HANDLE_CACHE: Lazy<Mutex<HashMap<String, ModelHandle>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

/// Resolve the model name for a query function.
///
/// If `explicit` is `Some`, use that.  Otherwise fall back to the
/// `larql.default_model` GUC.
pub fn resolve_model_name(explicit: Option<&str>) -> Result<String, PgLarqlError> {
    if let Some(name) = explicit {
        return Ok(name.to_string());
    }
    gucs::default_model().ok_or(PgLarqlError::NoDefaultModel)
}

/// Obtain a reference to the cached `ModelHandle` for `model_name`.
///
/// On cache miss the function queries `larql.models` for the vindex path,
/// loads the vindex, embeddings and tokenizer, then caches the result.
pub fn with_model<F, R>(model_name: &str, f: F) -> Result<R, PgLarqlError>
where
    F: FnOnce(&ModelHandle) -> Result<R, PgLarqlError>,
{
    let mut cache = HANDLE_CACHE
        .lock()
        .map_err(|e| PgLarqlError::Internal(format!("handle cache lock poisoned: {}", e)))?;

    if !cache.contains_key(model_name) {
        let handle = load_model(model_name)?;
        cache.insert(model_name.to_string(), handle);
    }

    let handle = cache
        .get(model_name)
        .expect("just inserted");
    f(handle)
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

/// Load a model by looking up its vindex path in the `larql.models` table,
/// then opening the vindex directory.
fn load_model(model_name: &str) -> Result<ModelHandle, PgLarqlError> {
    // Query the registry table for this model's vindex path.
    let vindex_path: String = Spi::get_one_with_args(
        "SELECT vindex_path FROM larql.models WHERE model_name = $1",
        &[DatumWithOid::from(model_name)],
    )?
    .ok_or_else(|| PgLarqlError::ModelNotFound {
        name: model_name.to_string(),
    })?;

    load_from_path(Path::new(&vindex_path))
}

/// Load a `ModelHandle` directly from a vindex directory path.
pub fn load_from_path(path: &Path) -> Result<ModelHandle, PgLarqlError> {
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
        vindex,
        embeddings,
        embed_scale,
        tokenizer,
        config,
        path: path.to_path_buf(),
    })
}
