use std::path::{Path, PathBuf};

use pgrx::datum::{DatumWithOid, TimestampWithTimeZone};
use pgrx::prelude::*;

use crate::backend::Backend;
use crate::error::PgInferError;
use crate::gucs;
use crate::registry;

/// Register a model so query functions can reference it by name.
///
/// Local (mmap) registration — the default:
/// ```sql
/// SELECT infer_create_model('qwen05b', '/data/qwen.vindex');
/// ```
///
/// Remote registration (points at a larql-server):
/// ```sql
/// SELECT infer_create_model_remote('qwen05b', 'http://localhost:8080');
/// ```
///
/// `source` is either:
/// - An absolute path to an existing `.vindex/` directory, or
/// - A relative path resolved under `infer.data_directory`.
/// HuggingFace auto-download is declared but not yet wired.
#[pg_extern]
fn infer_create_model(
    model_name: &str,
    source: &str,
    extract_level: default!(Option<&str>, "NULL"),
) -> Result<String, Box<dyn std::error::Error>> {
    let _level = extract_level.unwrap_or("browse");

    let vindex_path = resolve_source(source)?;

    // Validate the vindex loads before persisting anything.
    let handle = registry::load_from_path(Path::new(&vindex_path))?;
    let num_layers = handle.config.num_layers as i32;
    let hidden_size = handle.config.hidden_size as i32;
    let vocab_size = handle.config.vocab_size as i32;
    let level_str = format!("{:?}", handle.config.extract_level).to_lowercase();

    upsert_registry(
        model_name,
        &vindex_path,
        source,
        &level_str,
        num_layers,
        hidden_size,
        vocab_size,
        "local",
        None,
    )?;

    registry::evict(model_name);

    Ok(format!(
        "model '{}' registered (local, {} layers, hidden={}, level={})",
        model_name, num_layers, hidden_size, level_str
    ))
}

/// Register a model that lives on a remote `larql-server`.
///
/// The server is contacted at registration time so `/v1/stats` can be
/// cached into `infer.models`.  Subsequent queries run against the
/// server's shared vindex and activation cache.
///
/// ```sql
/// SELECT infer_create_model_remote('gemma4b', 'http://localhost:8080');
/// SELECT infer_create_model_remote('shared', 'http://pg-infer-server:8080');
/// ```
#[pg_extern]
fn infer_create_model_remote(
    model_name: &str,
    server_url: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    use crate::backend::remote::RemoteBackend;

    // Probe the server.  This also validates the URL.
    let backend = RemoteBackend::connect(server_url, gucs::remote_timeout())?;
    let num_layers = backend.num_layers as i32;
    let hidden_size = backend.hidden_size as i32;

    upsert_registry(
        model_name,
        "",
        server_url,
        "remote",
        num_layers,
        hidden_size,
        0,
        "remote",
        Some(server_url),
    )?;

    registry::evict(model_name);

    Ok(format!(
        "model '{}' registered (remote={}, {} layers, hidden={})",
        model_name, server_url, num_layers, hidden_size
    ))
}

#[allow(clippy::too_many_arguments)]
fn upsert_registry(
    model_name: &str,
    vindex_path: &str,
    source: &str,
    extract_level: &str,
    num_layers: i32,
    hidden_size: i32,
    vocab_size: i32,
    backend: &str,
    server_url: Option<&str>,
) -> Result<(), PgInferError> {
    Spi::run_with_args(
        "INSERT INTO infer.models \
             (model_name, vindex_path, source, extract_level, \
              num_layers, hidden_size, vocab_size, backend, server_url) \
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9) \
         ON CONFLICT (model_name) DO UPDATE SET \
             vindex_path   = EXCLUDED.vindex_path, \
             source        = EXCLUDED.source, \
             extract_level = EXCLUDED.extract_level, \
             num_layers    = EXCLUDED.num_layers, \
             hidden_size   = EXCLUDED.hidden_size, \
             vocab_size    = EXCLUDED.vocab_size, \
             backend       = EXCLUDED.backend, \
             server_url    = EXCLUDED.server_url, \
             registered_at = NOW()",
        &[
            DatumWithOid::from(model_name),
            DatumWithOid::from(vindex_path),
            DatumWithOid::from(source),
            DatumWithOid::from(extract_level),
            DatumWithOid::from(num_layers),
            DatumWithOid::from(hidden_size),
            DatumWithOid::from(vocab_size),
            DatumWithOid::from(backend),
            DatumWithOid::from(server_url),
        ],
    )?;
    Ok(())
}

/// Remove a model registration and evict it from the per-backend cache.
///
/// ```sql
/// SELECT infer_drop_model('qwen05b');
/// ```
#[pg_extern]
fn infer_drop_model(model_name: &str) -> Result<String, Box<dyn std::error::Error>> {
    Spi::run_with_args(
        "DELETE FROM infer.models WHERE model_name = $1",
        &[DatumWithOid::from(model_name)],
    )?;
    registry::evict(model_name);
    Ok(format!("model '{}' dropped", model_name))
}

/// List all registered models as a set-returning function.
#[pg_extern]
fn infer_models() -> Result<
    TableIterator<
        'static,
        (
            name!(model_name, String),
            name!(vindex_path, String),
            name!(source, String),
            name!(extract_level, String),
            name!(num_layers, Option<i32>),
            name!(hidden_size, Option<i32>),
            name!(vocab_size, Option<i32>),
            name!(registered_at, TimestampWithTimeZone),
        ),
    >,
    Box<dyn std::error::Error>,
> {
    let rows: Vec<_> = Spi::connect(|client| {
        let result = client.select(
            "SELECT model_name, vindex_path, source, extract_level, \
                    num_layers, hidden_size, vocab_size, registered_at \
             FROM infer.models ORDER BY registered_at",
            None,
            &[],
        )?;

        let mut rows = Vec::new();
        for row in result {
            let model_name: String = row.get(1)?.unwrap_or_default();
            let vindex_path: String = row.get(2)?.unwrap_or_default();
            let source: String = row.get(3)?.unwrap_or_default();
            let extract_level: String = row.get(4)?.unwrap_or_default();
            let num_layers: Option<i32> = row.get(5)?;
            let hidden_size: Option<i32> = row.get(6)?;
            let vocab_size: Option<i32> = row.get(7)?;
            let registered_at: TimestampWithTimeZone =
                row.get(8)?.unwrap_or_else(|| {
                    TimestampWithTimeZone::new_unchecked(2000, 1, 1, 0, 0, 0.0)
                });

            rows.push((
                model_name,
                vindex_path,
                source,
                extract_level,
                num_layers,
                hidden_size,
                vocab_size,
                registered_at,
            ));
        }

        Ok::<_, pgrx::spi::SpiError>(rows)
    })?;

    Ok(TableIterator::new(rows))
}

/// Register a model that routes through the grid (dynamic server pool).
///
/// The grid is contacted at registration time to verify the model is
/// available.  Subsequent queries are dispatched to discovered servers
/// with zero resolution latency via a background-refreshed route table.
///
/// ```sql
/// SET infer.grid_url = 'http://router:8080';
/// SELECT infer_create_model_grid('gemma4b');
/// SELECT infer_create_model_grid('custom_name', 'actual_model_id');
/// ```
#[pg_extern]
fn infer_create_model_grid(
    model_name: &str,
    model_id: default!(Option<&str>, "NULL"),
) -> Result<String, Box<dyn std::error::Error>> {
    use crate::backend::grid::GridBackend;

    let grid_url = gucs::grid_url().ok_or_else(|| {
        PgInferError::Internal(
            "infer.grid_url must be set before calling infer_create_model_grid()".into(),
        )
    })?;

    let effective_model_id = model_id.unwrap_or(model_name);

    // Probe the grid to verify the model is available.
    let backend = GridBackend::connect(
        effective_model_id,
        &grid_url,
        gucs::grid_poll_interval(),
        gucs::remote_timeout(),
    )?;

    let num_layers = backend.num_layers() as i32;
    let hidden_size = backend.hidden_size() as i32;

    upsert_registry(
        model_name,
        "",
        &grid_url,
        "grid",
        num_layers,
        hidden_size,
        0,
        "grid",
        Some(effective_model_id),
    )?;

    registry::evict(model_name);

    Ok(format!(
        "model '{}' registered (grid={}, model_id='{}', {} layers, hidden={})",
        model_name, grid_url, effective_model_id, num_layers, hidden_size
    ))
}

/// Register multiple models from a single remote server in one call.
///
/// Returns the count of successfully registered models.
///
/// ```sql
/// SELECT infer_create_models_remote(
///     ARRAY['model_a', 'model_b'],
///     'http://localhost:8080'
/// );
/// ```
#[pg_extern]
fn infer_create_models_remote(
    model_names: Vec<String>,
    server_url: &str,
) -> Result<i32, Box<dyn std::error::Error>> {
    use crate::backend::remote::RemoteBackend;

    // Probe once to get server stats.
    let backend = RemoteBackend::connect(server_url, gucs::remote_timeout())?;
    let num_layers = backend.num_layers as i32;
    let hidden_size = backend.hidden_size as i32;

    let mut count = 0i32;
    for model_name in &model_names {
        upsert_registry(
            model_name,
            "",
            server_url,
            "remote",
            num_layers,
            hidden_size,
            0,
            "remote",
            Some(server_url),
        )?;
        registry::evict(model_name);
        count += 1;
    }

    Ok(count)
}

/// Detect a colocated larql-server by probing well-known Unix socket paths.
///
/// Returns the detected socket URL (e.g. `uds:///tmp/larql.sock`) or NULL.
///
/// ```sql
/// SELECT infer_detect_server();
/// ```
#[pg_extern]
fn infer_detect_server() -> Option<String> {
    crate::backend::remote::detect_local_socket()
}

// ---------------------------------------------------------------------------
// Source resolution
// ---------------------------------------------------------------------------

/// Resolve the effective data directory as an absolute path.
///
/// If `infer.data_directory` is absolute, use it directly.
/// If relative, join with `$PGDATA` (fallback: `/var/lib/postgresql/data`).
fn resolve_data_directory() -> PathBuf {
    let data_dir = gucs::data_directory();
    let path = Path::new(&data_dir);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        let pgdata = std::env::var("PGDATA")
            .unwrap_or_else(|_| "/var/lib/postgresql/data".to_string());
        Path::new(&pgdata).join(path)
    }
}

/// Resolve a user-supplied source string into an absolute vindex path.
///
/// Local paths are restricted to be under `infer.data_directory` to prevent
/// arbitrary filesystem reads.
fn resolve_source(source: &str) -> Result<String, PgInferError> {
    // hf:// URI — pre-built vindex download.
    if source.starts_with("hf://") {
        if !crate::gucs::AUTO_DOWNLOAD.get() {
            return Err(PgInferError::Config(
                "infer.auto_download is disabled; cannot download from HuggingFace".into(),
            ));
        }
        let local_path = infer_vindex::format::huggingface::resolve_hf_vindex(source)
            .map_err(|e| PgInferError::Internal(format!("HuggingFace download failed: {e}")))?;
        return Ok(local_path.to_string_lossy().to_string());
    }

    // HuggingFace model ID (contains '/' but is not a local path).
    if source.contains('/') && !Path::new(source).exists() {
        return Err(PgInferError::Config(format!(
            "Model ID '{}' requires extraction to vindex format. \
             Use 'larql extract {}' CLI or provide a pre-built vindex via 'hf://<repo>'.",
            source, source
        )));
    }

    let data_dir = resolve_data_directory();

    let resolved = if Path::new(source).is_absolute() {
        PathBuf::from(source)
    } else {
        data_dir.join(source)
    };

    let canonical = resolved
        .canonicalize()
        .unwrap_or_else(|_| resolved.clone());

    let canonical_data_dir = data_dir
        .canonicalize()
        .unwrap_or_else(|_| data_dir.clone());

    if !canonical.starts_with(&canonical_data_dir) {
        return Err(PgInferError::PathNotPermitted {
            path: canonical.to_string_lossy().into_owned(),
            allowed: canonical_data_dir.to_string_lossy().into_owned(),
        });
    }

    Ok(canonical.to_string_lossy().into_owned())
}
