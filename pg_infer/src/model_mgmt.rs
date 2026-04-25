use std::path::Path;

use pgrx::datum::{DatumWithOid, TimestampWithTimeZone};
use pgrx::prelude::*;

use crate::error::PgInferError;
use crate::gucs;
use crate::registry;

/// Register a model so that query functions can reference it by name.
///
/// `source` is either:
/// - An absolute path to an existing `.vindex/` directory,
/// - A HuggingFace `hf://` URI pointing to a pre-built vindex, or
/// - A HuggingFace model ID (e.g. `google/gemma-3-4b-it`) which will
///   be downloaded and extracted if `infer.auto_download` is on.
///
/// ```sql
/// SELECT infer_create_model('qwen05b', '/data/qwen.vindex');
/// SELECT infer_create_model('gemma4b', 'hf://chrishayuk/gemma-3-4b-it-vindex');
/// ```
#[pg_extern]
fn infer_create_model(
    model_name: &str,
    source: &str,
    extract_level: default!(Option<&str>, "NULL"),
) -> Result<String, Box<dyn std::error::Error>> {
    let _level = extract_level.unwrap_or("browse");

    // Resolve the vindex path from the source.
    let vindex_path = resolve_source(source)?;

    // Validate that the vindex can actually be loaded.
    let handle = registry::load_from_path(Path::new(&vindex_path))?;
    let num_layers = handle.config.num_layers as i32;
    let hidden_size = handle.config.hidden_size as i32;
    let vocab_size = handle.config.vocab_size as i32;
    let level_str = format!("{:?}", handle.config.extract_level).to_lowercase();

    // Persist to registry table.
    Spi::run_with_args(
        "INSERT INTO infer.models \
             (model_name, vindex_path, source, extract_level, \
              num_layers, hidden_size, vocab_size) \
         VALUES ($1, $2, $3, $4, $5, $6, $7) \
         ON CONFLICT (model_name) DO UPDATE SET \
             vindex_path   = EXCLUDED.vindex_path, \
             source        = EXCLUDED.source, \
             extract_level = EXCLUDED.extract_level, \
             num_layers    = EXCLUDED.num_layers, \
             hidden_size   = EXCLUDED.hidden_size, \
             vocab_size    = EXCLUDED.vocab_size, \
             registered_at = NOW()",
        &[
            DatumWithOid::from(model_name),
            DatumWithOid::from(vindex_path.as_str()),
            DatumWithOid::from(source),
            DatumWithOid::from(level_str.as_str()),
            DatumWithOid::from(num_layers),
            DatumWithOid::from(hidden_size),
            DatumWithOid::from(vocab_size),
        ],
    )?;

    Ok(format!(
        "model '{}' registered ({} layers, hidden={}, level={})",
        model_name, num_layers, hidden_size, level_str
    ))
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

    // Evict from process-local cache (if present).
    registry::evict(model_name);

    Ok(format!("model '{}' dropped", model_name))
}

/// List all registered models as a set-returning function.
///
/// ```sql
/// SELECT * FROM infer_models();
/// ```
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
                    // Fallback: epoch timestamp (2000-01-01 is PG epoch).
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

// ---------------------------------------------------------------------------
// Source resolution
// ---------------------------------------------------------------------------

/// Resolve a user-supplied source string into an absolute vindex path.
fn resolve_source(source: &str) -> Result<String, PgInferError> {
    // Case 1: absolute or relative local path.
    let as_path = Path::new(source);
    if as_path.exists() {
        return Ok(as_path
            .canonicalize()
            .unwrap_or_else(|_| as_path.to_path_buf())
            .to_string_lossy()
            .into_owned());
    }

    // Case 2: hf:// URI — download the pre-built vindex.
    if source.starts_with("hf://") {
        if !gucs::AUTO_DOWNLOAD.get() {
            return Err(PgInferError::Internal(
                "infer.auto_download is off — cannot fetch from HuggingFace".into(),
            ));
        }
        return Err(PgInferError::Internal(
            "HuggingFace vindex download not yet implemented".into(),
        ));
    }

    // Case 3: HuggingFace model ID — download weights and extract.
    if source.contains('/') {
        if !gucs::AUTO_DOWNLOAD.get() {
            return Err(PgInferError::Internal(
                "infer.auto_download is off — cannot fetch from HuggingFace".into(),
            ));
        }
        return Err(PgInferError::Internal(
            "HuggingFace model download + extraction not yet implemented".into(),
        ));
    }

    Err(PgInferError::Internal(format!(
        "cannot resolve source '{}': not a local path, hf:// URI, or model ID",
        source
    )))
}
