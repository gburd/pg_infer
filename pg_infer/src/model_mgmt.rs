use std::path::{Path, PathBuf};

use pgrx::datum::{DatumWithOid, TimestampWithTimeZone};
use pgrx::prelude::*;

use crate::error::PgInferError;
use crate::gucs;
use crate::registry;

/// Register a model so that query functions can reference it by name.
///
/// Creates a PG index `USING infer` that stores the vindex data in
/// WAL-logged pages.  Also inserts a row into `infer.models` for backward
/// compatibility with the mmap path.
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

    // Persist to legacy registry table (backward compat with mmap path).
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

    // Create a PG index USING infer to store the vindex data in pages.
    // Drop any pre-existing index with the same name first.
    // Schema-qualify to ensure we find the index in the `infer` schema,
    // regardless of the current search_path.
    let drop_sql = format!(
        "DROP INDEX IF EXISTS infer.\"{}\"",
        model_name.replace('"', "\"\"")
    );
    let create_sql = format!(
        "CREATE INDEX \"{}\" ON infer._models USING infer (name) WITH (source = '{}')",
        model_name.replace('"', "\"\""),
        vindex_path.replace('\'', "''")
    );

    // These are DDL statements — they run in the current transaction.
    Spi::run(&drop_sql)?;
    Spi::run(&create_sql)?;

    Ok(format!(
        "model '{}' registered ({} layers, hidden={}, level={})",
        model_name, num_layers, hidden_size, level_str
    ))
}

/// Remove a model registration and evict it from the per-backend cache.
///
/// Drops the PG infer index (if one exists) and removes the registry
/// table entry.
///
/// ```sql
/// SELECT infer_drop_model('qwen05b');
/// ```
#[pg_extern]
fn infer_drop_model(model_name: &str) -> Result<String, Box<dyn std::error::Error>> {
    // Drop the PG infer index if it exists.
    // Schema-qualify to ensure we find the index in the `infer` schema.
    let drop_sql = format!(
        "DROP INDEX IF EXISTS infer.\"{}\"",
        model_name.replace('"', "\"\"")
    );
    Spi::run(&drop_sql)?;

    // Remove from legacy registry table.
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
/// Returns models from both the legacy `infer.models` table and PG
/// indexes using the `infer` AM.
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
        // Query from legacy table.
        let result = client.select(
            "SELECT model_name, vindex_path, source, extract_level, \
                    num_layers, hidden_size, vocab_size, registered_at \
             FROM infer.models ORDER BY registered_at",
            None,
            &[],
        )?;

        let mut rows = Vec::new();
        let mut seen = std::collections::HashSet::new();

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

            seen.insert(model_name.clone());
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

        // Also discover infer-AM indexes not in the legacy table.
        let idx_result = client.select(
            "SELECT c.relname::text \
             FROM pg_class c \
             JOIN pg_am a ON c.relam = a.oid \
             WHERE a.amname = 'infer' AND c.relkind = 'i' \
             ORDER BY c.relname",
            None,
            &[],
        )?;

        for row in idx_result {
            let idx_name: String = row.get(1)?.unwrap_or_default();
            if seen.contains(&idx_name) {
                continue;
            }
            // Minimal entry for index-only models.
            rows.push((
                idx_name,
                String::from("(stored in PG pages)"),
                String::new(),
                String::from("browse"),
                None,
                None,
                None,
                TimestampWithTimeZone::new_unchecked(2000, 1, 1, 0, 0, 0.0),
            ));
        }

        Ok::<_, pgrx::spi::SpiError>(rows)
    })?;

    Ok(TableIterator::new(rows))
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
    // Case 1: hf:// URI — download the pre-built vindex.
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

    // Case 2: HuggingFace model ID (contains '/' but is not a local path).
    if source.contains('/') && !Path::new(source).exists() {
        if !gucs::AUTO_DOWNLOAD.get() {
            return Err(PgInferError::Internal(
                "infer.auto_download is off — cannot fetch from HuggingFace".into(),
            ));
        }
        return Err(PgInferError::Internal(
            "HuggingFace model download + extraction not yet implemented".into(),
        ));
    }

    // Case 3: local path (absolute or relative).
    let data_dir = resolve_data_directory();

    let resolved = if Path::new(source).is_absolute() {
        PathBuf::from(source)
    } else {
        data_dir.join(source)
    };

    // Canonicalize the resolved path if it exists on disk.
    let canonical = resolved
        .canonicalize()
        .unwrap_or_else(|_| resolved.clone());

    // Canonicalize the data directory for comparison (may not exist yet).
    let canonical_data_dir = data_dir
        .canonicalize()
        .unwrap_or_else(|_| data_dir.clone());

    // Security check: the resolved path must be under the data directory.
    if !canonical.starts_with(&canonical_data_dir) {
        return Err(PgInferError::PathNotPermitted {
            path: canonical.to_string_lossy().into_owned(),
            allowed: canonical_data_dir.to_string_lossy().into_owned(),
        });
    }

    Ok(canonical.to_string_lossy().into_owned())
}
