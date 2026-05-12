//! SQL functions for server-side activation cache management.
//!
//! These expose the larql-server's activation cache to DBAs so they can
//! pre-warm hot entities and monitor cache efficiency.

use pgrx::prelude::*;

use crate::registry;

/// Pre-warm the server's activation cache for a list of entities.
///
/// Returns a status message with the count of warmed and already-cached
/// entities.  Against a local backend or an old server without `/v1/warmup`,
/// returns "(0 warmed, 0 cached)" without error.
///
/// ```sql
/// SELECT infer_warmup('my_model', ARRAY['Paris', 'France', 'Berlin']);
/// ```
#[pg_extern]
fn infer_warmup(
    model_name: &str,
    entities: Vec<String>,
) -> Result<String, Box<dyn std::error::Error>> {
    let (warmed, cached) = registry::with_backend(model_name, |b| b.warmup(&entities))?;
    Ok(format!("{} warmed, {} already cached", warmed, cached))
}

/// Return server-side cache statistics as a single row.
///
/// Returns an empty set for local backends or servers that don't support
/// `/v1/cache/stats`.
///
/// ```sql
/// SELECT * FROM infer_server_stats('my_model');
/// ```
#[pg_extern]
fn infer_server_stats(
    model_name: &str,
) -> Result<
    TableIterator<
        'static,
        (
            name!(entries, i64),
            name!(hits, i64),
            name!(misses, i64),
            name!(evictions, i64),
            name!(memory_mb, f64),
        ),
    >,
    Box<dyn std::error::Error>,
> {
    let stats = registry::with_backend(model_name, |b| b.cache_stats())?;
    match stats {
        Some(s) => Ok(TableIterator::new(vec![(
            s.entries as i64,
            s.hit_count as i64,
            s.miss_count as i64,
            s.eviction_count as i64,
            s.memory_bytes as f64 / (1024.0 * 1024.0),
        )])),
        None => Ok(TableIterator::new(vec![])),
    }
}
