//! SQL functions for cache management and backend statistics.
//!
//! These expose the larql-server's activation cache to DBAs so they can
//! pre-warm hot entities and monitor cache efficiency, and provide local
//! backend cache statistics for observability.

use pgrx::prelude::*;

use crate::registry;

/// Return per-backend cache statistics: hit/miss counters, total queries,
/// loaded model count, and approximate memory usage.
///
/// ```sql
/// SELECT * FROM infer_stats();
/// ```
#[pg_extern]
fn infer_stats() -> Result<
    TableIterator<
        'static,
        (
            name!(cache_hits, i64),
            name!(cache_misses, i64),
            name!(total_queries, i64),
            name!(loaded_models, i32),
            name!(memory_mb, f64),
        ),
    >,
    Box<dyn std::error::Error>,
> {
    let (hits, misses, total, models, bytes) = registry::cache_stats();
    Ok(TableIterator::new(vec![(
        hits as i64,
        misses as i64,
        total as i64,
        models as i32,
        bytes as f64 / (1024.0 * 1024.0),
    )]))
}

/// Pre-warm a remote server: prefetch layer pages and load inference
/// weights so the first query does not pay for them.
///
/// `layers` selects specific layers; `NULL` (the default) warms every
/// layer the server owns. Returns the server's own counters. Against a
/// local backend, or a server without `/v1/warmup`, returns a message
/// saying so rather than fabricating zeros.
///
/// The previous signature took `entities text[]` and reported
/// "N warmed, M already cached". larql-server has no per-entity
/// activation cache: `/v1/warmup` prefetches *layers* and loads weights,
/// and ignored the `entities` body entirely. The old counts were always
/// zero — not "nothing to do", but "those fields do not exist in the
/// response". Reporting the real fields is the only honest option, and
/// it changes the signature, so this is a breaking change.
///
/// ```sql
/// SELECT infer_warmup('my_model');                  -- all layers
/// SELECT infer_warmup('my_model', ARRAY[0,1,2,3]);  -- selected layers
/// ```
#[pg_extern]
fn infer_warmup(
    model_name: &str,
    layers: default!(Option<Vec<i32>>, "NULL"),
) -> Result<String, Box<dyn std::error::Error>> {
    // pgrx hands us i32 (SQL integer); the wire wants usize. Negative
    // layer numbers are a caller error, not something to silently clamp.
    let layers: Option<Vec<usize>> = match layers {
        Some(l) => {
            let mut out = Vec::with_capacity(l.len());
            for n in l {
                let Ok(n) = usize::try_from(n) else {
                    return Err(format!("layer index must be >= 0, got {n}").into());
                };
                out.push(n);
            }
            Some(out)
        }
        None => None,
    };
    let resp = registry::with_backend(model_name, |b| b.warmup(layers.as_deref()))?;
    let Some(r) = resp else {
        return Ok("warmup unsupported by this backend (no /v1/warmup)".to_string());
    };
    Ok(format!(
        "{} layers prefetched in {} ms, {} experts in {} ms, \
         weights_loaded={} ({} ms), hnsw_built={} ({} ms), total {} ms",
        r.layers_prefetched,
        r.prefetch_ms,
        r.experts_prefetched,
        r.expert_prefetch_ms,
        r.weights_loaded,
        r.weights_load_ms,
        r.hnsw_built,
        r.hnsw_warmup_ms,
        r.total_ms,
    ))
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
