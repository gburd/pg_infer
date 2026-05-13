# Tuning Guide

This chapter covers all pg_infer GUC (Grand Unified Configuration) parameters,
memory tuning, and performance recommendations.

## GUC Reference

### Model and Data

| GUC | Type | Default | Context | Description |
|-----|------|---------|---------|-------------|
| `infer.default_model` | string | unset | userset | Fallback model when queries omit the `model` parameter |
| `infer.data_directory` | string | `'infer'` | suset | Base directory for vindex files (relative to `$PGDATA` or absolute) |
| `infer.auto_download` | bool | `true` | suset | Allow downloads from HuggingFace on `infer_create_model` |
| `infer.max_memory` | int | 8192 | sighup | Maximum aggregate RSS (MB) for all loaded vindexes per backend |

### Query Behavior

| GUC | Type | Default | Context | Description |
|-----|------|---------|---------|-------------|
| `infer.gate_threshold` | float | 5.0 | userset | Gate score threshold for `describe()`/`implies()`. 0 = adaptive (max * 0.1) |
| `infer.describe_top_k` | int | 20 | userset | Features examined per layer in `describe()` |
| `infer.walk_embed_mode` | string | `'last'` | userset | Embedding mode: `'last'` (last token) or `'average'` (all tokens) |

### HNSW and Similarity

| GUC | Type | Default | Context | Description |
|-----|------|---------|---------|-------------|
| `infer.use_hnsw` | bool | `false` | userset | Enable HNSW approximate search for gate queries |
| `infer.hnsw_ef_search` | int | 200 | userset | HNSW beam width (50-500). Higher = more accurate, slower |
| `infer.warmup_on_load` | bool | `false` | userset | Pre-decode f16 gates to f32 on model load |
| `infer.build_hnsw_on_load` | bool | `false` | userset | Build HNSW indexes during registration (eager vs lazy) |
| `infer.similarity_max_layers` | int | 0 | userset | Max layers for `similar_to()`. 0 = all layers |
| `infer.parallel_similarity` | bool | `false` | userset | Use Rayon parallelism for similarity queries |
| `infer.gate_cache_max_layers` | int | 4 | userset | Max decoded f16-to-f32 gate layers kept in memory per model |

### Access Method (Index Scans)

| GUC | Type | Default | Context | Description |
|-----|------|---------|---------|-------------|
| `infer.am_hnsw_ef_search` | int | 64 | userset | Beam width for `ORDER BY <~>` index scans (16-4096) |
| `infer.rerank_oversampling` | int | 4 | userset | Oversampling factor for re-ranking after HNSW scan (1-32) |

### Remote Backend

| GUC | Type | Default | Context | Description |
|-----|------|---------|---------|-------------|
| `infer.default_backend` | string | `'local'` | userset | Default backend: `'local'`, `'remote'`, or `'grid'` |
| `infer.default_server_url` | string | unset | suset | Fallback larql-server URL for remote models |
| `infer.remote_timeout_ms` | int | 30000 | userset | Per-request timeout for remote calls (100-3600000 ms) |

### Grid Backend

| GUC | Type | Default | Context | Description |
|-----|------|---------|---------|-------------|
| `infer.grid_url` | string | unset | suset | Grid discovery URL (larql-router or seed server) |
| `infer.grid_poll_interval` | int | 30 | suset | Topology refresh interval in seconds (5-3600) |

### Observability

| GUC | Type | Default | Context | Description |
|-----|------|---------|---------|-------------|
| `infer.log_level` | string | `'info'` | suset | Tracing level: `error`, `warn`, `info`, `debug`, `trace` |

## GUC Context Levels

- **userset**: Any user can change per-session with `SET`
- **suset**: Superuser only (or via `ALTER SYSTEM` + reload)
- **sighup**: Requires `pg_reload_conf()` or SIGHUP to take effect

## Memory Tuning

### Understanding Memory Usage (Local Mode)

Each loaded model consumes memory in several tiers:

| Component | Size (3B model) | Notes |
|-----------|-----------------|-------|
| Mmap'd vindex | ~2 GB | Shared via OS page cache |
| Decoded gate cache | ~112 MB/layer | Bounded by `gate_cache_max_layers` |
| HNSW indexes | ~200 MB/layer | Only if `use_hnsw = true` |
| Feature metadata | ~50 MB | Loaded once per model |

**Key insight**: The mmap'd vindex is shared across all PostgreSQL backends via
the OS page cache. The decoded gate cache is per-backend (per-connection). This
is why `gate_cache_max_layers` is critical for multi-connection deployments.

### Recommended Settings by Deployment Size

**Single user, small model (< 1B params):**
```sql
SET infer.gate_cache_max_layers = 0;      -- unlimited (model fits in RAM)
SET infer.warmup_on_load = true;          -- decode everything upfront
SET infer.parallel_similarity = true;     -- use all cores
```

**Multi-user, medium model (1-7B params):**
```sql
ALTER SYSTEM SET infer.gate_cache_max_layers = 4;   -- ~450 MB per backend
ALTER SYSTEM SET infer.max_memory = 4096;           -- 4 GB total cap
ALTER SYSTEM SET infer.similarity_max_layers = 12;  -- sample layers for speed
SELECT pg_reload_conf();
```

**Production, large model (7B+ params):**
```sql
-- Use the remote backend to avoid per-backend memory pressure:
ALTER SYSTEM SET infer.default_backend = 'remote';
ALTER SYSTEM SET infer.default_server_url = 'http://larql-host:8080';
SELECT pg_reload_conf();
```

### OOM Prevention

The most common cause of OOM in local mode is unbounded gate cache growth.
With `gate_cache_max_layers = 0` (unlimited) and a 24-layer model, each
backend can consume 24 * 112 MB = ~2.7 GB of decoded gates alone.

For 10 concurrent connections: 10 * 2.7 GB = 27 GB just for gate caches.

**Fix**: Set `gate_cache_max_layers` to a reasonable value (4-8) or switch
to the remote backend.

## Performance Tuning

### Layer Sampling (`infer.similarity_max_layers`)

For `similar_to()` on models with many layers (24+), querying all layers is
expensive. Setting `similarity_max_layers = 8` samples 8 evenly-spaced layers
instead of all 24, giving a ~3x speedup with minimal accuracy loss for most
use cases.

```sql
-- Before: ~1.9s per similar_to on a 24-layer model
SET infer.similarity_max_layers = 0;

-- After: ~0.6s per similar_to (sampled)
SET infer.similarity_max_layers = 8;
```

### Parallel Similarity (`infer.parallel_similarity`)

Enables Rayon parallelism for layer iteration in `similar_to()`. Provides
4-8x speedup on multi-core systems at the cost of higher CPU utilization.

```sql
SET infer.parallel_similarity = true;
```

Best combined with `similarity_max_layers` to bound total work.

### HNSW Tuning

HNSW provides O(log N) approximate search instead of brute-force O(N) scan
over gate vectors. Disabled by default due to memory overhead and potential
stability issues on large models.

```sql
-- Enable HNSW:
SET infer.use_hnsw = true;
SET infer.hnsw_ef_search = 200;   -- accuracy/speed tradeoff

-- For index scans (ORDER BY <~>):
SET infer.am_hnsw_ef_search = 64;
SET infer.rerank_oversampling = 4;
```

**Warning**: Eager HNSW build (`build_hnsw_on_load = true`) can cause crashes
on large models. Prefer lazy build (default) combined with
`similarity_max_layers` for production.

### Connection Pooling (Remote Mode)

The remote backend uses HTTP/2 with connection reuse. Each PostgreSQL backend
maintains one persistent connection to larql-server. For pgBouncer or similar
poolers, this means connections are reused across sessions.

Recommended pgBouncer settings for pg_infer workloads:

```ini
[pgbouncer]
pool_mode = transaction      ; release connection after each transaction
default_pool_size = 20       ; match larql-server's concurrency capacity
reserve_pool_size = 5
server_idle_timeout = 300    ; keep warm connections alive
```

### Embedding Mode (`infer.walk_embed_mode`)

- `'last'` (default): Uses only the last token's embedding. Produces stronger,
  more interpretable activations for single concepts.
- `'average'`: Averages all token embeddings. Better for multi-word phrases
  where context is distributed across tokens.

```sql
-- For entity-style queries ("France", "Python"):
SET infer.walk_embed_mode = 'last';

-- For phrase-style queries ("capital cities of Europe"):
SET infer.walk_embed_mode = 'average';
```

## Diagnostic Queries

```sql
-- Check current GUC settings:
SHOW infer.gate_cache_max_layers;
SHOW infer.default_backend;

-- List all infer GUCs:
SELECT name, setting, context, short_desc
FROM pg_settings
WHERE name LIKE 'infer.%'
ORDER BY name;

-- Check model memory usage:
SELECT * FROM infer_show_models();

-- Remote backend cache stats:
SELECT * FROM infer_cache_stats();
```
