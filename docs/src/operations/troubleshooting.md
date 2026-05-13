# Troubleshooting

Common issues and fixes for pg_infer deployments.

## Remote Backend Issues

### `connection refused` on Registration

The server isn't listening on the URL you passed.

**Fix:**
1. Check the `larql-server` log output
2. Verify the server is running: `curl http://localhost:8080/v1/health`
3. Confirm the port matches what you passed to `infer_create_model_remote`
4. For UDS: check socket permissions and that the path exists

### `operation 'show_features' is not supported by the remote backend`

This is expected. Feature-level enumeration (`infer_show_features`,
`infer_diff`) has no larql-server endpoint today. Register a local
model for those queries or wait for a `/v1/features` endpoint upstream.

### Query Cancellation Takes > 200 ms

Your `infer.remote_timeout_ms` is shorter than the polling cadence (50 ms
tick + 20 ms token). Lowering it doesn't help; the tick cadence is the floor.

The client polls PostgreSQL's `InterruptPending` flag between 50 ms response
waits. Cancellation latency is bounded by this interval.

### `response body too large`

A `walk` returned more hits than the client's 64 MiB ceiling.

**Fix:** Cap `top` on the SQL side, or raise `MAX_BODY_BYTES` in
`crates/infer-client/src/transport.rs`.

## Performance Issues

### First `similar_to()` Takes 30+ Seconds

**Root cause:** HNSW indexes are built lazily on first use.

The first call triggers construction of HNSW graphs (one per layer), which
dominates query time. Subsequent calls are 1000x faster because HNSW is
cached.

**Fix:** Enable eager HNSW build during registration:
```sql
SET infer.build_hnsw_on_load = true;
SELECT infer_create_model('model', 'path.vindex');
```

Or switch to the [remote backend](../deployment/remote.md) which handles
caching server-side.

### OOM Killer Fires Under Load

**Root cause:** Unbounded gate cache growth in local mode.

With `gate_cache_max_layers = 0` (unlimited) and a 24-layer model, each
backend can consume 24 * 112 MB = ~2.7 GB of decoded gates alone.

**Fix:**
```sql
-- Bound the per-backend gate cache:
ALTER SYSTEM SET infer.gate_cache_max_layers = 4;
SELECT pg_reload_conf();
```

Or switch to the remote backend for production workloads.

### `ORDER BY <~>` Scans Are Very Slow

The custom access method performs one `similar_to` per candidate row. For
large tables, this is expensive.

**Fix:**
1. Pre-filter aggressively with `WHERE` clauses to reduce candidates
2. Use `LIMIT` to bound the scan
3. Enable HNSW: `SET infer.use_hnsw = true;`
4. Increase oversampling for accuracy: `SET infer.rerank_oversampling = 4;`

## Model Loading Issues

### `infer_create_model` Returns Error

Common causes:
- Vindex path doesn't exist or PostgreSQL can't read it
- Vindex is in an unsupported format (only v1 Q4_K/Q6_K supported)
- `infer.data_directory` is misconfigured

**Fix:** Check the path is accessible:
```sh
ls -la /var/lib/postgresql/18/data/infer/model.vindex/meta.json
```

### Model Registration Takes > 2 Minutes

Expected for large models. The `CREATE INDEX` step writes vindex data into
PostgreSQL pages through WAL:

| Model Size | Expected Registration Time |
|------------|--------------------------|
| < 1 GB | ~20s |
| 1-3 GB | ~60-90s |
| > 3 GB | ~2-3 min |

This is a one-time cost per model registration.

## Grid Backend Issues

### Grid Queries Return Errors After Server Restart

After restarting a shard server, it takes up to `infer.grid_poll_interval`
seconds (default 30s) for pg_infer to rediscover it.

**Fix:** Lower the poll interval for faster recovery:
```sql
ALTER SYSTEM SET infer.grid_poll_interval = 10;
SELECT pg_reload_conf();
```

### Not All Shards Appear in Route Table

**Diagnosis:**
```sh
curl http://router-host:9090/v1/models
```

If a shard is missing:
1. Check the shard server is running and healthy
2. Verify the router can reach the shard (network/firewall)
3. Check the router's `--servers` argument includes the shard URL

## Extension Issues

### `CREATE EXTENSION pg_infer` Fails

Common causes:
- Extension files not installed in the correct PostgreSQL lib directory
- PostgreSQL version mismatch (requires PG18+)
- Missing shared library dependencies

**Fix:**
```sh
# Verify files are in place:
ls /usr/lib/postgresql/18/lib/pg_infer.so
ls /usr/share/postgresql/18/extension/pg_infer.control

# Check library dependencies:
ldd /usr/lib/postgresql/18/lib/pg_infer.so
```
