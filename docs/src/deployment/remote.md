# Remote (larql-server) Deployment

Best for: production workloads, multiple concurrent users, models > 1B parameters.

## How It Works

```
psql --> PostgreSQL --> pg_infer --> HTTP/UDS --> larql-server --> vindex
```

A dedicated `larql-server` process owns the mmap and a three-tier activation
cache. Every pg_infer backend is a thin HTTP client. One copy of the model
serves all connections with 95%+ cache hit rate on recurring queries.

**Tradeoffs**: Requires running the external larql-server binary. Adds one
network hop (sub-millisecond over UDS). Feature-level enumeration
(`show_features`, `infer_diff`) is not available remotely.

## Comparison: Local vs Remote

| Aspect | Local | Remote (UDS) |
|--------|-------|--------------|
| `similar_to` cold | ~1.9 s | 8-15 ms |
| `similar_to` warm | ~1.9 s | < 1 ms (L2 hit) |
| 20-row re-rank | ~37 s | ~20-40 ms (one fan-out) |
| 32 concurrent | OOM after ~50 | ~10k req/s sustained |

## Install

```sh
# Build the extension:
cargo pgrx install --release                  # pg_infer.so -> $PG_LIBDIR

# Build larql-server from the upstream larql repository:
# git clone https://codeberg.org/gregburd/larql && cd larql
# cargo build --release -p larql-server
# cp target/release/larql-server /usr/local/bin/
```

## Start the Server

Colocated, over a Unix domain socket (lowest latency):

```sh
larql-server /data/qwen-0.5b.vindex \
    --uds-path /run/larql.sock \
    --max-gate-cache-layers 8
```

Colocated or LAN, over TCP:

```sh
larql-server /data/qwen-0.5b.vindex --port 8080
```

For multi-host sharding, point a `larql-router` at N servers and use
the router URL wherever you'd use a server URL; pg_infer doesn't care.

## Verify the Server

```sh
curl http://localhost:8080/v1/health
# {"status":"ok"}

curl http://localhost:8080/v1/stats
# {"model":"qwen-0.5b","num_layers":24,"hidden_size":896,...}
```

## Register the Model

```sql
CREATE EXTENSION pg_infer;

-- Remote over UDS (preferred on the same host):
SELECT infer_create_model_remote('qwen05b', 'uds:///run/larql.sock');

-- Or over TCP:
SELECT infer_create_model_remote('qwen05b', 'http://localhost:8080');

SET infer.default_model = 'qwen05b';
```

`infer_create_model_remote` issues one `GET /v1/stats` at registration
to cache `num_layers` and `hidden_size` in `infer.models`. Subsequent
queries use the cached shape; the server is hit for actual inference only.

## Query Shapes That Work Well

| Shape | Notes |
|-------|-------|
| `SELECT describe(entity)` | One GET per row; server L2 cache dominates |
| `SELECT walk(prompt, top => K)` | One GET per call |
| `SELECT similar_to(a, b)` | Two concurrent walks (one round-trip over HTTP/2 / UDS) |
| `SELECT unnest(similar_to_many(ARRAY[...], query))` | 1 + N concurrent walks in a single round-trip |
| `SELECT ... ORDER BY col <~> 'query' LIMIT K` | One `similar_to` per row -- pre-filter aggressively via WHERE |

For table-scan re-ranking, prefer `similar_to_many` over per-row
`similar_to`: it batches all the walks into a single concurrent fan-out,
so a 20-row re-rank is one network wait instead of twenty.

## Cancellation

In-flight remote calls respond to `pg_cancel_backend(...)` / Ctrl-C within
~100 ms. The client polls PostgreSQL's `InterruptPending` flag between
50 ms response waits; when set, the HTTP future is aborted and
`ProcessInterrupts()` is called at the unwind point for a clean SQL
`ERROR: canceling statement due to user request`.

## GUCs

| GUC | Default | Purpose |
|-----|---------|---------|
| `infer.default_model` | unset | Fallback when a query omits `model =>`. |
| `infer.default_backend` | `'local'` | Currently informational; per-model `backend` column overrides. |
| `infer.default_server_url` | unset | Used when a `backend='remote'` row omits `server_url`. |
| `infer.remote_timeout_ms` | `30000` | Per-request upper bound. |

## Verifying the Install

Run the mock-server integration test to confirm the client wire
contract matches larql-server's JSON API:

```sh
cd pg_infer && cargo test -p infer-client
```

With a real larql-server available, run the live smoke script:

```sh
LARQL_SERVER=/usr/local/bin/larql-server \
LARQL_VINDEX=/data/qwen-0.5b.vindex \
PGDATABASE=test \
bash scripts/live_server_test.sh
```

The script stands up the server on a free port, registers the model,
exercises every pg_infer function, and tears down.

## Benchmarking

Apply the schema (10k products + 8 bench phrases) then run pgbench:

```sh
psql -d test -f benches/schema.sql

# Single-call workload (measures the similar_to hot path):
pgbench -n -c 1  -T 30 -f benches/pgbench_similar_to.sql
pgbench -n -c 32 -T 60 -f benches/pgbench_similar_to.sql

# Table-scan re-ranking (measures similar_to_many batching):
pgbench -n -c 1  -T 30 -f benches/pgbench_semantic_rerank.sql
pgbench -n -c 32 -T 60 -f benches/pgbench_semantic_rerank.sql
```

### Expected Results

Results vary by hardware, vindex, and server warmup:

| Workload | `backend='local'` | `backend='remote'` (UDS) |
|----------|-------------------|--------------------------|
| `similar_to` cold | ~1.9 s | 8-15 ms |
| `similar_to` warm | ~1.9 s | < 1 ms (L2 hit) |
| 20-row re-rank | ~37 s | ~20-40 ms (one fan-out) |
| 32 concurrent | OOM after ~50 | ~10k req/s sustained |

The "cold vs warm" split is a measurement of the server's L2 activation
cache: once a query phrase has been walked for any client, subsequent
walks for the same phrase from any pg_infer backend hit the cache.
See ADR-0002 in the larql repo for why the cache is keyed on the sparse
feature-id set rather than the raw residual.

## Connection Pooling

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
