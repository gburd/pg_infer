# Grid (larql-router) Deployment

Best for: models too large for a single host, high-availability setups.

## Overview

```
pg_infer (PostgreSQL)
    |
    v HTTP round-robin
+---------------------------+
| larql-router              |  <-- optional: provides /v1/models discovery
| or seed server            |
+---------------------------+
    |           |           |
    v           v           v
server-A    server-B    server-C
(shard 0)   (shard 1)   (shard 2)
```

The grid backend enables multi-host model sharding for models too large to fit
on a single server. pg_infer discovers available servers via HTTP polling and
dispatches queries round-robin across them.

## How It Works

1. pg_infer calls `GET /v1/models` on the grid discovery URL
2. The response lists available models and their server URLs
3. pg_infer filters for the target model and connects to each server
4. Queries are dispatched round-robin with zero resolution latency
5. A background poller refreshes the route table every `infer.grid_poll_interval` seconds

## larql-router Setup

The `larql-router` binary aggregates multiple larql-server instances behind a
single discovery endpoint.

### Step 1: Start Model Shards

Each larql-server hosts a subset of the model's layers:

```sh
# Host A: layers 0-7
larql-server /data/model-shard-0.vindex --port 8080

# Host B: layers 8-15
larql-server /data/model-shard-1.vindex --port 8080

# Host C: layers 16-23
larql-server /data/model-shard-2.vindex --port 8080
```

### Step 2: Start the Router

```sh
larql-router \
    --servers http://hostA:8080,http://hostB:8080,http://hostC:8080 \
    --port 9090
```

The router exposes:
- `/v1/models` -- lists all registered servers and their models
- `/v1/health` -- aggregated health status

### Step 3: Register in PostgreSQL

```sql
CREATE EXTENSION pg_infer;

-- Register with grid backend:
SELECT infer_create_model_grid('bigmodel', 'http://router-host:9090');

-- Or set the grid URL globally:
SET infer.grid_url = 'http://router-host:9090';
SELECT infer_create_model_grid('bigmodel');
```

## Auto-Discovery

pg_infer uses HTTP-based auto-discovery (not gRPC). The discovery endpoint
(`/v1/models`) returns a JSON response:

```json
{
  "models": [
    {"model": "bigmodel", "url": "http://hostA:8080"},
    {"model": "bigmodel", "url": "http://hostB:8080"},
    {"model": "bigmodel", "url": "http://hostC:8080"}
  ]
}
```

pg_infer matches entries by `model` or `id` field against the registered model
name, then maintains a persistent connection to each discovered server.

### Discovery Without a Router

If you don't want to run larql-router, you can point pg_infer directly at a
seed server. If that server responds to `/v1/stats` with the target model
name, pg_infer uses it as the sole backend (effectively a single-server grid).

### Topology Changes

- New servers appear at the next poll interval (default 30s)
- Removed servers are dropped from the route table at the next poll
- Unreachable servers during discovery are skipped (retried next poll)
- If all servers disappear, queries return an error until at least one is rediscovered

## Configuration

| GUC | Default | Description |
|-----|---------|-------------|
| `infer.grid_url` | unset | Discovery URL (larql-router or seed server) |
| `infer.grid_poll_interval` | 30 | Seconds between topology refreshes (min: 5, max: 3600) |
| `infer.remote_timeout_ms` | 30000 | Per-request timeout for grid server calls |

## Round-Robin Dispatch

Queries are distributed across discovered servers using atomic round-robin
(fetch-and-add modulo server count). This ensures even load distribution
without external load balancers.

The dispatch is per-query, not per-connection: consecutive queries from the
same PostgreSQL backend may hit different servers. This is intentional -- it
maximizes activation cache utilization across the fleet.

## Failure Handling

- If a dispatched query fails (timeout, connection error), the error propagates
  to the SQL caller. There is no automatic retry or failover to another server.
- Stale servers (removed from `/v1/models`) are cleaned up at the next poll.
  In-flight queries to a stale server complete normally; only new queries stop
  routing to it.
- The background poller is best-effort: if a poll fails, the previous route
  table continues serving until the next successful poll.

## Limitations

The grid backend does not support:
- `show_features` / `infer_diff` (requires local mmap access)
- `embed()` (no server-side endpoint for raw embeddings)
- Weighted routing (all servers are treated equally)
- Automatic retry on per-query failures

## Example: 3-Host Grid Deployment

```sh
# Terminal 1 (host A):
larql-server /nfs/vindexes/llama-70b-shard0.vindex --port 8080

# Terminal 2 (host B):
larql-server /nfs/vindexes/llama-70b-shard1.vindex --port 8080

# Terminal 3 (host C):
larql-server /nfs/vindexes/llama-70b-shard2.vindex --port 8080

# Terminal 4 (router host):
larql-router \
    --servers http://hostA:8080,http://hostB:8080,http://hostC:8080 \
    --port 9090
```

```sql
-- In PostgreSQL:
CREATE EXTENSION pg_infer;
SET infer.grid_url = 'http://router-host:9090';
SET infer.grid_poll_interval = 15;  -- faster discovery for testing

SELECT infer_create_model_grid('llama70b');
SELECT * FROM describe('quantum computing');
```
