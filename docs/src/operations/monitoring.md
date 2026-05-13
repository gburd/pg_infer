# Monitoring

Health checks, diagnostics, and monitoring recommendations for pg_infer
deployments.

## larql-server Health Endpoint

```sh
curl http://localhost:8080/v1/health
# Returns: {"status":"ok"} with HTTP 200
# Returns: HTTP 503 if unhealthy
```

The `/v1/health` endpoint is suitable for:
- Docker health checks
- Kubernetes liveness/readiness probes
- Prometheus blackbox exporter
- Simple curl-based alerting

## Server Statistics

```sh
curl http://localhost:8080/v1/stats
# Returns: {"model":"qwen-0.5b","num_layers":24,"hidden_size":896,...}
```

The `/v1/stats` endpoint provides model metadata including layer count,
hidden size, and model name. Useful for verifying the correct model is loaded.

## PostgreSQL-Side Checks

```sql
-- Verify extension is loaded:
SELECT * FROM pg_extension WHERE extname = 'pg_infer';

-- Check registered models:
SELECT * FROM infer_show_models();

-- Verify remote connectivity (returns cache stats if connected):
SELECT * FROM infer_cache_stats('qwen05b');
```

## Grid Monitoring

For grid deployments, monitor the router's `/v1/models` endpoint to verify
all shards are registered:

```sh
curl http://router-host:9090/v1/models
# Returns list of all registered servers and their models
```

Check that the expected number of shards are present. Missing shards indicate
a server is down or unreachable from the router.

## Monitoring Recommendations

### What to Monitor

| Metric | Source | Alert Threshold |
|--------|--------|-----------------|
| Server health | `GET /v1/health` | HTTP != 200 |
| Request latency | PostgreSQL logs | > `infer.remote_timeout_ms` |
| Backend memory | OS metrics | > 80% of `infer.max_memory` |
| Grid shard count | `GET /v1/models` | < expected shard count |
| Connection errors | PostgreSQL logs | Any `connection refused` |

### Integration with Alerting Systems

**Prometheus blackbox exporter:**

```yaml
modules:
  larql_health:
    prober: http
    timeout: 5s
    http:
      valid_http_versions: ["HTTP/1.1", "HTTP/2.0"]
      valid_status_codes: [200]
      method: GET
      fail_if_body_not_matches_regexp:
        - '"status":"ok"'
```

**Simple cron check:**

```sh
#!/bin/sh
if ! curl -sf http://localhost:8080/v1/health > /dev/null; then
    echo "larql-server unhealthy" | mail -s "ALERT" ops@example.com
fi
```

## Diagnostic Queries

```sql
-- List all infer GUCs and their current values:
SELECT name, setting, context, short_desc
FROM pg_settings
WHERE name LIKE 'infer.%'
ORDER BY name;

-- Check model memory usage:
SELECT * FROM infer_show_models();

-- Remote backend cache statistics:
SELECT * FROM infer_cache_stats();

-- Verify a specific model is queryable:
SELECT * FROM describe('test') LIMIT 1;
```

## Log Levels

Control verbosity with `infer.log_level`:

```sql
-- Default: info
ALTER SYSTEM SET infer.log_level = 'info';

-- Debug: includes per-query timing and cache hit/miss
ALTER SYSTEM SET infer.log_level = 'debug';

-- Trace: full wire protocol logging (verbose)
ALTER SYSTEM SET infer.log_level = 'trace';

SELECT pg_reload_conf();
```

Timeout breaches are logged at `info` level. Look for messages about
`infer.remote_timeout_ms` in PostgreSQL logs to identify slow queries.
