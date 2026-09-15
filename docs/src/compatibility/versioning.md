# Versioning and Compatibility

This chapter describes pg_infer's version strategy, compatibility with the
upstream larql project, and upgrade procedures.

## Version Strategy

pg_infer uses [semantic versioning](https://semver.org/) (major.minor.patch):

- **Major**: Breaking SQL interface changes (function signatures, return types,
  removed functions)
- **Minor**: New SQL functions, new GUCs, new backend capabilities
- **Patch**: Bug fixes, performance improvements, internal refactors

Current version: **0.1.2-alpha**

## larql Server Compatibility

Compatibility between pg_infer and larql-server is defined by the `/v1` wire
protocol version. Any larql-server that speaks the `/v1` JSON API is compatible.

### Wire Protocol: /v1

The `/v1` protocol is a JSON-over-HTTP API. The endpoints pg_infer calls,
with the shapes the server actually sends (verified against its own
OpenAPI document -- see [Testing Compatibility](upstream.md#testing-compatibility)):

| Endpoint | Request | Response |
|----------|---------|----------|
| `GET /v1/health` | -- | `{"status":"ok"}` |
| `GET /v1/stats` | -- | Model metadata + `server` counters block |
| `GET /v1/describe` | `?entity=...&limit=N&min_score=F` | `{"entity","edges":[...],"latency_ms"}` |
| `GET /v1/walk` | `?prompt=...&top=N[&layers=...]` | `{"prompt","hits":[...],"latency_ms"}` |
| `GET /v1/relations` | -- | `{"relations":[{"name","count","min_layer","max_layer",...}],"total"}` |
| `POST /v1/infer` | `{prompt, top, mode}` | `{"prompt","mode","predictions":[...],"latency_ms"}` |
| `POST /v1/warmup` | `{layers?, skip_weights?}` | `{"layers_prefetched","prefetch_ms","total_ms",...}` |
| `GET /v1/models` | -- | `{"object":"list","data":[...]}` (OpenAI shape) |

Three of these were documented incorrectly here until the 23a56db1 sync,
and pg_infer's client matched the documentation rather than the server:

- `/v1/relations` entries are keyed `name`, not `token`, and report the
  layer span as `min_layer`/`max_layer` rather than a `layers` list;
- `/v1/models` is the OpenAI list envelope (`{object, data}`), never
  `{"models": [...]}`;
- `/v1/infer` is a `POST` with a JSON body, not a `GET` with query params;
- `/v1/warmup` takes `{layers, skip_weights}` and reports prefetch
  counters -- it has no notion of "entities" or a per-entity cache.

pg_infer's `infer-client` crate validates responses against the committed
OpenAPI fixture. Unknown fields are ignored (forward-compatible). Missing
required fields cause a parse error surfaced as a SQL `ERROR`.

## Supported Configurations

| pg_infer | PostgreSQL | larql-server | Vindex Format | Rust Toolchain |
|----------|------------|--------------|---------------|----------------|
| 0.1.2-alpha | 18+ | 23a56db1+ (2026-09-13) | VINDEX2 schema 1--2 (Q4_K/Q6_K) | 1.80+ |

### Vindex Format Versions

- **VINDEX2, `index.json` schema 1--2**: gate vectors in f16, Q4_K or Q6_K.
  Feature metadata as JSON. FFN weights as raw binary slices. The only
  generation pg_infer reads.
- **VINDEX3, schema 3+**: refused by `VindexConfig::validate_supported()`
  with an explicit error, not attempted. It is a different container
  generation, and reading it as a V2 would silently misinterpret it. See
  [Vindex generations](upstream.md#vindex-generations).
- Containers declaring `fp4` or `bitnet_layout` are likewise refused: those
  fields change how the weight bytes decode, and ignoring them yields
  plausible-looking wrong numbers.

## Upgrade Procedure

### Upgrading pg_infer (Extension Only)

No `pg_infer--<old>--<new>.sql` upgrade scripts ship yet, so
`ALTER EXTENSION pg_infer UPDATE` has nothing to apply and PostgreSQL
will refuse it ("extension has no update path"). During the alpha the
supported path is drop and recreate:

1. Stop active queries (or schedule during a maintenance window)
2. Build and install: `cargo pgrx install --release`
3. Restart PostgreSQL so the new `.so` is loaded
4. In PostgreSQL: `DROP EXTENSION pg_infer; CREATE EXTENSION pg_infer;`
5. Re-register models (registrations do not survive a drop):
   `SELECT infer_create_model_remote(...)`
6. Verify: `SELECT * FROM infer_show_models();`

This loses registered models and any index built on them, which is the
reason it is documented as an alpha limitation rather than presented as a
procedure. Upgrade scripts land with the first non-alpha release; until
then a version bump is a reinstall.

### Upgrading larql-server

1. Build new version from upstream: `cd ~/ws/larql && git fetch upstream && git merge upstream/main && cargo build --release -p larql-server`
2. Stop the running larql-server
3. Replace the binary: `cp target/release/larql-server /usr/local/bin/`
4. Start the new server with the same arguments
5. Verify: `curl http://localhost:8080/v1/health`
6. From PostgreSQL: `SELECT * FROM infer_cache_stats();`

### Full Upgrade (Both Components)

1. Build new pg_infer: `cargo pgrx install --release`
2. Build new larql-server from upstream
3. Stop larql-server
4. Restart PostgreSQL (loads new extension .so)
5. Start new larql-server
6. Verify: `SELECT describe('test');`

**Important**: Always upgrade larql-server first when a wire protocol change
is involved, since pg_infer is a client. A newer pg_infer against an older
server may fail on missing response fields.

## Breaking Change Policy

### SQL Interface

- Removing or renaming a SQL function is a major version bump
- Changing return column types is a major version bump
- Adding optional parameters to existing functions is a minor version bump
- Adding new functions is a minor version bump

### Wire Protocol

- pg_infer 1.x will always speak `/v1`
- If larql introduces `/v2`, pg_infer will support both `/v1` and `/v2`
  simultaneously during a transition period
- Dropping `/v1` support would require pg_infer 2.0

### GUCs

- Removing a GUC is a major version bump
- Changing a GUC's default is a minor version bump (documented in CHANGELOG)
- Adding new GUCs is a minor version bump

### Vindex Format

- Vindex format changes are guarded by the `meta.json` version field
- pg_infer 1.x reads vindex format v1 only
- A new vindex format would be supported alongside v1, not replacing it

## Evolution Strategy

pg_infer tracks the upstream larql project selectively. Not all larql changes
are ported -- only those relevant to the wire protocol, vindex format, or
server behavior that pg_infer depends on.

### What Triggers a Sync

- Wire protocol changes (`/v1/` endpoint additions or modifications)
- Vindex format changes (new quantization types, metadata schema)
- Server CLI flag changes that affect deployment
- Bug fixes in endpoints pg_infer calls

### What Does NOT Trigger a Sync

- larql-lql (query language) changes -- pg_infer uses native SQL
- larql-cli changes -- pg_infer uses psql
- larql-python changes -- pg_infer provides PL/pgSQL
- Internal larql refactors that don't affect the wire protocol

## Testing Compatibility

```sh
# Run the client wire-contract tests:
cargo test -p infer-client

# With a live larql-server:
LARQL_SERVER=/usr/local/bin/larql-server \
LARQL_VINDEX=/data/model.vindex \
bash scripts/live_server_test.sh
```

The `infer-client` tests include a mock server that validates request/response
shapes against the `/v1` protocol specification. These tests catch wire
protocol regressions without requiring a running larql-server.
