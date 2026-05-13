# Versioning and Compatibility

This chapter describes pg_infer's version strategy, compatibility with the
upstream larql project, and upgrade procedures.

## Version Strategy

pg_infer uses [semantic versioning](https://semver.org/) (major.minor.patch):

- **Major**: Breaking SQL interface changes (function signatures, return types,
  removed functions)
- **Minor**: New SQL functions, new GUCs, new backend capabilities
- **Patch**: Bug fixes, performance improvements, internal refactors

Current version: **1.0.0**

## larql Server Compatibility

Compatibility between pg_infer and larql-server is defined by the `/v1` wire
protocol version. Any larql-server that speaks the `/v1` JSON API is compatible.

### Wire Protocol: /v1

The `/v1` protocol is a JSON-over-HTTP API with these endpoints:

| Endpoint | Request | Response | Since |
|----------|---------|----------|-------|
| `GET /v1/health` | -- | `{"status":"ok"}` | Initial |
| `GET /v1/stats` | -- | Model metadata JSON | Initial |
| `GET /v1/describe` | `?entity=...` | `{"edges":[...]}` | Initial |
| `GET /v1/walk` | `?prompt=...&top_k=N` | `{"hits":[...]}` | Initial |
| `GET /v1/infer` | `?prompt=...&top_k=N` | `{"predictions":[...]}` | Initial |
| `GET /v1/models` | -- | `{"models":[...]}` | Grid support |

pg_infer's `infer-client` crate validates responses against expected shapes.
Unknown fields are ignored (forward-compatible). Missing required fields cause
a parse error surfaced as a SQL `ERROR`.

## Supported Configurations

| pg_infer | PostgreSQL | larql-server | Vindex Format | Rust Toolchain |
|----------|------------|--------------|---------------|----------------|
| 1.0.0 | 18+ | c880fb7+ (2026-05-12) | v1 (Q4_K/Q6_K) | 1.80+ |

### Vindex Format Versions

- **v1**: Gate vectors in f16, Q4_K, or Q6_K quantization. Feature metadata as
  JSON. FFN weights as raw binary slices. This is the only format pg_infer 1.0
  supports.

## Upgrade Procedure

### Upgrading pg_infer (Extension Only)

1. Stop active queries (or schedule during maintenance window)
2. Build new version: `cargo pgrx install --release`
3. In PostgreSQL: `ALTER EXTENSION pg_infer UPDATE;`
4. Verify: `SELECT * FROM infer_show_models();`

### Upgrading larql-server

1. Build new version from upstream: `cd larql && git pull && cargo build --release -p larql-server`
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
