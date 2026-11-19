# Versioning and Compatibility

This chapter describes pg_infer's version strategy, compatibility with the
upstream larql project, and upgrade procedures.

## Version Strategy

pg_infer uses [semantic versioning](https://semver.org/) (major.minor.patch):

- **Major**: Breaking SQL interface changes (function signatures, return types,
  removed functions)
- **Minor**: New SQL functions, new GUCs, new backend capabilities
- **Patch**: Bug fixes, performance improvements, internal refactors

During the `0.x` alpha that strategy is relaxed in one direction: a renamed
result column ships as a MINOR bump rather than a major one, because there
are no upgrade scripts yet (see [Upgrading pg_infer](#upgrading-pg_infer-extension-only))
and a version bump is already a reinstall.  META.json says the same thing
less formally -- "the SQL API and the vindex format may change between
releases".  Once the first non-alpha release lands, a column rename becomes
a major bump like any other signature change.

Current version: **0.1.5-alpha**

### Renamed result columns

| release | function(s) | was | now |
|---|---|---|---|
| 0.1.5-alpha | `describe`, `describe_many` | `confidence` | `gate_score` |
| 0.1.5-alpha | `walk` | `activation` | `gate_score` |
| 0.1.5-alpha | `nearest_to` | `score` | `gate_score` |

One quantity had four names across the surface, and `confidence` was the
actively misleading one: it holds an unnormalised dot product between a
pooled embedding and a gate vector, whose scale moves with the model (20--45
on Gemma-class weights, 300+ on BitNet b1.58).  Reading it as a probability,
or carrying a threshold from one model to another, both gave wrong answers.

Queries that referenced the old names by position (`SELECT * FROM
describe(...)`) are unaffected.  Queries naming the column need a one-word
edit:

```sql
-- before
SELECT target, confidence FROM describe('France');
-- after
SELECT target, gate_score FROM describe('France');
```

`infer_show_features` still returns `score` and `infer_diff` still returns
`score_a`/`score_b`; neither changed.

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
| 0.1.5-alpha | 18, 19 | 3b5dd3d2+ (2026-11-18) | VINDEX2 schema 1--2 (Q4_K/Q6_K) | 1.80+ |

### PostgreSQL 19

Built and tested on 19 as of 2026-11-18, where the newest release is
**rc2** -- there is no 19.0 GA tarball yet, so treat 19 support as ready
rather than shipped.  Both majors run the same 55 pgrx tests in CI
(`pg: [18, 19]`).

Two PG19 source changes reach an index access method, and pg_infer handles
both without a behaviour change:

- `table_beginscan()` takes a new trailing `flags` bitmask.  It is OR'd on
  top of the internal defaults earlier versions always applied
  (`SO_TYPE_SEQSCAN | SO_ALLOW_STRAT | SO_ALLOW_SYNC | SO_ALLOW_PAGEMODE`),
  so passing 0 requests exactly the PG18 behaviour.
- `BUFFER_LOCK_*` became a `BufferLockMode` enum over `c_uint`, where it
  had been plain `#define`s reaching `LockBuffer()` as a signed `c_int`.
  The eight call sites go through `am_pages::lock_buffer()`, which casts to
  the parameter's own type and so compiles on both.  The values are
  unchanged.

This also required pgrx 0.17 -> 0.19.4: 0.17 has no `pg19` feature at all.
pgrx 0.19 drops the `pgrx_embed!` macro and the `src/bin/pgrx_embed.rs`
binary that called it, so that file is gone.

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
