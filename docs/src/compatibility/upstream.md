# Upstream Relationship

## Summary

pg_infer is an independent PostgreSQL extension inspired by the larql project.
It reimplements larql's core concepts (vindex, gate KNN, feature labeling) as
SQL-accessible functions inside PostgreSQL.

Upstream is **<https://github.com/chrishayuk/larql>**.

## Pinned Version

| Component | Commit | Date | Branch |
|-----------|--------|------|--------|
| larql upstream | 23a56db1 | 2026-09-13 | main (PR #476) |

The wire contract is pinned mechanically, not by prose: the server's own
OpenAPI document is committed at
`crates/infer-client/tests/fixtures/larql-openapi.json` and
`cargo test -p infer-client --test openapi_contract` checks every DTO
against it. See [Testing Compatibility](#testing-compatibility).

## What We Take from larql

- **larql-server**: external binary (deployed alongside pg_infer)
- **Wire protocol**: `/v1/describe`, `/v1/walk`, `/v1/stats`, `/v1/infer`,
  `/v1/relations`, `/v1/models`, `/v1/warmup`, `/v1/health`
- **Vindex format**: VINDEX2 (`index.json` schema 1--2) -- gate vectors,
  feature metadata, FFN weights
- **Conceptual**: activation cache keying, layer sharding, router protocol

## What We Don't Take

- larql-lql (SQL-like language -- pg_infer uses native SQL)
- larql-cli (pg_infer uses psql)
- larql-python (pg_infer provides PL/pgSQL interface)
- VINDEX3 (see [Vindex generations](#vindex-generations))

## pg_infer-Local Endpoints

These are **not** part of larql's `/v1` contract. They have never existed
upstream -- not at any pinned commit -- and are pg_infer inventions kept
behind 404 fallbacks so a real server simply reports them as unavailable:

| Endpoint | Status | Replacement |
|----------|--------|-------------|
| `/v1/cache/stats` | never upstream | `GET /v1/stats` -> `server` block (`v3_kv.{hits,misses,entries}`, `uptime_secs`, `requests_served`) |
| `/v1/rank` | never upstream | falls back to batch walks |
| `/v1/features` | never upstream | none; feature introspection needs a local vindex |
| `/v1/layers` | never upstream | derived from `/v1/stats` |

An earlier version of this page listed these alongside the real endpoints,
which implied a compatibility guarantee that does not exist.
`openapi_contract.rs::pg_infer_local_endpoints_are_absent_upstream` now
asserts their absence, so if upstream ever adds one the test fails and the
fallback can be dropped.

## Vindex Generations

pg_infer reads **VINDEX2 only** (`index.json` schema 1--2). Its
`VindexConfig` still matches upstream's V2 config field-for-field; the
required fields are identical.

VINDEX3 (schema 3+) is a different container *generation*, not a newer
revision of the same format: its own index shape, the LYRW v2 physical
layout, a system graph, and a fail-closed contract stack. Consuming it is a
project rather than a sync, and V2 remains first-class upstream
(`V2_MIN_SCHEMA=1..=2`, `detect_generation()` dispatches on it).

`VindexConfig::validate_supported()` therefore refuses:

- **format version outside 1--2** -- a V3 index deserialized against the V2
  struct would default every unshared field and then be read as a
  malformed V2;
- **`fp4`** and **`bitnet_layout`** -- both declare block geometry, scale
  dtypes and per-projection precision that determine how weight bytes
  decode. Upstream's spec is explicit that readers must dispatch on the
  declared tag and must not sniff filenames. Nothing in `infer-vindex` sets
  `deny_unknown_fields`, so before this check those fields were silently
  ignored and the bytes decoded under the wrong geometry. Note that
  upstream's `--keep-quant` extract writes `bitnet_layout`, so this is a
  container you can actually produce.

A refusal is cheap; a confident wrong answer is not.

## Compatibility Matrix

| pg_infer version | larql commit | Server API | Vindex format |
|-----------------|--------------|------------|---------------|
| 1.0.0 | 23a56db1+ | /v1 JSON | VINDEX2 (schema 1--2, Q4_K/Q6_K) |

## Sync Procedure

1. Fetch upstream: `cd ~/ws/larql && git fetch upstream`
2. Review `git log` since the pinned commit
3. Re-dump the OpenAPI fixture and diff it -- this is the mechanical part:

   ```sh
   larql-server <vindex> &
   curl -s localhost:8080/v1/openapi.json | jq . \
     > crates/infer-client/tests/fixtures/larql-openapi.json
   git diff crates/infer-client/tests/fixtures/larql-openapi.json
   ```

4. `cargo test -p infer-client` -- a wire change now fails a test instead of
   being noticed in production
5. Port relevant changes to the `infer-*` crates
6. Update the pinned commit in this file

## Known Divergences

| Area | pg_infer | larql | Notes |
|------|----------|-------|-------|
| Embedding server | Built-in `embed()` | `POST /v1/embed` | Endpoint exists upstream; the remote backend does not use it yet |
| VindexPatch | Not integrated | Full CRUD via `/v1/patches` | Future work |
| Grid discovery | HTTP `/v1/models` poll | Router proxies `/v1/walk-ffn` itself | See below |
| Predicate pushdown | Client-side filtering | `POST /v1/select` | Future work |
| Boundary codec | Not used | larql-boundary crate | Binary wire format, future |
| Capabilities | 404 probing | `GET /v1/capabilities` | Future work |

### Grid Discovery

larql-router's `/v1/models` reports model **ids only** -- no per-model URL.
The router is a proxy that fans `/v1/walk-ffn` out across shards itself, not
a shard directory, and it does **not** serve `/v1/describe`, `/v1/walk`,
`/v1/relations` or `/v1/infer` at all. So a grid URL is best pointed at a
seed `larql-server`, which does serve them; when a discovery entry names the
target model without a URL, pg_infer treats the discovery endpoint itself as
the server.

## Sync Decision Criteria

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
# Wire contract: DTOs vs the server's own OpenAPI document.
cargo test -p infer-client --test openapi_contract

# Everything, including the (self-written) mock round-trip.
cargo test -p infer-client
```

`openapi_contract.rs` exists because `mock_server_integration.rs` cannot
catch a wire divergence: both the mock and the parser are ours, so they
agree by construction. Three bugs shipped behind that green test -- the
`/v1/relations` field rename, the `/v1/models` envelope, and `/v1/warmup`
silently reading zeros. Two of them had never worked against a real server.

See [Versioning](versioning.md) for the full compatibility policy.
