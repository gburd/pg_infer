# Upstream Relationship

## Summary

pg_infer is an independent PostgreSQL extension inspired by the larql project.
It reimplements larql's core concepts (vindex, gate KNN, feature labeling) as
SQL-accessible functions inside PostgreSQL.

## Pinned Version

| Component | Commit | Date | Branch |
|-----------|--------|------|--------|
| larql upstream | c880fb7 | 2026-05-12 | main (PR #87) |

## What We Take from larql

- **larql-server**: external binary (deployed alongside pg_infer)
- **Wire protocol**: /v1/describe, /v1/walk, /v1/stats, /v1/infer, /v1/health
- **Vindex format**: gate vectors, feature metadata, FFN weights
- **Conceptual**: activation cache keying, layer sharding, router protocol

## What We Don't Take

- larql-lql (SQL-like language -- pg_infer uses native SQL)
- larql-cli (pg_infer uses psql)
- larql-python (pg_infer provides PL/pgSQL interface)

## Compatibility Matrix

| pg_infer version | larql commit | Server API | Vindex format |
|-----------------|--------------|------------|---------------|
| 1.0.0 | c880fb7+ | /v1 JSON | v1 (Q4_K/Q6_K) |

## Sync Procedure

1. Fetch upstream: `cd _/larql && git pull`
2. Review CHANGELOG / git log since pinned commit
3. Identify breaking changes to: wire protocol, vindex format, server CLI flags
4. Port relevant changes to infer-* crates
5. Update pinned commit in this file
6. Run compatibility tests: `cargo test -p infer-client`

## Known Divergences

| Area | pg_infer | larql | Notes |
|------|----------|-------|-------|
| Embedding server | Built-in embed() | Separate /v1/embed endpoint | pg_infer can use either |
| VindexPatch | Not integrated | Full CRUD via /v1/patches | Future work |
| Grid protocol | Simple HTTP round-robin | gRPC bidirectional streaming | pg_infer uses HTTP discovery |
| Boundary codec | Not used | larql-boundary crate | Binary wire format, future |

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

See [Versioning](versioning.md) for the full compatibility policy.
