# Contributing

Build instructions, test commands, and development workflow for pg_infer.

## Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| Rust | 1.80+ | `rustup update stable` |
| PostgreSQL | 18+ | Development headers required |
| pgrx | 0.17 | `cargo install cargo-pgrx --version 0.17` |
| cargo-pgrx init | -- | `cargo pgrx init --pg18 $(which pg_config)` |

## Building

### Build the Extension

```sh
# Development build (faster compile, debug symbols):
cargo pgrx run pg18

# Release build (optimized, for benchmarking):
cargo pgrx install --release
```

### Build larql-server (for remote/grid testing)

```sh
cd _/larql
cargo build --release -p larql-server
cp target/release/larql-server /usr/local/bin/
```

## Crate Structure

```
pg_infer/                  PostgreSQL extension (pgrx)
  crates/
    infer-core/            Core types, vindex loading, gate-KNN
    infer-inference/       Full forward pass (optional, "inference" feature)
    infer-compute/         SIMD kernels: dequant, f16<->f32, dot products
    infer-vindex/          Vindex format parser, mmap management
    infer-models/          Model architecture detection from GGUF
    infer-client/          HTTP client for remote/grid backends
```

## Running Tests

```sh
# All unit tests:
cargo test --all

# Specific crate:
cargo test -p infer-core
cargo test -p infer-client
cargo test -p infer-compute

# With the inference feature (heavy dependencies):
cargo test --features inference

# Extension integration tests (requires PostgreSQL):
cargo pgrx test pg18

# Client wire-contract tests (mock server):
cargo test -p infer-client

# Live server integration test:
LARQL_SERVER=/usr/local/bin/larql-server \
LARQL_VINDEX=/data/model.vindex \
PGDATABASE=test \
bash scripts/live_server_test.sh
```

## Development Workflow

### Adding a New SQL Function

1. Create the function in `src/fn_<name>.rs`
2. Register it with `#[pg_extern]` (pgrx macro)
3. Add it to the SQL migration file
4. Write a test in the same file or `tests/`
5. Run `cargo pgrx test pg18`

### Adding a New GUC

1. Define the GUC in `src/gucs.rs`
2. Register it in `_PG_init()`
3. Add documentation to `docs/src/operations/tuning.md`
4. Write a test verifying the default and bounds

### Modifying the Wire Protocol

1. Update types in `crates/infer-client/src/types.rs`
2. Update mock server in `crates/infer-client/src/mock.rs`
3. Run `cargo test -p infer-client`
4. Update [compatibility documentation](../compatibility/versioning.md)

## Code Style

- Follow `cargo fmt` and `cargo clippy`
- Use `unsafe` only in SIMD kernels (`infer-compute`) with safety comments
- PostgreSQL-facing code uses pgrx idioms (SPI, GUC macros, `#[pg_extern]`)
- Internal crates use standard Rust patterns

## Benchmarking

```sh
# Apply benchmark schema:
psql -d test -f benches/schema.sql

# Single-call workload:
pgbench -n -c 1  -T 30 -f benches/pgbench_similar_to.sql
pgbench -n -c 32 -T 60 -f benches/pgbench_similar_to.sql

# Table-scan re-ranking:
pgbench -n -c 1  -T 30 -f benches/pgbench_semantic_rerank.sql
pgbench -n -c 32 -T 60 -f benches/pgbench_semantic_rerank.sql
```

See [Benchmarks](../performance/benchmarks.md) for expected results and methodology.

## Architecture Overview

For a detailed understanding of the system design, see [Architecture](../architecture.md).

Key concepts:
- **Vindex**: On-disk interpretability data (gate vectors + feature metadata)
- **Gate KNN**: Core query primitive (find top-K activated features for a residual)
- **Backend**: Abstraction over local/remote/grid execution
- **Access Method**: PostgreSQL custom AM for `ORDER BY <~>` index scans
