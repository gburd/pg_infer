# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

pg_larql is a PostgreSQL extension (pgrx 0.17.0, PG18+ only) that exposes LARQL's neural network weight querying as SQL functions. It lets you query transformer model knowledge directly from SQL via `walk()`, `describe()`, `infer()`, `similar_to()`, and `implies()`.

## Build Commands

The build requires bypassing Nix compiler wrappers (they SIGSEGV) and providing explicit paths for OpenSSL and openblas:

```bash
# One-time setup: create openblas linker shim (no openblas-devel package installed)
mkdir -p /tmp/claude-1000/openblas-shim
ln -sf /usr/lib64/libopenblas.so.0 /tmp/claude-1000/openblas-shim/libopenblas.so

# Set build environment (required for every cargo command)
export CC=/usr/bin/clang
export CXX=/usr/bin/clang++
export AR=/usr/bin/ar
export LIBCLANG_PATH="/nix/store/czbh4a0ybqfrfx1c62xf2yry9yizx00l-clang-21.1.0-lib/lib"
export OPENSSL_LIB_DIR="/nix/store/llswcygvgv9x2sa3z6j7i0g5iqqmn5gn-openssl-3.6.0/lib"
export OPENSSL_INCLUDE_DIR="/nix/store/ydrckgnllgg8nmhdwni81h7xhcpnrlhd-openssl-3.6.0-dev/include"
export LD_LIBRARY_PATH="/usr/lib64:/nix/store/llswcygvgv9x2sa3z6j7i0g5iqqmn5gn-openssl-3.6.0/lib:$LD_LIBRARY_PATH"
export RUSTFLAGS="-L/tmp/claude-1000/openblas-shim -C linker=/usr/bin/cc"

# Check compilation (fast)
cargo check

# Full build
cargo build

# Build with inference support (heavy: pulls in wasmtime + protobuf-src)
cargo build --features inference

# Run in PostgreSQL (requires pgrx init)
cargo pgrx run pg18

# Run tests (10 pgrx integration tests)
cargo pgrx test pg18
```

All cargo commands must be run from `pg_larql/` (the crate directory), not the workspace root.

**Why bypass Nix wrappers:** The Nix `cc` and `ar` wrappers at `/nix/store/.../gcc-wrapper-.../bin/` crash with SIGSEGV. Using `/usr/bin/clang`, `/usr/bin/ar`, and `-C linker=/usr/bin/cc` avoids this entirely.

## Architecture

```
pg_larql/src/
  lib.rs          - pg_module_magic, extension_sql (schema + models table), _PG_init
  gucs.rs         - GUC parameters: larql.default_model, data_directory, max_memory, auto_download
  error.rs        - PgLarqlError enum with thiserror, maps to pgrx::error!()
  registry.rs     - Per-backend ModelHandle cache (Lazy<Mutex<HashMap>>), loads VectorIndex via mmap
  model_mgmt.rs   - larql_create_model(), larql_drop_model(), larql_models() SQL functions
  fn_walk.rs      - walk() SRF: raw gate activations per layer
  fn_describe.rs  - describe() SRF: labeled knowledge edges (deduped, filtered)
  fn_infer.rs     - infer() SRF: forward pass predictions (feature-gated behind "inference")
  fn_similar.rs   - similar_to() scalar + <~> operator: semantic similarity via shared gate activations
  fn_implies.rs   - implies() scalar: directional relationship test via describe
```

## Key Dependencies

- **LARQL crates** live in `_/larql/crates/` as path dependencies (larql-vindex, larql-compute, larql-models, optionally larql-inference)
- **pgrx 0.17.0** provides the PostgreSQL extension framework
- There is no workspace root Cargo.toml; pg_larql is a standalone crate (avoids Cargo workspace merge with LARQL's workspace)

## pgrx 0.17.0 API Notes

- String GUCs use `GucSetting<Option<CString>>` with C string literals (`c"value"`)
- SPI args use `&[DatumWithOid::from(value)]` (not the old `Option<Vec<(PgOid, Option<Datum>)>>`)
- `client.select()` takes `&[DatumWithOid]` for args, not `Option`
- `_PG_init` must use `extern "C-unwind"` (not `extern "C"`) with `#[pg_guard]`
- `TimestampWithTimeZone` is at `pgrx::datum::TimestampWithTimeZone`

## Model Registry

Models are registered in the `larql.models` table (created by extension_sql during CREATE EXTENSION). Each backend caches loaded VectorIndex handles in a process-local HashMap; the OS kernel shares the underlying mmap pages across PostgreSQL backends.

## Clippy Lints

`unwrap_used`, `panic`, and `panic_in_result_fn` are denied. All functions return `Result`.
