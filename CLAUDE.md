# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

pg_infer is a PostgreSQL extension (pgrx 0.17.0, PG18+ only) that exposes transformer model weight querying as SQL functions. It lets you query transformer model knowledge directly from SQL via `walk()`, `describe()`, `infer()`, `similar_to()`, and `implies()`. Inspired by the LARQL project.

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

All cargo commands must be run from `pg_infer/` (the workspace root).

**Why bypass Nix wrappers:** The Nix `cc` and `ar` wrappers at `/nix/store/.../gcc-wrapper-.../bin/` crash with SIGSEGV. Using `/usr/bin/clang`, `/usr/bin/ar`, and `-C linker=/usr/bin/cc` avoids this entirely.

## Architecture

```
pg_infer/
  Cargo.toml        - Workspace root + extension package
  src/
    lib.rs          - pg_module_magic, extension_sql (schema + models table), _PG_init
    gucs.rs         - GUC parameters: infer.default_model, data_directory, max_memory, auto_download, gate_threshold
    error.rs        - PgInferError enum with thiserror, maps to pgrx::error!()
    registry.rs     - Per-backend ModelHandle cache (Lazy<Mutex<HashMap>>), loads VectorIndex via mmap
    model_mgmt.rs   - infer_create_model(), infer_drop_model(), infer_models() SQL functions
    fn_walk.rs      - walk() SRF: raw gate activations per layer
    fn_describe.rs  - describe() SRF: labeled knowledge edges (deduped, filtered)
    fn_infer.rs     - infer() SRF: forward pass predictions (feature-gated behind "inference")
    fn_similar.rs   - similar_to() scalar + <~> operator: semantic similarity via shared gate activations
    fn_implies.rs   - implies() scalar: directional relationship test via describe
    am.rs           - Index access method: infer_am_handler() + IndexAmRoutine callbacks
    build.rs        - ambuild: reads vindex files, writes PG pages via GenericXLog
    pages.rs        - #[repr(C)] page format structs (meta, layer_dir, gate, embed, blob)
    scan.rs         - Scan callbacks (minimal — index queried via walk/describe functions)
    options.rs      - Reloptions parsing: WITH (source = '...')
  crates/
    infer-models/   - Model architecture definitions, config parsing, tensor key mappings
    infer-compute/  - Compute backends (CPU/BLAS, Metal GPU optional), matmul dispatch, Q4 kernels
    infer-vindex/   - VectorIndex loading, gate KNN queries, mmap management, tokenizer
    infer-core/     - Core graph engine, knowledge graph data types
    infer-inference/ - (optional) Transformer inference engine, forward pass
```

## Key Dependencies

- **Internal crates** live in `pg_infer/crates/` as workspace members (infer-vindex, infer-compute, infer-models, infer-core, optionally infer-inference)
- **pgrx 0.17.0** provides the PostgreSQL extension framework
- pg_infer is a Cargo workspace rooted at `pg_infer/Cargo.toml` with the extension as the main crate and internal crates under `crates/`

## pgrx 0.17.0 API Notes

- String GUCs use `GucSetting<Option<CString>>` with C string literals (`c"value"`)
- SPI args use `&[DatumWithOid::from(value)]` (not the old `Option<Vec<(PgOid, Option<Datum>)>>`)
- `client.select()` takes `&[DatumWithOid]` for args, not `Option`
- `_PG_init` must use `extern "C-unwind"` (not `extern "C"`) with `#[pg_guard]`
- `TimestampWithTimeZone` is at `pgrx::datum::TimestampWithTimeZone`

## Model Registry

Models are registered in the `infer.models` table (created by extension_sql during CREATE EXTENSION). Each backend caches loaded VectorIndex handles in a process-local HashMap; the OS kernel shares the underlying mmap pages across PostgreSQL backends.

## Clippy Lints

`unwrap_used`, `panic`, and `panic_in_result_fn` are denied. All functions return `Result`.
