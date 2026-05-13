# Architecture

pg_infer is a PostgreSQL extension that provides SQL-accessible neural network
inference. It exposes transformer model internals (gate vectors, sparse
features, activation patterns) as queryable relations within PostgreSQL.

## Deployment Topologies

pg_infer supports three deployment modes, selectable per model:

### Local (mmap)

```
psql --> PostgreSQL --> pg_infer --> mmap vindex file
```

Each PostgreSQL backend memory-maps the vindex directory directly. Simple to
deploy but memory-intensive: every connection holds its own f16-to-f32 decode
cache. Best for single-user exploration or small models.

**Tradeoffs**: No shared cache. Each `similar_to` call redoes gate-KNN from
scratch. Tops out at 2-3 concurrent users on a 3B-parameter model before OOM.

### Remote (larql-server)

```
psql --> PostgreSQL --> pg_infer --> HTTP/UDS --> larql-server --> vindex
```

A dedicated `larql-server` process owns the mmap and a three-tier activation
cache. Every pg_infer backend is a thin HTTP client. One copy of the model
serves all connections with 95%+ cache hit rate on recurring queries.

**Tradeoffs**: Requires running the external larql-server binary. Adds one
network hop (sub-millisecond over UDS). Feature-level enumeration
(`show_features`, `infer_diff`) is not available remotely.

### Grid (larql-router)

```
psql --> PostgreSQL --> pg_infer --> HTTP round-robin
                                        |
                         +--------------+---------------+
                         v              v               v
                   larql-server    larql-server    larql-server
                   (layers 0-7)   (layers 8-15)  (layers 16-23)
```

Multiple larql-server instances, each hosting a shard of the model's layers.
A `larql-router` exposes `/v1/models` for auto-discovery. pg_infer discovers
servers via HTTP polling and dispatches queries round-robin.

**Tradeoffs**: Enables models too large for a single host. Adds discovery
latency on first query. Grid topology changes are detected within the poll
interval (default 30s).

## Data Flow

A typical query like `SELECT * FROM describe('France')` follows this path:

1. **SQL function** (`src/fn_describe.rs`) receives the call
2. **Model registry** (`src/registry.rs`) resolves the model name to a `Backend`
3. **Backend dispatch** (`src/backend/mod.rs`) routes to local, remote, or grid
4. **Execution**:
   - Local: gate-KNN across layers, feature metadata lookup, edge assembly
   - Remote: single `GET /v1/describe?entity=France` to larql-server
   - Grid: round-robin selection, then same HTTP call as remote
5. **Result assembly**: backend returns `Vec<Edge>`, SQL function formats as rows

## Crate Architecture

```
pg_infer (PostgreSQL extension, pgrx)
+-- infer-core        Core types, vindex loading, gate-KNN, feature metadata
+-- infer-inference   Full forward pass (optional, behind "inference" feature)
+-- infer-compute     SIMD kernels: Q4_K/Q6_K dequant, f16<->f32, dot products
+-- infer-vindex      Vindex format parser, mmap management, layer iteration
+-- infer-models      Model detection (architecture sniffing from GGUF metadata)
+-- infer-client      HTTP client for remote/grid backends (CancellableClient)
```

### pg_infer (root crate)

The PostgreSQL extension itself. Contains:
- SQL function implementations (`fn_describe`, `fn_walk`, `fn_similar`, etc.)
- GUC definitions (`gucs.rs`)
- Backend trait and dispatch (`backend/mod.rs`)
- Custom access method for `ORDER BY <~>` index scans (`am_*.rs`)
- Model management and registry (`model_mgmt.rs`, `registry.rs`)

### infer-core

Shared types and the local execution engine. Loads vindex directories, runs
gate-KNN queries, resolves feature metadata to human-readable tokens.

### infer-inference

Optional crate (behind the `inference` feature flag) that provides full
transformer forward-pass inference via GGUF model weights. Heavy dependency
chain (wasmtime, protobuf-src).

### infer-compute

Low-level SIMD compute kernels:
- Q4_K and Q6_K dequantization (AVX2, NEON, scalar fallbacks)
- f16 to f32 conversion (batch)
- Dot product and cosine similarity
- SQ8 quantized distance for HNSW index scans

### infer-vindex

Vindex format parser. A vindex is the on-disk representation of a model's
interpretability data: gate vectors per layer, feature metadata (top tokens,
c-scores), and FFN weight slices for inference.

### infer-models

Model architecture detection. Reads GGUF metadata to identify model family
(Qwen, Llama, Mistral, DeepSeek, etc.) and extract structural parameters
(num_layers, hidden_size, num_experts).

### infer-client

HTTP client library for communicating with larql-server. Provides:
- `CancellableClient`: HTTP/2 client with PostgreSQL interrupt integration
- `CancelToken`: cooperative cancellation for in-flight requests
- Wire protocol types for `/v1/describe`, `/v1/walk`, `/v1/stats`, `/v1/infer`

## Wire Protocol

The remote and grid backends communicate with larql-server over a JSON HTTP
API (the `/v1` protocol):

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/v1/health` | GET | Liveness check |
| `/v1/stats` | GET | Model metadata (num_layers, hidden_size, model name) |
| `/v1/describe` | GET | Knowledge edges for an entity |
| `/v1/walk` | GET | Top-K feature activations for a prompt |
| `/v1/infer` | GET | Next-token predictions |
| `/v1/models` | GET | Grid: list available models and their server URLs |

## Vindex Format

A vindex directory contains the preprocessed interpretability data for one
model checkpoint:

```
model.vindex/
+-- meta.json          Model metadata, layer count, hidden size
+-- gates/
|   +-- layer_00.bin   Gate vectors (f16, one per feature per layer)
|   +-- layer_01.bin
|   +-- ...
+-- features/
|   +-- layer_00.json  Feature metadata (top token, c-score, also-tokens)
|   +-- ...
+-- ffn/               (optional) FFN weight slices for inference
    +-- layer_00.bin
    +-- ...
```

Gate vectors are stored as f16 and decoded to f32 on demand (with an LRU
cache bounded by `infer.gate_cache_max_layers`). The Q4_K and Q6_K
quantization formats are supported for reduced storage.
