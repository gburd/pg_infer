# Serving BitNet (bitnet-b1.58-2B-4T)

[`microsoft/bitnet-b1.58-2B-4T`](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T)
is a 2B-parameter model whose linear weights are ternary `{-1, 0, +1}` with a
per-channel `f32` scale (the "b1.58" / I2_S layout). At full precision the
working set is ~5 GB; served from its native I2_S bytes it stays around
~1.1 GB resident. This page covers the two ways pg_infer can serve it.

| Path | Status | Where inference runs |
|------|--------|----------------------|
| **A — Remote** (recommended for production today) | Ready | External `larql-server` |
| **B — Native local** | Forward path in place; vindex loader deferred | In-process (`infer-inference`) |

## Path A — Remote via larql-server (production)

Upstream `larql-server` already serves BitNet natively. It detects the I2_S
layout from the vindex's `bitnet_layout` field, loads the ternary weights into
a `BitnetModel`, and dispatches generation through its ternary kernels. pg_infer
talks to it exactly like any other remote model — over `/v1` JSON — so nothing
in pg_infer needs to know the model is ternary.

```
psql --> PostgreSQL --> pg_infer --> HTTP/UDS --> larql-server --> bitnet/ I2_S weights
```

### 1. Build the vindex with `--keep-quant`

The BitNet vindex is built by the **upstream larql CLI**, not by pg_infer. The
`--keep-quant` flag copies the I2_S bytes verbatim instead of dequantizing them,
producing a `bitnet/` subdirectory and stamping `bitnet_layout` into
`index.json`:

```sh
# From the upstream larql checkout:
larql gguf-to-vindex \
    --input  bitnet-b1.58-2B-4T.i2_s.gguf \
    --output /data/bitnet-2b.vindex \
    --keep-quant
```

This writes, per layer, `bitnet/blk.N.{attn_q,attn_k,attn_v,attn_o,ffn_gate,ffn_up,ffn_down}.weight.i2s`
plus a single `bitnet/scales.f32` (all per-channel scales concatenated) and a
top-level `bitnet_layout.json`. `--keep-quant` is a no-op (with a warning) for
non-BitNet architectures.

### 2. Start the server

```sh
larql-server /data/bitnet-2b.vindex \
    --uds-path /run/larql.sock \
    --max-gate-cache-layers 8
```

The server loads the BitNet model lazily on first inference request. Confirm it
came up:

```sh
curl --unix-socket /run/larql.sock http://localhost/v1/health
# {"status":"ok"}
curl --unix-socket /run/larql.sock http://localhost/v1/stats
# {"model":"bitnet-b1.58-2B-4T","num_layers":30,"hidden_size":2560,...}
```

### 3. Register in PostgreSQL

```sql
CREATE EXTENSION pg_infer;

SELECT infer_create_model_remote('bitnet2b', 'uds:///run/larql.sock');
SET infer.default_model = 'bitnet2b';
```

`infer_create_model_remote` issues one `GET /v1/stats` at registration to cache
`num_layers` and `hidden_size`; the server is hit for actual inference only. From
here every pg_infer function (`describe`, `walk`, `similar_to`, …) works against
the ternary model with no further configuration — the ternary dispatch is
entirely server-side.

See [Remote (larql-server) Deployment](remote.md) for GUCs, cancellation,
pooling, and benchmarking that apply unchanged here.

## Path B — Native local inference (in-process)

For a single-process deployment with no separate server, pg_infer carries its
own BitNet forward path in `infer-inference` (`ternary.rs`): I2_S BitLinear
matvec with per-channel scale (`infer-compute`'s `BitLinearWeight` /
`matvec_i2s_f32_into`), squared-ReLU FFN with `ffn_sub_norm`, GQA attention with
`attn_sub_norm`, a KV cache, and greedy generation. It mirrors the upstream
ternary core rather than depending on it.

**Status — what works and what is pending:**

- The forward path (prefill, `decode_step`, `generate_greedy`) is implemented
  and its pure-Rust logic has been validated in a standalone harness
  (decode/prefill equivalence, GQA, FFN, rmsnorm, argmax). It has **not** been
  run through the in-tree `cargo test`, because the sandbox toolchain cannot
  link `infer-compute`'s BLAS/C kernels (see the contributing notes); in-tree
  verification on real hardware is still required before relying on this path.
- The vindex-side loader that materializes a `BitnetModel` from the `bitnet/`
  artifacts (`load_bitnet_model`) is **deferred**. Until it lands, Path B cannot
  be wired to a registered model end-to-end, so Path A is the supported route
  for production.

When the loader is complete, a locally-registered BitNet model
(`infer_create_model('bitnet2b', '/data/bitnet-2b.vindex')`) will run entirely
in the PostgreSQL backend with no server hop.

## Which path should I use?

- **Production, now:** Path A. It is complete, shares the activation cache
  across all connections, and isolates the model's memory in one process.
- **Embedded / single-backend, later:** Path B, once the vindex loader is wired
  and the forward path has been verified in-tree on real hardware.
