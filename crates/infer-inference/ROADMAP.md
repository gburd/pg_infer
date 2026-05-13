# Roadmap — infer-inference

## Current: 4.9 tok/s honest (real model) | 59 tok/s GPU synthetic | Ollama: 97 tok/s

## P0: Close Ollama Gap

### Fix GPU prefill for post-norm models (Gemma3)
**Impact**: 203ms → ~17ms honest with GPU prefill  
**Effort**: Medium  
**Status**: In progress — Metal done, CUDA port complete (hardware validation pending)

The GPU `prefill_q4` path produces wrong output for Gemma3 post-norm architecture.
Root cause: `prefill.rs` doesn't mirror `full_pipeline.rs`'s post-norm handling.
CPU fallback is correct. See infer-compute ADR-009.

### Wire KV-cached decode into honest path — DONE
**Impact**: 4.9 tok/s → 59+ tok/s decode
**Effort**: Low
**Status**: ✅ Complete (2026-05)

`kv_generate.rs` implements `generate_cached()`, `generate_cached_backend()`, and
`generate_cached_with_window()`. Prefill → decode loop with KV cache persistence.

### Merge per-layer dispatches
**Impact**: ~30% speedup on GPU path  
**Effort**: Medium  
**Status**: Identified in compute component profiling

Currently 7 encoders per layer. Merging norm+QKV+attend+O+FFN into fewer encoders
would save ~8ms on the 34-layer GPU path.

## P1: Production Hardening

### Lift MarkovResidualEngine into infer-inference — DONE
**Impact**: First-class KV-cache-free decode path; unblocks long-context use cases where KV memory is the bottleneck.
**Effort**: Medium
**Status**: ✅ Complete (2026-05)

Implemented in `engines/markov_residual/`. Stores per-layer pre-attention residuals; recomputes K/V on the fly from stored residuals during decode. Bit-identical to standard forward on TinyModel (validated by `markov_rs_prefill_matches_standard_forward` test). Public API via `generate_cached_with_window()`.

### Clean up experimental FFN backends — N/A
**Effort**: Low
**Status**: ✅ Stale — no `ffn/experimental/` directory exists. Production backends only.

### Example reorganization
**Effort**: Low
**Status**: ✅ Complete (2026-05) — examples renamed with prefix convention

49 examples organized with prefix-based naming:
`demo_` (8), `bench_` (17), `profile_` (10), `test_` (14)

### Add doc tests — DONE
**Effort**: Low
**Status**: ✅ Complete (2026-05)

18 doc tests across `attention/` (rope, decode, block, mod), `forward/` (embed, mod, predict, ple),
`layer_graph/` (mod, cached, dense, template), `error.rs`, and `tokenizer.rs`.

## P2: Research

### Template-guided walk (restrict feature universe)
Pre-compute per-template feature sets. Only score features in the template's universe.
Reduces gate KNN work for known entity types.

### Multi-token generation loop — DONE
Implemented in `forward/kv_generate.rs`. `generate_cached()` performs prefill → decode loop
with persistent KV cache. Sliding-window variant available via `generate_cached_with_window()`.

## Completed

| Item | Date | Impact |
|------|------|--------|
| KV-cached generation loop | 2026-05 | Prefill + decode with KV cache |
| MarkovResidualEngine | 2026-05 | KV-free decode via residual storage |
| Attention backward (target-delta) | 2026-05 | Last-pos gradient for optimization |
| CachedLayerGraph predict wiring | 2026-05 | Skip layers via cached residuals |
| Config validation | 2026-05 | Architecture invariant checks |
| Forward pass (CPU BLAS) | 2026-03 | Foundation |
| BLAS-fused attention | 2026-04-03 | Online softmax, O(seq) memory |
| WalkFfn (sparse FFN via vindex) | 2026-04-03 | Gate KNN + top-K |
| CachedLayerGraph | 2026-04-04 | Skip L0-12, 0.999 cosine |
| LayerGraph trait | 2026-04-04 | Pluggable per-layer routing |
| predict_honest | 2026-04-06 | Production path, GPU+CPU hybrid |
| GPU prefill pipeline | 2026-04-06 | seq>1 on GPU (pre-norm models) |
| Q4_K FFN format wiring | 2026-04-07 | Vindex Q4_K FFN → FullPipelineLayer |
| GELU-tanh activation | 2026-04-07 | Gemma3 correct on GPU |
| Post-norm guard | 2026-04-07 | Gemma3 falls to CPU correctly |
| Zero warnings | 2026-04-07 | Clean build |
| PERFORMANCE.md | 2026-04-07 | Benchmark data documented |
| Doc tests | 2026-05 | 18 examples across attention, forward, layer_graph |
