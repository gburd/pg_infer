# Roadmap — infer-vindex

## Current State

- 146 tests passing, 0 build warnings
- 3 storage formats: f32, Q8, Q4_K/Q6_K (Ollama-compatible)
- Mmap zero-copy with adaptive residency
- HNSW graph index for sub-linear KNN
- Patch system for editable knowledge

## P0: Support Cached Layer Decode

### Store pre-computed residuals for template-fixed layers (L0-12) — DONE
**Impact**: Enables 155+ tok/s decode (skip 13 of 21 layers)
**Effort**: Medium
**Status**: ✅ Complete (2026-05) — on-disk format + extraction function implemented

The vindex stores cached residuals per template. During extraction, `extract_residual_cache()` runs one forward pass per template through L0-12 and saves the output residual. At decode time, look up the cached residual instead of computing 13 layers.

### Wire Q4_K FFN consumption (interleaved_q4k.bin) — DONE
**Impact**: Match Ollama's exact FFN quantization  
**Effort**: Medium  
**Status**: ✅ Complete (2026-04-07)

Added `load_interleaved_q4k()`, `has_interleaved_q4k()`, `interleaved_q4k_mmap_ref()` to vindex.
Inference `predict_honest` now prefers Q4_K FFN (`interleaved_q4k.bin`) over Q4_0.
Format tag (`ffn_format`) passed through `FullPipelineLayer` to compute for shader dispatch.

### GGUF Q4_K format option (144 bytes vs 148 bytes) — DONE
**Impact**: Direct compatibility with llama.cpp weight files
**Effort**: Low
**Status**: ✅ Complete (2026-05)

`quantize_q4_k_gguf` + `dequantize_q4_k_gguf` in infer-compute, `write_attention_weights_q4k_gguf` in infer-vindex. Manifest entries tagged `Q4_K_GGUF` for explicit GGUF-compatibility identification. Note: `quantize_q4_k` already produced the GGUF-compatible 144-byte layout; the `_gguf` variants make this explicit in the API and documentation.

## P1: Production Hardening

### HuggingFace resolution in Vindexfile — DONE
**Effort**: Medium
**Status**: ✅ Complete (2026-05)

FROM directive in Vindexfile resolves `hf://user/repo` paths via `resolve_hf_vindex()`.

### Streaming extraction checkpoints — DONE
**Effort**: Medium
**Status**: ✅ Complete (2026-05)

`write_checkpoint()` / `has_checkpoint()` + `build_vindex_resume()` entry point for interrupted builds.

### Q4_K FFN in vindex — DONE
**Effort**: Low
**Status**: ✅ Complete (2026-05)

Extraction now emits `interleaved_q4k.bin` (Q4_K gate/up + Q6_K down) by default for all
dense models regardless of `--quant` setting. Both the streaming and non-streaming build
paths call `write_ffn_interleaved_q4k()`. Inference already prefers Q4_K when the file is
present; legacy Q4_0 (`interleaved_q4.bin`) remains a supported fallback for old vindexes.

## P2: Research

### Multi-model vindex
Store features from multiple models in one vindex. Compare representations across architectures.

### Incremental extraction
Add new layers/features to an existing vindex without full rebuild.

## Completed

| Item | Date | Impact |
|------|------|--------|
| HuggingFace vindexfile resolution | 2026-05 | `hf://` paths in FROM directives |
| Streaming extraction checkpoints | 2026-05 | Resume interrupted builds |
| Relation cluster loading | 2026-05 | Feature→relation label mapping |
| Core VectorIndex with mmap | 2026-03 | Foundation |
| Gate KNN (brute-force + BLAS) | 2026-03 | Walk engine |
| Walk FFN (per-feature down/up vectors) | 2026-03 | Sparse inference |
| Binary down_meta format | 2026-03 | 5x compression vs JSONL |
| F16 storage + decode cache | 2026-03 | 2x smaller gate vectors |
| Interleaved layout (gate\|up\|down packed) | 2026-04 | Reduced TLB thrash |
| Q4_0 gate vectors + interleaved | 2026-04 | 7x smaller gates |
| HNSW graph index | 2026-04 | Sub-linear KNN |
| Adaptive residency (pin/evict) | 2026-04 | Memory budget management |
| Patch system (PatchedVindex) | 2026-04 | Editable knowledge |
| MoE expert routing | 2026-04 | Mixtral/DeepSeek support |
| Q4_K/Q6_K attention weights | 2026-04 | Ollama-compatible |
| Q8 attention weights | 2026-04 | Higher precision option |
| Streaming extraction (mmap, per-layer) | 2026-04 | ~2 GB peak RAM |
| Safety doc for mmap_optimized | 2026-04-07 | Clippy compliance |
| VindexPatch::is_empty() | 2026-04-07 | API completeness |
| Q4_K FFN loader + wiring | 2026-04-07 | `interleaved_q4k.bin` end-to-end |
| Quantizer single source of truth | 2026-04-07 | Builder uses infer-compute (ADR-008) |
| Example cleanup (13→11) | 2026-04-07 | Removed Q4_0 attn + Q4_0 interleaved |
| 8 ADRs documented | 2026-04-07 | All major decisions recorded |
| PERFORMANCE.md + format alignment | 2026-04-07 | Fresh benchmarks, verified pipeline |
| Q4_K FFN in vindex | 2026-05 | Default Q4_K FFN for dense models |
| GGUF Q4_K format option | 2026-05 | `quantize_q4_k_gguf` + vindex writer |
