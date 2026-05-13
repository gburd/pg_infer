# Roadmap — infer-models

## Current: 12 architectures, 130 tests, safetensors + GGUF loading

## P0: Complete Gemma 4 Support

### Wire v_shares_k into inference forward pass — DONE
**Impact**: Correct K=V handling without runtime tensor probing
**Effort**: Low
**Status**: ✅ Complete (2026-05)

Architecture config is now authoritative. `attention/block.rs` uses `arch.v_shares_k(layer)` directly with a `debug_assert` validating tensor presence.

### Validate PLE (per-layer embeddings) end-to-end — DONE
**Impact**: Correct Gemma 4 E2B inference
**Effort**: Medium
**Status**: ✅ Complete (2026-05) — forward pass wired, E2E test validates precompute + gate + contribution

PLE adds a gated embedding lookup per layer. Keys (`per_layer_embed_key`, `per_layer_input_gate_key`, `per_layer_projection_key`, `post_per_layer_input_norm_key`) are all implemented. Forward pass integration in `forward/ple.rs` with `precompute_per_layer_inputs` + `apply_per_layer_embedding`. E2E test at `infer-inference/tests/test_ple_e2e.rs`.

### KV layer sharing in inference — DONE
**Impact**: Memory savings for Gemma 4 (20 shared layers = 20 fewer KV caches)
**Effort**: Medium
**Status**: ✅ Complete (2026-05) — decode loop uses source layer's cache for shared layers

`run_attention_block_decode_step_shared` computes only Q + attention against source layer's
KV cache. Decode loop in `kv_generate.rs` skips K/V projection and cache storage for shared
layers. Prefill skips cache storage for shared layers. Saves ~60% KV memory on Gemma 4.

## P1: Architecture Coverage

### Phi-3 / Phi-4 — DONE
**Effort**: Low
**Status**: ✅ Complete (2026-05)

`architectures/phi.rs` — partial RoPE, SuRoPE scaling. Registered in detect.rs.

### Command R / Cohere — DONE
**Effort**: Medium
**Status**: ✅ Complete (2026-05)

`architectures/cohere.rs` — different attention key pattern, LayerNorm. Registered in detect.rs.

### Mamba / state-space models
**Effort**: Large  
**Status**: Research

Would require extending the trait beyond transformer assumptions (no attention keys, no KV cache). May warrant a separate trait hierarchy.

## P2: Loading Improvements

### Streaming safetensors loading — DONE
**Effort**: Medium
**Status**: ✅ Complete (2026-05)

Implemented in infer-vindex `extract/streaming.rs` — mmap + on-demand per-tensor deserialize. Peak memory = 1 layer + embeddings.

### GGUF quantized inference (skip dequant)
**Effort**: Large  
**Status**: Not started

Currently GGUF tensors are dequantized to f32 during loading. For Q4_K/Q6_K formats, keep data in quantized form and pass directly to `infer-compute` Q4_K shaders. Requires a `QuantizedWeights` variant alongside `ModelWeights`.

### MLX npz/safetensors hybrid
**Effort**: Low  
**Status**: Partial (MLX safetensors work, npz not yet)

Apple MLX models sometimes use `.npz` format. Add npz parsing alongside safetensors.

## P3: Trait Evolution

### Per-layer FFN type — DONE
**Effort**: Low
**Status**: ✅ Complete (2026-05)

`FfnLayerType` enum (`Dense` / `MoE { num_experts, top_k }`) and `ffn_type_for_layer(layer)` method on `ModelArchitecture`. Default returns `Dense`; overridden in Mixtral (all MoE), GPT-OSS (all MoE), DeepSeek v2/v3 (layer 0 dense, rest MoE), DeepSeek-V4 (same), and Gemma 4 (hybrid MoE when `enable_moe_block`).

### Attention pattern abstraction
**Effort**: Medium  
**Status**: Research

Current sliding window is boolean per layer. Future models may have more complex patterns (local + global hybrid, dilated attention, prefix caching hints). Consider a richer `AttentionPattern` enum.

### Config validation — DONE
**Effort**: Low
**Status**: ✅ Complete (2026-05)

`ModelConfig::validate()` checks num_layers, hidden_size, head_dim, head divisibility, MoE consistency, layer_types length, and partial_rotary_factor bounds. Called from detect.rs after config parsing.

## Completed

| Item | Date | Impact |
|------|------|--------|
| v_shares_k inference wiring | 2026-05 | Config authoritative, no tensor probing |
| Phi-3/4 architecture | 2026-05 | Partial RoPE, SuRoPE |
| Cohere/Command R architecture | 2026-05 | LayerNorm, different key pattern |
| Config validation | 2026-05 | `ModelConfig::validate()` |
| Streaming safetensors loading | 2026-05 | Per-layer mmap, ~2 GB peak |
| ModelArchitecture trait | 2026-03 | Foundation — 82 methods with defaults |
| Gemma 2/3 support | 2026-03 | QK-norm, softcapping, sliding window |
| Llama/Mistral/Qwen/DeepSeek | 2026-03 | Core architecture coverage |
| Mixtral MoE (PerExpert) | 2026-03 | Expert key patterns |
| GPT-OSS (PackedMxfp4) | 2026-03 | MXFP4 dequantization, packed expert keys |
| Granite (scaling multipliers) | 2026-03 | Embedding/residual/attention/logits scaling |
| StarCoder2 | 2026-03 | LayerNorm, bias, GELU |
| GGUF loading | 2026-03 | Q4_0/Q4_1/Q8_0/F16/BF16 dequantization |
| Safetensors mmap + HF cache | 2026-03 | Zero-copy loading, cache resolution |
| drop_ffn_weights | 2026-04 | Walk-only mode saves ~13GB |
| Gemma 4 architecture | 2026-04 | Per-layer geometry, PLE, KV sharing, V-norm, layer scalars |
| Gemma 4 31B + E2B configs | 2026-04 | Both variants tested with real config.json |
| Gemma4Arch re-export | 2026-04-07 | Public API complete |
| v_shares_k from config | 2026-04-07 | Uses attention_k_eq_v flag instead of hardcoded false |
| Gemma 3 qk_norm_weight_offset | 2026-04-07 | Was missing (Gemma 2 had it, Gemma 3 didn't) |
| Full test coverage (130 tests) | 2026-04-07 | All 12 architectures tested: Gemma 2/3/4, Llama, Mistral, Mixtral, Qwen, DeepSeek, GPT-OSS, Granite, StarCoder2, Generic |
| Clippy clean (zero warnings) | 2026-04-07 | lib + examples + tests all pass `-D warnings` |
| Documentation suite | 2026-04-07 | README, ROADMAP, PERFORMANCE, 3 docs, 6 ADRs |
| Example suite (3 demos) | 2026-04-07 | architecture_demo (all 12), demo_tensor_keys (all 12), demo_loading |
| KV layer sharing in decode | 2026-05 | Shared layers skip K/V projection, reuse source cache |
