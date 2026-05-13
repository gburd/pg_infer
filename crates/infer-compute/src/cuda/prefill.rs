//! CUDA prefill pipeline: full inference for seq_len > 1 with post-norm support.
//!
//! Processes all positions through all layers on GPU, handling both standard
//! (pre-norm) and post-norm (Gemma3/4) architectures. Uses cuBLAS for f32 gemv
//! and custom CUDA kernels for Q4_K/Q6_K dequant+matvec.
//!
//! ## Pipeline per layer
//!
//! 1. Input RMS norm (per position)
//! 2. Q/K/V projections (quantized matvec per position)
//! 3. Scaled dot-product attention (causal, multi-head, with RoPE)
//! 4. O projection (quantized matvec per position)
//! 5. Post-attention residual (post-norm or pre-norm path)
//! 6. FFN norm (weight selection based on post-norm flag)
//! 7. FFN: gate + up -> activation -> down
//! 8. Post-FFN residual (with optional post-FFN norm)

use super::{CudaBackend, CudaError};
use super::kernels;
use crate::backend::ComputeBackend;
use crate::pipeline::{FullPipelineLayer, QuantFormat, Activation, FfnType};

// ── Helper: RMS norm (CPU-side, applied per position) ──

/// RMS normalize a single vector in-place semantics (returns new vec).
/// out[i] = x[i] / sqrt(mean(x^2) + eps) * (w[i] + offset)
#[inline]
fn rms_norm_vec(x: &[f32], w: &[f32], eps: f32, offset: f32) -> Vec<f32> {
    let n = x.len();
    let rms = (x.iter().map(|&v| v * v).sum::<f32>() / n as f32 + eps).sqrt();
    x.iter()
        .zip(w.iter())
        .map(|(&xi, &wi)| xi / rms * (wi + offset))
        .collect()
}

/// Apply RMS norm to each position in a batched hidden state.
/// Input: flat [seq_len * hidden], norm_weight: [hidden].
/// Returns: flat [seq_len * hidden].
fn rms_norm_batch(
    h: &[f32],
    norm_weight: &[f32],
    hidden: usize,
    seq_len: usize,
    eps: f32,
    offset: f32,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(seq_len * hidden);
    for s in 0..seq_len {
        let pos = &h[s * hidden..(s + 1) * hidden];
        out.extend(rms_norm_vec(pos, norm_weight, eps, offset));
    }
    out
}

// ── Helper: Quantized matvec dispatch (per position) ──

/// Dispatch a quantized matvec for a single position's input vector.
/// Routes to Q4_K, Q6_K, or CPU fallback based on format.
fn quant_matvec(
    backend: &CudaBackend,
    data: &[u8],
    format: QuantFormat,
    x: &[f32],
    num_rows: usize,
    hidden: usize,
) -> Result<Vec<f32>, CudaError> {
    match format {
        QuantFormat::Q4_K | QuantFormat::Q4_KF => {
            kernels::q4k_matvec_cuda(
                &backend.device,
                &backend.stream,
                &backend.buffer_pool,
                data,
                x,
                num_rows,
                hidden,
            )
        }
        QuantFormat::Q6_K => {
            kernels::q6k_matvec_cuda(
                &backend.device,
                &backend.stream,
                &backend.buffer_pool,
                data,
                x,
                num_rows,
                hidden,
            )
        }
        QuantFormat::Q4_0 | QuantFormat::Q8_0 => {
            // Q4_0/Q8_0: fall back to CPU for now (requires Q8 quantized input)
            let result = backend.cpu_fallback.q4k_matvec(data, x, num_rows, hidden);
            result.ok_or_else(|| {
                CudaError::KernelLaunch("Q4_0/Q8_0 fallback failed".into())
            })
        }
    }
}

// ── Helper: RoPE application ──

/// Apply RoPE to Q or K vectors for all positions.
/// q_or_k: [seq_len, dim] flat. Applies rotary embeddings in-place.
/// `rotary_dim`: how many dimensions to rotate (0 = full dim = head_dim).
fn apply_rope(
    q_or_k: &mut [f32],
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    rope_base: f32,
    rotary_dim: usize,
) {
    let dim = if rotary_dim == 0 { head_dim } else { rotary_dim };
    let half = dim / 2;

    for s in 0..seq_len {
        for h in 0..num_heads {
            let offset = s * num_heads * head_dim + h * head_dim;
            for i in 0..half {
                let freq = 1.0 / rope_base.powf(2.0 * i as f32 / dim as f32);
                let theta = s as f32 * freq;
                let cos_t = theta.cos();
                let sin_t = theta.sin();

                let re = q_or_k[offset + i];
                let im = q_or_k[offset + i + half];
                q_or_k[offset + i] = re * cos_t - im * sin_t;
                q_or_k[offset + i + half] = re * sin_t + im * cos_t;
            }
        }
    }
}

// ── Helper: Scaled dot-product attention (causal) ──

/// Multi-head scaled dot-product attention with causal mask.
/// Q: [seq_len, num_q_heads * head_dim]
/// K: [seq_len, num_kv_heads * head_dim]
/// V: [seq_len, num_kv_heads * head_dim]
/// Returns: [seq_len, num_q_heads * head_dim]
///
/// Supports GQA (grouped-query attention) where num_q_heads > num_kv_heads.
fn scaled_dot_product_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    softcap: f32,
) -> Vec<f32> {
    let q_dim = num_q_heads * head_dim;
    let heads_per_group = num_q_heads / num_kv_heads;
    let mut out = vec![0.0f32; seq_len * q_dim];

    for s in 0..seq_len {
        for qh in 0..num_q_heads {
            let kv_head = qh / heads_per_group;
            let q_off = s * q_dim + qh * head_dim;

            // Compute attention scores for positions 0..=s (causal)
            let mut scores = Vec::with_capacity(s + 1);
            for t in 0..=s {
                let k_off = t * num_kv_heads * head_dim + kv_head * head_dim;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[q_off + d] * k[k_off + d];
                }
                dot *= scale;

                // Softcap (logit capping): tanh(score / cap) * cap
                if softcap > 0.0 {
                    dot = (dot / softcap).tanh() * softcap;
                }

                scores.push(dot);
            }

            // Softmax
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
            let sum: f32 = exp_scores.iter().sum();
            for e in exp_scores.iter_mut() {
                *e /= sum;
            }

            // Weighted sum of V
            let out_off = s * q_dim + qh * head_dim;
            for (t, &weight) in exp_scores.iter().enumerate() {
                let v_off = t * num_kv_heads * head_dim + kv_head * head_dim;
                for d in 0..head_dim {
                    out[out_off + d] += weight * v[v_off + d];
                }
            }
        }
    }

    out
}

// ── Helper: QK norm ──

/// Apply per-head RMS norm to Q or K vectors (Gemma 3/4 QK-norm).
/// vectors: [seq_len, num_heads * head_dim] flat.
/// norm_weight: [head_dim].
fn apply_qk_norm(
    vectors: &mut [f32],
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    norm_weight: &[f32],
    eps: f32,
    offset: f32,
) {
    for s in 0..seq_len {
        for h in 0..num_heads {
            let base = s * num_heads * head_dim + h * head_dim;
            let head_slice = &vectors[base..base + head_dim];

            let rms = (head_slice.iter().map(|&v| v * v).sum::<f32>()
                / head_dim as f32
                + eps)
                .sqrt();

            for d in 0..head_dim {
                vectors[base + d] = vectors[base + d] / rms * (norm_weight[d] + offset);
            }
        }
    }
}

// ── Helper: V-norm (parameter-free, Gemma 4) ──

/// Apply parameter-free RMS norm per V head (Gemma 4).
/// Normalizes each head's V vector to unit RMS.
fn apply_v_norm(
    v: &mut [f32],
    seq_len: usize,
    num_kv_heads: usize,
    head_dim: usize,
    eps: f32,
) {
    for s in 0..seq_len {
        for h in 0..num_kv_heads {
            let base = s * num_kv_heads * head_dim + h * head_dim;
            let rms = (v[base..base + head_dim]
                .iter()
                .map(|&x| x * x)
                .sum::<f32>()
                / head_dim as f32
                + eps)
                .sqrt();
            for d in 0..head_dim {
                v[base + d] /= rms;
            }
        }
    }
}

// ── Helper: Activation functions ──

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline]
fn gelu_tanh(x: f32) -> f32 {
    let c = 0.797_884_6_f32;
    0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
}

// ── Helper: FFN forward pass ──

/// FFN forward for one position. Supports gated (gate+up->act->down) and
/// standard (up->act->down) topologies.
fn ffn_one_position(
    backend: &CudaBackend,
    layer: &FullPipelineLayer<'_>,
    input: &[f32],
    hidden: usize,
) -> Result<Vec<f32>, CudaError> {
    let activation = layer.activation;

    match layer.ffn_type {
        FfnType::Gated => {
            // Compute intermediate dimension from weight tensor size
            let inter = infer_rows(layer.gate.data, layer.gate.format, hidden);

            // Gate projection: [inter] = gate_weights[inter, hidden] @ input[hidden]
            let gate_out = quant_matvec(
                backend, layer.gate.data, layer.gate.format,
                input, inter, hidden,
            )?;

            // Up projection: [inter] = up_weights[inter, hidden] @ input[hidden]
            let up_out = quant_matvec(
                backend, layer.up.data, layer.up.format,
                input, inter, hidden,
            )?;

            // Activation: act(gate) * up
            let mut act_out = vec![0.0f32; inter];
            match activation {
                Activation::Silu => {
                    for i in 0..inter {
                        act_out[i] = silu(gate_out[i]) * up_out[i];
                    }
                }
                Activation::GeluTanh => {
                    for i in 0..inter {
                        act_out[i] = gelu_tanh(gate_out[i]) * up_out[i];
                    }
                }
            }

            // Down projection: [hidden] = down_weights[hidden, inter] @ act[inter]
            let down_out = quant_matvec(
                backend, layer.down.data, layer.down.format,
                &act_out, hidden, inter,
            )?;

            Ok(down_out)
        }
        FfnType::Standard => {
            // Standard FFN: up -> activation -> down
            let inter = infer_rows(layer.up.data, layer.up.format, hidden);
            let mut up_out = quant_matvec(
                backend, layer.up.data, layer.up.format,
                input, inter, hidden,
            )?;

            // Add bias if present
            if let Some(bias) = layer.ffn_up_bias {
                for (o, &b) in up_out.iter_mut().zip(bias.iter()) {
                    *o += b;
                }
            }

            // Activation (in-place)
            match activation {
                Activation::Silu => {
                    for v in up_out.iter_mut() {
                        *v = silu(*v);
                    }
                }
                Activation::GeluTanh => {
                    for v in up_out.iter_mut() {
                        *v = gelu_tanh(*v);
                    }
                }
            }

            // Down projection
            let mut down_out = quant_matvec(
                backend, layer.down.data, layer.down.format,
                &up_out, hidden, inter,
            )?;

            // Add bias if present
            if let Some(bias) = layer.ffn_down_bias {
                for (o, &b) in down_out.iter_mut().zip(bias.iter()) {
                    *o += b;
                }
            }

            Ok(down_out)
        }
    }
}

// ── Helper: Infer output rows from quantized weight data size ──

/// Infer the number of output rows from a quantized weight tensor.
/// For Q4_K: bytes_per_row = (hidden/256) * 144, num_rows = data.len() / bytes_per_row.
/// For Q6_K: bytes_per_row = (hidden/256) * 210.
/// For Q4_0: bytes_per_row = hidden * 18 / 32 = hidden * 9 / 16.
/// For Q8_0: bytes_per_row = hidden (int8 values, scales stored separately).
fn infer_rows(data: &[u8], format: QuantFormat, hidden: usize) -> usize {
    let bytes_per_row = match format {
        QuantFormat::Q4_K | QuantFormat::Q4_KF => (hidden / 256) * 144,
        QuantFormat::Q6_K => (hidden / 256) * 210,
        QuantFormat::Q4_0 => hidden * 9 / 16, // 18 bytes per 32 values
        QuantFormat::Q8_0 => hidden,           // 1 byte per value (scales separate)
    };
    if bytes_per_row == 0 {
        return 0;
    }
    data.len() / bytes_per_row
}

// ── Helper: Residual add ──

/// Element-wise addition: out[i] = a[i] + b[i].
#[inline]
fn add_residual(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(&ai, &bi)| ai + bi).collect()
}

// ── Main prefill entry point ──

/// Run the CUDA prefill pipeline for seq_len > 1 tokens.
///
/// Processes each layer with the full transformer block computation:
/// norm -> Q/K/V projection -> attention -> O projection -> residual ->
/// FFN norm -> FFN -> residual.
///
/// Handles both pre-norm (standard) and post-norm (Gemma 2/3/4) architectures.
///
/// Returns the final hidden state as a flat f32 vec of length `seq_len * hidden`.
pub fn prefill_cuda(
    backend: &CudaBackend,
    layers: &[FullPipelineLayer<'_>],
    x: &[f32],
    seq_len: usize,
) -> Result<Vec<f32>, CudaError> {
    let num_layers = layers.len();
    if num_layers == 0 || seq_len == 0 {
        return Ok(x.to_vec());
    }
    let hidden = x.len() / seq_len;

    // Working buffer: current hidden state [seq_len, hidden]
    let mut h = x.to_vec();

    for l in 0..num_layers {
        let layer = &layers[l];

        // Skip cached layers: use pre-computed residual as hidden state.
        if let Some(cached) = layer.cached_residual {
            h[..cached.len()].copy_from_slice(cached);
            continue;
        }

        let has_post_norms = layer.has_post_norms;
        let eps = layer.eps;
        let norm_offset = layer.norm_offset;
        let qk_norm_offset = layer.qk_norm_offset;
        let head_dim = layer.head_dim;
        let num_q_heads = layer.num_q_heads;
        let num_kv_heads = layer.num_kv_heads;
        let rope_base = layer.rope_base;
        let rotary_dim = layer.rotary_dim;
        let scale = layer.attn_scale;
        let softcap = 0.0f32; // Softcap is passed at the trait level, not per-layer; default 0

        let q_dim = num_q_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        // ── 1. Input RMS norm per position ──
        let normed = rms_norm_batch(&h, layer.input_norm, hidden, seq_len, eps, norm_offset);

        // ── 2. Q/K/V projections per position ──
        let mut q_all = Vec::with_capacity(seq_len * q_dim);
        let mut k_all = Vec::with_capacity(seq_len * kv_dim);
        let mut v_all = Vec::with_capacity(seq_len * kv_dim);

        for s in 0..seq_len {
            let input_pos = &normed[s * hidden..(s + 1) * hidden];

            // Q projection
            let q = quant_matvec(
                backend, layer.wq.data, layer.wq.format,
                input_pos, q_dim, hidden,
            )?;
            q_all.extend_from_slice(&q);

            // K projection
            let k = quant_matvec(
                backend, layer.wk.data, layer.wk.format,
                input_pos, kv_dim, hidden,
            )?;
            k_all.extend_from_slice(&k);

            // V projection
            let v = quant_matvec(
                backend, layer.wv.data, layer.wv.format,
                input_pos, kv_dim, hidden,
            )?;
            v_all.extend_from_slice(&v);
        }

        // ── 2b. QK-norm (Gemma 3/4) ──
        if let Some(q_norm) = layer.q_norm_weight {
            apply_qk_norm(
                &mut q_all, seq_len, num_q_heads, head_dim,
                q_norm, eps, qk_norm_offset,
            );
        }
        if let Some(k_norm) = layer.k_norm_weight {
            apply_qk_norm(
                &mut k_all, seq_len, num_kv_heads, head_dim,
                k_norm, eps, qk_norm_offset,
            );
        }

        // ── 2c. V-norm (Gemma 4, parameter-free) ──
        if layer.has_v_norm {
            apply_v_norm(&mut v_all, seq_len, num_kv_heads, head_dim, eps);
        }

        // ── 3. RoPE ──
        apply_rope(&mut q_all, seq_len, num_q_heads, head_dim, rope_base, rotary_dim);
        apply_rope(&mut k_all, seq_len, num_kv_heads, head_dim, rope_base, rotary_dim);

        // ── 4. Attention ──
        let attn_out = scaled_dot_product_attention(
            &q_all, &k_all, &v_all,
            seq_len, num_q_heads, num_kv_heads, head_dim,
            scale, softcap,
        );

        // ── 5. O projection per position ──
        let mut o_out = Vec::with_capacity(seq_len * hidden);
        for s in 0..seq_len {
            let attn_pos = &attn_out[s * q_dim..(s + 1) * q_dim];
            let o = quant_matvec(
                backend, layer.wo.data, layer.wo.format,
                attn_pos, hidden, q_dim,
            )?;
            o_out.extend_from_slice(&o);
        }

        // ── 5b. Layer scalar (Gemma 4) ──
        if layer.layer_scalar != 0.0 && layer.layer_scalar != 1.0 {
            for v in o_out.iter_mut() {
                *v *= layer.layer_scalar;
            }
        }

        // ── 6. Post-attention residual ──
        let h_post_attn = if has_post_norms {
            // Post-norm: norm(O) + residual
            let o_normed = rms_norm_batch(
                &o_out, layer.post_attn_norm, hidden, seq_len, eps, norm_offset,
            );
            add_residual(&h, &o_normed)
        } else {
            // Standard pre-norm: residual + O
            add_residual(&h, &o_out)
        };

        // ── 7. FFN norm ──
        // Post-norm models use pre_ffn_norm (if available) for the FFN input norm.
        // Pre-norm models use post_attn_norm as the FFN input norm.
        let ffn_norm_weight = if has_post_norms {
            layer.pre_ffn_norm.unwrap_or(layer.post_attn_norm)
        } else {
            layer.post_attn_norm
        };
        let ffn_input = rms_norm_batch(
            &h_post_attn, ffn_norm_weight, hidden, seq_len, eps, norm_offset,
        );

        // ── 8. FFN: gate + up -> activation -> down (per position) ──
        let mut ffn_out = Vec::with_capacity(seq_len * hidden);
        for s in 0..seq_len {
            let pos_input = &ffn_input[s * hidden..(s + 1) * hidden];
            let pos_out = ffn_one_position(backend, layer, pos_input, hidden)?;
            ffn_out.extend_from_slice(&pos_out);
        }

        // ── 9. Post-FFN residual ──
        h = if has_post_norms {
            if let Some(post_ffn_norm) = layer.post_ffn_norm {
                // Post-norm: norm(FFN output) + residual
                let ffn_normed = rms_norm_batch(
                    &ffn_out, post_ffn_norm, hidden, seq_len, eps, norm_offset,
                );
                add_residual(&h_post_attn, &ffn_normed)
            } else {
                add_residual(&h_post_attn, &ffn_out)
            }
        } else {
            add_residual(&h_post_attn, &ffn_out)
        };
    }

    Ok(h)
}
