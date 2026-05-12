//! Core residual-stream compute: prefill, decode step, K/V recomputation.

use infer_compute::{dot_proj_gpu, ComputeBackend};
use ndarray::{s, Array2};

use super::store::RsStore;
use crate::attention::SharedKV;
use crate::attention::{
    apply_rope_partial_at, run_attention_with_kv_backend, run_attention_block_decode_step_backend,
};
use crate::ffn::FfnBackend;
use crate::forward::{add_bias, apply_norm, embed_tokens_pub, run_ffn};
use crate::model::ModelWeights;
use crate::residual::{rms_norm_heads, rms_norm_heads_no_weight};

/// Result of `rs_prefill` — contains the final hidden state and the populated store.
pub struct RsPrefillResult {
    pub hidden: Array2<f32>,
    pub store: RsStore,
    pub memory_bytes: usize,
    pub window_tokens: usize,
}

/// Backend-dispatched dense FFN. Routes matmuls through `ComputeBackend`.
pub struct BackendFfn<'a, 'b> {
    pub weights: &'a ModelWeights,
    pub backend: &'b dyn ComputeBackend,
}

impl<'a, 'b> FfnBackend for BackendFfn<'a, 'b> {
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> {
        self.forward_with_activation(layer, x).0
    }

    fn forward_with_activation(&self, layer: usize, x: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        dense_ffn_forward_backend(self.weights, layer, x, self.backend)
    }

    fn name(&self) -> &str {
        "weights+backend"
    }
}

/// Dense FFN forward with backend dispatch for projections.
fn dense_ffn_forward_backend(
    weights: &ModelWeights,
    layer: usize,
    x: &Array2<f32>,
    backend: &dyn ComputeBackend,
) -> (Array2<f32>, Array2<f32>) {
    let arch = &*weights.arch;
    let be = Some(backend);

    let w_gate = weights.tensors.get(&arch.ffn_gate_key(layer));
    let w_up = weights
        .tensors
        .get(&arch.ffn_up_key(layer))
        .expect("FFN up tensor missing — this may be a compact vindex");
    let w_down = weights
        .tensors
        .get(&arch.ffn_down_key(layer))
        .expect("FFN down tensor missing");

    let activation = if let Some(gate) = w_gate {
        // Gated FFN: SiLU(x @ gate.T) * (x @ up.T)
        let gate_out = dot_proj_gpu(x, gate, be);
        let up_out = dot_proj_gpu(x, w_up, be);
        let mut result = Array2::zeros(gate_out.raw_dim());
        ndarray::Zip::from(&mut result)
            .and(&gate_out)
            .and(&up_out)
            .for_each(|r, &g, &u| {
                let silu = g / (1.0 + (-g).exp());
                *r = silu * u;
            });
        result
    } else {
        // Non-gated FFN: activation(x @ up.T)
        let mut up_out = dot_proj_gpu(x, w_up, be);
        // Apply activation in-place (SiLU by default)
        up_out.mapv_inplace(|v| v / (1.0 + (-v).exp()));
        up_out
    };

    let output = dot_proj_gpu(&activation, w_down, be);

    // Add bias if present
    let mut final_output = output;
    if let Some(bias_key) = arch.ffn_down_bias_key(layer) {
        if let Some(bias) = weights.vectors.get(&bias_key) {
            add_bias(&mut final_output, bias);
        }
    }

    (final_output, activation)
}

/// Prefill: process all tokens, storing per-layer residuals in the RsStore.
///
/// Returns the last-position hidden state (shape [1, hidden]) plus the
/// populated store ready for subsequent decode steps.
pub fn rs_prefill(
    weights: &ModelWeights,
    token_ids: &[u32],
    max_window: Option<usize>,
    backend: &dyn ComputeBackend,
) -> RsPrefillResult {
    let num_layers = weights.num_layers;
    let seq_len = token_ids.len();
    let mut h = embed_tokens_pub(weights, token_ids);
    let mut stored: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
    let be = Some(backend);

    for layer in 0..num_layers {
        stored.push(h.clone());
        let (h_post_attn, _k, _v) = run_attention_with_kv_backend(weights, &h, layer, be)
            .expect("attention failed during MarkovRS prefill");
        let bffn = BackendFfn { weights, backend };
        let (h_out, _) = run_ffn(weights, &h_post_attn, layer, &bffn, false);
        h = h_out;
    }

    let mut rs = RsStore {
        stored,
        cold_residuals: None,
        cold_kv: None,
        cold_abs_start: 0,
        next_position: seq_len,
        max_window,
    };

    // Clip hot store to window and pre-compute cold K/V
    let mut cold: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        rs.clip_layer(layer, &mut cold);
    }
    if cold.first().map_or(0, |c| c.shape()[0]) > 0 {
        let cold_kv: Vec<SharedKV> = (0..num_layers)
            .map(|layer| {
                recompute_kv(weights, &cold[layer], layer, 0, backend)
                    .expect("cold K/V pre-computation failed")
            })
            .collect();
        rs.cold_residuals = Some(cold);
        rs.cold_kv = Some(cold_kv);
        rs.cold_abs_start = 0;
    }

    let window_tokens = rs.window_tokens();
    let memory_bytes = rs.memory_bytes();
    RsPrefillResult {
        hidden: last_row(&h),
        store: rs,
        memory_bytes,
        window_tokens,
    }
}

/// Single decode step: embed new token, recompute K/V from stored residuals,
/// run attention + FFN through all layers, append new residual to store.
pub fn rs_decode_step(
    weights: &ModelWeights,
    new_token_id: u32,
    rs: RsStore,
    backend: &dyn ComputeBackend,
) -> Option<(Array2<f32>, RsStore)> {
    let num_layers = weights.num_layers;
    let abs_position = rs.next_position;
    let mut h_new = embed_tokens_pub(weights, &[new_token_id]);
    let mut new_stored: Vec<Array2<f32>> = Vec::with_capacity(num_layers);

    for layer in 0..num_layers {
        let h_hot = &rs.stored[layer];
        let s_hot = h_hot.shape()[0];
        let hot_abs_start = abs_position.saturating_sub(s_hot);

        // Recompute K/V from stored residuals
        let (k_full, v_full) = if let Some(cold_kv) = &rs.cold_kv {
            // Cold K/V is pre-computed; only recompute hot portion
            let (k_cold, v_cold) = &cold_kv[layer];
            let (k_hot, v_hot) = recompute_kv(weights, h_hot, layer, hot_abs_start, backend)?;
            let c = k_cold.shape()[0];
            let kv_dim = k_cold.shape()[1];
            let mut k_combined = Array2::<f32>::zeros((c + s_hot, kv_dim));
            k_combined.slice_mut(s![..c, ..]).assign(k_cold);
            k_combined.slice_mut(s![c.., ..]).assign(&k_hot);
            let mut v_combined = Array2::<f32>::zeros((c + s_hot, kv_dim));
            v_combined.slice_mut(s![..c, ..]).assign(v_cold);
            v_combined.slice_mut(s![c.., ..]).assign(&v_hot);
            (k_combined, v_combined)
        } else {
            // No pre-computed cold K/V — recompute from all residuals
            let (h_full, full_abs_start) = if let Some(cold) = &rs.cold_residuals {
                let h_cold = &cold[layer];
                let s_cold = h_cold.shape()[0];
                if s_cold > 0 {
                    let hidden = h_hot.shape()[1];
                    let mut combined = Array2::<f32>::zeros((s_cold + s_hot, hidden));
                    combined.slice_mut(s![..s_cold, ..]).assign(h_cold);
                    combined.slice_mut(s![s_cold.., ..]).assign(h_hot);
                    (combined, rs.cold_abs_start)
                } else {
                    (h_hot.clone(), hot_abs_start)
                }
            } else {
                (h_hot.clone(), hot_abs_start)
            };
            recompute_kv(weights, &h_full, layer, full_abs_start, backend)?
        };

        // Store the new token's residual for this layer
        new_stored.push(h_new.clone());

        // Attention with full context K/V
        let (h_post_attn, _new_kv) = run_attention_block_decode_step_backend(
            weights,
            &h_new,
            layer,
            Some(&(k_full, v_full)),
            abs_position,
            Some(backend),
        )?;

        // FFN
        let bffn = BackendFfn { weights, backend };
        let (h_out, _) = run_ffn(weights, &h_post_attn, layer, &bffn, false);
        h_new = h_out;
    }

    // Append new residuals to existing stored
    let mut updated_stored: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
    for (stored, new_row) in rs.stored.iter().zip(new_stored.iter()) {
        let s_old = stored.shape()[0];
        let hidden_dim = stored.shape()[1];
        let mut combined = Array2::<f32>::zeros((s_old + 1, hidden_dim));
        combined.slice_mut(s![..s_old, ..]).assign(stored);
        combined.slice_mut(s![s_old.., ..]).assign(new_row);
        updated_stored.push(combined);
    }

    let mut updated_rs = RsStore {
        stored: updated_stored,
        cold_residuals: rs.cold_residuals,
        cold_kv: rs.cold_kv,
        cold_abs_start: rs.cold_abs_start,
        next_position: abs_position + 1,
        max_window: rs.max_window,
    };

    // Clip overflow beyond window
    let mut overflow: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        updated_rs.clip_layer(layer, &mut overflow);
    }
    if overflow.first().map_or(0, |c| c.shape()[0]) > 0 {
        match updated_rs.cold_residuals.as_mut() {
            Some(cold) => {
                for layer in 0..num_layers {
                    let hidden = cold[layer].shape()[1];
                    let c_old = cold[layer].shape()[0];
                    let c_new = overflow[layer].shape()[0];
                    let mut merged = Array2::<f32>::zeros((c_old + c_new, hidden));
                    merged.slice_mut(s![..c_old, ..]).assign(&cold[layer]);
                    merged.slice_mut(s![c_old.., ..]).assign(&overflow[layer]);
                    cold[layer] = merged;
                }
            }
            None => {
                updated_rs.cold_residuals = Some(overflow);
            }
        }
        // Invalidate pre-computed cold K/V — will be recomputed on next decode
        updated_rs.cold_kv = None;
    }

    Some((last_row(&h_new), updated_rs))
}

/// Recompute K/V from stored pre-layer residuals.
///
/// `h_stored` is [seq_len, hidden_size], `abs_start` is the absolute position
/// of the first row (used for RoPE). Returns (K_rope, V) each [seq_len, kv_dim].
pub fn recompute_kv(
    weights: &ModelWeights,
    h_stored: &Array2<f32>,
    layer: usize,
    abs_start: usize,
    backend: &dyn ComputeBackend,
) -> Option<(Array2<f32>, Array2<f32>)> {
    let arch = &*weights.arch;
    let head_dim = arch.head_dim_for_layer(layer);
    let num_kv = arch.num_kv_heads_for_layer(layer);
    let norm_offset = arch.norm_weight_offset();
    let qk_offset = arch.qk_norm_weight_offset();
    let qk_norm_off = if qk_offset != 0.0 {
        qk_offset
    } else {
        norm_offset
    };
    let be = Some(backend);

    let h_norm = apply_norm(
        weights,
        h_stored,
        &arch.input_layernorm_key(layer),
        norm_offset,
    );
    let w_k = weights.tensors.get(&arch.attn_k_key(layer))?;
    let v_from_k = arch.v_shares_k(layer);
    let w_v = if v_from_k {
        w_k
    } else {
        weights.tensors.get(&arch.attn_v_key(layer))?
    };

    let mut k = dot_proj_gpu(&h_norm, w_k, be);
    let mut v = dot_proj_gpu(&h_norm, w_v, be);

    if let Some(bias) = arch
        .attn_k_bias_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        add_bias(&mut k, bias);
    }
    if let Some(bias) = arch
        .attn_v_bias_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        add_bias(&mut v, bias);
    }
    if arch.has_v_norm() {
        v = rms_norm_heads_no_weight(&v, num_kv, head_dim);
    }
    let k_normed = match arch
        .attn_k_norm_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        Some(norm_w) => rms_norm_heads(&k, norm_w, num_kv, head_dim, qk_norm_off),
        None => k,
    };
    let k_rope = apply_rope_partial_at(
        &k_normed,
        num_kv,
        head_dim,
        arch.rope_base_for_layer(layer),
        arch.rotary_fraction_for_layer(layer),
        abs_start,
    );
    Some((k_rope, v))
}

/// Equivalent Standard KV memory in bytes for `seq_len` tokens (FP16 K+V).
pub fn kv_memory_bytes_for_seq(weights: &ModelWeights, seq_len: usize) -> usize {
    let arch = &*weights.arch;
    (0..weights.num_layers)
        .map(|l| {
            let kv_dim = arch.num_kv_heads_for_layer(l) * arch.head_dim_for_layer(l);
            // K + V, each seq_len * kv_dim * 2 bytes (FP16)
            seq_len * kv_dim * 2 * 2
        })
        .sum()
}

/// Extract last row as [1, hidden] matrix.
pub(super) fn last_row(h: &Array2<f32>) -> Array2<f32> {
    let last = h.shape()[0] - 1;
    h.slice(s![last..=last, ..]).to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use infer_compute::CpuBackend;
    use crate::model::test_utils::make_test_weights;

    #[test]
    fn recompute_kv_returns_some_with_valid_weights() {
        let weights = make_test_weights();
        let h = Array2::from_elem((3, weights.hidden_size), 0.5f32);
        let result = recompute_kv(&weights, &h, 0, 0, &CpuBackend);
        assert!(
            result.is_some(),
            "recompute_kv should return Some with valid weights"
        );
    }

    #[test]
    fn recompute_kv_output_shape_correct() {
        let weights = make_test_weights();
        let seq_len = 4;
        let h = Array2::from_elem((seq_len, weights.hidden_size), 1.0f32);
        let (k, v) = recompute_kv(&weights, &h, 0, 0, &CpuBackend).unwrap();
        let kv_dim = weights.num_kv_heads * weights.head_dim;
        assert_eq!(k.shape(), &[seq_len, kv_dim], "K shape mismatch");
        assert_eq!(v.shape(), &[seq_len, kv_dim], "V shape mismatch");
    }

    #[test]
    fn recompute_kv_output_is_finite() {
        let weights = make_test_weights();
        let h = Array2::from_elem((2, weights.hidden_size), 0.1f32);
        let (k, v) = recompute_kv(&weights, &h, 0, 0, &CpuBackend).unwrap();
        assert!(k.iter().all(|x| x.is_finite()), "K contains non-finite values");
        assert!(v.iter().all(|x| x.is_finite()), "V contains non-finite values");
    }

    #[test]
    fn recompute_kv_abs_start_shifts_rope() {
        let weights = make_test_weights();
        let h = Array2::from_elem((1, weights.hidden_size), 0.5f32);
        let (k0, _) = recompute_kv(&weights, &h, 0, 0, &CpuBackend).unwrap();
        let (k5, _) = recompute_kv(&weights, &h, 0, 5, &CpuBackend).unwrap();
        let diff: f32 = k0.iter().zip(k5.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 0.0, "RoPE at different positions should produce different K");
    }

    #[test]
    fn rs_prefill_returns_correct_shape() {
        let weights = make_test_weights();
        let result = rs_prefill(&weights, &[0u32, 1, 2], None, &CpuBackend);
        assert_eq!(result.hidden.shape(), &[1, weights.hidden_size]);
        assert!(result.hidden.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn rs_prefill_stores_all_layers() {
        let weights = make_test_weights();
        let result = rs_prefill(&weights, &[0u32], None, &CpuBackend);
        assert_eq!(result.store.stored.len(), weights.num_layers);
        assert_eq!(result.store.next_position, 1);
    }

    #[test]
    fn rs_prefill_with_window_clips_hot_store() {
        let weights = make_test_weights();
        let result = rs_prefill(&weights, &[0u32, 1, 2, 3, 4], Some(2), &CpuBackend);
        assert!(
            result.window_tokens <= 2,
            "window_tokens={} > 2",
            result.window_tokens
        );
    }

    #[test]
    fn rs_decode_step_produces_finite_hidden() {
        let weights = make_test_weights();
        let prefill = rs_prefill(&weights, &[0u32], None, &CpuBackend);
        let (h, _) = rs_decode_step(&weights, 1, prefill.store, &CpuBackend).expect("decode step");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(h.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn rs_decode_step_advances_position() {
        let weights = make_test_weights();
        let prefill = rs_prefill(&weights, &[0u32, 1], None, &CpuBackend);
        assert_eq!(prefill.store.next_position, 2);
        let (_, rs2) = rs_decode_step(&weights, 2, prefill.store, &CpuBackend).unwrap();
        assert_eq!(rs2.next_position, 3);
        let (_, rs3) = rs_decode_step(&weights, 3, rs2, &CpuBackend).unwrap();
        assert_eq!(rs3.next_position, 4);
    }
}
