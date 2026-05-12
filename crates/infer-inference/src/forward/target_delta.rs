//! Per-fact target-delta optimisation — the MEMIT Phase-3 primitive.
//!
//! Given a model, a prompt, and a target token id, find the residual
//! delta `δ ∈ R^hidden` such that adding `δ` to the FFN output at
//! `install_layer`'s last-position makes `target_id` the top logit,
//! with a KL regulariser keeping the distribution close to baseline.
//!
//! This is the per-fact pre-compute that the Python reference at
//! `experiments/15_v11_model/vindex_compile_rome_v11.py::optimise_target_delta`
//! runs before MEMIT's closed-form W-edit. Without it, MEMIT's V*
//! defaults to `target_alpha × embed(target)` — a rough direction
//! that doesn't account for how downstream layers transform the
//! residual. With it, MEMIT's V* is the exact signal that, added
//! to the L_install FFN output, produces the target at logits.
//!
//! ## Algorithm (Python reference)
//!
//! ```text
//! base_logits = model(x)[-1]             # no-edit baseline
//! base_probs  = softmax(base_logits)
//! δ = zeros(hidden); opt = Adam([δ], lr)
//!
//! for step in 0..60:
//!     h = embed(x) * sqrt(dim)
//!     for layer in 0..n_layers:
//!         h = h + attn(attn_norm(h))
//!         ffn_out = ffn(ffn_norm(h))
//!         if layer == install_layer:
//!             ffn_out[-1] += δ            # the perturbation
//!         h = h + ffn_out
//!     logits = lm_head(norm(h))[-1]
//!
//!     loss = cross_entropy(logits, target_id) +
//!            kl_weight · KL(base_probs, softmax(logits))
//!     loss.backward(); opt.step()
//!
//! return δ
//! ```
//!
//! ## Native port status (WIP)
//!
//! The forward pass already exists in `forward/` (via `WalkFfn` and
//! the dense path). What's missing is the **backward pass**: to
//! compute `∂loss/∂δ` we need gradients through:
//!
//!   lm_head @ final_norm(h) @ layers[install..]  ← every op gets a transpose-multiply
//!
//! Hand-rolled backward implementations for each layer (attention,
//! FFN, RMSNorm) are landing in this module as `backward_*` helpers.
//! Scope for this first drop: infrastructure + cross-entropy gradient
//! + lm_head backward (tied embedding). Per-layer transformer-block
//!   backward is the remaining ~80% of the work — tracked as follow-ups
//!   on each `backward_*` stub.
//!
//! Once complete, `optimise_target_delta` runs 60-80 Adam iters per
//! fact in pure Rust; `run_memit` calls it and feeds the optimised
//! deltas into `rome_batch_update` as V*.

use ndarray::{Array1, ArrayView1, ArrayView2};

use crate::model::ModelWeights;

/// Hyperparameters for target-delta optimisation. Defaults match the
/// Python reference (`vindex_compile_rome_v11.py::optimise_target_delta`).
#[derive(Debug, Clone, Copy)]
pub struct TargetDeltaOpts {
    pub steps: usize,
    pub lr: f32,
    pub kl_weight: f32,
    /// If true, the returned delta is normalised to unit norm — useful
    /// when the downstream MEMIT solve scales its own magnitude.
    pub normalise: bool,
}

impl Default for TargetDeltaOpts {
    fn default() -> Self {
        Self {
            steps: 60,
            lr: 0.5,
            kl_weight: 0.0625,
            normalise: false,
        }
    }
}

/// Result of a single target-delta optimisation.
#[derive(Debug, Clone)]
pub struct TargetDelta {
    pub layer: usize,
    pub delta: Array1<f32>,
    /// Cross-entropy loss on the final step (lower is better; 0 means
    /// the target is the argmax with very high probability).
    pub final_loss: f32,
    /// Baseline loss on the target under the no-edit forward pass —
    /// useful for diagnostics ("did the optimisation actually move?").
    pub baseline_loss: f32,
}

// ── Autograd tape: minimal reverse-mode for the ops we need ────────
//
// Rather than pull in a full autograd crate (candle/burn/dfdx), we
// implement a focused reverse-mode that supports exactly the ops on
// the critical path from `δ` to `loss`. Each forward op returns both
// an output tensor AND appends a closure to a tape that, given the
// upstream gradient, contributes to the inputs' gradients.
//
// This avoids a full refactor of the model forward and keeps us in
// ndarray throughout.
//
// NOTE: this is a structural sketch. The tape record types and
// closures for each layer's backward are filled in piece-by-piece as
// the backward functions below are implemented.

/// Softmax cross-entropy loss for a 1-D logits vector and a single
/// target id. Returns `(loss, dlogits)` where `dlogits[j] = softmax[j] - onehot[target][j]`.
/// Used at the output end — no tape needed since this is the loss itself.
pub(crate) fn cross_entropy_and_grad(logits: ArrayView1<f32>, target_id: u32) -> (f32, Array1<f32>) {
    // Numerically stable log-softmax
    let max = logits.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let shifted: Array1<f32> = logits.map(|&v| v - max);
    let exp_sum: f32 = shifted.iter().map(|v| v.exp()).sum();
    let log_sum = exp_sum.ln();
    let loss = -(shifted[target_id as usize] - log_sum);

    // Gradient: softmax(logits) - onehot(target)
    let mut dlogits = shifted.map(|v| (v - log_sum).exp());
    dlogits[target_id as usize] -= 1.0;
    (loss, dlogits)
}

/// Backward through the tied-embedding lm_head: `logits = embed @ h`
/// so `∂loss/∂h = embed.T @ dlogits`. For tied embeddings
/// `lm_head.weight == embed.weight`, so we use the same matrix.
pub(crate) fn lm_head_backward(
    embed_weight: ArrayView2<f32>, // (vocab, hidden)
    dlogits: ArrayView1<f32>,       // (vocab,)
) -> Array1<f32> {
    // ∂loss/∂h[i] = Σ_v dlogits[v] · embed[v, i]
    // = embed.T @ dlogits  →  shape (hidden,)
    let hidden = embed_weight.ncols();
    let mut dh = Array1::<f32>::zeros(hidden);
    for (v_idx, &dl) in dlogits.iter().enumerate() {
        if dl == 0.0 {
            continue;
        }
        let row = embed_weight.row(v_idx);
        for i in 0..hidden {
            dh[i] += dl * row[i];
        }
    }
    dh
}

/// Backward through the final RMSNorm at the last position:
///
///   y = (x / rms(x)) * weight           where rms(x) = sqrt(mean(x^2) + eps)
///
/// The gradient form for RMSNorm (per-position) is:
///
///   ∂L/∂x = (weight / rms) * [dy - (x · (weight · dy)) / (rms^2 · d)]
///
/// where `·` is dot product and `d = hidden`. Returns `dx`, not `dweight`
/// (we don't update the norm weights during target-delta opt — they
/// aren't in the optimisation path).
pub(crate) fn rmsnorm_backward_pos(
    x: ArrayView1<f32>,
    weight: ArrayView1<f32>,
    dy: ArrayView1<f32>,
    eps: f32,
) -> Array1<f32> {
    let d = x.len() as f32;
    let ms = x.iter().map(|v| v * v).sum::<f32>() / d;
    let rms = (ms + eps).sqrt();

    // inner = (weight · dy) dotted element-wise
    let wdy: Array1<f32> = weight.iter().zip(dy.iter()).map(|(&w, &g)| w * g).collect();
    // xwdy = x · wdy  (scalar)
    let xwdy: f32 = x.iter().zip(wdy.iter()).map(|(&xi, &w)| xi * w).sum();

    // dx[i] = (1/rms) * (wdy[i] - x[i] * xwdy / (d * rms^2))
    let inv_rms = 1.0 / rms;
    let coef = xwdy / (d * rms * rms);
    let mut dx = Array1::<f32>::zeros(x.len());
    for i in 0..x.len() {
        dx[i] = inv_rms * (wdy[i] - x[i] * coef);
    }
    dx
}

/// Backward through a gated FFN block at one position.
///
/// Forward:
///   g_pre = gate_w @ x         (gate_w: ffn_dim × hidden)
///   g     = silu(g_pre)         silu(z) = z · σ(z)
///   u     = up_w @ x            (up_w:   ffn_dim × hidden)
///   act   = g * u               (ffn_dim)
///   out   = down_w @ act        (down_w: hidden × ffn_dim)
///
/// Backward (given d_out):
///   d_act = down_w.T @ d_out
///   d_g   = d_act * u
///   d_u   = d_act * g
///   silu'(z) = σ(z) · (1 + z · (1 - σ(z)))
///   d_g_pre = d_g * silu'(g_pre)
///   d_x = gate_w.T @ d_g_pre + up_w.T @ d_u
#[allow(dead_code)] // reserved primitive for mid-layer target-delta; FD-tested
pub(crate) fn gated_ffn_backward(
    x: ArrayView1<f32>,
    gate_w: ArrayView2<f32>,
    up_w: ArrayView2<f32>,
    down_w: ArrayView2<f32>,
    d_out: ArrayView1<f32>,
) -> Array1<f32> {
    let hidden = x.len();
    let ffn_dim = gate_w.nrows();
    assert_eq!(gate_w.ncols(), hidden);
    assert_eq!(up_w.nrows(), ffn_dim);
    assert_eq!(up_w.ncols(), hidden);
    assert_eq!(down_w.nrows(), hidden);
    assert_eq!(down_w.ncols(), ffn_dim);
    assert_eq!(d_out.len(), hidden);

    // Forward activations we need again for backward.
    let mut g_pre = Array1::<f32>::zeros(ffn_dim);
    let mut u = Array1::<f32>::zeros(ffn_dim);
    for i in 0..ffn_dim {
        let mut gp = 0.0_f32;
        let mut up = 0.0_f32;
        for j in 0..hidden {
            gp += gate_w[[i, j]] * x[j];
            up += up_w[[i, j]] * x[j];
        }
        g_pre[i] = gp;
        u[i] = up;
    }
    // silu and σ
    let sigma: Array1<f32> = g_pre.map(|&z| 1.0 / (1.0 + (-z).exp()));
    let g: Array1<f32> = g_pre.iter().zip(sigma.iter()).map(|(&z, &s)| z * s).collect();

    // d_act = down_w.T @ d_out → shape ffn_dim
    let mut d_act = Array1::<f32>::zeros(ffn_dim);
    for i in 0..ffn_dim {
        let mut s = 0.0_f32;
        for k in 0..hidden {
            s += down_w[[k, i]] * d_out[k];
        }
        d_act[i] = s;
    }

    // d_g = d_act * u ; d_u = d_act * g
    let d_g: Array1<f32> = d_act.iter().zip(u.iter()).map(|(&a, &b)| a * b).collect();
    let d_u: Array1<f32> = d_act.iter().zip(g.iter()).map(|(&a, &b)| a * b).collect();

    // silu'(z) = σ(z) * (1 + z * (1 - σ(z)))
    let d_g_pre: Array1<f32> = g_pre
        .iter()
        .zip(sigma.iter())
        .zip(d_g.iter())
        .map(|((&z, &s), &dg)| dg * s * (1.0 + z * (1.0 - s)))
        .collect();

    // d_x = gate_w.T @ d_g_pre + up_w.T @ d_u
    let mut d_x = Array1::<f32>::zeros(hidden);
    for j in 0..hidden {
        let mut s = 0.0_f32;
        for i in 0..ffn_dim {
            s += gate_w[[i, j]] * d_g_pre[i] + up_w[[i, j]] * d_u[i];
        }
        d_x[j] = s;
    }
    d_x
}

/// Backward through a single attention block at the last position.
///
/// Given the upstream gradient `d_residual_out` at the last position
/// (shape: hidden), computes `d_h_in` — the gradient at the input
/// residual's last position.
///
/// Requires forward-pass intermediates:
/// - `h_norm_last`: the post-input-norm hidden state at last position (used for Q/K/V proj)
/// - `q_rope_last`: Q after RoPE at last position, shape (num_q * head_dim,)
/// - `k_rope`: K after RoPE, shape (seq_len, num_kv * head_dim) — all rows for attention scores
/// - `v`: V values, shape (seq_len, num_kv * head_dim)
/// - Weight matrices: W_q, W_k, W_v, W_o
/// - RoPE parameters for Q/K at last position
/// - `input_norm_weight`: the input layernorm weight vector
/// - `h_last`: raw input hidden state at last position (for RMSNorm backward)
///
/// Returns: d_h at the last position (shape: hidden)
///
/// The forward at one layer is:
///   h_norm = RMSNorm(h, norm_weight)
///   q = h_norm @ W_q.T, k = h_norm @ W_k.T, v = h_norm @ W_v.T
///   apply RoPE to Q, K
///   attn_out = GQA(q, k, v)  [at last pos, attends to all positions]
///   o_out = attn_out @ W_o.T
///   h_post = h + o_out  [residual]
///
/// Backward:
///   d_h = d_residual_out                     (skip connection)
///   d_o_out = d_residual_out
///   d_attn_out = W_o.T @ d_o_out             (O-proj backward)
///   d_q, d_k_last, d_v_last = attention_backward(d_attn_out)
///   d_q_pre_rope = rope_backward(d_q)
///   d_k_pre_rope = rope_backward(d_k_last)
///   d_h_norm = W_q.T @ d_q_pre_rope + W_k.T @ d_k_pre_rope + W_v.T @ d_v_last
///   d_h += rmsnorm_backward(d_h_norm)
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention_backward_last_pos(
    d_residual_out: ArrayView1<f32>,    // (hidden,) upstream gradient
    h_last: ArrayView1<f32>,            // (hidden,) input residual at last pos
    _h_norm_last: ArrayView1<f32>,      // (hidden,) normed input at last pos (retained for future use)
    w_q: ArrayView2<f32>,               // (num_q * head_dim, hidden)
    w_k: ArrayView2<f32>,               // (num_kv * head_dim, hidden)
    w_v: ArrayView2<f32>,               // (num_kv * head_dim, hidden)
    w_o: ArrayView2<f32>,               // (hidden, num_q * head_dim)
    input_norm_weight: ArrayView1<f32>, // (hidden,)
    q_rope_last: ArrayView1<f32>,       // (num_q * head_dim,) Q after RoPE at last pos
    k_rope: ArrayView2<f32>,            // (seq_len, num_kv * head_dim)
    v: ArrayView2<f32>,                 // (seq_len, num_kv * head_dim)
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,                         // 1/sqrt(head_dim) or custom
    rope_base: f64,
    rotary_fraction: f64,
    seq_len: usize,
    norm_eps: f32,
) -> Array1<f32> {
    let hidden = h_last.len();
    let reps = num_q_heads / num_kv_heads;
    let q_dim = num_q_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;

    // ── Step 1: O-projection backward ──────────────────────────────────
    // Forward: o_out = attn_out @ W_o.T  where W_o is (hidden, q_dim)
    // So: d_attn_out = W_o.T @ d_residual_out → shape (q_dim,)
    // W_o is (hidden, q_dim), so W_o.T is (q_dim, hidden)
    let mut d_attn_out = Array1::<f32>::zeros(q_dim);
    for i in 0..q_dim {
        let mut s = 0.0_f32;
        for j in 0..hidden {
            s += w_o[[j, i]] * d_residual_out[j];
        }
        d_attn_out[i] = s;
    }

    // ── Step 2: GQA attention backward at last position ────────────────
    // For each Q head h (kv_h = h / reps):
    //   scores[j] = (q_last[h] · k[j, kv_h]) * scale  for j=0..seq_len
    //   probs = softmax(scores)  (causal mask: all positions visible at last pos)
    //   attn_out[h] = Σ_j probs[j] * v[j, kv_h]
    //
    // Backward:
    //   d_probs[j] = d_attn_out_h · v[j, kv_h]
    //   d_scores = softmax_backward(probs, d_probs)
    //   d_q_h = scale * Σ_j d_scores[j] * k[j, kv_h]
    //   d_k_last[kv_h] += scale * d_scores[seq-1] * q_last[h]
    //   d_v_last[kv_h] += probs[seq-1] * d_attn_out_h

    let mut d_q_rope = Array1::<f32>::zeros(q_dim);
    let mut d_k_rope_last = Array1::<f32>::zeros(kv_dim);
    let mut d_v_last = Array1::<f32>::zeros(kv_dim);

    let last_pos = seq_len - 1;
    let mut scores_buf = vec![0.0_f32; seq_len];
    let mut probs_buf = vec![0.0_f32; seq_len];
    let mut d_probs_buf = vec![0.0_f32; seq_len];

    for h in 0..num_q_heads {
        let kv_h = h / reps;
        let q_off = h * head_dim;
        let kv_off = kv_h * head_dim;

        // Recompute scores for last position (attends to ALL positions, causal)
        for j in 0..seq_len {
            let mut dot = 0.0_f32;
            for d in 0..head_dim {
                dot += q_rope_last[q_off + d] * k_rope[[j, kv_off + d]];
            }
            scores_buf[j] = dot * scale;
        }

        // Softmax
        let max_score = scores_buf[..seq_len]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut exp_sum = 0.0_f64;
        for j in 0..seq_len {
            let e = ((scores_buf[j] - max_score) as f64).exp();
            probs_buf[j] = e as f32;
            exp_sum += e;
        }
        let inv_sum = (1.0 / exp_sum) as f32;
        for p in probs_buf.iter_mut().take(seq_len) {
            *p *= inv_sum;
        }

        // d_probs[j] = d_attn_out_h · v[j, kv_h]
        for j in 0..seq_len {
            let mut dot = 0.0_f32;
            for d in 0..head_dim {
                dot += d_attn_out[q_off + d] * v[[j, kv_off + d]];
            }
            d_probs_buf[j] = dot;
        }

        // Softmax backward: d_scores[j] = probs[j] * (d_probs[j] - Σ_k probs[k]*d_probs[k])
        let weighted_sum: f32 = (0..seq_len)
            .map(|j| probs_buf[j] * d_probs_buf[j])
            .sum();
        // Reuse scores_buf for d_scores
        for j in 0..seq_len {
            scores_buf[j] = probs_buf[j] * (d_probs_buf[j] - weighted_sum);
        }

        // d_q_rope[h] = scale * Σ_j d_scores[j] * k[j, kv_h]
        for j in 0..seq_len {
            let ds_scaled = scale * scores_buf[j];
            for d in 0..head_dim {
                d_q_rope[q_off + d] += ds_scaled * k_rope[[j, kv_off + d]];
            }
        }

        // d_k_rope_last[kv_h] += scale * d_scores[last_pos] * q_last[h]
        let ds_last_scaled = scale * scores_buf[last_pos];
        for d in 0..head_dim {
            d_k_rope_last[kv_off + d] += ds_last_scaled * q_rope_last[q_off + d];
        }

        // d_v_last[kv_h] += probs[last_pos] * d_attn_out_h
        let p_last = probs_buf[last_pos];
        for d in 0..head_dim {
            d_v_last[kv_off + d] += p_last * d_attn_out[q_off + d];
        }
    }

    // ── Step 3: RoPE backward on Q at last position ────────────────────
    // Forward RoPE: out[i] = x[i]*cos - x[i+half]*sin
    //               out[i+half] = x[i]*sin + x[i+half]*cos
    // Backward (inverse rotation — transpose of rotation matrix):
    //   d_pre[i]      = d_post[i]*cos + d_post[i+half]*sin
    //   d_pre[i+half] = -d_post[i]*sin + d_post[i+half]*cos

    let rotary_dim = ((head_dim as f64 * rotary_fraction) as usize).max(2);
    let half_rotary = rotary_dim / 2;

    // Precompute inv_freq
    let inv_freq: Vec<f64> = (0..half_rotary)
        .map(|i| 1.0 / rope_base.powf(2.0 * i as f64 / rotary_dim as f64))
        .collect();

    // Q RoPE backward at position = last_pos
    let mut d_q_pre_rope = Array1::<f32>::zeros(q_dim);
    for h in 0..num_q_heads {
        let offset = h * head_dim;
        for i in 0..half_rotary {
            let theta = last_pos as f64 * inv_freq[i];
            let cos_t = theta.cos() as f32;
            let sin_t = theta.sin() as f32;

            let dq_i = d_q_rope[offset + i];
            let dq_ih = d_q_rope[offset + half_rotary + i];

            d_q_pre_rope[offset + i] = dq_i * cos_t + dq_ih * sin_t;
            d_q_pre_rope[offset + half_rotary + i] = -dq_i * sin_t + dq_ih * cos_t;
        }
        // Pass-through dims (beyond rotary_dim) have identity RoPE
        for i in rotary_dim..head_dim {
            d_q_pre_rope[offset + i] = d_q_rope[offset + i];
        }
    }

    // ── Step 4: RoPE backward on K at last position ────────────────────
    let mut d_k_pre_rope = Array1::<f32>::zeros(kv_dim);
    for h in 0..num_kv_heads {
        let offset = h * head_dim;
        for i in 0..half_rotary {
            let theta = last_pos as f64 * inv_freq[i];
            let cos_t = theta.cos() as f32;
            let sin_t = theta.sin() as f32;

            let dk_i = d_k_rope_last[offset + i];
            let dk_ih = d_k_rope_last[offset + half_rotary + i];

            d_k_pre_rope[offset + i] = dk_i * cos_t + dk_ih * sin_t;
            d_k_pre_rope[offset + half_rotary + i] = -dk_i * sin_t + dk_ih * cos_t;
        }
        // Pass-through dims
        for i in rotary_dim..head_dim {
            d_k_pre_rope[offset + i] = d_k_rope_last[offset + i];
        }
    }

    // ── Step 5: QKV projection backward ────────────────────────────────
    // Forward: q = h_norm @ W_q.T  where W_q is (q_dim, hidden)
    //          k = h_norm @ W_k.T  where W_k is (kv_dim, hidden)
    //          v = h_norm @ W_v.T  where W_v is (kv_dim, hidden)
    // Backward: d_h_norm = W_q.T @ d_q_pre_rope + W_k.T @ d_k_pre_rope + W_v.T @ d_v_last
    //   W_q.T is (hidden, q_dim), so d_h_norm[j] = Σ_i W_q[i,j] * d_q[i]
    let mut d_h_norm = Array1::<f32>::zeros(hidden);
    for j in 0..hidden {
        let mut s = 0.0_f32;
        for i in 0..q_dim {
            s += w_q[[i, j]] * d_q_pre_rope[i];
        }
        for i in 0..kv_dim {
            s += w_k[[i, j]] * d_k_pre_rope[i];
            s += w_v[[i, j]] * d_v_last[i];
        }
        d_h_norm[j] = s;
    }

    // ── Step 6: Input RMSNorm backward ─────────────────────────────────
    let d_h_from_norm = rmsnorm_backward_pos(
        h_last,
        input_norm_weight,
        d_h_norm.view(),
        norm_eps,
    );

    // ── Step 7: Skip connection ────────────────────────────────────────
    // h_post = h + o_proj(attention(norm(h)))
    // d_h = d_residual_out (skip) + d_h_from_norm (through attention path)
    let mut d_h = Array1::<f32>::zeros(hidden);
    for i in 0..hidden {
        d_h[i] = d_residual_out[i] + d_h_from_norm[i];
    }

    d_h
}

/// Per-fact target delta optimisation.
///
/// CURRENT SUPPORT: `install_layer = n_layers - 1` (last layer). The
/// perturbation at the last block's output flows through only
/// `final_norm` + `lm_head` to logits, both of which have verified
/// backward primitives in this module. For earlier layers the
/// backward through intermediate transformer blocks is still being
/// ported (see `gated_ffn_backward` and `attention_backward_last_pos`
/// stubs).
///
/// Runs Adam for `opts.steps` iterations on a delta ∈ R^hidden,
/// minimising `CE(logits, target_id) + kl_weight · KL(baseline, current)`.
pub fn optimise_target_delta(
    weights: &ModelWeights,
    tokens: &[u32],
    target_id: u32,
    install_layer: usize,
    opts: TargetDeltaOpts,
) -> Result<TargetDelta, String> {
    let n_layers = weights.arch.config().num_layers;
    if install_layer >= n_layers {
        return Err(format!(
            "install_layer {install_layer} ≥ n_layers {n_layers}"
        ));
    }
    if install_layer != n_layers - 1 {
        return Err(format!(
            "optimise_target_delta: only install_layer = n_layers-1 = {} is \
             supported in this build (got {install_layer}). Mid-layer backward \
             through attention+FFN is pending (target_delta.rs stubs).",
            n_layers - 1
        ));
    }

    let hidden = weights.arch.config().hidden_size;
    let norm_offset = weights.arch.norm_weight_offset();
    let final_norm_key = weights.arch.final_norm_key();
    let norm_weight_vec: Vec<f32> = weights
        .vectors
        .get(final_norm_key)
        .map(|v| {
            let mut w = v.to_vec();
            for x in w.iter_mut() {
                *x += norm_offset;
            }
            w
        })
        .ok_or_else(|| format!("missing final norm weight key: {final_norm_key}"))?;
    let norm_weight = Array1::from(norm_weight_vec);
    let inv_scale = 1.0 / weights.arch.logits_scaling();
    if weights.arch.final_logit_softcapping().is_some() {
        return Err(
            "target-delta opt doesn't yet handle logit softcap — port required".into(),
        );
    }

    // Baseline forward (no perturbation) for KL regulariser.
    let baseline = crate::forward::predict::forward_raw_logits(weights, tokens, None);
    let base_probs = softmax_1d(&baseline.logits);
    let baseline_loss = {
        let (l, _) = cross_entropy_and_grad(baseline.logits.view(), target_id);
        l
    };

    // Adam state.
    let mut delta = Array1::<f32>::zeros(hidden);
    let mut m = Array1::<f32>::zeros(hidden);
    let mut v = Array1::<f32>::zeros(hidden);
    const BETA1: f32 = 0.9;
    const BETA2: f32 = 0.999;
    const ADAM_EPS: f32 = 1e-8;
    const RMS_EPS: f32 = 1e-6;

    let mut final_loss = f32::NAN;
    for step in 1..=opts.steps {
        let out = crate::forward::predict::forward_raw_logits(
            weights,
            tokens,
            Some((install_layer, delta.view())),
        );

        // Loss: CE(target) + kl_weight · KL(base || current)
        let (ce, mut dlogits) = cross_entropy_and_grad(out.logits.view(), target_id);
        let cur_probs = softmax_1d(&out.logits);

        // KL(p || q) gradient on logits: q - p (where q is current probs, p is baseline).
        // Add to dlogits weighted by kl_weight.
        if opts.kl_weight != 0.0 {
            for i in 0..dlogits.len() {
                dlogits[i] += opts.kl_weight * (cur_probs[i] - base_probs[i]);
            }
        }

        // KL value for diagnostics (not strictly needed for backprop).
        let kl_val: f32 = if opts.kl_weight != 0.0 {
            base_probs
                .iter()
                .zip(cur_probs.iter())
                .map(|(&p, &q)| {
                    if p < 1e-12 {
                        0.0
                    } else {
                        p * (p.max(1e-12).ln() - q.max(1e-12).ln())
                    }
                })
                .sum()
        } else {
            0.0
        };
        final_loss = ce + opts.kl_weight * kl_val;

        // Backward: logits ← lm_head ← h_final ← final_norm ← h_pre_norm.
        // Scale gradient by inv_scale since logits = raw / scale.
        for d in dlogits.iter_mut() {
            *d *= inv_scale;
        }

        let last_final = out.h_final.row(out.h_final.nrows() - 1);
        let _last_pre_norm = out.h_pre_norm.row(out.h_pre_norm.nrows() - 1);
        let _ = last_final;

        // lm_head backward: weights.lm_head shape (vocab, hidden); logits = lm_head @ h_last
        let lm = &weights.lm_head;
        let d_h_final = lm_head_backward(lm.view(), dlogits.view());

        // RMSNorm backward at the last position:
        // h_pre_norm[-1] is input; norm_weight is scale; d_h_final is upstream grad.
        let last_pre = out.h_pre_norm.row(out.h_pre_norm.nrows() - 1).to_owned();
        let d_h_pre_norm =
            rmsnorm_backward_pos(last_pre.view(), norm_weight.view(), d_h_final.view(), RMS_EPS);

        // For install_layer = n_layers - 1, δ is added directly to
        // h[-1] after the last block. So ∂loss/∂δ = d_h_pre_norm.
        let grad = d_h_pre_norm;

        // Adam update.
        let s = step as f32;
        let bc1 = 1.0 - BETA1.powi(step as i32);
        let bc2 = 1.0 - BETA2.powi(step as i32);
        for i in 0..hidden {
            m[i] = BETA1 * m[i] + (1.0 - BETA1) * grad[i];
            v[i] = BETA2 * v[i] + (1.0 - BETA2) * grad[i] * grad[i];
            let m_hat = m[i] / bc1;
            let v_hat = v[i] / bc2;
            delta[i] -= opts.lr * m_hat / (v_hat.sqrt() + ADAM_EPS);
        }
        let _ = s;
    }

    if opts.normalise {
        let norm: f32 = delta.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in delta.iter_mut() {
                *x /= norm;
            }
        }
    }

    Ok(TargetDelta {
        layer: install_layer,
        delta,
        final_loss,
        baseline_loss,
    })
}

/// Softmax over a 1-D vector (numerically stable).
fn softmax_1d(logits: &Array1<f32>) -> Array1<f32> {
    let max = logits.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exps: Array1<f32> = logits.map(|&v| (v - max).exp());
    let sum: f32 = exps.iter().sum();
    exps.map(|&v| v / sum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2, Array2};

    #[test]
    fn cross_entropy_and_grad_matches_numerical() {
        // Reference: with logits [1.0, 2.0, 0.5], target=1
        // softmax(logits) ≈ [0.2312, 0.6285, 0.1402]
        // loss = -log(0.6285) ≈ 0.4644
        // dlogits = softmax - onehot(1) = [0.2312, -0.3715, 0.1402]
        let logits = arr1(&[1.0_f32, 2.0, 0.5]);
        let (loss, dlogits) = cross_entropy_and_grad(logits.view(), 1);
        assert!((loss - 0.4644).abs() < 1e-3, "loss {loss}");
        assert!((dlogits[0] - 0.2312).abs() < 1e-3);
        assert!((dlogits[1] - (-0.3715)).abs() < 1e-3);
        assert!((dlogits[2] - 0.1402).abs() < 1e-3);
    }

    #[test]
    fn lm_head_backward_shape_and_values() {
        // embed shape (vocab=3, hidden=4), dlogits (3,) → dh (4,)
        let embed = arr2(&[
            [1.0_f32, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ]);
        let dlogits = arr1(&[0.5_f32, -0.3, 0.2]);
        let dh = lm_head_backward(embed.view(), dlogits.view());
        // dh[i] = Σ_v dlogits[v] * embed[v,i]
        //  dh[0] = 0.5*1 + -0.3*0 + 0.2*1 = 0.7
        //  dh[1] = 0.5*0 + -0.3*1 + 0.2*1 = -0.1
        //  dh[2] = 0.2
        //  dh[3] = 0.2
        assert!((dh[0] - 0.7).abs() < 1e-5);
        assert!((dh[1] - (-0.1)).abs() < 1e-5);
        assert!((dh[2] - 0.2).abs() < 1e-5);
        assert!((dh[3] - 0.2).abs() < 1e-5);
    }

    #[test]
    fn gated_ffn_backward_finite_difference() {
        // Small hand-sized case: hidden=3, ffn_dim=4
        let x = arr1(&[0.5_f32, -0.3, 1.0]);
        let gate_w = arr2(&[
            [0.1_f32, -0.2, 0.3],
            [0.4, 0.5, -0.1],
            [-0.3, 0.2, 0.4],
            [0.1, 0.1, -0.2],
        ]);
        let up_w = arr2(&[
            [0.2_f32, 0.1, -0.3],
            [-0.1, 0.4, 0.2],
            [0.3, -0.2, 0.1],
            [0.0, 0.3, 0.2],
        ]);
        let down_w = arr2(&[
            [0.1_f32, 0.2, -0.1, 0.3],
            [0.4, -0.2, 0.1, 0.1],
            [-0.3, 0.1, 0.2, -0.1],
        ]);
        // Forward helper
        let fwd = |xi: &Array1<f32>| -> Array1<f32> {
            let g_pre: Array1<f32> = (0..gate_w.nrows())
                .map(|i| (0..xi.len()).map(|j| gate_w[[i, j]] * xi[j]).sum())
                .collect();
            let u: Array1<f32> = (0..up_w.nrows())
                .map(|i| (0..xi.len()).map(|j| up_w[[i, j]] * xi[j]).sum())
                .collect();
            let g: Array1<f32> = g_pre.map(|&z| z / (1.0 + (-z).exp()));
            let act: Array1<f32> = g.iter().zip(u.iter()).map(|(&a, &b)| a * b).collect();
            (0..down_w.nrows())
                .map(|k| {
                    (0..down_w.ncols())
                        .map(|i| down_w[[k, i]] * act[i])
                        .sum()
                })
                .collect()
        };
        // Loss = sum(out) so d_out = ones
        let d_out = Array1::from_elem(3, 1.0_f32);
        let dx_analytical =
            gated_ffn_backward(x.view(), gate_w.view(), up_w.view(), down_w.view(), d_out.view());
        let h = 1e-4_f32;
        for i in 0..x.len() {
            let mut xp = x.clone();
            xp[i] += h;
            let mut xm = x.clone();
            xm[i] -= h;
            let lp: f32 = fwd(&xp).iter().sum();
            let lm: f32 = fwd(&xm).iter().sum();
            let num = (lp - lm) / (2.0 * h);
            let err = (dx_analytical[i] - num).abs();
            assert!(err < 1e-2, "dx[{i}]: analytical {} vs numerical {num}", dx_analytical[i]);
        }
    }

    #[test]
    fn rmsnorm_backward_finite_difference() {
        // Analytical gradient should match numerical at a random point.
        let x = arr1(&[0.5_f32, 1.0, -0.5, 2.0]);
        let w = arr1(&[1.0_f32, 0.5, 2.0, 1.5]);
        let eps = 1e-5_f32;

        // Forward helper
        let fwd = |xi: &Array1<f32>| -> Array1<f32> {
            let d = xi.len() as f32;
            let ms = xi.iter().map(|v| v * v).sum::<f32>() / d;
            let rms = (ms + eps).sqrt();
            xi.iter()
                .zip(w.iter())
                .map(|(xv, wv)| (xv / rms) * wv)
                .collect()
        };

        // Loss = sum of y (so dy = ones)
        let dy = Array1::from_elem(x.len(), 1.0_f32);
        let dx_analytical = rmsnorm_backward_pos(x.view(), w.view(), dy.view(), eps);

        // Numerical dx via finite difference
        let h = 1e-4_f32;
        for i in 0..x.len() {
            let mut xp = x.clone();
            xp[i] += h;
            let mut xm = x.clone();
            xm[i] -= h;
            let loss_p: f32 = fwd(&xp).iter().sum();
            let loss_m: f32 = fwd(&xm).iter().sum();
            let num = (loss_p - loss_m) / (2.0 * h);
            let err = (dx_analytical[i] - num).abs();
            assert!(err < 1e-2, "dx[{i}]: analytical {} vs numerical {num} (err {err})", dx_analytical[i]);
        }
    }

    #[test]
    fn attention_backward_last_pos_finite_difference() {
        // Small attention block: hidden=4, num_q=2, num_kv=2, head_dim=2, seq_len=3
        // (no GQA grouping for simplicity — reps=1)
        let hidden = 4;
        let num_q = 2;
        let num_kv = 2;
        let head_dim = 2;
        let seq_len = 3;
        let last_pos = seq_len - 1;
        let q_dim = num_q * head_dim;
        let kv_dim = num_kv * head_dim;
        let norm_eps = 1e-5_f32;
        let rope_base = 10000.0_f64;
        let rotary_fraction = 1.0_f64;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Fixed weight matrices (random-ish but deterministic)
        let w_q = arr2(&[
            [0.1_f32, -0.2, 0.3, 0.1],
            [0.4, 0.2, -0.1, 0.3],
            [-0.2, 0.1, 0.4, -0.3],
            [0.3, -0.1, 0.2, 0.1],
        ]);
        let w_k = arr2(&[
            [0.2_f32, 0.1, -0.3, 0.2],
            [-0.1, 0.3, 0.1, -0.2],
            [0.3, -0.2, 0.1, 0.4],
            [0.1, 0.2, -0.1, 0.3],
        ]);
        let w_v = arr2(&[
            [-0.1_f32, 0.2, 0.3, -0.1],
            [0.2, -0.3, 0.1, 0.2],
            [0.1, 0.1, -0.2, 0.3],
            [-0.2, 0.3, 0.2, -0.1],
        ]);
        let w_o = arr2(&[
            [0.1_f32, 0.2, -0.1, 0.3],
            [-0.2, 0.1, 0.3, -0.1],
            [0.3, -0.1, 0.2, 0.1],
            [0.1, 0.3, -0.2, 0.2],
        ]);
        let norm_weight = arr1(&[1.0_f32, 0.8, 1.2, 0.9]);

        // Input residual stream at all positions — we only perturb last row.
        let h_all = arr2(&[
            [0.5_f32, -0.3, 1.0, 0.2],
            [0.1, 0.8, -0.5, 0.4],
            [0.7, -0.2, 0.3, 0.9],
        ]);

        // Forward function: given h_last (last row), produce the output
        // scalar loss = sum(h_post_last) where h_post_last = h_last + o_proj(attn_out_last)
        // This exercises the entire attention block at the last position.
        let forward_full = |h_last_vec: &Array1<f32>| -> f32 {
            // Build h_all with modified last row
            let mut h_mod = h_all.clone();
            for j in 0..hidden {
                h_mod[[last_pos, j]] = h_last_vec[j];
            }

            // RMSNorm all positions
            let mut h_normed = Array2::<f32>::zeros((seq_len, hidden));
            for pos in 0..seq_len {
                let d = hidden as f32;
                let row_slice = h_mod.row(pos);
                let ms: f32 = row_slice.iter().map(|x| x * x).sum::<f32>() / d;
                let rms = (ms + norm_eps).sqrt();
                for j in 0..hidden {
                    h_normed[[pos, j]] = (row_slice[j] / rms) * norm_weight[j];
                }
            }

            // Q/K/V projections: proj = h_normed @ W.T
            let mut q_full = Array2::<f32>::zeros((seq_len, q_dim));
            let mut k_full = Array2::<f32>::zeros((seq_len, kv_dim));
            let mut v_full = Array2::<f32>::zeros((seq_len, kv_dim));
            for pos in 0..seq_len {
                for i in 0..q_dim {
                    let mut s = 0.0_f32;
                    for j in 0..hidden {
                        s += w_q[[i, j]] * h_normed[[pos, j]];
                    }
                    q_full[[pos, i]] = s;
                }
                for i in 0..kv_dim {
                    let mut sk = 0.0_f32;
                    let mut sv = 0.0_f32;
                    for j in 0..hidden {
                        sk += w_k[[i, j]] * h_normed[[pos, j]];
                        sv += w_v[[i, j]] * h_normed[[pos, j]];
                    }
                    k_full[[pos, i]] = sk;
                    v_full[[pos, i]] = sv;
                }
            }

            // RoPE
            let rotary_dim = ((head_dim as f64 * rotary_fraction) as usize).max(2);
            let half_rotary = rotary_dim / 2;
            let inv_freq: Vec<f64> = (0..half_rotary)
                .map(|i| 1.0 / rope_base.powf(2.0 * i as f64 / rotary_dim as f64))
                .collect();

            let mut q_rope = q_full.clone();
            let mut k_rope_all = k_full.clone();
            for pos in 0..seq_len {
                for hh in 0..num_q {
                    let off = hh * head_dim;
                    for i in 0..half_rotary {
                        let theta = pos as f64 * inv_freq[i];
                        let cos_t = theta.cos() as f32;
                        let sin_t = theta.sin() as f32;
                        let x0 = q_full[[pos, off + i]];
                        let x1 = q_full[[pos, off + half_rotary + i]];
                        q_rope[[pos, off + i]] = x0 * cos_t - x1 * sin_t;
                        q_rope[[pos, off + half_rotary + i]] = x0 * sin_t + x1 * cos_t;
                    }
                }
                for hh in 0..num_kv {
                    let off = hh * head_dim;
                    for i in 0..half_rotary {
                        let theta = pos as f64 * inv_freq[i];
                        let cos_t = theta.cos() as f32;
                        let sin_t = theta.sin() as f32;
                        let x0 = k_full[[pos, off + i]];
                        let x1 = k_full[[pos, off + half_rotary + i]];
                        k_rope_all[[pos, off + i]] = x0 * cos_t - x1 * sin_t;
                        k_rope_all[[pos, off + half_rotary + i]] = x0 * sin_t + x1 * cos_t;
                    }
                }
            }

            // GQA at last position only
            let reps = num_q / num_kv;
            let mut attn_out_last = vec![0.0_f32; q_dim];
            for hh in 0..num_q {
                let kv_h = hh / reps;
                let q_off = hh * head_dim;
                let kv_off = kv_h * head_dim;

                // scores: q_last · k[j] * scale for all j
                let mut scores = vec![0.0_f32; seq_len];
                for j in 0..seq_len {
                    let mut dot = 0.0_f32;
                    for d in 0..head_dim {
                        dot += q_rope[[last_pos, q_off + d]] * k_rope_all[[j, kv_off + d]];
                    }
                    scores[j] = dot * scale;
                }
                // softmax
                let max_s = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut exp_sum = 0.0_f64;
                for s in scores.iter_mut() {
                    let e = ((*s - max_s) as f64).exp();
                    *s = e as f32;
                    exp_sum += e;
                }
                let inv_sum = (1.0 / exp_sum) as f32;
                for s in scores.iter_mut() {
                    *s *= inv_sum;
                }
                // weighted sum of V
                for d in 0..head_dim {
                    let mut acc = 0.0_f32;
                    for j in 0..seq_len {
                        acc += scores[j] * v_full[[j, kv_off + d]];
                    }
                    attn_out_last[q_off + d] = acc;
                }
            }

            // O projection: o_out = attn_out @ W_o.T where W_o is (hidden, q_dim)
            let mut o_out = vec![0.0_f32; hidden];
            for j in 0..hidden {
                let mut s = 0.0_f32;
                for i in 0..q_dim {
                    s += w_o[[j, i]] * attn_out_last[i];
                }
                o_out[j] = s;
            }

            // Residual + loss = sum(h_last + o_out)
            let mut loss = 0.0_f32;
            for j in 0..hidden {
                loss += h_last_vec[j] + o_out[j];
            }
            loss
        };

        // Compute the analytical gradient using our function.
        // The upstream gradient for loss=sum(h_post) is d_residual_out = ones.
        let d_residual_out = Array1::from_elem(hidden, 1.0_f32);
        let h_last = h_all.row(last_pos).to_owned();

        // We need intermediates: h_norm_last, q_rope_last, k_rope, v
        // Compute them from the base h_all.
        let h_norm_last = {
            let d = hidden as f32;
            let ms: f32 = h_last.iter().map(|x| x * x).sum::<f32>() / d;
            let rms = (ms + norm_eps).sqrt();
            let normed: Array1<f32> = h_last.iter().zip(norm_weight.iter())
                .map(|(&x, &w)| (x / rms) * w)
                .collect();
            normed
        };

        // Compute Q/K/V and apply RoPE for the intermediates
        let rotary_dim = ((head_dim as f64 * rotary_fraction) as usize).max(2);
        let half_rotary = rotary_dim / 2;
        let inv_freq: Vec<f64> = (0..half_rotary)
            .map(|i| 1.0 / rope_base.powf(2.0 * i as f64 / rotary_dim as f64))
            .collect();

        // h_normed for all positions
        let mut h_normed_all = Array2::<f32>::zeros((seq_len, hidden));
        for pos in 0..seq_len {
            let d = hidden as f32;
            let row = h_all.row(pos);
            let ms: f32 = row.iter().map(|x| x * x).sum::<f32>() / d;
            let rms = (ms + norm_eps).sqrt();
            for j in 0..hidden {
                h_normed_all[[pos, j]] = (row[j] / rms) * norm_weight[j];
            }
        }

        // Full projections + RoPE
        let mut k_rope_full = Array2::<f32>::zeros((seq_len, kv_dim));
        let mut v_full = Array2::<f32>::zeros((seq_len, kv_dim));

        for pos in 0..seq_len {
            // K projection
            for i in 0..kv_dim {
                let mut sk = 0.0_f32;
                let mut sv = 0.0_f32;
                for j in 0..hidden {
                    sk += w_k[[i, j]] * h_normed_all[[pos, j]];
                    sv += w_v[[i, j]] * h_normed_all[[pos, j]];
                }
                // Apply RoPE to K
                // (store pre-rope temporarily, then rotate)
                k_rope_full[[pos, i]] = sk;
                v_full[[pos, i]] = sv;
            }
            // RoPE on K
            for hh in 0..num_kv {
                let off = hh * head_dim;
                for i in 0..half_rotary {
                    let theta = pos as f64 * inv_freq[i];
                    let cos_t = theta.cos() as f32;
                    let sin_t = theta.sin() as f32;
                    let x0 = k_rope_full[[pos, off + i]];
                    let x1 = k_rope_full[[pos, off + half_rotary + i]];
                    k_rope_full[[pos, off + i]] = x0 * cos_t - x1 * sin_t;
                    k_rope_full[[pos, off + half_rotary + i]] = x0 * sin_t + x1 * cos_t;
                }
            }
        }

        // Q at last position
        let mut q_pre_rope_last = Array1::<f32>::zeros(q_dim);
        for i in 0..q_dim {
            let mut s = 0.0_f32;
            for j in 0..hidden {
                s += w_q[[i, j]] * h_normed_all[[last_pos, j]];
            }
            q_pre_rope_last[i] = s;
        }
        // RoPE on Q at last pos
        let mut q_rope_last_vec = q_pre_rope_last.clone();
        for hh in 0..num_q {
            let off = hh * head_dim;
            for i in 0..half_rotary {
                let theta = last_pos as f64 * inv_freq[i];
                let cos_t = theta.cos() as f32;
                let sin_t = theta.sin() as f32;
                let x0 = q_pre_rope_last[off + i];
                let x1 = q_pre_rope_last[off + half_rotary + i];
                q_rope_last_vec[off + i] = x0 * cos_t - x1 * sin_t;
                q_rope_last_vec[off + half_rotary + i] = x0 * sin_t + x1 * cos_t;
            }
        }

        // Analytical gradient
        let dh_analytical = attention_backward_last_pos(
            d_residual_out.view(),
            h_last.view(),
            h_norm_last.view(),
            w_q.view(),
            w_k.view(),
            w_v.view(),
            w_o.view(),
            norm_weight.view(),
            q_rope_last_vec.view(),
            k_rope_full.view(),
            v_full.view(),
            num_q,
            num_kv,
            head_dim,
            scale,
            rope_base,
            rotary_fraction,
            seq_len,
            norm_eps,
        );

        // Numerical gradient via finite differences
        let eps_fd = 1e-3_f32;
        for i in 0..hidden {
            let mut hp = h_last.clone();
            hp[i] += eps_fd;
            let mut hm = h_last.clone();
            hm[i] -= eps_fd;
            let loss_p = forward_full(&hp);
            let loss_m = forward_full(&hm);
            let numerical = (loss_p - loss_m) / (2.0 * eps_fd);
            let err = (dh_analytical[i] - numerical).abs();
            let rel_err = err / (numerical.abs().max(dh_analytical[i].abs()).max(1e-6));
            assert!(
                rel_err < 0.05 || err < 1e-3,
                "d_h[{i}]: analytical={} vs numerical={} (abs_err={err}, rel_err={rel_err})",
                dh_analytical[i], numerical
            );
        }
    }
}
