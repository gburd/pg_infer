//! BitNet 1.58 native-ternary inference building blocks.
//!
//! Ported from the upstream larql project's `ternary.rs`.  This
//! module assembles the scaled ternary matvec kernel from
//! [`infer_compute::cpu::ops::bitlinear_matvec`] into the
//! higher-level pieces a BitNet forward pass needs:
//!
//! - [`BitNetFfn`]: a complete FFN block — RMSnorm → gate / up
//!   BitLinear projections → squared-ReLU activation (BitNet b1.58
//!   uses ReLU², not SwiGLU) → element-wise multiply →
//!   post-FFN-norm → down BitLinear → residual addition.
//! - [`BitnetModel`] / [`BitnetLayer`]: the full per-layer weight
//!   set (q/k/v/o BitLinear projections + RMSnorm scales) plus the
//!   embed table, output norm, and LM head.
//! - [`predict_bitnet`]: single-shot prefill returning top-K
//!   next-token predictions.
//! - [`BitnetKvCache`] + [`prefill`] / [`decode_step`] /
//!   [`generate_greedy`]: KV-cached autoregressive decode.
//!
//! All weights are held as [`BitLinearWeight`] (typed ternary
//! container with per-channel scale).  No f16 / f32 weight tensors
//! are ever materialised — the arithmetic stays in trit accumulation
//! plus one f32 scale per output channel.  At BitNet b1.58 2B 4T this
//! drops the resident working set from ~5 GB (f16-after-dequant) to
//! ~1.1 GB (I2_S bytes + scales + norms).
//!
//! ## Scope note (pg_infer port)
//!
//! The upstream module also shipped `generate_sampled` /
//! `generate_streaming_bitnet` (built on a `Sampler` / `SamplingConfig`
//! stack) and `load_bitnet_model` (built on a `bitnet_loader` vindex
//! path).  Neither dependency exists in pg_infer yet, so this port
//! provides a self-contained greedy [`generate_greedy`] in their place
//! and defers the loader to vindex-side follow-up.

use infer_compute::cpu::ops::bitlinear_matvec::{matvec_i2s_f32_into, BitLinearWeight};
use ndarray::{Array1, Array2, ArrayView2};

/// One BitLinear-FFN block.  Holds three ternary weight tensors
/// (gate, up, down) and the two RMSnorm scales (input, post-gate-up).
///
/// Layer ordering (BitNet b1.58 architecture):
///
/// ```text
///   x        : input residual                                  [hidden]
///   x_norm   = rmsnorm(x, ffn_norm.weight, eps)                [hidden]
///   gate     = matvec_i2s(gate.weight, x_norm) (* gate_scale)   [inter]
///   up       = matvec_i2s(up.weight,   x_norm) (* up_scale)     [inter]
///   hid      = (gate * gate) * up                                [inter]
///   hid_norm = rmsnorm(hid, ffn_sub_norm.weight, eps)            [inter]
///   y        = matvec_i2s(down.weight, hid_norm) (* down_scale)  [hidden]
///   x_out    = x + y                                              [hidden]
/// ```
///
/// `gate_scale`, `up_scale`, and `down_scale` are baked into the
/// [`BitLinearWeight::channel_scales`] of each tensor, so the matvec
/// call already returns scaled outputs.
pub struct BitNetFfn {
    pub gate: BitLinearWeight,
    pub up: BitLinearWeight,
    pub down: BitLinearWeight,
    /// Per-channel weight for the input RMSnorm (`ffn_norm.weight`),
    /// length = `hidden_size`.
    pub ffn_norm: Vec<f32>,
    /// Per-channel weight for the post-gate-up RMSnorm
    /// (`ffn_sub_norm.weight`), length = `intermediate_size`.
    pub ffn_sub_norm: Vec<f32>,
    /// RMSnorm epsilon (typically 1e-5).
    pub eps: f32,
}

impl BitNetFfn {
    /// Run one forward step: `x_out = x + ffn(rmsnorm(x))`.
    ///
    /// Allocates scratch buffers.  For per-token-allocations-matter
    /// callers, see [`Self::forward_into`].
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let hidden = x.len();
        let inter = self.gate.rows;
        let mut gate = vec![0.0f32; inter];
        let mut up = vec![0.0f32; inter];
        let mut hid = vec![0.0f32; inter];
        let mut y = vec![0.0f32; hidden];
        self.forward_into(x, &mut gate, &mut up, &mut hid, &mut y);
        // Residual addition: y already holds the FFN output.
        for (yo, xi) in y.iter_mut().zip(x.iter()) {
            *yo += xi;
        }
        y
    }

    /// In-place variant that uses caller-provided scratch buffers.
    ///
    /// `gate`, `up`, `hid` must each be length `intermediate_size`.
    /// `y` must be length `hidden_size`.  All four buffers are
    /// overwritten.  Caller is responsible for the residual-add step.
    pub fn forward_into(
        &self,
        x: &[f32],
        gate: &mut [f32],
        up: &mut [f32],
        hid: &mut [f32],
        y: &mut [f32],
    ) {
        let hidden = x.len();
        let inter = self.gate.rows;
        debug_assert_eq!(self.up.rows, inter);
        debug_assert_eq!(self.down.cols, inter);
        debug_assert_eq!(self.down.rows, hidden);
        debug_assert_eq!(gate.len(), inter);
        debug_assert_eq!(up.len(), inter);
        debug_assert_eq!(hid.len(), inter);
        debug_assert_eq!(y.len(), hidden);
        debug_assert_eq!(self.ffn_norm.len(), hidden);
        debug_assert_eq!(self.ffn_sub_norm.len(), inter);

        // 1. Input RMSnorm.
        let mut x_norm = vec![0.0f32; hidden];
        rmsnorm_into(x, &self.ffn_norm, self.eps, &mut x_norm);

        // 2. gate / up projections via ternary matvec.
        matvec_i2s_f32_into(&self.gate, &x_norm, gate).expect("gate shape");
        matvec_i2s_f32_into(&self.up, &x_norm, up).expect("up shape");

        // 3. Squared-ReLU activation (BitNet b1.58 spec) +
        //    element-wise multiply with up.
        for ((g, u), h) in gate.iter().zip(up.iter()).zip(hid.iter_mut()) {
            let relu = g.max(0.0);
            *h = relu * relu * u;
        }

        // 4. Post-gate-up RMSnorm.
        let mut hid_norm = vec![0.0f32; inter];
        rmsnorm_into(hid, &self.ffn_sub_norm, self.eps, &mut hid_norm);

        // 5. y = ternary(down.weight) · hid_norm
        matvec_i2s_f32_into(&self.down, &hid_norm, y).expect("down shape");
    }
}

/// RMS normalisation: `out[i] = (x[i] / rms(x)) * weight[i]`,
/// where `rms(x) = sqrt(mean(x_i^2) + eps)`.
pub fn rmsnorm_into(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    debug_assert_eq!(x.len(), weight.len());
    debug_assert_eq!(out.len(), x.len());
    let mut ss = 0.0f64;
    for &v in x {
        ss += (v as f64) * (v as f64);
    }
    let inv = (1.0 / (ss / (x.len() as f64) + eps as f64).sqrt()) as f32;
    for ((o, &xi), &wi) in out.iter_mut().zip(x.iter()).zip(weight.iter()) {
        *o = xi * inv * wi;
    }
}

/// Complete BitNet 1.58 model — every tensor needed for a forward
/// pass.  Construct from a `--keep-quant` vindex (loader is vindex-side
/// follow-up) and feed into [`predict_bitnet`] / [`prefill`].
pub struct BitnetModel {
    /// Per-layer BitLinear projections + RMSnorm weights.
    pub layers: Vec<BitnetLayer>,
    /// Token embedding table, shape [vocab, hidden], f32.
    pub embed: Array2<f32>,
    /// Optional embed scale (most BitNet builds = 1.0).
    pub embed_scale: f32,
    /// Output RMSnorm weight, length = hidden_size.
    pub output_norm: Vec<f32>,
    /// LM head matrix, shape [vocab, hidden], f32.
    pub lm_head: Array2<f32>,
    /// RMSnorm epsilon used everywhere.
    pub eps: f32,
    /// Per-head dimension (= hidden / n_q_heads typically).
    pub head_dim: usize,
    /// Number of query heads.
    pub n_q_heads: usize,
    /// Number of key/value heads (GQA: usually < n_q_heads).
    pub n_kv_heads: usize,
    /// RoPE base (theta) — read from GGUF metadata.
    pub rope_base: f64,
}

/// One transformer block's worth of BitLinear weights + norms.
pub struct BitnetLayer {
    pub attn_norm: Vec<f32>,
    pub attn_q: BitLinearWeight,
    pub attn_k: BitLinearWeight,
    pub attn_v: BitLinearWeight,
    pub attn_sub_norm: Vec<f32>,
    pub attn_o: BitLinearWeight,
    pub ffn: BitNetFfn,
}

/// One top-K prediction.
#[derive(Debug, Clone, PartialEq)]
pub struct TernaryPrediction {
    pub token: String,
    pub probability: f64,
}

/// Run a full BitNet forward pass and return top-K next-token
/// predictions for the position immediately after `token_ids`.
///
/// Single-shot prefill, no KV cache.  Memory profile at BitNet b1.58
/// 2B 4T: ~1.1 GB weights resident, ~10 MB per-call working set.
pub fn predict_bitnet(
    model: &BitnetModel,
    tokenizer: &infer_vindex::tokenizers::Tokenizer,
    token_ids: &[u32],
    top_k: usize,
) -> Vec<TernaryPrediction> {
    if token_ids.is_empty() {
        return Vec::new();
    }
    let logits = run_full_forward(model, token_ids, None, None);

    // Top-K softmax.  Stable softmax: subtract max before exp.
    let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut probs: Vec<(usize, f64)> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, ((v - max_logit) as f64).exp()))
        .collect();
    let sum: f64 = probs.iter().map(|(_, p)| p).sum();
    if sum > 0.0 {
        for (_, p) in probs.iter_mut() {
            *p /= sum;
        }
    }
    probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    probs
        .into_iter()
        .take(top_k)
        .filter_map(|(token_id, prob)| {
            tokenizer
                .id_to_token(token_id as u32)
                .map(|s| TernaryPrediction {
                    token: s,
                    probability: prob,
                })
        })
        .collect()
}

/// FFN forward pass that skips the input RMSnorm (caller already did
/// it).  Used to avoid double-norming when the input norm is computed
/// once per layer.
fn ffn_forward_after_input_norm(
    ffn: &BitNetFfn,
    x_norm: &[f32],
    eps: f32,
    gate: &mut [f32],
    up: &mut [f32],
    hid: &mut [f32],
    y: &mut [f32],
) {
    let inter = ffn.gate.rows;
    debug_assert_eq!(gate.len(), inter);
    debug_assert_eq!(up.len(), inter);
    debug_assert_eq!(hid.len(), inter);

    matvec_i2s_f32_into(&ffn.gate, x_norm, gate).expect("gate shape");
    matvec_i2s_f32_into(&ffn.up, x_norm, up).expect("up shape");

    for ((g, u), h) in gate.iter().zip(up.iter()).zip(hid.iter_mut()) {
        let relu = g.max(0.0);
        *h = relu * relu * u;
    }

    let mut hid_norm = vec![0.0f32; inter];
    rmsnorm_into(hid, &ffn.ffn_sub_norm, eps, &mut hid_norm);

    matvec_i2s_f32_into(&ffn.down, &hid_norm, y).expect("down shape");
}

/// Causal-masked scaled-dot-product attention with GQA support.
///
/// `q` is `[seq_len, n_q_heads * head_dim]`, `k` and `v` are
/// `[seq_len, n_kv_heads * head_dim]`.  Each q-head maps to k/v head
/// `head_idx / groups` (standard GQA); when `n_kv_heads == n_q_heads`
/// this is plain MHA.  Mask is causal: position `i` only attends to
/// `0..=i`.
fn scaled_dot_product_attention_gqa(
    q: ArrayView2<f32>,
    k: ArrayView2<f32>,
    v: ArrayView2<f32>,
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    mut out: ndarray::ArrayViewMut2<f32>,
) {
    let seq_len = q.shape()[0];
    let scale = 1.0 / (head_dim as f32).sqrt();
    let groups = n_q_heads / n_kv_heads.max(1);

    out.fill(0.0);

    for h_q in 0..n_q_heads {
        let h_kv = h_q / groups.max(1);
        let q_off = h_q * head_dim;
        let kv_off = h_kv * head_dim;

        for i in 0..seq_len {
            // 1. scores[j] = (q[i] · k[j]) * scale for j <= i.
            let mut scores = vec![f32::NEG_INFINITY; seq_len];
            for (j, score) in scores.iter_mut().enumerate().take(i + 1) {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[(i, q_off + d)] * k[(j, kv_off + d)];
                }
                *score = dot * scale;
            }

            // 2. Stable softmax over the unmasked (j <= i) prefix.
            let max_score = scores[..=i]
                .iter()
                .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut sum = 0.0f32;
            for s in scores[..=i].iter_mut() {
                *s = (*s - max_score).exp();
                sum += *s;
            }
            if sum > 0.0 {
                for s in scores[..=i].iter_mut() {
                    *s /= sum;
                }
            }

            // 3. out[i] += Σ_j weights[j] * v[j]
            for (j, &w) in scores.iter().enumerate().take(i + 1) {
                if w == 0.0 {
                    continue;
                }
                for d in 0..head_dim {
                    out[(i, q_off + d)] += w * v[(j, kv_off + d)];
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  KV-cached decode
// ─────────────────────────────────────────────────────────────────────────────

/// Per-layer K/V projections accumulated across all positions seen so
/// far.  Held by the caller across decode steps.
#[derive(Clone)]
pub struct BitnetKvCache {
    /// `k[layer]` is `[past_len, n_kv_heads * head_dim]` f32,
    /// RoPE-applied at the position each row represents.
    pub k: Vec<Array2<f32>>,
    /// `v[layer]` is `[past_len, n_kv_heads * head_dim]` f32.
    pub v: Vec<Array2<f32>>,
    /// Number of positions accumulated so far.
    pub seq_len: usize,
}

impl BitnetKvCache {
    /// Empty cache sized for `n_layers`.  Rows are appended one at a
    /// time as `decode_step` or `prefill` runs.
    pub fn new(n_layers: usize, n_kv_heads: usize, head_dim: usize) -> Self {
        let kv_width = n_kv_heads * head_dim;
        Self {
            k: (0..n_layers).map(|_| Array2::zeros((0, kv_width))).collect(),
            v: (0..n_layers).map(|_| Array2::zeros((0, kv_width))).collect(),
            seq_len: 0,
        }
    }
}

/// Run the full prompt through every layer, accumulating K/V into a
/// fresh cache.  Returns the cache + raw logits at the last position.
///
/// Equivalent to `predict_bitnet` minus the top-K extraction.
pub fn prefill(model: &BitnetModel, token_ids: &[u32]) -> (BitnetKvCache, Vec<f32>) {
    let n_layers = model.layers.len();
    let mut cache = BitnetKvCache::new(n_layers, model.n_kv_heads, model.head_dim);
    if token_ids.is_empty() {
        let vocab = model.lm_head.shape()[0];
        return (cache, vec![0.0; vocab]);
    }
    let logits = run_full_forward(model, token_ids, Some(&mut cache), None);
    (cache, logits)
}

/// Append one new token to an existing cache and return the logits for
/// that position.  Caller picks the sampling strategy.
pub fn decode_step(model: &BitnetModel, cache: &mut BitnetKvCache, new_token: u32) -> Vec<f32> {
    let position = cache.seq_len;
    let hidden = model.embed.shape()[1];
    let head_dim = model.head_dim;
    let n_q_heads = model.n_q_heads;
    let n_kv_heads = model.n_kv_heads;
    let kv_width = n_kv_heads * head_dim;
    let q_width = n_q_heads * head_dim;
    debug_assert_eq!(q_width, hidden, "hidden = n_q_heads * head_dim");

    // 1. Embed the new token.
    let mut h = Array1::<f32>::zeros(hidden);
    let row = model.embed.row(new_token as usize % model.embed.shape()[0]);
    for (dst, &src) in h.iter_mut().zip(row.iter()) {
        *dst = src * model.embed_scale;
    }

    let mut x_norm = vec![0.0f32; hidden];
    let mut q = vec![0.0f32; q_width];
    let mut k = vec![0.0f32; kv_width];
    let mut v = vec![0.0f32; kv_width];
    let mut attn_pool = vec![0.0f32; hidden];
    let mut attn_pool_norm = vec![0.0f32; hidden];
    let mut attn_out = vec![0.0f32; hidden];
    let mut ffn_x_norm = vec![0.0f32; hidden];
    let mut ffn_gate = vec![0.0f32; model.layers[0].ffn.gate.rows];
    let mut ffn_up = vec![0.0f32; model.layers[0].ffn.up.rows];
    let mut ffn_hid = vec![0.0f32; model.layers[0].ffn.gate.rows];
    let mut ffn_out_row = vec![0.0f32; hidden];

    for (layer_idx, layer) in model.layers.iter().enumerate() {
        // a. attn_norm.
        rmsnorm_into(h.as_slice().unwrap(), &layer.attn_norm, model.eps, &mut x_norm);

        // b. Q/K/V projections.
        matvec_i2s_f32_into(&layer.attn_q, &x_norm, &mut q).expect("attn_q shape");
        matvec_i2s_f32_into(&layer.attn_k, &x_norm, &mut k).expect("attn_k shape");
        matvec_i2s_f32_into(&layer.attn_v, &x_norm, &mut v).expect("attn_v shape");

        // c. RoPE on the new token's Q + K only.  The cached K already
        //    carries RoPE for positions 0..position-1.
        let q_arr = Array2::from_shape_vec((1, q_width), q.clone()).expect("q shape");
        let k_arr = Array2::from_shape_vec((1, kv_width), k.clone()).expect("k shape");
        let q_rotated = crate::attention::rope::apply_rope_partial_at(
            &q_arr,
            n_q_heads,
            head_dim,
            model.rope_base,
            1.0,
            position,
        );
        let k_rotated = crate::attention::rope::apply_rope_partial_at(
            &k_arr,
            n_kv_heads,
            head_dim,
            model.rope_base,
            1.0,
            position,
        );

        // d. Append K/V rows to the per-layer cache.
        let new_k_row = k_rotated.row(0).to_owned();
        let new_v_row = Array1::from(v.clone());
        cache.k[layer_idx] = stack_one_row(&cache.k[layer_idx], &new_k_row);
        cache.v[layer_idx] = stack_one_row(&cache.v[layer_idx], &new_v_row);

        // e. Causal-masked GQA attention: new Q vs full cached K/V.
        let q_view = q_rotated.row(0);
        attention_decode_into(
            q_view.as_slice().unwrap(),
            cache.k[layer_idx].view(),
            cache.v[layer_idx].view(),
            n_q_heads,
            n_kv_heads,
            head_dim,
            &mut attn_pool,
        );

        // f. Sub-norm + O projection.
        rmsnorm_into(&attn_pool, &layer.attn_sub_norm, model.eps, &mut attn_pool_norm);
        matvec_i2s_f32_into(&layer.attn_o, &attn_pool_norm, &mut attn_out).expect("attn_o shape");

        // g. Residual + FFN + residual.
        for (dst, &src) in h.iter_mut().zip(attn_out.iter()) {
            *dst += src;
        }
        rmsnorm_into(h.as_slice().unwrap(), &layer.ffn.ffn_norm, model.eps, &mut ffn_x_norm);
        ffn_forward_after_input_norm(
            &layer.ffn,
            &ffn_x_norm,
            model.eps,
            &mut ffn_gate,
            &mut ffn_up,
            &mut ffn_hid,
            &mut ffn_out_row,
        );
        for (dst, &src) in h.iter_mut().zip(ffn_out_row.iter()) {
            *dst += src;
        }
    }

    cache.seq_len += 1;

    // h_final = output_norm(h)
    let mut h_final = vec![0.0f32; hidden];
    rmsnorm_into(h.as_slice().unwrap(), &model.output_norm, model.eps, &mut h_final);
    let h_arr = Array1::from(h_final);
    model.lm_head.dot(&h_arr).to_vec()
}

/// Greedily generate up to `max_new_tokens` from `prompt`.  Stops
/// early if `stop_token` is produced.  Returns the raw token-id stream.
///
/// Self-contained (argmax) greedy decode — the upstream sampler stack
/// (temperature / top-k / top-p) is not yet present in pg_infer.
pub fn generate_greedy(
    model: &BitnetModel,
    prompt_token_ids: &[u32],
    max_new_tokens: usize,
    stop_token: Option<u32>,
) -> Vec<u32> {
    if prompt_token_ids.is_empty() || max_new_tokens == 0 {
        return Vec::new();
    }
    let (mut cache, last_logits) = prefill(model, prompt_token_ids);
    let mut generated = Vec::with_capacity(max_new_tokens);

    let mut next = argmax(&last_logits);
    for _ in 0..max_new_tokens {
        if let Some(stop) = stop_token {
            if next == stop {
                break;
            }
        }
        generated.push(next);
        let logits = decode_step(model, &mut cache, next);
        next = argmax(&logits);
    }
    generated
}

/// Decode-time attention: one Q-row vs the full cached K/V history.
///
/// `q` is `[n_q_heads * head_dim]`, `k` and `v` are `[seq_len,
/// n_kv_heads * head_dim]`.  Result is written to `out` (length
/// `n_q_heads * head_dim`).
fn attention_decode_into(
    q: &[f32],
    k: ArrayView2<f32>,
    v: ArrayView2<f32>,
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    out: &mut [f32],
) {
    let seq_len = k.shape()[0];
    debug_assert_eq!(v.shape()[0], seq_len);
    let scale = 1.0 / (head_dim as f32).sqrt();
    let groups = n_q_heads / n_kv_heads.max(1);

    for o in out.iter_mut() {
        *o = 0.0;
    }

    for h_q in 0..n_q_heads {
        let h_kv = h_q / groups.max(1);
        let q_off = h_q * head_dim;
        let kv_off = h_kv * head_dim;

        // Scores over the full cached K (position is at the end, so it
        // attends to the whole 0..seq_len prefix — no extra mask).
        let mut scores = vec![0.0f32; seq_len];
        for (j, score) in scores.iter_mut().enumerate() {
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q[q_off + d] * k[(j, kv_off + d)];
            }
            *score = dot * scale;
        }

        // Stable softmax.
        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for s in scores.iter_mut() {
            *s = (*s - max_score).exp();
            sum += *s;
        }
        if sum > 0.0 {
            for s in scores.iter_mut() {
                *s /= sum;
            }
        }

        // out[q_head] = Σ_j w[j] * v[j, kv_head]
        for (j, &w) in scores.iter().enumerate() {
            if w == 0.0 {
                continue;
            }
            for d in 0..head_dim {
                out[q_off + d] += w * v[(j, kv_off + d)];
            }
        }
    }
}

/// Append one row to a 2D ndarray.  ndarray has no built-in append; we
/// rebuild and copy.
fn stack_one_row(prev: &Array2<f32>, new_row: &Array1<f32>) -> Array2<f32> {
    let cols = prev.shape()[1];
    debug_assert_eq!(new_row.len(), cols);
    let new_rows = prev.shape()[0] + 1;
    let mut out = Array2::<f32>::zeros((new_rows, cols));
    if !prev.is_empty() {
        out.slice_mut(ndarray::s![..new_rows - 1, ..]).assign(prev);
    }
    out.row_mut(new_rows - 1).assign(new_row);
    out
}

/// Index of the maximum logit (first occurrence on ties).
fn argmax(logits: &[f32]) -> u32 {
    let mut best_idx = 0u32;
    let mut best = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best {
            best = v;
            best_idx = i as u32;
        }
    }
    best_idx
}

/// Shared-spine forward used by both `prefill` (when `cache=Some`) and
/// the single-shot `predict_bitnet`.  When `cache` is `Some`, per-layer
/// K/V are pushed in (after RoPE) so a subsequent `decode_step` can
/// extend the sequence.  When `residuals` is `Some`, the last-token
/// post-FFN residual is captured per layer (for walk-inference KNN
/// override).  Returns logits at the last position only.
fn run_full_forward(
    model: &BitnetModel,
    token_ids: &[u32],
    mut cache: Option<&mut BitnetKvCache>,
    mut residuals: Option<&mut Vec<(usize, Vec<f32>)>>,
) -> Vec<f32> {
    let seq_len = token_ids.len();
    let hidden = model.embed.shape()[1];
    let head_dim = model.head_dim;
    let n_q_heads = model.n_q_heads;
    let n_kv_heads = model.n_kv_heads;
    debug_assert_eq!(n_q_heads * head_dim, hidden);

    let mut h = Array2::<f32>::zeros((seq_len, hidden));
    for (i, &tok) in token_ids.iter().enumerate() {
        let row = model.embed.row(tok as usize % model.embed.shape()[0]);
        let mut h_row = h.row_mut(i);
        for (dst, &src) in h_row.iter_mut().zip(row.iter()) {
            *dst = src * model.embed_scale;
        }
    }

    let mut x_norm = Array2::<f32>::zeros((seq_len, hidden));
    let mut q = Array2::<f32>::zeros((seq_len, n_q_heads * head_dim));
    let mut k = Array2::<f32>::zeros((seq_len, n_kv_heads * head_dim));
    let mut v = Array2::<f32>::zeros((seq_len, n_kv_heads * head_dim));
    let mut attn_pool = Array2::<f32>::zeros((seq_len, hidden));
    let mut attn_pool_norm = Array2::<f32>::zeros((seq_len, hidden));
    let mut attn_out = Array2::<f32>::zeros((seq_len, hidden));
    let mut ffn_x_norm = Array2::<f32>::zeros((seq_len, hidden));
    let mut ffn_gate = vec![0.0f32; model.layers[0].ffn.gate.rows];
    let mut ffn_up = vec![0.0f32; model.layers[0].ffn.up.rows];
    let mut ffn_hid = vec![0.0f32; model.layers[0].ffn.gate.rows];
    let mut ffn_out_row = vec![0.0f32; hidden];

    for (layer_idx, layer) in model.layers.iter().enumerate() {
        for i in 0..seq_len {
            rmsnorm_into(
                h.row(i).as_slice().unwrap(),
                &layer.attn_norm,
                model.eps,
                x_norm.row_mut(i).as_slice_mut().unwrap(),
            );
        }
        for i in 0..seq_len {
            matvec_i2s_f32_into(
                &layer.attn_q,
                x_norm.row(i).as_slice().unwrap(),
                q.row_mut(i).as_slice_mut().unwrap(),
            )
            .expect("attn_q shape");
            matvec_i2s_f32_into(
                &layer.attn_k,
                x_norm.row(i).as_slice().unwrap(),
                k.row_mut(i).as_slice_mut().unwrap(),
            )
            .expect("attn_k shape");
            matvec_i2s_f32_into(
                &layer.attn_v,
                x_norm.row(i).as_slice().unwrap(),
                v.row_mut(i).as_slice_mut().unwrap(),
            )
            .expect("attn_v shape");
        }

        let q_rot = crate::attention::rope::apply_rope(&q, n_q_heads, head_dim, model.rope_base);
        let k_rot = crate::attention::rope::apply_rope(&k, n_kv_heads, head_dim, model.rope_base);

        attn_pool.fill(0.0);
        scaled_dot_product_attention_gqa(
            q_rot.view(),
            k_rot.view(),
            v.view(),
            n_q_heads,
            n_kv_heads,
            head_dim,
            attn_pool.view_mut(),
        );

        // If a cache is being built, capture the prefill K/V for this
        // layer (post-RoPE for K, pre-anything for V).
        if let Some(c) = cache.as_deref_mut() {
            c.k[layer_idx] = k_rot.clone();
            c.v[layer_idx] = v.clone();
        }

        for i in 0..seq_len {
            rmsnorm_into(
                attn_pool.row(i).as_slice().unwrap(),
                &layer.attn_sub_norm,
                model.eps,
                attn_pool_norm.row_mut(i).as_slice_mut().unwrap(),
            );
            matvec_i2s_f32_into(
                &layer.attn_o,
                attn_pool_norm.row(i).as_slice().unwrap(),
                attn_out.row_mut(i).as_slice_mut().unwrap(),
            )
            .expect("attn_o shape");
        }
        h += &attn_out;

        for i in 0..seq_len {
            rmsnorm_into(
                h.row(i).as_slice().unwrap(),
                &layer.ffn.ffn_norm,
                model.eps,
                ffn_x_norm.row_mut(i).as_slice_mut().unwrap(),
            );
            ffn_forward_after_input_norm(
                &layer.ffn,
                ffn_x_norm.row(i).as_slice().unwrap(),
                model.eps,
                &mut ffn_gate,
                &mut ffn_up,
                &mut ffn_hid,
                &mut ffn_out_row,
            );
            for (dst, &src) in h.row_mut(i).iter_mut().zip(ffn_out_row.iter()) {
                *dst += src;
            }
        }

        // Capture the last-token residual at this layer for walk
        // inference's KNN-store override.
        if let Some(r) = residuals.as_deref_mut() {
            r.push((layer_idx, h.row(seq_len - 1).to_vec()));
        }
    }

    if let Some(c) = cache {
        c.seq_len = seq_len;
    }

    let last_h = h.row(seq_len - 1).to_owned();
    let mut h_final = vec![0.0f32; hidden];
    rmsnorm_into(last_h.as_slice().unwrap(), &model.output_norm, model.eps, &mut h_final);
    let h_arr = Array1::from(h_final);
    model.lm_head.dot(&h_arr).to_vec()
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a tiny synthetic BitLinearWeight from a list of (row, col,
    /// trit) triples plus per-row scales.
    fn build_weight(
        rows: usize,
        cols: usize,
        trits: &[(usize, usize, i8)],
        scales: Vec<f32>,
    ) -> BitLinearWeight {
        assert!(cols.is_multiple_of(4));
        let mut bytes = vec![0u8; rows * cols / 4];
        for &(r, c, t) in trits {
            let bits: u8 = match t {
                1 => 0b01,
                -1 => 0b10,
                _ => 0b00,
            };
            let byte_idx = r * (cols / 4) + c / 4;
            let slot = (c % 4) as u8;
            bytes[byte_idx] |= bits << (2 * slot);
        }
        BitLinearWeight::new(rows, cols, bytes, scales).unwrap()
    }

    #[test]
    fn rmsnorm_zero_input_zero_output() {
        let x = vec![0.0f32; 8];
        let w = vec![1.0f32; 8];
        let mut out = vec![0.0f32; 8];
        rmsnorm_into(&x, &w, 1e-6, &mut out);
        assert!(out.iter().all(|&v| v.abs() < 1e-3));
    }

    #[test]
    fn rmsnorm_with_unit_weight_normalises() {
        let x = vec![2.0f32, 2.0, 2.0, 2.0]; // rms = 2.0
        let w = vec![1.0f32; 4];
        let mut out = vec![0.0f32; 4];
        rmsnorm_into(&x, &w, 0.0, &mut out);
        let post_rms = (out.iter().map(|v| v * v).sum::<f32>() / (out.len() as f32)).sqrt();
        assert!(
            (post_rms - 1.0).abs() < 1e-5,
            "post-norm rms should be ~1, got {post_rms}"
        );
    }

    #[test]
    fn rmsnorm_weight_scales_per_channel() {
        let x = vec![1.0f32; 4];
        let w = vec![2.0f32, 0.5, 1.0, -1.0];
        let mut out = vec![0.0f32; 4];
        rmsnorm_into(&x, &w, 0.0, &mut out);
        assert!((out[0] - 2.0).abs() < 1e-5);
        assert!((out[1] - 0.5).abs() < 1e-5);
        assert!((out[2] - 1.0).abs() < 1e-5);
        assert!((out[3] - (-1.0)).abs() < 1e-5);
    }

    /// Squared-ReLU: a positive gate squares, a negative gate zeros.
    #[test]
    fn bitnet_ffn_squared_relu_zeros_negative_gates() {
        let hidden = 4;
        let inter = 4;
        let gate = build_weight(inter, hidden, &[(0, 0, 1)], vec![1.0; inter]);
        let up = build_weight(inter, hidden, &[(0, 0, 1)], vec![1.0; inter]);
        let down = build_weight(hidden, inter, &[(0, 0, 1)], vec![1.0; hidden]);

        let ffn = BitNetFfn {
            gate,
            up,
            down,
            ffn_norm: vec![1.0; hidden],
            ffn_sub_norm: vec![1.0; inter],
            eps: 1e-5,
        };

        // x=[4,0,0,0] (rms 2) → x_norm[0]=2 → gate/up[0]=2 →
        // hid[0]=relu(2)^2*2=8 → sub_norm rms=4 → hid_norm[0]=2 →
        // y[0]=2 → x_out[0]=4+2=6.
        let x_pos = vec![4.0f32, 0.0, 0.0, 0.0];
        let out_pos = ffn.forward(&x_pos);
        assert!(
            (out_pos[0] - 6.0).abs() < 1e-3,
            "positive input: expected x_out[0]=6, got {}",
            out_pos[0]
        );

        // Negative input: ReLU zeros gate, residual passes through.
        let x_neg = vec![-4.0f32, 0.0, 0.0, 0.0];
        let out_neg = ffn.forward(&x_neg);
        assert!(
            (out_neg[0] - (-4.0)).abs() < 1e-3,
            "negative input: residual passthrough; got {}",
            out_neg[0]
        );
    }

    #[test]
    fn forward_and_forward_into_agree() {
        let hidden = 4;
        let inter = 8;
        let gate = build_weight(
            inter,
            hidden,
            &[(0, 0, 1), (1, 1, -1), (2, 2, 1), (3, 3, 1), (4, 0, 1)],
            vec![0.5; inter],
        );
        let up = build_weight(
            inter,
            hidden,
            &[(0, 0, 1), (1, 0, 1), (2, 1, 1), (3, 2, -1), (4, 3, 1)],
            vec![0.5; inter],
        );
        let down = build_weight(
            hidden,
            inter,
            &[(0, 0, 1), (1, 1, 1), (2, 2, 1), (3, 4, -1)],
            vec![0.7; hidden],
        );

        let ffn = BitNetFfn {
            gate,
            up,
            down,
            ffn_norm: vec![1.0, 1.5, 0.8, 1.2],
            ffn_sub_norm: vec![1.0; inter],
            eps: 1e-6,
        };
        let x = vec![0.7f32, -0.3, 0.5, -0.1];

        let out_a = ffn.forward(&x);

        let mut gate_buf = vec![0.0; inter];
        let mut up_buf = vec![0.0; inter];
        let mut hid_buf = vec![0.0; inter];
        let mut y_buf = vec![0.0; hidden];
        ffn.forward_into(&x, &mut gate_buf, &mut up_buf, &mut hid_buf, &mut y_buf);
        for (b, xi) in y_buf.iter_mut().zip(x.iter()) {
            *b += xi;
        }

        for (a, b) in out_a.iter().zip(y_buf.iter()) {
            assert!((a - b).abs() < 1e-5, "forward {a} vs into+resid {b}");
        }
    }

    /// Reusable tiny model factory: hidden=4, vocab=8, 1 head, 1 layer.
    fn tiny_model() -> BitnetModel {
        let hidden = 4;
        let inter = 4;
        let vocab = 8;
        let n_heads = 1;
        let head_dim = hidden / n_heads;
        let mk_w = |rows: usize, cols: usize, scale: f32| {
            let mut bytes = vec![0u8; rows * cols / 4];
            for (i, b) in bytes.iter_mut().enumerate() {
                *b = match i % 4 {
                    0 => 0b01_10_00_01,
                    1 => 0b10_01_01_00,
                    2 => 0b00_01_10_01,
                    _ => 0b01_00_01_10,
                };
            }
            BitLinearWeight::new(rows, cols, bytes, vec![scale; rows]).unwrap()
        };
        let layer = BitnetLayer {
            attn_norm: vec![1.0; hidden],
            attn_q: mk_w(hidden, hidden, 0.3),
            attn_k: mk_w(hidden, hidden, 0.4),
            attn_v: mk_w(hidden, hidden, 0.5),
            attn_sub_norm: vec![1.0; hidden],
            attn_o: mk_w(hidden, hidden, 0.6),
            ffn: BitNetFfn {
                gate: mk_w(inter, hidden, 0.2),
                up: mk_w(inter, hidden, 0.3),
                down: mk_w(hidden, inter, 0.7),
                ffn_norm: vec![1.0; hidden],
                ffn_sub_norm: vec![1.0; inter],
                eps: 1e-5,
            },
        };
        BitnetModel {
            layers: vec![layer],
            embed: Array2::from_shape_fn((vocab, hidden), |(i, j)| {
                ((i * 7 + j * 3) as f32 % 5.0) - 2.0
            }),
            embed_scale: 1.0,
            output_norm: vec![1.0; hidden],
            lm_head: Array2::from_shape_fn((vocab, hidden), |(i, j)| {
                ((i * 11 + j * 5) as f32 % 4.0) - 1.5
            }),
            eps: 1e-5,
            head_dim,
            n_q_heads: n_heads,
            n_kv_heads: n_heads,
            rope_base: 10000.0,
        }
    }

    /// Causal mask self-test: position 0 attends only to itself, so its
    /// output equals v[0].
    #[test]
    fn scaled_dot_product_attention_position_zero_is_self_attended() {
        let n_heads = 1;
        let head_dim = 4;
        let q = Array2::from_shape_vec((1, head_dim), vec![1.0, 0.5, -0.5, 0.25]).unwrap();
        let k = q.clone();
        let v = Array2::from_shape_vec((1, head_dim), vec![3.0, -1.0, 2.5, 0.0]).unwrap();
        let mut out = Array2::<f32>::zeros((1, head_dim));
        scaled_dot_product_attention_gqa(
            q.view(),
            k.view(),
            v.view(),
            n_heads,
            n_heads,
            head_dim,
            out.view_mut(),
        );
        for (a, b) in out.row(0).iter().zip(v.row(0).iter()) {
            assert!((a - b).abs() < 1e-5, "expected v, got {a} vs {b}");
        }
    }

    /// Prefill cache should hold one row per token in K and V.
    #[test]
    fn prefill_populates_cache_rows() {
        let model = tiny_model();
        let tokens = vec![0u32, 1, 2, 3, 4];
        let (cache, _logits) = prefill(&model, &tokens);
        assert_eq!(cache.seq_len, tokens.len());
        for (k_layer, v_layer) in cache.k.iter().zip(cache.v.iter()) {
            assert_eq!(k_layer.shape()[0], tokens.len());
            assert_eq!(v_layer.shape()[0], tokens.len());
        }
    }

    /// A decode_step appends one row to each layer's K and V cache.
    #[test]
    fn decode_step_grows_cache_by_one() {
        let model = tiny_model();
        let tokens = vec![0u32, 1, 2];
        let (mut cache, _) = prefill(&model, &tokens);
        let before = cache.seq_len;
        let logits = decode_step(&model, &mut cache, 5);
        assert_eq!(cache.seq_len, before + 1);
        assert_eq!(cache.k[0].shape()[0], before + 1);
        assert_eq!(cache.v[0].shape()[0], before + 1);
        assert_eq!(logits.len(), model.lm_head.shape()[0]);
    }

    /// Decode equivalence: prefilling N tokens then decoding one must
    /// produce the same logits as prefilling all N+1 tokens for the
    /// last position.  Load-bearing correctness test for the cache.
    #[test]
    fn decode_step_matches_full_prefill_at_last_position() {
        let model = tiny_model();
        let tokens = vec![0u32, 1, 2, 3];

        let (_, logits_full) = prefill(&model, &tokens);

        let (mut cache, _) = prefill(&model, &tokens[..tokens.len() - 1]);
        let logits_decoded = decode_step(&model, &mut cache, *tokens.last().unwrap());

        assert_eq!(logits_full.len(), logits_decoded.len());
        for (i, (a, b)) in logits_full.iter().zip(logits_decoded.iter()).enumerate() {
            let diff = (a - b).abs();
            assert!(diff < 1e-3, "logit {i}: prefill={a} decoded={b} diff={diff}");
        }
    }

    #[test]
    fn argmax_picks_max() {
        assert_eq!(argmax(&[1.0, 3.0, 2.0]), 1);
        assert_eq!(argmax(&[5.0, 0.0, -1.0]), 0);
        assert_eq!(argmax(&[2.0, 2.0, 2.0]), 0);
    }

    #[test]
    fn generate_greedy_produces_max_new_tokens() {
        let model = tiny_model();
        let prompt = vec![0u32, 1];
        let out = generate_greedy(&model, &prompt, 4, None);
        assert_eq!(out.len(), 4);
        for &id in &out {
            assert!(id < 8, "vocab=8");
        }
    }

    #[test]
    fn generate_greedy_empty_prompt_returns_empty() {
        let model = tiny_model();
        let out = generate_greedy(&model, &[], 5, None);
        assert!(out.is_empty());
    }

    #[test]
    fn generate_greedy_stops_on_stop_token() {
        let model = tiny_model();
        let prompt = vec![0u32, 1];
        let (_, logits) = prefill(&model, &prompt);
        let first_pred = argmax(&logits);
        let out = generate_greedy(&model, &prompt, 10, Some(first_pred));
        assert!(
            !out.contains(&first_pred),
            "stop_token leaked into output: {out:?}"
        );
    }
}
