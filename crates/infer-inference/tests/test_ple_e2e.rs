//! End-to-end validation tests for Per-Layer Embeddings (PLE).
//!
//! Exercises the PLE precompute and forward-pass integration to verify that:
//! - `precompute_per_layer_inputs` returns correct shapes and non-zero values
//! - PLE contributes to the hidden state during a full forward pass
//! - With zero gate weights, PLE contribution is blocked (output ≈ input)
//! - Without PLE tensors, `precompute_per_layer_inputs` returns empty Vec

use std::collections::HashMap;
use infer_inference::forward::ple::precompute_per_layer_inputs;
use infer_inference::infer_models::{detect_from_json, ModelWeights, WeightArray};
use infer_inference::ndarray::Array2;

const VOCAB: usize = 32;
const HIDDEN: usize = 16;
const INTER: usize = 32;
const NUM_Q: usize = 2;
const NUM_KV: usize = 1;
const HEAD_DIM: usize = 8;
const NUM_LAYERS: usize = 2;
const PLE_DIM: usize = 8;

/// Simple LCG PRNG for reproducible random weight initialization.
fn rand_mat(rows: usize, cols: usize, seed: u64, scale: f32) -> WeightArray {
    let mut rng_state = seed;
    let data: Vec<f32> = (0..rows * cols)
        .map(|_| {
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (rng_state as u32) as f32 / u32::MAX as f32 * 2.0 * scale - scale
        })
        .collect();
    Array2::from_shape_vec((rows, cols), data)
        .unwrap()
        .into_shared()
}

/// Build a synthetic `ModelWeights` with PLE support.
///
/// Creates a tinymodel-style architecture with `hidden_size_per_layer_input`
/// set so `has_per_layer_embeddings()` returns true. Populates all PLE tensors.
fn make_ple_weights() -> ModelWeights {
    let arch_json = serde_json::json!({
        "model_type": "tinymodel",
        "hidden_size": HIDDEN,
        "num_hidden_layers": NUM_LAYERS,
        "intermediate_size": INTER,
        "head_dim": HEAD_DIM,
        "num_attention_heads": NUM_Q,
        "num_key_value_heads": NUM_KV,
        "vocab_size": VOCAB,
        "hidden_size_per_layer_input": PLE_DIM,
    });
    let arch = detect_from_json(&arch_json);

    let mut tensors: HashMap<String, WeightArray> = HashMap::new();
    let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();

    // Embed + lm_head
    let embed = rand_mat(VOCAB, HIDDEN, 0xdead0001, 0.1);
    let lm_head = rand_mat(VOCAB, HIDDEN, 0xdead0002, 0.1);
    tensors.insert(arch.embed_key().to_string(), embed.clone());

    // Final norm
    vectors.insert(arch.final_norm_key().to_string(), vec![1.0; HIDDEN]);

    let q_dim = NUM_Q * HEAD_DIM;
    let kv_dim = NUM_KV * HEAD_DIM;

    for layer in 0..NUM_LAYERS {
        // Attention projections
        tensors.insert(arch.attn_q_key(layer), rand_mat(q_dim, HIDDEN, 100 + layer as u64 * 10, 0.1));
        tensors.insert(arch.attn_k_key(layer), rand_mat(kv_dim, HIDDEN, 200 + layer as u64 * 10, 0.1));
        tensors.insert(arch.attn_v_key(layer), rand_mat(kv_dim, HIDDEN, 300 + layer as u64 * 10, 0.1));
        tensors.insert(arch.attn_o_key(layer), rand_mat(HIDDEN, q_dim, 400 + layer as u64 * 10, 0.1));
        // FFN
        tensors.insert(arch.ffn_gate_key(layer), rand_mat(INTER, HIDDEN, 500 + layer as u64 * 10, 0.1));
        tensors.insert(arch.ffn_up_key(layer), rand_mat(INTER, HIDDEN, 600 + layer as u64 * 10, 0.1));
        tensors.insert(arch.ffn_down_key(layer), rand_mat(HIDDEN, INTER, 700 + layer as u64 * 10, 0.1));
        // Layer norms
        vectors.insert(arch.input_layernorm_key(layer), vec![1.0; HIDDEN]);
        vectors.insert(arch.post_attention_layernorm_key(layer), vec![1.0; HIDDEN]);

        // PLE per-layer tensors:
        // input_gate: [ple_dim, hidden]
        tensors.insert(
            arch.per_layer_input_gate_key(layer).unwrap(),
            rand_mat(PLE_DIM, HIDDEN, 800 + layer as u64 * 10, 0.1),
        );
        // projection: [hidden, ple_dim]
        tensors.insert(
            arch.per_layer_projection_key(layer).unwrap(),
            rand_mat(HIDDEN, PLE_DIM, 900 + layer as u64 * 10, 0.1),
        );
        // post-PLE norm
        vectors.insert(
            arch.post_per_layer_input_norm_key(layer).unwrap(),
            vec![1.0; HIDDEN],
        );
    }

    // PLE shared tensors:
    // per_layer_model_projection: [num_layers * ple_dim, hidden]
    tensors.insert(
        "per_layer_model_projection.weight".to_string(),
        rand_mat(NUM_LAYERS * PLE_DIM, HIDDEN, 0xdead0003, 0.1),
    );
    // embed_tokens_per_layer: [vocab, num_layers * ple_dim]
    tensors.insert(
        "embed_tokens_per_layer.weight".to_string(),
        rand_mat(VOCAB, NUM_LAYERS * PLE_DIM, 0xdead0004, 0.1),
    );
    // per_layer_projection_norm
    vectors.insert(
        "per_layer_projection_norm.weight".to_string(),
        vec![1.0; PLE_DIM],
    );

    ModelWeights {
        tensors,
        vectors,
        raw_bytes: HashMap::new(),
        packed_mmaps: HashMap::new(),
        packed_byte_ranges: HashMap::new(),
        embed,
        lm_head,
        arch,
        num_layers: NUM_LAYERS,
        hidden_size: HIDDEN,
        intermediate_size: INTER,
        vocab_size: VOCAB,
        head_dim: HEAD_DIM,
        num_q_heads: NUM_Q,
        num_kv_heads: NUM_KV,
        rope_base: 10_000.0,
    }
}

/// Build a synthetic `ModelWeights` WITHOUT PLE support (no per_layer_embed_dim).
fn make_non_ple_weights() -> ModelWeights {
    let arch_json = serde_json::json!({
        "model_type": "tinymodel",
        "hidden_size": HIDDEN,
        "num_hidden_layers": NUM_LAYERS,
        "intermediate_size": INTER,
        "head_dim": HEAD_DIM,
        "num_attention_heads": NUM_Q,
        "num_key_value_heads": NUM_KV,
        "vocab_size": VOCAB,
    });
    let arch = detect_from_json(&arch_json);

    let mut tensors: HashMap<String, WeightArray> = HashMap::new();
    let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();

    let embed = rand_mat(VOCAB, HIDDEN, 0xbeef0001, 0.1);
    let lm_head = rand_mat(VOCAB, HIDDEN, 0xbeef0002, 0.1);
    tensors.insert(arch.embed_key().to_string(), embed.clone());
    vectors.insert(arch.final_norm_key().to_string(), vec![1.0; HIDDEN]);

    let q_dim = NUM_Q * HEAD_DIM;
    let kv_dim = NUM_KV * HEAD_DIM;

    for layer in 0..NUM_LAYERS {
        tensors.insert(arch.attn_q_key(layer), rand_mat(q_dim, HIDDEN, 100 + layer as u64, 0.1));
        tensors.insert(arch.attn_k_key(layer), rand_mat(kv_dim, HIDDEN, 200 + layer as u64, 0.1));
        tensors.insert(arch.attn_v_key(layer), rand_mat(kv_dim, HIDDEN, 300 + layer as u64, 0.1));
        tensors.insert(arch.attn_o_key(layer), rand_mat(HIDDEN, q_dim, 400 + layer as u64, 0.1));
        tensors.insert(arch.ffn_gate_key(layer), rand_mat(INTER, HIDDEN, 500 + layer as u64, 0.1));
        tensors.insert(arch.ffn_up_key(layer), rand_mat(INTER, HIDDEN, 600 + layer as u64, 0.1));
        tensors.insert(arch.ffn_down_key(layer), rand_mat(HIDDEN, INTER, 700 + layer as u64, 0.1));
        vectors.insert(arch.input_layernorm_key(layer), vec![1.0; HIDDEN]);
        vectors.insert(arch.post_attention_layernorm_key(layer), vec![1.0; HIDDEN]);
    }

    ModelWeights {
        tensors,
        vectors,
        raw_bytes: HashMap::new(),
        packed_mmaps: HashMap::new(),
        packed_byte_ranges: HashMap::new(),
        embed,
        lm_head,
        arch,
        num_layers: NUM_LAYERS,
        hidden_size: HIDDEN,
        intermediate_size: INTER,
        vocab_size: VOCAB,
        head_dim: HEAD_DIM,
        num_q_heads: NUM_Q,
        num_kv_heads: NUM_KV,
        rope_base: 10_000.0,
    }
}

// ── Tests ──

#[test]
fn precompute_returns_correct_count_and_shapes() {
    let weights = make_ple_weights();
    let token_ids: Vec<u32> = vec![0, 1, 2, 3];
    let seq_len = token_ids.len();

    // Embed tokens to get main_embeds (seq_len, hidden)
    let main_embeds = Array2::from_shape_fn((seq_len, HIDDEN), |(s, d)| {
        weights.embed[[token_ids[s] as usize, d]]
    });

    let ple_inputs = precompute_per_layer_inputs(&weights, &main_embeds, &token_ids);

    // Must return exactly num_layers entries
    assert_eq!(
        ple_inputs.len(),
        NUM_LAYERS,
        "expected {} per-layer inputs, got {}",
        NUM_LAYERS,
        ple_inputs.len()
    );

    // Each entry has shape [seq_len, ple_dim]
    for (i, input) in ple_inputs.iter().enumerate() {
        assert_eq!(
            input.shape(),
            &[seq_len, PLE_DIM],
            "layer {} PLE input has wrong shape: {:?}",
            i,
            input.shape()
        );
    }
}

#[test]
fn precompute_produces_nonzero_values() {
    let weights = make_ple_weights();
    let token_ids: Vec<u32> = vec![1, 5, 10];
    let seq_len = token_ids.len();

    let main_embeds = Array2::from_shape_fn((seq_len, HIDDEN), |(s, d)| {
        weights.embed[[token_ids[s] as usize, d]]
    });

    let ple_inputs = precompute_per_layer_inputs(&weights, &main_embeds, &token_ids);

    // At least some values should be non-zero (the computation produces output)
    for (i, input) in ple_inputs.iter().enumerate() {
        let max_abs = input.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        assert!(
            max_abs > 1e-8,
            "layer {} PLE input is all zeros (max_abs = {})",
            i,
            max_abs
        );
    }
}

#[test]
fn precompute_returns_empty_without_ple_config() {
    let weights = make_non_ple_weights();
    let token_ids: Vec<u32> = vec![0, 1, 2];
    let seq_len = token_ids.len();

    let main_embeds = Array2::from_shape_fn((seq_len, HIDDEN), |(s, d)| {
        weights.embed[[token_ids[s] as usize, d]]
    });

    let ple_inputs = precompute_per_layer_inputs(&weights, &main_embeds, &token_ids);

    assert!(
        ple_inputs.is_empty(),
        "expected empty PLE inputs for non-PLE model, got {} entries",
        ple_inputs.len()
    );
}

#[test]
fn ple_contributes_to_forward_pass() {
    // Run forward_raw_logits with PLE tensors and verify PLE changes the output
    let weights_with_ple = make_ple_weights();
    let token_ids: Vec<u32> = vec![0, 2, 4];

    let result_with_ple = infer_inference::forward_raw_logits(&weights_with_ple, &token_ids, None);

    // Now build identical weights but remove the PLE model projection
    // so precompute_per_layer_inputs returns empty (missing tensor early-exit).
    let mut weights_without_ple_tensors = make_ple_weights();
    weights_without_ple_tensors
        .tensors
        .remove("per_layer_model_projection.weight");

    let result_without_ple =
        infer_inference::forward_raw_logits(&weights_without_ple_tensors, &token_ids, None);

    // Logits must differ — PLE contributes to the hidden state
    let differs = result_with_ple
        .logits
        .iter()
        .zip(result_without_ple.logits.iter())
        .any(|(a, b)| (a - b).abs() > 1e-6);

    assert!(
        differs,
        "forward pass logits are identical with and without PLE — PLE did not contribute"
    );
}

#[test]
fn zero_gate_blocks_ple_contribution() {
    // With all-zero gate weights, GELU(0) = 0, so the gated product is zero
    // and PLE should not change the hidden state (output ≈ no-PLE output).
    let mut weights = make_ple_weights();
    let token_ids: Vec<u32> = vec![1, 3, 5];

    // Zero out all per-layer input gate weights
    for layer in 0..NUM_LAYERS {
        let gate_key = weights.arch.per_layer_input_gate_key(layer).unwrap();
        let zero_gate = Array2::<f32>::zeros((PLE_DIM, HIDDEN)).into_shared();
        weights.tensors.insert(gate_key, zero_gate);
    }

    let result_zero_gate = infer_inference::forward_raw_logits(&weights, &token_ids, None);

    // Compare against weights with PLE projection removed (no PLE at all)
    let mut weights_no_ple = make_ple_weights();
    weights_no_ple
        .tensors
        .remove("per_layer_model_projection.weight");
    // Also zero gates in the no-ple version for consistency (though it won't matter
    // since precompute returns empty)
    let result_no_ple = infer_inference::forward_raw_logits(&weights_no_ple, &token_ids, None);

    // Logits should be approximately equal (zero gate blocks all PLE contribution)
    let max_diff = result_zero_gate
        .logits
        .iter()
        .zip(result_no_ple.logits.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_diff < 1e-4,
        "zero-gate PLE output differs from no-PLE output by {max_diff} (expected ≈ 0)"
    );
}

#[test]
fn precompute_different_tokens_produce_different_outputs() {
    let weights = make_ple_weights();
    let seq_len = 3;

    let tokens_a: Vec<u32> = vec![0, 1, 2];
    let tokens_b: Vec<u32> = vec![10, 20, 30];

    let embeds_a = Array2::from_shape_fn((seq_len, HIDDEN), |(s, d)| {
        weights.embed[[tokens_a[s] as usize, d]]
    });
    let embeds_b = Array2::from_shape_fn((seq_len, HIDDEN), |(s, d)| {
        weights.embed[[tokens_b[s] as usize, d]]
    });

    let ple_a = precompute_per_layer_inputs(&weights, &embeds_a, &tokens_a);
    let ple_b = precompute_per_layer_inputs(&weights, &embeds_b, &tokens_b);

    assert_eq!(ple_a.len(), NUM_LAYERS);
    assert_eq!(ple_b.len(), NUM_LAYERS);

    // Different tokens should produce different PLE inputs
    let differs = ple_a
        .iter()
        .zip(ple_b.iter())
        .any(|(a, b)| a.iter().zip(b.iter()).any(|(x, y)| (x - y).abs() > 1e-8));

    assert!(
        differs,
        "different token sequences produced identical PLE inputs"
    );
}

#[test]
fn forward_pass_with_ple_produces_finite_values() {
    let weights = make_ple_weights();
    let token_ids: Vec<u32> = vec![0, 5, 10, 15, 20];

    let result = infer_inference::forward_raw_logits(&weights, &token_ids, None);

    // All logits must be finite
    for (i, &logit) in result.logits.iter().enumerate() {
        assert!(
            logit.is_finite(),
            "logit[{i}] is not finite: {logit} — PLE introduced NaN/Inf"
        );
    }

    // Hidden state must be finite
    for &v in result.h_final.iter() {
        assert!(v.is_finite(), "hidden state value is not finite: {v}");
    }
}
