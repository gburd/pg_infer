//! Full-pipeline integration tests for the inference engine.
//!
//! Exercises the complete forward pass: mock model creation → load → tokenize →
//! forward_raw_logits / predict → verify shapes, value ranges, and generation.

use std::collections::HashMap;
use std::path::Path;

/// Create a synthetic model directory with safetensors weights, config.json,
/// and tokenizer.json. Small enough for fast CI (2 layers, hidden=16, vocab=32).
fn create_test_model(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();

    let hidden = 16usize;
    let intermediate = 32usize;
    let vocab = 32usize;
    let head_dim = 8usize;
    let num_q_heads = 2usize;
    let num_kv_heads = 1usize;
    let num_layers = 2usize;

    let mut tensors: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();

    // Embedding: (vocab, hidden)
    tensors.insert(
        "model.embed_tokens.weight".into(),
        (random_f32(vocab * hidden, 1), vec![vocab, hidden]),
    );

    // Final norm
    tensors.insert(
        "model.norm.weight".into(),
        (vec![1.0f32; hidden], vec![hidden]),
    );

    for layer in 0..num_layers {
        let p = format!("model.layers.{layer}");

        // Layer norms
        tensors.insert(
            format!("{p}.input_layernorm.weight"),
            (vec![1.0f32; hidden], vec![hidden]),
        );
        tensors.insert(
            format!("{p}.post_attention_layernorm.weight"),
            (vec![1.0f32; hidden], vec![hidden]),
        );

        // Attention projections
        let q_dim = num_q_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        tensors.insert(
            format!("{p}.self_attn.q_proj.weight"),
            (random_f32(q_dim * hidden, layer * 100 + 10), vec![q_dim, hidden]),
        );
        tensors.insert(
            format!("{p}.self_attn.k_proj.weight"),
            (random_f32(kv_dim * hidden, layer * 100 + 20), vec![kv_dim, hidden]),
        );
        tensors.insert(
            format!("{p}.self_attn.v_proj.weight"),
            (random_f32(kv_dim * hidden, layer * 100 + 30), vec![kv_dim, hidden]),
        );
        tensors.insert(
            format!("{p}.self_attn.o_proj.weight"),
            (random_f32(hidden * q_dim, layer * 100 + 40), vec![hidden, q_dim]),
        );

        // FFN (gate/up/down)
        tensors.insert(
            format!("{p}.mlp.gate_proj.weight"),
            (random_f32(intermediate * hidden, layer * 100 + 50), vec![intermediate, hidden]),
        );
        tensors.insert(
            format!("{p}.mlp.up_proj.weight"),
            (random_f32(intermediate * hidden, layer * 100 + 60), vec![intermediate, hidden]),
        );
        tensors.insert(
            format!("{p}.mlp.down_proj.weight"),
            (random_f32(hidden * intermediate, layer * 100 + 70), vec![hidden, intermediate]),
        );
    }

    // Write safetensors
    write_safetensors(dir, &tensors);

    // Write config.json (llama-style)
    let config = serde_json::json!({
        "model_type": "llama",
        "num_hidden_layers": num_layers,
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "head_dim": head_dim,
        "num_attention_heads": num_q_heads,
        "num_key_value_heads": num_kv_heads,
        "vocab_size": vocab,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-5
    });
    std::fs::write(dir.join("config.json"), serde_json::to_string_pretty(&config).unwrap())
        .unwrap();

    // Write tokenizer
    write_mock_tokenizer(dir, vocab);
}

fn random_f32(n: usize, seed: usize) -> Vec<f32> {
    let mut vals = Vec::with_capacity(n);
    let mut x = seed as u64 * 2654435761 + 1;
    for _ in 0..n {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let f = ((x >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0;
        vals.push(f * 0.1); // small values for stable forward pass
    }
    vals
}

fn write_safetensors(dir: &Path, tensors: &HashMap<String, (Vec<f32>, Vec<usize>)>) {
    let mut byte_bufs: HashMap<String, Vec<u8>> = HashMap::new();
    for (name, (values, _shape)) in tensors {
        let bytes: Vec<u8> = values.iter().flat_map(|f| f.to_le_bytes()).collect();
        byte_bufs.insert(name.clone(), bytes);
    }

    let mut data_map: HashMap<String, safetensors::tensor::TensorView<'_>> = HashMap::new();
    for (name, (_, shape)) in tensors {
        let bytes = &byte_bufs[name];
        data_map.insert(
            name.clone(),
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape.clone(), bytes)
                .unwrap(),
        );
    }

    let serialized = safetensors::tensor::serialize(&data_map, &None).unwrap();
    std::fs::write(dir.join("model.safetensors"), serialized).unwrap();
}

fn write_mock_tokenizer(dir: &Path, vocab_size: usize) {
    let tokens = [
        "the", "a", "is", "of", "France", "Paris", "Germany", "Berlin",
        "capital", "Europe", "language", "French", "city", "country", "and", "in",
        "cat", "dog", "runs", "fast", "big", "small", "red", "blue",
        "one", "two", "three", "four", "five", "six", "seven", "eight",
    ];

    let mut vocab = serde_json::Map::new();
    for (i, tok) in tokens.iter().enumerate().take(vocab_size) {
        vocab.insert(tok.to_string(), serde_json::json!(i));
    }

    let tokenizer_json = serde_json::json!({
        "version": "1.0",
        "model": {
            "type": "WordLevel",
            "vocab": vocab,
            "unk_token": "the"
        },
        "pre_tokenizer": {
            "type": "Whitespace"
        }
    });

    std::fs::write(
        dir.join("tokenizer.json"),
        serde_json::to_string_pretty(&tokenizer_json).unwrap(),
    )
    .unwrap();
}

// ── Tests ──

#[test]
fn forward_raw_logits_produces_correct_shapes() {
    let dir = std::env::temp_dir().join("infer_test_full_pipeline_shapes");
    let _ = std::fs::remove_dir_all(&dir);
    create_test_model(&dir);

    let weights = infer_inference::load_model_dir(dir.to_str().unwrap()).unwrap();

    // Verify model dimensions
    assert_eq!(weights.num_layers, 2);
    assert_eq!(weights.hidden_size, 16);
    assert_eq!(weights.vocab_size, 32);

    // Run forward pass with a short token sequence
    let token_ids: Vec<u32> = vec![0, 4, 5]; // "the France Paris"
    let result = infer_inference::forward_raw_logits(&weights, &token_ids, None);

    // h_pre_norm: (seq_len, hidden_size)
    assert_eq!(result.h_pre_norm.shape(), &[3, 16]);
    // h_final: (seq_len, hidden_size)
    assert_eq!(result.h_final.shape(), &[3, 16]);
    // logits: (vocab_size,)
    assert_eq!(result.logits.len(), 32);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn forward_raw_logits_values_are_finite() {
    let dir = std::env::temp_dir().join("infer_test_full_pipeline_finite");
    let _ = std::fs::remove_dir_all(&dir);
    create_test_model(&dir);

    let weights = infer_inference::load_model_dir(dir.to_str().unwrap()).unwrap();
    let token_ids: Vec<u32> = vec![1, 2, 3];
    let result = infer_inference::forward_raw_logits(&weights, &token_ids, None);

    // All logits must be finite (no NaN, no Inf)
    for &logit in result.logits.iter() {
        assert!(logit.is_finite(), "logit is not finite: {logit}");
    }

    // Hidden states must also be finite
    for &v in result.h_final.iter() {
        assert!(v.is_finite(), "hidden state is not finite: {v}");
    }

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn predict_returns_valid_top_k() {
    let dir = std::env::temp_dir().join("infer_test_full_pipeline_predict");
    let _ = std::fs::remove_dir_all(&dir);
    create_test_model(&dir);

    let weights = infer_inference::load_model_dir(dir.to_str().unwrap()).unwrap();
    let tokenizer = infer_inference::load_tokenizer(&dir).unwrap();

    let token_ids: Vec<u32> = vec![0, 4, 8]; // "the France capital"
    let result = infer_inference::predict(&weights, &tokenizer, &token_ids, 5);

    // Should return up to 5 predictions
    assert!(!result.predictions.is_empty());
    assert!(result.predictions.len() <= 5);

    // Probabilities should sum to <= 1.0 (they're top-k, not all)
    let prob_sum: f64 = result.predictions.iter().map(|(_, p)| p).sum();
    assert!(prob_sum <= 1.0 + 1e-6, "prob sum {prob_sum} > 1.0");

    // Each probability should be in [0, 1]
    for (token, prob) in &result.predictions {
        assert!(*prob >= 0.0, "negative probability for {token}: {prob}");
        assert!(*prob <= 1.0, "probability > 1 for {token}: {prob}");
    }

    // Token IDs should be valid vocabulary indices
    for &id in &result.token_ids {
        assert!((id as usize) < 32, "token id {id} out of vocab range");
    }

    // predictions and token_ids should be parallel
    assert_eq!(result.predictions.len(), result.token_ids.len());

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn predict_probabilities_sorted_descending() {
    let dir = std::env::temp_dir().join("infer_test_full_pipeline_sorted");
    let _ = std::fs::remove_dir_all(&dir);
    create_test_model(&dir);

    let weights = infer_inference::load_model_dir(dir.to_str().unwrap()).unwrap();
    let tokenizer = infer_inference::load_tokenizer(&dir).unwrap();

    let token_ids: Vec<u32> = vec![6, 7]; // "Germany Berlin"
    let result = infer_inference::predict(&weights, &tokenizer, &token_ids, 10);

    // Predictions should be sorted by probability descending
    for window in result.predictions.windows(2) {
        assert!(
            window[0].1 >= window[1].1,
            "predictions not sorted: {} ({}) before {} ({})",
            window[0].0, window[0].1, window[1].0, window[1].1,
        );
    }

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn single_token_forward_pass() {
    let dir = std::env::temp_dir().join("infer_test_full_pipeline_single");
    let _ = std::fs::remove_dir_all(&dir);
    create_test_model(&dir);

    let weights = infer_inference::load_model_dir(dir.to_str().unwrap()).unwrap();
    let token_ids: Vec<u32> = vec![0]; // single token
    let result = infer_inference::forward_raw_logits(&weights, &token_ids, None);

    assert_eq!(result.h_pre_norm.shape(), &[1, 16]);
    assert_eq!(result.logits.len(), 32);

    // Even with a single token, logits should be finite
    assert!(result.logits.iter().all(|l| l.is_finite()));

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn autoregressive_generation_loop() {
    let dir = std::env::temp_dir().join("infer_test_full_pipeline_generate");
    let _ = std::fs::remove_dir_all(&dir);
    create_test_model(&dir);

    let weights = infer_inference::load_model_dir(dir.to_str().unwrap()).unwrap();
    let tokenizer = infer_inference::load_tokenizer(&dir).unwrap();

    // Start with a prompt and generate 5 tokens autoregressively
    let mut token_ids: Vec<u32> = vec![0, 4]; // "the France"

    for _ in 0..5 {
        let result = infer_inference::predict(&weights, &tokenizer, &token_ids, 1);
        assert!(!result.token_ids.is_empty(), "predict returned no tokens");

        let next_id = result.token_ids[0];
        assert!((next_id as usize) < 32, "generated token {next_id} out of vocab");
        token_ids.push(next_id);
    }

    // Should have generated 5 new tokens
    assert_eq!(token_ids.len(), 7); // 2 prompt + 5 generated

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn deterministic_forward_pass() {
    let dir = std::env::temp_dir().join("infer_test_full_pipeline_deterministic");
    let _ = std::fs::remove_dir_all(&dir);
    create_test_model(&dir);

    let weights = infer_inference::load_model_dir(dir.to_str().unwrap()).unwrap();
    let token_ids: Vec<u32> = vec![2, 5, 8]; // "is Paris capital"

    let r1 = infer_inference::forward_raw_logits(&weights, &token_ids, None);
    let r2 = infer_inference::forward_raw_logits(&weights, &token_ids, None);

    // Same input → same output (no randomness in forward pass)
    for (a, b) in r1.logits.iter().zip(r2.logits.iter()) {
        assert_eq!(a, b, "forward pass is non-deterministic");
    }

    std::fs::remove_dir_all(&dir).ok();
}
