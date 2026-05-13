//! Model loading — imports from infer-models.

pub use infer_models::ModelWeights;
pub use infer_models::QuantizedModelWeights;
pub use infer_models::{load_model_dir, load_model_dir_walk_only, resolve_model_path};
pub use infer_models::load_gguf_quantized;

/// Enum to hold either f32-expanded weights or native quantized weights.
///
/// This allows the inference engine to work with both loading paths:
/// - `F32`: standard load path where all tensors are dequantized to f32 arrays
/// - `Quantized`: skip-dequant path where 2D weights stay in native format (Q4_K, Q6_K, etc.)
///
/// The forward pass checks which variant is active and uses quantized matvec kernels
/// when weights are `Quantized`, avoiding the f32 intermediate entirely.
pub enum ModelWeightsVariant {
    /// Standard f32 weights (from load_gguf, load_model_dir, etc.)
    F32(ModelWeights),
    /// Native quantized weights (from load_gguf_quantized) — ~6x memory savings.
    Quantized(QuantizedModelWeights),
}

impl ModelWeightsVariant {
    /// Get the number of layers regardless of variant.
    pub fn num_layers(&self) -> usize {
        match self {
            Self::F32(w) => w.num_layers,
            Self::Quantized(w) => w.num_layers(),
        }
    }

    /// Get hidden size regardless of variant.
    pub fn hidden_size(&self) -> usize {
        match self {
            Self::F32(w) => w.hidden_size,
            Self::Quantized(w) => w.hidden_size(),
        }
    }

    /// Get a 1D vector (norm weights, biases) by key — available in both variants.
    pub fn get_vector(&self, key: &str) -> Option<&[f32]> {
        match self {
            Self::F32(w) => w.vectors.get(key).map(|v| v.as_slice()),
            Self::Quantized(w) => w.get_vector(key),
        }
    }

    /// Returns true if this is the quantized variant.
    pub fn is_quantized(&self) -> bool {
        matches!(self, Self::Quantized(_))
    }

    /// Get the f32 variant, or None if quantized.
    pub fn as_f32(&self) -> Option<&ModelWeights> {
        match self {
            Self::F32(w) => Some(w),
            Self::Quantized(_) => None,
        }
    }

    /// Get the quantized variant, or None if f32.
    pub fn as_quantized(&self) -> Option<&QuantizedModelWeights> {
        match self {
            Self::F32(_) => None,
            Self::Quantized(w) => Some(w),
        }
    }
}

/// Synthetic test fixtures for unit tests that need a functional `ModelWeights`
/// without loading from disk.
#[cfg(test)]
pub mod test_utils {
    use infer_models::{detect_from_json, ModelWeights, WeightArray};
    use ndarray::Array2;
    use std::collections::HashMap;

    /// Build a synthetic `ModelWeights` with all tensors populated.
    ///
    /// Dimensions: vocab=32, hidden=16, intermediate=32, 2 q-heads, 1 kv-head,
    /// head_dim=8, 2 layers. Forward pass ≈ 10 ms on CPU.
    pub fn make_test_weights() -> ModelWeights {
        const VOCAB: usize = 32;
        const HIDDEN: usize = 16;
        const INTER: usize = 32;
        const NUM_Q: usize = 2;
        const NUM_KV: usize = 1;
        const HEAD_DIM: usize = 8;
        const NUM_LAYERS: usize = 2;

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
        let mut rng_state = 0xdeadbeef_u64;

        // LCG giving values in [-scale, +scale]
        let mut rand_mat = |rows: usize, cols: usize, scale: f32| -> WeightArray {
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
        };

        // Embed + lm_head
        let embed = rand_mat(VOCAB, HIDDEN, 0.1);
        let lm_head = rand_mat(VOCAB, HIDDEN, 0.1);
        tensors.insert(arch.embed_key().to_string(), embed.clone());

        // Final norm
        vectors.insert(arch.final_norm_key().to_string(), vec![1.0; HIDDEN]);

        let q_dim = NUM_Q * HEAD_DIM;
        let kv_dim = NUM_KV * HEAD_DIM;

        for layer in 0..NUM_LAYERS {
            // Attention projections
            tensors.insert(arch.attn_q_key(layer), rand_mat(q_dim, HIDDEN, 0.1));
            tensors.insert(arch.attn_k_key(layer), rand_mat(kv_dim, HIDDEN, 0.1));
            tensors.insert(arch.attn_v_key(layer), rand_mat(kv_dim, HIDDEN, 0.1));
            tensors.insert(arch.attn_o_key(layer), rand_mat(HIDDEN, q_dim, 0.1));
            // FFN
            tensors.insert(arch.ffn_gate_key(layer), rand_mat(INTER, HIDDEN, 0.1));
            tensors.insert(arch.ffn_up_key(layer), rand_mat(INTER, HIDDEN, 0.1));
            tensors.insert(arch.ffn_down_key(layer), rand_mat(HIDDEN, INTER, 0.1));
            // Layer norms
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
}
