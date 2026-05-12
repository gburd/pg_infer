//! Phi-3/4 architecture.
//!
//! Phi is a Llama variant with partial rotary embeddings (SuRoPE),
//! attention biases, and FFN biases. Uses the same `model.layers.{l}.`
//! tensor key prefix and SiLU activation as Llama.

use crate::config::{ModelArchitecture, ModelConfig};

pub struct PhiArch {
    config: ModelConfig,
}

impl PhiArch {
    pub fn from_config(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArchitecture for PhiArch {
    fn family(&self) -> &str {
        "phi"
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    // Phi uses partial rotary embeddings (only partial_rotary_factor of head_dim gets RoPE).
    fn rotary_fraction_for_layer(&self, _layer: usize) -> f64 {
        self.config.partial_rotary_factor.unwrap_or(0.5)
    }

    // Phi models have attention biases on Q/K/V projections.
    fn attn_q_bias_key(&self, layer: usize) -> Option<String> {
        Some(format!("{}self_attn.q_proj.bias", self.layer_prefix(layer)))
    }

    fn attn_k_bias_key(&self, layer: usize) -> Option<String> {
        Some(format!("{}self_attn.k_proj.bias", self.layer_prefix(layer)))
    }

    fn attn_v_bias_key(&self, layer: usize) -> Option<String> {
        Some(format!("{}self_attn.v_proj.bias", self.layer_prefix(layer)))
    }

    // Phi uses FFN biases on up and down projections.
    fn ffn_up_bias_key(&self, layer: usize) -> Option<String> {
        Some(format!("{}mlp.up_proj.bias", self.layer_prefix(layer)))
    }

    fn ffn_down_bias_key(&self, layer: usize) -> Option<String> {
        Some(format!("{}mlp.down_proj.bias", self.layer_prefix(layer)))
    }
}
