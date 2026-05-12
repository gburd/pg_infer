//! BitNet b1.58 architecture (microsoft/BitNet-b1.58-2B-4T).
//!
//! Weights quantized to {-1, 0, +1} via AbsMean. FFN uses squared ReLU
//! (relu²) with gate/up/down projections plus an ffn_sub_norm (RMSNorm)
//! before down_proj. Forward: down(sub_norm(relu²(gate(x)) * up(x))).

use crate::config::{Activation, ModelArchitecture, ModelConfig};

pub struct BitNetArch {
    config: ModelConfig,
}

impl BitNetArch {
    pub fn from_config(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArchitecture for BitNetArch {
    fn family(&self) -> &str {
        "bitnet"
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn activation(&self) -> Activation {
        Activation::SquaredRelu
    }

    fn has_ffn_sub_norm(&self) -> bool {
        true
    }

    fn ffn_sub_norm_key(&self, layer: usize) -> Option<String> {
        Some(format!("{}mlp.ffn_norm.weight", self.layer_prefix(layer)))
    }

    fn preferred_gate_dtype(&self) -> &str {
        "ternary"
    }
}
