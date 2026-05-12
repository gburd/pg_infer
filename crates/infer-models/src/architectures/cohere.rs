//! Cohere / Command R architecture.
//!
//! Key differences from standard Llama:
//! - Uses LayerNorm instead of RMSNorm
//! - Has tied word embeddings (lm_head shares embed_tokens)
//! - Uses SiLU activation and gated FFN (same as Llama)

use crate::config::{ModelArchitecture, ModelConfig, NormType};

pub struct CohereArch {
    config: ModelConfig,
}

impl CohereArch {
    pub fn from_config(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArchitecture for CohereArch {
    fn family(&self) -> &str {
        "cohere"
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn norm_type(&self) -> NormType {
        NormType::LayerNorm
    }
}
