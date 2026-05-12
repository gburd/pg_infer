//! MarkovResidualEngine — main engine struct.

use infer_compute::{cpu_backend, ComputeBackend};
use ndarray::Array2;

use super::compute::{rs_decode_step, rs_prefill};
use super::store::RsStore;
use crate::model::ModelWeights;

/// MarkovResidualEngine — KV-cache-free decode using stored residuals.
///
/// Instead of caching K/V tensors for all layers × all positions (which grows
/// as O(layers × seq_len × kv_dim × 2)), this engine stores the pre-layer
/// residual vectors and recomputes K/V on-the-fly during decode. The residual
/// stream is the complete Markov state of the transformer — storing it is both
/// necessary and sufficient for exact reconstruction of the forward pass.
///
/// Memory: O(layers × window × hidden) + cold tier.
/// Correctness: KL = 0.0 vs full-KV baseline (validated on Gemma 3 4B).
pub struct MarkovResidualEngine {
    window_size: Option<usize>,
    store: Option<RsStore>,
    backend: Box<dyn ComputeBackend>,
}

impl MarkovResidualEngine {
    /// Create with CPU backend and optional window size.
    /// `None` means unlimited window (store all residuals).
    pub fn new(window_size: Option<usize>) -> Self {
        Self::with_backend(window_size, cpu_backend())
    }

    /// Create with a specific compute backend.
    pub fn with_backend(window_size: Option<usize>, backend: Box<dyn ComputeBackend>) -> Self {
        Self {
            window_size,
            store: None,
            backend,
        }
    }

    /// Human-readable engine name.
    pub fn name(&self) -> &str {
        "markov-rs"
    }

    /// Total memory in bytes (hot + cold).
    pub fn memory_bytes(&self) -> usize {
        self.store.as_ref().map_or(0, |s| s.memory_bytes())
    }

    /// Number of tokens in the hot window.
    pub fn window_tokens(&self) -> usize {
        self.store.as_ref().map_or(0, |s| s.window_tokens())
    }

    /// Cold-tier memory in bytes.
    pub fn cold_bytes(&self) -> usize {
        self.store.as_ref().map_or(0, |s| s.cold_bytes())
    }

    /// Prefill: process prompt tokens, populate residual store.
    /// Returns hidden state at last position [1, hidden_size].
    pub fn prefill(&mut self, weights: &ModelWeights, token_ids: &[u32]) -> Option<Array2<f32>> {
        let result = rs_prefill(weights, token_ids, self.window_size, self.backend.as_ref());
        let hidden = result.hidden.clone();
        self.store = Some(result.store);
        Some(hidden)
    }

    /// Decode one token: recompute K/V from stored residuals, run forward pass.
    /// Returns hidden state at the new position [1, hidden_size].
    pub fn decode_step(&mut self, weights: &ModelWeights, token_id: u32) -> Option<Array2<f32>> {
        let rs = self.store.take()?;
        let (hidden, new_rs) = rs_decode_step(weights, token_id, rs, self.backend.as_ref())?;
        self.store = Some(new_rs);
        Some(hidden)
    }

    /// Configuration string for logging.
    pub fn config_string(&self) -> String {
        match self.window_size {
            Some(w) => format!("window={w}"),
            None => "window=full".into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forward::hidden_to_raw_logits;
    use crate::model::test_utils::make_test_weights;

    #[test]
    fn engine_name() {
        assert_eq!(MarkovResidualEngine::new(None).name(), "markov-rs");
    }

    #[test]
    fn engine_memory_zero_before_prefill() {
        let eng = MarkovResidualEngine::new(None);
        assert_eq!(eng.memory_bytes(), 0);
        assert_eq!(eng.window_tokens(), 0);
        assert_eq!(eng.cold_bytes(), 0);
    }

    #[test]
    fn engine_config_full_window() {
        let eng = MarkovResidualEngine::new(None);
        assert!(eng.config_string().contains("full"));
    }

    #[test]
    fn engine_config_fixed_window() {
        let eng = MarkovResidualEngine::new(Some(16));
        assert!(eng.config_string().contains("16"));
    }

    #[test]
    fn prefill_stores_residuals() {
        let weights = make_test_weights();
        let mut engine = MarkovResidualEngine::new(None);
        let h = engine.prefill(&weights, &[0u32, 1, 2]).expect("prefill");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(engine.memory_bytes() > 0);
    }

    #[test]
    fn decode_step_produces_finite_logits() {
        let weights = make_test_weights();
        let mut engine = MarkovResidualEngine::new(None);
        engine.prefill(&weights, &[0u32, 1]).expect("prefill");
        let h = engine.decode_step(&weights, 2).expect("decode");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(hidden_to_raw_logits(&weights, &h).iter().all(|v| v.is_finite()));
    }

    #[test]
    fn memory_grows_with_decode_steps() {
        let weights = make_test_weights();
        let mut engine = MarkovResidualEngine::new(None);
        engine.prefill(&weights, &[0u32]).expect("prefill");
        let mem0 = engine.memory_bytes();
        engine.decode_step(&weights, 1).expect("decode 1");
        let mem1 = engine.memory_bytes();
        engine.decode_step(&weights, 2).expect("decode 2");
        let mem2 = engine.memory_bytes();
        assert!(mem1 > mem0, "memory should grow");
        assert!(mem2 > mem1, "memory should grow");
    }

    #[test]
    fn window_clipping_limits_hot_store() {
        let weights = make_test_weights();
        let mut engine = MarkovResidualEngine::new(Some(2));
        engine.prefill(&weights, &[0u32, 1, 2, 3, 4]).expect("prefill");
        assert!(engine.window_tokens() <= 2);
        assert!(engine.cold_bytes() > 0);
    }

    #[test]
    fn multiple_decode_steps_consistent_shapes() {
        let weights = make_test_weights();
        let mut engine = MarkovResidualEngine::new(None);
        engine.prefill(&weights, &[0u32]).expect("prefill");
        for step in 0..3 {
            let h = engine.decode_step(&weights, step as u32).expect("decode");
            assert_eq!(h.shape(), &[1, weights.hidden_size], "step {step}");
        }
    }
}
