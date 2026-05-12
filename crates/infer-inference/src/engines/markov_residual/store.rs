//! RsStore — per-layer residual buffer for MarkovResidualEngine.

use ndarray::{s, Array2};

use crate::attention::SharedKV;

/// Per-layer pre-attention residuals for all stored positions.
pub struct RsStore {
    /// One entry per layer: [seq_len, hidden_size] residuals in the hot window.
    pub stored: Vec<Array2<f32>>,
    /// Cold-tier residuals (overflow beyond window). One entry per layer.
    pub cold_residuals: Option<Vec<Array2<f32>>>,
    /// Pre-computed K/V for cold-tier positions (avoids re-recompute each step).
    pub cold_kv: Option<Vec<SharedKV>>,
    /// Absolute token position where the cold tier starts.
    pub cold_abs_start: usize,
    /// Next absolute token position (total tokens processed so far).
    pub next_position: usize,
    /// Maximum hot-window size. None = unlimited (store all residuals).
    pub max_window: Option<usize>,
}

impl RsStore {
    /// Total memory used by this store (hot + cold) in bytes.
    pub fn memory_bytes(&self) -> usize {
        let hot: usize = self.stored.iter().map(|s| s.len() * 4).sum();
        let cold_res: usize = self
            .cold_residuals
            .as_ref()
            .map(|c| c.iter().map(|s| s.len() * 4).sum())
            .unwrap_or(0);
        let cold_kv: usize = self
            .cold_kv
            .as_ref()
            .map(|kv| kv.iter().map(|(k, v)| (k.len() + v.len()) * 4).sum())
            .unwrap_or(0);
        hot + cold_res + cold_kv
    }

    /// Cold-tier memory in bytes.
    pub fn cold_bytes(&self) -> usize {
        let cold_res: usize = self
            .cold_residuals
            .as_ref()
            .map(|c| c.iter().map(|s| s.len() * 4).sum())
            .unwrap_or(0);
        let cold_kv: usize = self
            .cold_kv
            .as_ref()
            .map(|kv| kv.iter().map(|(k, v)| (k.len() + v.len()) * 4).sum())
            .unwrap_or(0);
        cold_res + cold_kv
    }

    /// Number of tokens in the hot window.
    pub fn window_tokens(&self) -> usize {
        self.stored.first().map_or(0, |s| s.shape()[0])
    }

    /// Clip a single layer's hot store to `max_window`, moving overflow to `cold`.
    pub(crate) fn clip_layer(&mut self, layer: usize, cold: &mut Vec<Array2<f32>>) {
        let window = match self.max_window {
            Some(w) => w,
            None => return,
        };
        let s = &self.stored[layer];
        let rows = s.shape()[0];
        if rows <= window {
            cold.push(Array2::zeros((0, s.shape()[1])));
            return;
        }
        let start = rows - window;
        cold.push(s.slice(s![..start, ..]).to_owned());
        self.stored[layer] = s.slice(s![start.., ..]).to_owned();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_store(num_layers: usize, seq_len: usize, hidden: usize) -> RsStore {
        let stored = (0..num_layers)
            .map(|_| Array2::from_elem((seq_len, hidden), 1.0f32))
            .collect();
        RsStore {
            stored,
            cold_residuals: None,
            cold_kv: None,
            cold_abs_start: 0,
            next_position: seq_len,
            max_window: None,
        }
    }

    #[test]
    fn memory_bytes_hot_only() {
        let store = make_store(2, 5, 16);
        assert_eq!(store.memory_bytes(), 2 * 5 * 16 * 4);
    }

    #[test]
    fn memory_bytes_empty_store_is_zero() {
        let store = make_store(0, 0, 16);
        assert_eq!(store.memory_bytes(), 0);
    }

    #[test]
    fn cold_bytes_zero_when_no_cold() {
        let store = make_store(2, 5, 16);
        assert_eq!(store.cold_bytes(), 0);
    }

    #[test]
    fn window_tokens_matches_stored_rows() {
        let store = make_store(3, 7, 8);
        assert_eq!(store.window_tokens(), 7);
    }

    #[test]
    fn window_tokens_zero_for_empty_store() {
        let store = make_store(0, 0, 8);
        assert_eq!(store.window_tokens(), 0);
    }

    #[test]
    fn clip_layer_no_window_is_noop() {
        let mut store = make_store(1, 10, 4);
        let mut cold = Vec::new();
        store.clip_layer(0, &mut cold);
        assert!(cold.is_empty());
        assert_eq!(store.stored[0].shape()[0], 10);
    }

    #[test]
    fn clip_layer_within_window_pushes_empty_cold() {
        let mut store = make_store(1, 4, 4);
        store.max_window = Some(8);
        let mut cold = Vec::new();
        store.clip_layer(0, &mut cold);
        assert_eq!(cold.len(), 1);
        assert_eq!(cold[0].shape()[0], 0);
        assert_eq!(store.stored[0].shape()[0], 4);
    }

    #[test]
    fn clip_layer_excess_rows_moved_to_cold() {
        let mut store = make_store(1, 10, 4);
        store.max_window = Some(3);
        let mut cold = Vec::new();
        store.clip_layer(0, &mut cold);
        assert_eq!(cold[0].shape()[0], 7);
        assert_eq!(store.stored[0].shape()[0], 3);
    }

    #[test]
    fn clip_layer_exactly_at_window_no_cold() {
        let mut store = make_store(1, 5, 4);
        store.max_window = Some(5);
        let mut cold = Vec::new();
        store.clip_layer(0, &mut cold);
        assert_eq!(cold[0].shape()[0], 0);
        assert_eq!(store.stored[0].shape()[0], 5);
    }
}
