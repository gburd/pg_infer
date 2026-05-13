//! Token embedding — lookup + architecture-specific scaling.

use ndarray::Array2;
use crate::model::ModelWeights;

/// Embed token IDs with architecture-specific scaling (internal).
pub(super) fn embed_tokens(weights: &ModelWeights, token_ids: &[u32]) -> Array2<f32> {
    embed_tokens_pub(weights, token_ids)
}

/// Embed token IDs with architecture-specific scaling.
///
/// Looks up each token's row in the embedding matrix and multiplies by the
/// architecture's embedding scale factor (e.g., sqrt(hidden_size) for Gemma).
/// Returns a `[seq_len, hidden_size]` array.
///
/// # Examples
///
/// ```no_run
/// use infer_inference::{load_model_dir, forward::embed_tokens_pub};
/// use std::path::Path;
///
/// let weights = load_model_dir(Path::new("/path/to/model")).unwrap();
/// let token_ids = vec![1, 450, 5765, 315];
///
/// let embeddings = embed_tokens_pub(&weights, &token_ids);
/// assert_eq!(embeddings.shape()[0], token_ids.len());
/// assert_eq!(embeddings.shape()[1], weights.hidden_size);
/// ```
pub fn embed_tokens_pub(weights: &ModelWeights, token_ids: &[u32]) -> Array2<f32> {
    let seq_len = token_ids.len();
    let hidden = weights.hidden_size;
    let scale = weights.arch.embed_scale();

    let mut h = Array2::<f32>::zeros((seq_len, hidden));
    for (i, &tok_id) in token_ids.iter().enumerate() {
        let row = weights.embed.row(tok_id as usize);
        for j in 0..hidden {
            h[[i, j]] = row[j] * scale;
        }
    }
    h
}
