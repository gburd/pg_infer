use ndarray::Array1;
use pgrx::prelude::*;

use crate::error::PgInferError;
use crate::registry;

/// Semantic similarity between two texts using the model's internal
/// representation.
///
/// Returns the maximum gate activation score across all layers, computed
/// by averaging token embeddings for each input and taking the dot
/// product with gate vectors.  Higher scores indicate stronger semantic
/// association.
///
/// ```sql
/// SELECT similar_to('France', 'Paris');        -- high score
/// SELECT similar_to('France', 'banana');       -- low score
/// SELECT * FROM products
///     WHERE similar_to(category, 'AI') > 15.0;
/// ```
#[pg_extern]
fn similar_to(
    a: &str,
    b: &str,
    model: default!(Option<&str>, "NULL"),
) -> Result<f64, Box<dyn std::error::Error>> {
    let model_name = registry::resolve_model_name(model)?;

    let score = registry::with_model(&model_name, |handle| {
        similar_to_impl(handle, a, b)
    })?;

    Ok(score)
}

/// Compute the similarity score between two texts.
///
/// Algorithm:
/// 1. Embed both texts (average token embeddings, scaled).
/// 2. For each layer, compute gate_knn for both embeddings.
/// 3. Find features that appear in both top-K sets.
/// 4. Return the maximum shared gate score.
fn similar_to_impl(
    handle: &registry::ModelHandle,
    a: &str,
    b: &str,
) -> Result<f64, PgInferError> {
    let embed_a = embed_text(handle, a)?;
    let embed_b = embed_text(handle, b)?;

    // Compute cosine similarity between the averaged embeddings as a
    // baseline, then boost with shared gate activations.
    let cosine = cosine_similarity(&embed_a, &embed_b);

    // Walk layers looking for shared feature activations.
    let top_k = 50_usize;
    let num_layers = handle.config.num_layers;
    let mut max_shared_score: f32 = 0.0;

    for layer in 0..num_layers {
        let hits_a = handle.gate_knn(layer, &embed_a, top_k);
        let hits_b = handle.gate_knn(layer, &embed_b, top_k);

        // Build a set of feature indices activated by B.
        let set_b: std::collections::HashSet<usize> =
            hits_b.iter().map(|&(idx, _)| idx).collect();

        // Find overlapping features and accumulate scores.
        for &(idx, score_a) in &hits_a {
            if set_b.contains(&idx) {
                if let Some(&(_, score_b)) = hits_b.iter().find(|&&(i, _)| i == idx) {
                    let shared = score_a.min(score_b);
                    if shared > max_shared_score {
                        max_shared_score = shared;
                    }
                }
            }
        }
    }

    // Combine embedding cosine similarity with gate activation overlap.
    // The gate score dominates when features overlap; cosine provides
    // a baseline when they don't.
    let score = if max_shared_score > 0.0 {
        max_shared_score as f64
    } else {
        cosine * 10.0 // scale cosine to comparable range
    };

    Ok(score)
}

/// Distance function for the `<~>` operator (lower = more similar).
#[pg_extern]
fn infer_distance(a: &str, b: &str) -> Result<f64, Box<dyn std::error::Error>> {
    let model_name = registry::resolve_model_name(None)?;
    let score = registry::with_model(&model_name, |handle| {
        similar_to_impl(handle, a, b)
    })?;

    Ok(if score > 0.0 { 1.0 / score } else { f64::MAX })
}

// Register the <~> operator.
extension_sql!(
    r#"
CREATE OPERATOR <~> (
    LEFTARG  = text,
    RIGHTARG = text,
    FUNCTION = infer_distance,
    COMMUTATOR = <~>
);
"#,
    name = "infer_distance_operator",
    requires = [infer_distance],
);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Embed a text string into a single vector by averaging token embeddings.
fn embed_text(
    handle: &registry::ModelHandle,
    text: &str,
) -> Result<Array1<f32>, PgInferError> {
    let encoding = handle
        .tokenizer
        .encode(text, false)
        .map_err(|e| PgInferError::Tokenize(e.to_string()))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();

    if token_ids.is_empty() {
        return Err(PgInferError::EmptyPrompt);
    }

    let hidden = handle.config.hidden_size;
    if token_ids.len() == 1 {
        Ok(handle
            .embeddings
            .row(token_ids[0] as usize)
            .mapv(|v| v * handle.embed_scale))
    } else {
        let mut avg = Array1::<f32>::zeros(hidden);
        for &tok in &token_ids {
            avg += &handle
                .embeddings
                .row(tok as usize)
                .mapv(|v| v * handle.embed_scale);
        }
        avg /= token_ids.len() as f32;
        Ok(avg)
    }
}

/// Cosine similarity between two 1-D vectors.
fn cosine_similarity(a: &Array1<f32>, b: &Array1<f32>) -> f64 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        (dot / (norm_a * norm_b)) as f64
    }
}
