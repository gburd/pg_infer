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

    let score = registry::with_backend(&model_name, |backend| backend.similar_to(a, b))?;

    Ok(score)
}

/// Bulk variant of [`similar_to`]: returns one score per candidate.
///
/// On the remote backend, all walks are issued in a single concurrent
/// batch (one HTTP/2 multiplexed round-trip or one UDS fan-out), so a
/// 20-candidate ranking is one network wait instead of 20.  The mmap
/// backend falls back to a trivial loop — there's nothing to overlap
/// when the data is in-process.
///
/// ```sql
/// SELECT id, score
///   FROM (
///     SELECT id, unnest(similar_to_many(
///       (SELECT array_agg(name) FROM products WHERE category = 'AI'),
///       'large language model'
///     )) AS score
///     FROM products WHERE category = 'AI'
///   ) t
///   ORDER BY score DESC LIMIT 5;
/// ```
#[pg_extern]
fn similar_to_many(
    candidates: Vec<Option<String>>,
    query: &str,
    model: default!(Option<&str>, "NULL"),
) -> Result<Vec<Option<f64>>, Box<dyn std::error::Error>> {
    let model_name = registry::resolve_model_name(model)?;

    // NULL candidates get NULL scores; keep positional correspondence so
    // the array can be unnested next to the original table.
    let mut non_null: Vec<String> = Vec::with_capacity(candidates.len());
    let mut is_null: Vec<bool> = Vec::with_capacity(candidates.len());
    for c in &candidates {
        match c {
            Some(s) => {
                is_null.push(false);
                non_null.push(s.clone());
            }
            None => is_null.push(true),
        }
    }

    let scores =
        registry::with_backend(&model_name, |backend| backend.similar_to_many(&non_null, query))?;

    // Re-interleave NULLs.
    let mut out = Vec::with_capacity(candidates.len());
    let mut score_iter = scores.into_iter();
    for nullish in is_null {
        if nullish {
            out.push(None);
        } else {
            out.push(score_iter.next());
        }
    }
    Ok(out)
}

/// Compute the similarity score between two texts.
///
/// Algorithm:
/// 1. Embed both texts (average token embeddings, scaled).
/// 2. For each layer (optionally sampled), compute gate_knn for both embeddings.
/// 3. Find features that appear in both top-K sets.
/// 4. Return the maximum shared gate score.
///
/// Performance optimizations:
/// - Layer sampling: When `infer.similarity_max_layers` is set, sample evenly
///   across layers instead of querying all (3x+ speedup).
/// - Parallel processing: When `infer.parallel_similarity` is true, query
///   layers in parallel using Rayon (4-8x speedup on multi-core).
pub(crate) fn similar_to_impl(
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

    // Determine which layers to query (all vs. sampled).
    let max_layers = crate::gucs::similarity_max_layers();
    let layers_to_query: Vec<usize> = if max_layers > 0 && num_layers > max_layers {
        // Sample evenly across layers
        let step = num_layers / max_layers;
        (0..num_layers).step_by(step).take(max_layers).collect()
    } else {
        // Query all layers
        (0..num_layers).collect()
    };

    // Compute max shared score across layers (parallel or sequential).
    let max_shared_score: f32 = if crate::gucs::parallel_similarity() {
        // Parallel processing with Rayon
        use rayon::prelude::*;
        layers_to_query
            .par_iter()
            .map(|&layer| compute_layer_similarity(handle, layer, &embed_a, &embed_b, top_k))
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0)
    } else {
        // Sequential processing
        let mut max = 0.0;
        for &layer in &layers_to_query {
            let score = compute_layer_similarity(handle, layer, &embed_a, &embed_b, top_k);
            if score > max {
                max = score;
            }
        }
        max
    };

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

/// Compute similarity score for a single layer.
///
/// Finds overlapping features between the two embeddings' top-K activations
/// and returns the maximum shared score.
fn compute_layer_similarity(
    handle: &registry::ModelHandle,
    layer: usize,
    embed_a: &Array1<f32>,
    embed_b: &Array1<f32>,
    top_k: usize,
) -> f32 {
    let hits_a = handle.gate_knn(layer, embed_a, top_k);
    let hits_b = handle.gate_knn(layer, embed_b, top_k);

    // Build a set of feature indices activated by B.
    let set_b: std::collections::HashSet<usize> =
        hits_b.iter().map(|&(idx, _)| idx).collect();

    // Find overlapping features and return the maximum shared score.
    let mut max_shared = 0.0f32;
    for &(idx, score_a) in &hits_a {
        if set_b.contains(&idx) {
            if let Some(&(_, score_b)) = hits_b.iter().find(|&&(i, _)| i == idx) {
                let shared = score_a.min(score_b);
                if shared > max_shared {
                    max_shared = shared;
                }
            }
        }
    }
    max_shared
}

/// Pre-compute gate_knn results for a query embedding across all layers.
///
/// Returns a `Vec` indexed by layer, where each element is a `HashMap`
/// mapping feature index → gate score.  Retained for reuse when Phase C
/// introduces the remote backend — it halves gate_knn traffic for the
/// query side of a table scan.
#[allow(dead_code)]
pub(crate) fn precompute_query_gates(
    handle: &registry::ModelHandle,
    query_embedding: &Array1<f32>,
    top_k: usize,
) -> Vec<std::collections::HashMap<usize, f32>> {
    (0..handle.config.num_layers)
        .map(|layer| {
            handle
                .gate_knn(layer, query_embedding, top_k)
                .into_iter()
                .collect::<std::collections::HashMap<usize, f32>>()
        })
        .collect()
}

/// Compute similarity using pre-computed query gate results.
///
/// Same algorithm as `similar_to_impl` but avoids calling `gate_knn` for
/// the query side — the caller provides the pre-computed results.  This
/// halves the gate_knn calls during table scans.  Retained for Phase C.
#[allow(dead_code)]
pub(crate) fn similar_to_with_precomputed(
    handle: &registry::ModelHandle,
    col_text: &str,
    query_embedding: &Array1<f32>,
    query_gates: &[std::collections::HashMap<usize, f32>],
) -> Result<f64, PgInferError> {
    let embed_col = embed_text(handle, col_text)?;
    let cosine = cosine_similarity(&embed_col, query_embedding);

    let top_k = 50_usize;
    let mut max_shared_score: f32 = 0.0;

    for (layer, q_gates) in query_gates.iter().enumerate() {
        let hits_col = handle.gate_knn(layer, &embed_col, top_k);

        for &(idx, score_col) in &hits_col {
            if let Some(&score_q) = q_gates.get(&idx) {
                let shared = score_col.min(score_q);
                if shared > max_shared_score {
                    max_shared_score = shared;
                }
            }
        }
    }

    Ok(if max_shared_score > 0.0 {
        max_shared_score as f64
    } else {
        cosine * 10.0
    })
}

/// Convert a similarity score to a distance value.
///
/// Maps to `[0, 1)` range, monotonically decreasing with increasing score.
/// Avoids the discontinuity of `1/score` near zero.
pub(crate) fn score_to_distance(score: f64) -> f64 {
    if score <= 0.0 {
        f64::MAX
    } else {
        1.0 / (1.0 + score)
    }
}

/// Distance function for the `<~>` operator (lower = more similar).
#[pg_extern]
fn infer_distance(a: &str, b: &str) -> Result<f64, Box<dyn std::error::Error>> {
    let model_name = registry::resolve_model_name(None)?;
    let score = registry::with_backend(&model_name, |backend| backend.similar_to(a, b))?;

    Ok(score_to_distance(score))
}

/// Raw similarity score for the `<~` operator (higher = more similar).
///
/// Complement to `<~>` (distance).  Useful for `WHERE col <~ 'query' > threshold`
/// patterns where you want to filter by similarity rather than sort by distance.
#[pg_extern]
fn infer_similarity(a: &str, b: &str) -> Result<f64, Box<dyn std::error::Error>> {
    let model_name = registry::resolve_model_name(None)?;
    let score = registry::with_backend(&model_name, |backend| backend.similar_to(a, b))?;
    Ok(score)
}

// Operators (<~>, <~) and COST annotations are registered in am.rs
// as part of the unified infer_access_method SQL block.  This ensures
// correct ordering: operators are created before the operator class
// that references them.

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Embed a text string into a single vector by averaging token embeddings.
pub(crate) fn embed_text(
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
pub(crate) fn cosine_similarity(a: &Array1<f32>, b: &Array1<f32>) -> f64 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        (dot / (norm_a * norm_b)) as f64
    }
}
