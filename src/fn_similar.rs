use ndarray::Array1;
use pgrx::prelude::*;

use crate::error::PgInferError;
use crate::registry;

/// Semantic similarity between two texts using the model's internal
/// representation.
///
/// Returns an UNNORMALISED, MODEL-DEPENDENT magnitude: higher means more
/// similar, but it is not a probability and has no fixed upper bound.  The
/// sign tells you which kind of evidence it rests on:
///
/// - **Positive** -- the two inputs activate at least one gate feature in
///   common, and the result is the strongest such shared activation, on the
///   same open-ended scale as `gate_score` elsewhere.  The range depends on
///   the model: a Gemma-class vindex lands around 20--45, BitNet b1.58
///   reaches 300+.
/// - **Zero or negative** -- the model's features connect the two inputs
///   nowhere, so there is no gate magnitude to report and the result falls
///   back to `cosine - 1.0`, in `[-2, 0]`.  That band sits entirely below
///   the positive one, so a gate-backed match always outranks a
///   cosine-only one, and cosine ordering is preserved within the band.
///
/// A sign change is therefore meaningful: it marks the point where the
/// model stopped recognising a connection.  Treat a negative score as "no
/// learned association, here is the raw geometry instead".
///
/// NB: thresholds do not transfer between models, because the positive band
/// is scaled by the model's own gate magnitudes.  Compare scores only
/// against others from the same model.  [`similar_to_many`] ranks candidates
/// against one query and is the better tool when a caller wants a comparison.
///
/// ```sql
/// SELECT similar_to('France', 'Paris');        -- positive, shares features
/// SELECT similar_to('France', 'banana');       -- likely negative
/// -- Require a real learned association, not just embedding geometry:
/// SELECT * FROM products WHERE similar_to(category, 'AI') > 0;
/// -- 15.0 is a Gemma-class figure; calibrate per model rather than
/// -- copying it.
/// SELECT * FROM products
///     WHERE similar_to(category, 'AI') > 15.0;
/// ```
#[pg_extern(stable, parallel_safe)]
#[tracing::instrument(skip_all, fields(a = a, b = b, model = model.unwrap_or("default")))]
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
#[pg_extern(stable, parallel_safe)]
#[tracing::instrument(skip_all, fields(num_candidates = candidates.len(), query = query, model = model.unwrap_or("default")))]
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

    let scores = registry::with_backend(&model_name, |backend| {
        backend.similar_to_many(&non_null, query)
    })?;

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

    // The two branches answer different questions, and the old `cosine * 10`
    // let the weaker answer outrank the stronger one.  A shared gate feature
    // means the model's own learned features connect these two texts; a bare
    // cosine means they did not, and the raw embeddings merely point a
    // similar way.  The second is strictly weaker evidence, but at x10 a
    // cosine of 0.95 scored 9.5 and beat a real-but-weak gate activation of
    // 3.0.  Which branch fired, rather than how similar the texts were,
    // decided the ranking.
    //
    // So the fallback now lands in a band strictly below every value the gate
    // branch can produce.  Gate scores are > 0 by construction (the branch
    // only fires when `max_shared_score` exceeds zero), and `cosine - 1.0` is
    // in [-2, 0], so the two can never interleave while cosine ordering is
    // preserved inside the fallback band.
    //
    // NB: the gate branch is byte-for-byte unchanged, so thresholds already
    // tuned against it (`WHERE similar_to(...) > 15.0`) keep their meaning.
    // Only the no-shared-feature case moves, and it moves from a misleading
    // positive number to a clearly-weaker negative one.
    let score = if max_shared_score > 0.0 {
        max_shared_score as f64
    } else {
        cosine - 1.0
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
    let set_b: std::collections::HashSet<usize> = hits_b.iter().map(|&(idx, _)| idx).collect();

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

    // Same two-band scheme as `similar_to_impl`; see the note there.
    Ok(if max_shared_score > 0.0 {
        max_shared_score as f64
    } else {
        cosine - 1.0
    })
}

/// Convert a similarity score to a distance value, for `<~>` and the index
/// access method's ordering.
///
/// Strictly decreasing in `score` across the whole range a score can take,
/// which is what the index AM relies on: ordering by distance ascending has
/// to equal ordering by score descending, or `ORDER BY ... <~> ...` returns
/// rows in the wrong order.
///
/// `1 / (3 + score)` rather than the earlier `1 / (1 + score)` because
/// [`similar_to_impl`]'s no-shared-feature fallback now returns
/// `cosine - 1.0`, i.e. down to -2.  The old form had its pole at -1, inside
/// that range: it went to infinity there and NEGATIVE below, which would have
/// inverted the order for the most dissimilar pairs.  Moving the pole to -3
/// keeps the result finite and positive for every score >= -2, and the
/// branch that mapped `score <= 0` to `f64::MAX` is gone -- it collapsed
/// every fallback row to one distance and threw away their ordering.
///
/// Scores above zero land in `(0, 1/3)` and fallback scores in `[1/3, 1]`,
/// so a gate-backed match always sorts ahead of a cosine-only one.
pub(crate) fn score_to_distance(score: f64) -> f64 {
    1.0 / (3.0 + score)
}

/// Distance function for the `<~>` operator (lower = more similar).
#[pg_extern(stable, parallel_safe)]
fn infer_distance(a: &str, b: &str) -> Result<f64, Box<dyn std::error::Error>> {
    let model_name = registry::resolve_model_name(None)?;
    let score = registry::with_backend(&model_name, |backend| backend.similar_to(a, b))?;

    Ok(score_to_distance(score))
}

/// Raw similarity score for the `<~` operator (higher = more similar).
///
/// Complement to `<~>` (distance).  Useful for `WHERE col <~ 'query' > threshold`
/// patterns where you want to filter by similarity rather than sort by distance.
#[pg_extern(stable, parallel_safe)]
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

#[cfg(test)]
mod tests {
    use super::score_to_distance;

    /// The smallest and largest scores each branch can emit.
    ///
    /// The gate branch is open-ended above and strictly positive; the
    /// fallback is `cosine - 1.0` for a cosine in [-1, 1], so [-2, 0].
    const GATE: &[f64] = &[0.001, 3.0, 25.0, 332.8, 1.0e6];
    const FALLBACK: &[f64] = &[0.0, -0.01, -0.05, -0.5, -1.0, -2.0];

    #[test]
    fn distance_is_finite_and_positive_over_the_whole_range() {
        for &s in GATE.iter().chain(FALLBACK) {
            let d = score_to_distance(s);
            assert!(
                d.is_finite() && d > 0.0,
                "score {s} gave a distance of {d}; the index AM needs a \
                 finite positive distance for every score"
            );
        }
    }

    #[test]
    fn distance_ascending_equals_score_descending() {
        // This is the property the index access method depends on: it orders
        // by distance and promises the caller score order.
        let mut scores: Vec<f64> = GATE.iter().chain(FALLBACK).copied().collect();
        scores.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let mut by_distance = scores.clone();
        by_distance.sort_by(|a, b| {
            score_to_distance(*a)
                .partial_cmp(&score_to_distance(*b))
                .unwrap()
        });
        assert_eq!(
            scores, by_distance,
            "ordering by distance ascending must match score descending"
        );
    }

    /// The bug this replaced: `cosine * 10.0` let a cosine-only pair outrank
    /// a weaker-but-real gate match, so the branch decided the ranking.
    #[test]
    fn every_gate_score_outranks_every_fallback_score() {
        let weakest_gate = GATE.iter().copied().fold(f64::MAX, f64::min);
        let strongest_fallback = FALLBACK.iter().copied().fold(f64::MIN, f64::max);
        assert!(
            weakest_gate > strongest_fallback,
            "the weakest gate score ({weakest_gate}) must still beat the \
             strongest fallback ({strongest_fallback})"
        );
        // And the same ordering must survive the distance mapping.
        assert!(score_to_distance(weakest_gate) < score_to_distance(strongest_fallback));
    }

    #[test]
    fn fallback_preserves_cosine_ordering() {
        // Within the fallback band, a higher cosine must still rank higher;
        // the old `score <= 0 -> f64::MAX` branch collapsed all of these.
        let cosines = [1.0, 0.99, 0.5, 0.0, -1.0];
        let distances: Vec<f64> = cosines.iter().map(|c| score_to_distance(c - 1.0)).collect();
        for w in distances.windows(2) {
            assert!(
                w[0] < w[1],
                "a higher cosine must give a smaller distance: {w:?}"
            );
        }
    }
}
