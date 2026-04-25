use std::collections::HashMap;

use ndarray::Array1;
use pgrx::prelude::*;

use crate::error::PgInferError;
use crate::gucs;
use crate::registry;

/// Return relationships the model knows about an entity.
///
/// Walks the vindex gate vectors using the entity's embedding as a query,
/// collects activated features, deduplicates by target token, and returns
/// edges ranked by gate score.
///
/// The `threshold` parameter controls the minimum gate score for a feature
/// to be included.  When omitted (or set to 0), an adaptive threshold is
/// used: `max_score × 0.1`, where `max_score` is the highest activation
/// observed across all layers for this query.  The `infer.gate_threshold`
/// GUC provides a session-level default.
///
/// ```sql
/// SELECT * FROM describe('France');
/// SELECT * FROM describe('Einstein', model => 'llama3_8b');
/// SELECT * FROM describe('France', threshold => 0.01);
/// ```
#[pg_extern]
fn describe(
    entity: &str,
    model: default!(Option<&str>, "NULL"),
    threshold: default!(Option<f64>, "NULL"),
) -> Result<
    TableIterator<
        'static,
        (
            name!(relation, String),
            name!(target, String),
            name!(confidence, f64),
            name!(layer, i32),
        ),
    >,
    Box<dyn std::error::Error>,
> {
    let model_name = registry::resolve_model_name(model)?;

    let rows = registry::with_model(&model_name, |handle| {
        describe_impl(handle, entity, threshold)
    })?;

    Ok(TableIterator::new(rows))
}

/// Internal describe implementation.
///
/// Mirrors the algorithm in `larql-lql/src/executor/query/describe.rs`:
/// 1. Tokenize entity, compute average token embedding (scaled).
/// 2. For each layer, run gate_knn to find top features.
/// 3. Aggregate results by target token, keeping the highest score.
/// 4. Filter out trivial tokens and self-references.
///
/// `explicit_threshold`: caller-provided threshold.  `None` or `Some(0.0)`
/// means "use adaptive thresholding" (checked after the GUC fallback).
pub(crate) fn describe_impl(
    handle: &registry::ModelHandle,
    entity: &str,
    explicit_threshold: Option<f64>,
) -> Result<Vec<(String, String, f64, i32)>, PgInferError> {
    let entity_lower = entity.to_lowercase();

    // 1. Tokenize and build query vector.
    let encoding = handle
        .tokenizer
        .encode(entity, false)
        .map_err(|e| PgInferError::Tokenize(e.to_string()))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();

    if token_ids.is_empty() {
        return Err(PgInferError::EmptyPrompt);
    }

    let hidden = handle.config.hidden_size;
    let query: Array1<f32> = if token_ids.len() == 1 {
        handle
            .embeddings
            .row(token_ids[0] as usize)
            .mapv(|v| v * handle.embed_scale)
    } else {
        // Average token embeddings for multi-token entities.
        let mut avg = Array1::<f32>::zeros(hidden);
        for &tok in &token_ids {
            avg += &handle
                .embeddings
                .row(tok as usize)
                .mapv(|v| v * handle.embed_scale);
        }
        avg /= token_ids.len() as f32;
        avg
    };

    // 2. Walk all layers, collect raw hits first (we need them to compute
    //    the adaptive threshold before filtering).
    let top_k_per_layer = 20_usize;
    let num_layers = handle.config.num_layers;

    // Collect all (layer, feature_idx, gate_score) tuples.
    let mut all_hits: Vec<(usize, usize, f32)> = Vec::new();
    for layer in 0..num_layers {
        let hits = handle.gate_knn(layer, &query, top_k_per_layer);
        for (feature_idx, gate_score) in hits {
            all_hits.push((layer, feature_idx, gate_score));
        }
    }

    // 3. Determine effective threshold.
    let gate_threshold = resolve_threshold(explicit_threshold, &all_hits);

    // 4. Filter and accumulate edges by target token.
    // Map: lowercased target → (original target, best gate score, best layer, count)
    let mut edges: HashMap<String, (String, f32, usize, usize)> = HashMap::new();

    for &(layer, feature_idx, gate_score) in &all_hits {
        if gate_score < gate_threshold {
            continue;
        }
        let meta = match handle.feature_meta(layer, feature_idx) {
            Some(m) => m,
            None => continue,
        };

        let tok = &meta.top_token;

        // Skip non-content tokens and self-references.
        if !is_content_token(tok) {
            continue;
        }
        if tok.to_lowercase() == entity_lower {
            continue;
        }

        let key = tok.to_lowercase();
        let entry = edges
            .entry(key)
            .or_insert_with(|| (tok.clone(), 0.0, layer, 0));

        if gate_score > entry.1 {
            entry.1 = gate_score;
            entry.2 = layer;
        }
        entry.3 += 1;
    }

    // 5. Rank by gate score descending and format output.
    let mut ranked: Vec<_> = edges.into_values().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let results: Vec<(String, String, f64, i32)> = ranked
        .into_iter()
        .map(|(target, score, layer, _count)| {
            // Relation labelling requires a trained classifier (Phase 2).
            // For now we leave the relation column as an empty string.
            let relation = String::new();
            (relation, target, score as f64, layer as i32)
        })
        .collect();

    Ok(results)
}

/// Resolve the effective gate threshold from (in priority order):
/// 1. Explicit function parameter (if > 0)
/// 2. `infer.gate_threshold` GUC (if > 0)
/// 3. Adaptive: `max_score × 0.1` computed from the query's actual hits
fn resolve_threshold(explicit: Option<f64>, hits: &[(usize, usize, f32)]) -> f32 {
    // Explicit parameter takes priority.
    if let Some(t) = explicit {
        if t > 0.0 {
            return t as f32;
        }
    }

    // GUC fallback.
    let guc_val = gucs::GATE_THRESHOLD.get();
    if guc_val > 0.0 {
        return guc_val as f32;
    }

    // Adaptive: 10% of the maximum observed score.
    let max_score = hits
        .iter()
        .map(|&(_, _, s)| s)
        .fold(0.0_f32, f32::max);

    if max_score > 0.0 {
        max_score * 0.1
    } else {
        0.0
    }
}

/// Heuristic: a token is "content" if it contains at least one alphabetic
/// character and is longer than one byte.
fn is_content_token(tok: &str) -> bool {
    let trimmed = tok.trim();
    trimmed.len() > 1 && trimmed.chars().any(|c| c.is_alphabetic())
}
