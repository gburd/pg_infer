//! `nearest_to()` — single-layer gate KNN probe.
//!
//! Implements LARQL's `SELECT ... NEAREST TO <entity> AT LAYER <n>`.

use ndarray::Array1;
use pgrx::prelude::*;

use crate::error::PgInferError;
use crate::helpers;
use crate::registry;

/// Probe a single layer's gate vectors for the nearest features to an entity.
///
/// Returns raw results — no filtering, no dedup.  This is a low-level
/// diagnostic tool for exploring what a specific layer encodes.
///
/// ```sql
/// SELECT * FROM nearest_to('France', layer => 20, top => 20);
/// SELECT * FROM nearest_to('France', layer => 20, model => 'llama8b');
/// ```
#[pg_extern]
fn nearest_to(
    entity: &str,
    layer: i32,
    top: default!(Option<i32>, "NULL"),
    model: default!(Option<&str>, "NULL"),
) -> Result<
    TableIterator<
        'static,
        (
            name!(feature, i32),
            name!(token, String),
            name!(score, f64),
            name!(also, String),
        ),
    >,
    Box<dyn std::error::Error>,
> {
    let model_name = registry::resolve_model_name(model)?;
    let top_k = top.unwrap_or(20) as usize;
    let layer_idx = layer as usize;

    let rows = registry::with_backend(&model_name, |backend| {
        let hits = backend.nearest_to(entity, layer_idx, top_k)?;
        Ok(hits
            .into_iter()
            .map(|h| (h.feature, h.concept, h.gate_score, h.also))
            .collect::<Vec<_>>())
    })?;

    Ok(TableIterator::new(rows))
}

pub(crate) fn mmap_nearest_to(
    handle: &registry::ModelHandle,
    entity: &str,
    layer: usize,
    top_k: usize,
) -> Result<Vec<crate::backend::Hit>, PgInferError> {
    if layer >= handle.config.num_layers {
        return Err(PgInferError::Internal(format!(
            "layer {} out of range (model has {} layers)",
            layer, handle.config.num_layers
        )));
    }

    // 1. Tokenize entity, compute average token embedding (scaled).
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

    // 2. gate_knn at the specified layer.
    let hits = handle.gate_knn(layer, &query, top_k);

    // 3. Build result rows with secondary token info.
    let mut results = Vec::with_capacity(hits.len());
    for (feature_idx, gate_score) in hits {
        let meta = match handle.feature_meta(layer, feature_idx) {
            Some(m) => m,
            None => continue,
        };

        // "also" column: top 3 readable secondary tokens with positive logit.
        let also: String = meta
            .top_k
            .iter()
            .filter(|e| e.logit > 0.0 && helpers::is_readable_token(&e.token))
            .take(3)
            .map(|e| e.token.clone())
            .collect::<Vec<_>>()
            .join(", ");

        results.push(crate::backend::Hit {
            layer: layer as i32,
            feature: feature_idx as i32,
            gate_score: gate_score as f64,
            concept: meta.top_token,
            also,
        });
    }

    Ok(results)
}
