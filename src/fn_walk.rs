use ndarray::Array1;
use pgrx::prelude::*;

use crate::error::PgInferError;
use crate::helpers;
use crate::registry;

/// Trace model activations for a prompt, returning the top-K features
/// that fire at each layer.
///
/// ```sql
/// SELECT * FROM walk('The capital of France is', top => 10);
/// SELECT * FROM walk('Hello world', top => 5, model => 'qwen05b');
/// ```
#[pg_extern]
fn walk(
    prompt: &str,
    top: default!(Option<i32>, "NULL"),
    model: default!(Option<&str>, "NULL"),
) -> Result<
    TableIterator<
        'static,
        (
            name!(layer, i32),
            name!(feature, i32),
            name!(activation, f64),
            name!(concept, String),
        ),
    >,
    Box<dyn std::error::Error>,
> {
    let model_name = registry::resolve_model_name(model)?;
    let top_k = top.unwrap_or(20) as usize;

    let rows = registry::with_model(&model_name, |handle| {
        walk_impl(handle, prompt, top_k)
    })?;

    Ok(TableIterator::new(rows))
}

fn walk_impl(
    handle: &registry::ModelHandle,
    prompt: &str,
    top_k: usize,
) -> Result<Vec<(i32, i32, f64, String)>, PgInferError> {
    // 1. Tokenize the prompt.
    let encoding = handle
        .tokenizer
        .encode(prompt, true)
        .map_err(|e| PgInferError::Tokenize(e.to_string()))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();

    if token_ids.is_empty() {
        return Err(PgInferError::EmptyPrompt);
    }

    // 2. Build the query vector.
    //    By default, use the last token's embedding (controlled by
    //    `infer.walk_embed_mode` GUC, default "last").  This matches
    //    the LARQL CLI behaviour.  "average" mode averages all token
    //    embeddings instead.
    let query: Array1<f32> = if crate::gucs::walk_embed_mode_is_last() {
        let last_tok = token_ids[token_ids.len() - 1];
        handle
            .embeddings
            .row(last_tok as usize)
            .mapv(|v| v * handle.embed_scale)
    } else {
        crate::fn_similar::embed_text(handle, prompt)?
    };

    // 3. Scan every owned layer, collect top-K features.
    let num_layers = handle.config.num_layers;
    let mut results = Vec::new();

    for layer in 0..num_layers {
        let hits = handle.gate_knn(layer, &query, top_k);
        for (feature_idx, gate_score) in hits {
            let concept = handle
                .feature_meta(layer, feature_idx)
                .map(|m| m.top_token.clone())
                .unwrap_or_default();

            results.push((
                layer as i32,
                feature_idx as i32,
                gate_score as f64,
                concept,
            ));
        }
    }

    Ok(results)
}

// ---------------------------------------------------------------------------
// infer_explain_walk()
// ---------------------------------------------------------------------------

/// Annotated walk trace: same as `walk()` but adds band and secondary token
/// columns for richer exploration.
///
/// ```sql
/// SELECT * FROM infer_explain_walk('The capital of France is', top => 5);
/// SELECT * FROM infer_explain_walk('France', top => 3, model => 'llama8b');
/// ```
#[pg_extern]
fn infer_explain_walk(
    prompt: &str,
    top: default!(Option<i32>, "NULL"),
    model: default!(Option<&str>, "NULL"),
) -> Result<
    TableIterator<
        'static,
        (
            name!(layer, i32),
            name!(band, String),
            name!(feature, i32),
            name!(activation, f64),
            name!(token, String),
            name!(also, String),
        ),
    >,
    Box<dyn std::error::Error>,
> {
    let model_name = registry::resolve_model_name(model)?;
    let top_k = top.unwrap_or(5) as usize;

    let rows = registry::with_model(&model_name, |handle| {
        explain_walk_impl(handle, prompt, top_k)
    })?;

    Ok(TableIterator::new(rows))
}

fn explain_walk_impl(
    handle: &registry::ModelHandle,
    prompt: &str,
    top_k: usize,
) -> Result<Vec<(i32, String, i32, f64, String, String)>, PgInferError> {
    // 1. Tokenize the prompt.
    let encoding = handle
        .tokenizer
        .encode(prompt, true)
        .map_err(|e| PgInferError::Tokenize(e.to_string()))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();

    if token_ids.is_empty() {
        return Err(PgInferError::EmptyPrompt);
    }

    // 2. Build the query vector (same as walk).
    let query: Array1<f32> = if crate::gucs::walk_embed_mode_is_last() {
        let last_tok = token_ids[token_ids.len() - 1];
        handle
            .embeddings
            .row(last_tok as usize)
            .mapv(|v| v * handle.embed_scale)
    } else {
        crate::fn_similar::embed_text(handle, prompt)?
    };

    // 3. Scan every layer, collect top-K with band + also annotations.
    let num_layers = handle.config.num_layers;
    let bands = &handle.config.layer_bands;
    let mut results = Vec::new();

    for layer in 0..num_layers {
        let band = bands
            .as_ref()
            .map(|b| b.band_for_layer(layer).to_string())
            .unwrap_or_default();

        let hits = handle.gate_knn(layer, &query, top_k);
        for (feature_idx, gate_score) in hits {
            let meta = match handle.feature_meta(layer, feature_idx) {
                Some(m) => m,
                None => continue,
            };

            let also: String = meta
                .top_k
                .iter()
                .filter(|e| e.logit > 0.0 && helpers::is_readable_token(&e.token))
                .take(3)
                .map(|e| e.token.clone())
                .collect::<Vec<_>>()
                .join(", ");

            results.push((
                layer as i32,
                band.clone(),
                feature_idx as i32,
                gate_score as f64,
                meta.top_token,
                also,
            ));
        }
    }

    Ok(results)
}
