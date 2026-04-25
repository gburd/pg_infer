use ndarray::Array1;
use pgrx::prelude::*;

use crate::error::PgInferError;
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

    // 2. Build the query vector from the last token's embedding, scaled.
    //    For multi-token prompts we use the last token (matching LQL's walk
    //    behaviour); a future version could average all tokens.
    let last_tok = token_ids[token_ids.len() - 1];
    let embed_row = handle.embeddings.row(last_tok as usize);
    let query: Array1<f32> = embed_row.mapv(|v| v * handle.embed_scale);

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
