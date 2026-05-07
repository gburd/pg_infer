use pgrx::prelude::*;

use crate::error::PgInferError;
use crate::registry;

/// Run a full forward pass on the prompt and return the top-K predicted
/// next tokens with their probabilities.
///
/// Requires the model to have been extracted at the `inference` level
/// (i.e., full FFN weights are available in the vindex).
///
/// **Build note:** This function requires the `inference` cargo feature
/// (`--features inference`) which pulls in `infer-inference`.  Without
/// that feature the function always returns an error.
///
/// ```sql
/// SELECT * FROM infer('The capital of France is', top => 5);
/// SELECT * FROM infer('Product category:', top => 3, model => 'gemma4b');
/// ```
#[pg_extern]
fn infer(
    prompt: &str,
    top: default!(Option<i32>, "NULL"),
    model: default!(Option<&str>, "NULL"),
) -> Result<
    TableIterator<
        'static,
        (
            name!(token, String),
            name!(probability, f64),
            name!(rank, i32),
        ),
    >,
    Box<dyn std::error::Error>,
> {
    let model_name = registry::resolve_model_name(model)?;
    let top_k = top.unwrap_or(5) as usize;

    let rows = registry::with_model(&model_name, |handle| {
        let preds = mmap_infer(handle, prompt, top_k)?;
        Ok(preds
            .into_iter()
            .map(|p| (p.token, p.probability, p.rank))
            .collect::<Vec<_>>())
    })?;

    Ok(TableIterator::new(rows))
}

#[cfg(feature = "inference")]
pub(crate) fn mmap_infer(
    handle: &registry::ModelHandle,
    prompt: &str,
    top_k: usize,
) -> Result<Vec<crate::backend::Prediction>, PgInferError> {
    use infer_vindex::ExtractLevel;

    // Verify the vindex has inference-level weights.
    let level = handle.config.extract_level;
    if level != ExtractLevel::Inference && level != ExtractLevel::All {
        return Err(PgInferError::InsufficientExtractLevel {
            needed: "inference".to_string(),
            have: format!("{:?}", level).to_lowercase(),
        });
    }

    // Tokenize the prompt.
    let encoding = handle
        .tokenizer
        .encode(prompt, true)
        .map_err(|e| PgInferError::Tokenize(e.to_string()))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();

    if token_ids.is_empty() {
        return Err(PgInferError::EmptyPrompt);
    }

    // Load the full model weights for inference.
    let weights = infer_inference::load_model_dir(&handle.path)
        .map_err(|e| PgInferError::Internal(format!("failed to load model weights: {}", e)))?;

    // Run the forward pass.
    let result = infer_inference::predict(&weights, &handle.tokenizer, &token_ids, top_k);

    // Format as Prediction records.
    let rows: Vec<crate::backend::Prediction> = result
        .predictions
        .into_iter()
        .enumerate()
        .map(|(i, (token, prob))| crate::backend::Prediction {
            token,
            probability: prob,
            rank: (i + 1) as i32,
        })
        .collect();

    Ok(rows)
}

#[cfg(not(feature = "inference"))]
pub(crate) fn mmap_infer(
    _handle: &registry::ModelHandle,
    _prompt: &str,
    _top_k: usize,
) -> Result<Vec<crate::backend::Prediction>, PgInferError> {
    Err(PgInferError::Internal(
        "infer() requires the 'inference' feature — \
         rebuild with: cargo pgrx run --features inference"
            .to_string(),
    ))
}
