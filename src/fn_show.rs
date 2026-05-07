//! SHOW functions: layer metadata, feature enumeration, relation discovery.

use std::collections::HashMap;

use pgrx::prelude::*;

use crate::error::PgInferError;
use crate::helpers;
use crate::registry;

// ---------------------------------------------------------------------------
// infer_show_layers()
// ---------------------------------------------------------------------------

/// Show layer metadata including band classification and feature counts.
///
/// ```sql
/// SELECT * FROM infer_show_layers();
/// SELECT * FROM infer_show_layers(model => 'qwen05b');
/// ```
#[pg_extern]
fn infer_show_layers(
    model: default!(Option<&str>, "NULL"),
) -> Result<
    TableIterator<
        'static,
        (
            name!(layer, i32),
            name!(band, String),
            name!(num_features, i32),
        ),
    >,
    Box<dyn std::error::Error>,
> {
    let model_name = registry::resolve_model_name(model)?;

    let rows = registry::with_model(&model_name, |handle| {
        let infos = mmap_show_layers(handle)?;
        Ok(infos
            .into_iter()
            .map(|i| (i.layer, i.band, i.num_features))
            .collect::<Vec<_>>())
    })?;

    Ok(TableIterator::new(rows))
}

pub(crate) fn mmap_show_layers(
    handle: &registry::ModelHandle,
) -> Result<Vec<crate::backend::LayerInfo>, PgInferError> {
    let num_layers = handle.config.num_layers;
    let bands = &handle.config.layer_bands;

    let mut results = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        let nf = handle.num_features(layer);
        let band = bands
            .as_ref()
            .map(|b| b.band_for_layer(layer).to_string())
            .unwrap_or_default();

        results.push(crate::backend::LayerInfo {
            layer: layer as i32,
            band,
            num_features: nf as i32,
        });
    }
    Ok(results)
}

// ---------------------------------------------------------------------------
// infer_show_features()
// ---------------------------------------------------------------------------

/// Enumerate features at a given layer, optionally filtered by token name.
///
/// ```sql
/// SELECT * FROM infer_show_features(20);
/// SELECT * FROM infer_show_features(20, filter => 'capital', top => 50);
/// ```
#[pg_extern]
fn infer_show_features(
    layer: i32,
    filter: default!(Option<&str>, "NULL"),
    min_score: default!(Option<f64>, "NULL"),
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
    let layer_idx = layer as usize;
    let limit = top.unwrap_or(50) as usize;
    let filter_lower = filter.map(|f| f.to_lowercase());
    let min_c = min_score.unwrap_or(0.0) as f32;

    let rows = registry::with_model(&model_name, |handle| {
        let feats = mmap_show_features(handle, layer_idx, filter_lower.as_deref(), min_c, limit)?;
        Ok(feats
            .into_iter()
            .map(|f| (f.feature, f.token, f.score, f.also))
            .collect::<Vec<_>>())
    })?;

    Ok(TableIterator::new(rows))
}

pub(crate) fn mmap_show_features(
    handle: &registry::ModelHandle,
    layer: usize,
    filter: Option<&str>,
    min_score: f32,
    limit: usize,
) -> Result<Vec<crate::backend::FeatureRow>, PgInferError> {
    if layer >= handle.config.num_layers {
        return Err(PgInferError::Internal(format!(
            "layer {} out of range (model has {} layers)",
            layer, handle.config.num_layers
        )));
    }

    let nf = handle.num_features(layer);
    let mut results = Vec::new();

    for feat in 0..nf {
        let meta = match handle.feature_meta(layer, feat) {
            Some(m) => m,
            None => continue,
        };

        if meta.c_score < min_score {
            continue;
        }

        if let Some(f) = filter {
            if !meta.top_token.to_lowercase().contains(f) {
                continue;
            }
        }

        let also: String = meta
            .top_k
            .iter()
            .filter(|e| e.logit > 0.0 && helpers::is_readable_token(&e.token))
            .take(3)
            .map(|e| e.token.clone())
            .collect::<Vec<_>>()
            .join(", ");

        results.push(crate::backend::FeatureRow {
            feature: feat as i32,
            token: meta.top_token,
            score: meta.c_score as f64,
            also,
        });

        if results.len() >= limit {
            break;
        }
    }

    Ok(results)
}

// ---------------------------------------------------------------------------
// infer_show_relations()
// ---------------------------------------------------------------------------

/// Discover relation-like tokens across the model's features.
///
/// Iterates all layers (or the knowledge band if available), aggregates
/// content tokens by lowercased form, and returns the most frequent ones.
///
/// ```sql
/// SELECT * FROM infer_show_relations();
/// SELECT * FROM infer_show_relations(model => 'llama8b');
/// ```
#[pg_extern]
fn infer_show_relations(
    model: default!(Option<&str>, "NULL"),
) -> Result<
    TableIterator<
        'static,
        (
            name!(relation, String),
            name!(count, i32),
            name!(max_score, f64),
            name!(layers, String),
            name!(examples, String),
        ),
    >,
    Box<dyn std::error::Error>,
> {
    let model_name = registry::resolve_model_name(model)?;

    let rows = registry::with_model(&model_name, |handle| {
        let rels = mmap_show_relations(handle)?;
        Ok(rels
            .into_iter()
            .map(|r| (r.relation, r.count, r.max_score, r.layers, r.examples))
            .collect::<Vec<_>>())
    })?;

    Ok(TableIterator::new(rows))
}

/// Aggregation state for a discovered relation token.
struct RelationAgg {
    count: usize,
    max_score: f32,
    layers: Vec<usize>,
    examples: Vec<String>,
}

pub(crate) fn mmap_show_relations(
    handle: &registry::ModelHandle,
) -> Result<Vec<crate::backend::RelationRow>, PgInferError> {
    let num_layers = handle.config.num_layers;

    // Determine layer range: prefer knowledge band if available.
    let (start, end) = match &handle.config.layer_bands {
        Some(bands) => (bands.knowledge.0, bands.knowledge.1 + 1),
        None => (0, num_layers),
    };

    let mut agg: HashMap<String, RelationAgg> = HashMap::new();

    for layer in start..end.min(num_layers) {
        let nf = handle.num_features(layer);
        for feat in 0..nf {
            let meta = match handle.feature_meta(layer, feat) {
                Some(m) => m,
                None => continue,
            };

            if meta.c_score < 0.2 {
                continue;
            }

            let tok = &meta.top_token;
            if !helpers::is_content_token(tok) {
                continue;
            }

            let key = tok.to_lowercase();
            let entry = agg.entry(key).or_insert_with(|| RelationAgg {
                count: 0,
                max_score: 0.0,
                layers: Vec::new(),
                examples: Vec::new(),
            });

            entry.count += 1;
            if meta.c_score > entry.max_score {
                entry.max_score = meta.c_score;
            }
            if !entry.layers.contains(&layer) {
                entry.layers.push(layer);
            }
            // Collect example secondaries (up to 3 total).
            if entry.examples.len() < 3 {
                for e in meta.top_k.iter().take(2) {
                    if e.logit > 0.0
                        && helpers::is_readable_token(&e.token)
                        && !entry.examples.contains(&e.token)
                        && entry.examples.len() < 3
                    {
                        entry.examples.push(e.token.clone());
                    }
                }
            }
        }
    }

    // Sort by count descending, limit to 30.
    let mut ranked: Vec<_> = agg.into_iter().collect();
    ranked.sort_by(|a, b| b.1.count.cmp(&a.1.count));
    ranked.truncate(30);

    let results = ranked
        .into_iter()
        .map(|(token, agg)| {
            let layers_str = agg
                .layers
                .iter()
                .map(|l| l.to_string())
                .collect::<Vec<_>>()
                .join(",");
            let examples_str = agg.examples.join(", ");
            crate::backend::RelationRow {
                relation: token,
                count: agg.count as i32,
                max_score: agg.max_score as f64,
                layers: layers_str,
                examples: examples_str,
            }
        })
        .collect();

    Ok(results)
}
