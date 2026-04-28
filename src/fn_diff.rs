//! `infer_diff()` — compare two models' feature metadata.

use pgrx::prelude::*;

use crate::error::PgInferError;
use crate::registry;

/// Compare feature metadata between two models.
///
/// For each feature at each layer (or a single specified layer), reports
/// differences in the top token label or c_score.
///
/// ```sql
/// SELECT * FROM infer_diff('model_a', 'model_b');
/// SELECT * FROM infer_diff('model_a', 'model_b', layer => 14, top => 50);
/// ```
#[pg_extern]
fn infer_diff(
    model_a: &str,
    model_b: &str,
    layer: default!(Option<i32>, "NULL"),
    top: default!(Option<i32>, "NULL"),
) -> Result<
    TableIterator<
        'static,
        (
            name!(layer, i32),
            name!(feature, i32),
            name!(token_a, String),
            name!(token_b, String),
            name!(score_a, f64),
            name!(score_b, f64),
            name!(status, String),
        ),
    >,
    Box<dyn std::error::Error>,
> {
    let limit = top.unwrap_or(20) as usize;

    // Load model A's metadata into a local vector first, then release the
    // lock before loading model B.
    let a_data = registry::with_model(model_a, |handle_a| {
        collect_model_features(handle_a, layer)
    })?;

    let rows = registry::with_model(model_b, |handle_b| {
        diff_impl(&a_data, handle_b, layer, limit)
    })?;

    Ok(TableIterator::new(rows))
}

/// Per-feature snapshot from one model: (layer, feature, top_token, c_score).
struct FeatureSnapshot {
    layer: usize,
    feature: usize,
    top_token: String,
    c_score: f32,
}

/// Collect feature metadata from a model for comparison.
fn collect_model_features(
    handle: &registry::ModelHandle,
    layer_filter: Option<i32>,
) -> Result<Vec<FeatureSnapshot>, PgInferError> {
    let num_layers = handle.config.num_layers;
    let (start, end) = match layer_filter {
        Some(l) => {
            let l = l as usize;
            if l >= num_layers {
                return Err(PgInferError::Internal(format!(
                    "layer {} out of range (model has {} layers)",
                    l, num_layers
                )));
            }
            (l, l + 1)
        }
        None => (0, num_layers),
    };

    let mut features = Vec::new();
    for layer in start..end {
        let nf = handle.num_features(layer);
        for feat in 0..nf {
            if let Some(meta) = handle.feature_meta(layer, feat) {
                features.push(FeatureSnapshot {
                    layer,
                    feature: feat,
                    top_token: meta.top_token,
                    c_score: meta.c_score,
                });
            }
        }
    }

    Ok(features)
}

/// Compare model A's collected features against model B.
fn diff_impl(
    a_data: &[FeatureSnapshot],
    handle_b: &registry::ModelHandle,
    layer_filter: Option<i32>,
    limit: usize,
) -> Result<Vec<(i32, i32, String, String, f64, f64, String)>, PgInferError> {
    let num_layers_b = handle_b.config.num_layers;
    let mut results = Vec::new();

    for snap_a in a_data {
        if results.len() >= limit {
            break;
        }

        let layer = snap_a.layer;
        // Skip if model B doesn't have this layer.
        if layer >= num_layers_b {
            results.push((
                layer as i32,
                snap_a.feature as i32,
                snap_a.top_token.clone(),
                String::new(),
                snap_a.c_score as f64,
                0.0,
                "removed".to_string(),
            ));
            continue;
        }

        let nf_b = handle_b.num_features(layer);
        if snap_a.feature >= nf_b {
            results.push((
                layer as i32,
                snap_a.feature as i32,
                snap_a.top_token.clone(),
                String::new(),
                snap_a.c_score as f64,
                0.0,
                "removed".to_string(),
            ));
            continue;
        }

        match handle_b.feature_meta(layer, snap_a.feature) {
            Some(meta_b) => {
                let token_differs = snap_a.top_token != meta_b.top_token;
                let score_differs = (snap_a.c_score - meta_b.c_score).abs() > 0.01;

                if token_differs || score_differs {
                    results.push((
                        layer as i32,
                        snap_a.feature as i32,
                        snap_a.top_token.clone(),
                        meta_b.top_token,
                        snap_a.c_score as f64,
                        meta_b.c_score as f64,
                        "modified".to_string(),
                    ));
                }
            }
            None => {
                results.push((
                    layer as i32,
                    snap_a.feature as i32,
                    snap_a.top_token.clone(),
                    String::new(),
                    snap_a.c_score as f64,
                    0.0,
                    "removed".to_string(),
                ));
            }
        }
    }

    // Also check for features in B that don't exist in A (at the same layers).
    // Only if we haven't hit the limit yet.
    if results.len() < limit {
        let (start, end) = match layer_filter {
            Some(l) => (l as usize, l as usize + 1),
            None => (0, num_layers_b),
        };

        // Build a set of (layer, feature) pairs from A for fast lookup.
        let a_set: std::collections::HashSet<(usize, usize)> =
            a_data.iter().map(|s| (s.layer, s.feature)).collect();

        for layer in start..end.min(num_layers_b) {
            if results.len() >= limit {
                break;
            }
            // Only look for "added" features in layers beyond A's range
            // or features beyond A's count at that layer.
            let nf_b = handle_b.num_features(layer);
            for feat in 0..nf_b {
                if results.len() >= limit {
                    break;
                }
                if a_set.contains(&(layer, feat)) {
                    continue;
                }
                // Feature exists in B but not in A.
                if let Some(meta_b) = handle_b.feature_meta(layer, feat) {
                    results.push((
                        layer as i32,
                        feat as i32,
                        String::new(),
                        meta_b.top_token,
                        0.0,
                        meta_b.c_score as f64,
                        "added".to_string(),
                    ));
                }
            }
        }
    }

    Ok(results)
}
