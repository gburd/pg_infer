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

    let a_data = registry::with_backend(model_a, |backend| {
        backend.snapshot_features(layer)
    })?;

    // We need per-feature metadata from B to synthesize diff rows.  The
    // trait exposes `feature_meta_at` — remote backends return `None`
    // there, which the differ treats as "feature missing" and records
    // as a removal.  Snapshot call also rejects remote backends, which
    // is correct: diff requires enumeration only the mmap path offers.
    let rows = registry::with_backend(model_b, |backend_b| {
        diff_impl(&a_data, backend_b, layer, limit)
    })?;

    Ok(TableIterator::new(rows))
}

/// Collect feature metadata from a model for comparison.
pub(crate) fn mmap_snapshot_features(
    handle: &registry::ModelHandle,
    layer_filter: Option<i32>,
) -> Result<Vec<crate::backend::FeatureSnapshot>, PgInferError> {
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
                features.push(crate::backend::FeatureSnapshot {
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
    a_data: &[crate::backend::FeatureSnapshot],
    backend_b: &dyn crate::backend::Backend,
    layer_filter: Option<i32>,
    limit: usize,
) -> Result<Vec<(i32, i32, String, String, f64, f64, String)>, PgInferError> {
    let num_layers_b = backend_b.num_layers();
    let mut results = Vec::new();

    for snap_a in a_data {
        if results.len() >= limit {
            break;
        }

        let layer = snap_a.layer;
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

        match backend_b.feature_meta_at(layer, snap_a.feature) {
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

    // Also check for features in B that don't exist in A.  Uses B's
    // snapshot; rejects if B is remote.
    if results.len() < limit {
        let b_snapshot = backend_b.snapshot_features(layer_filter)?;
        let a_set: std::collections::HashSet<(usize, usize)> =
            a_data.iter().map(|s| (s.layer, s.feature)).collect();

        for snap_b in &b_snapshot {
            if results.len() >= limit {
                break;
            }
            if a_set.contains(&(snap_b.layer, snap_b.feature)) {
                continue;
            }
            results.push((
                snap_b.layer as i32,
                snap_b.feature as i32,
                String::new(),
                snap_b.top_token.clone(),
                0.0,
                snap_b.c_score as f64,
                "added".to_string(),
            ));
        }
    }

    Ok(results)
}
