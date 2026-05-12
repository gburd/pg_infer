//! Thread safety tests for VectorIndex under concurrent access.
//!
//! Verifies that multiple threads can call gate_knn and feature_meta
//! simultaneously without data races, panics, or incorrect results.

use std::sync::Arc;

use infer_vindex::{FeatureMeta, VectorIndex};
use ndarray::{Array1, Array2};

fn make_meta(token: &str, id: u32, score: f32) -> FeatureMeta {
    FeatureMeta {
        top_token: token.to_string(),
        top_token_id: id,
        c_score: score,
        top_k: vec![infer_models::TopKEntry {
            token: token.to_string(),
            token_id: id,
            logit: score,
        }],
        relation: None,
    }
}

/// Build a larger in-memory VectorIndex suitable for concurrent stress testing.
fn create_test_index() -> VectorIndex {
    let hidden = 32;
    let num_features = 64;
    let num_layers = 4;

    let mut gate_vectors = Vec::new();
    let mut down_meta = Vec::new();

    for layer in 0..num_layers {
        // Deterministic but varied gate vectors per layer
        let mut gate = Array2::<f32>::zeros((num_features, hidden));
        for feat in 0..num_features {
            for dim in 0..hidden {
                let seed = (layer * 1000 + feat * 100 + dim) as f32;
                gate[[feat, dim]] = (seed * 0.01).sin();
            }
        }
        gate_vectors.push(Some(gate));

        let metas: Vec<Option<FeatureMeta>> = (0..num_features)
            .map(|f| {
                Some(make_meta(
                    &format!("token_L{}_F{}", layer, f),
                    (layer * 100 + f) as u32,
                    0.5 + (f as f32) * 0.01,
                ))
            })
            .collect();
        down_meta.push(Some(metas));
    }

    VectorIndex::new(gate_vectors, down_meta, num_layers, hidden)
}

#[test]
fn concurrent_gate_knn_no_panic() {
    let index = Arc::new(create_test_index());
    let num_threads = 8;
    let iterations = 100;

    let handles: Vec<_> = (0..num_threads)
        .map(|thread_id| {
            let idx = Arc::clone(&index);
            std::thread::spawn(move || {
                for iter in 0..iterations {
                    // Generate a different query each iteration
                    let query = Array1::from_vec(
                        (0..32)
                            .map(|d| ((thread_id * 1000 + iter * 10 + d) as f32 * 0.03).cos())
                            .collect(),
                    );

                    // Query all layers
                    for layer in 0..4 {
                        let results = idx.gate_knn(layer, &query, 10);
                        // Basic sanity: results are non-empty and bounded
                        assert!(
                            results.len() <= 10,
                            "gate_knn returned more than top_k results"
                        );
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().expect("thread panicked during concurrent gate_knn");
    }
}

#[test]
fn concurrent_feature_meta_consistent() {
    let index = Arc::new(create_test_index());
    let num_threads = 8;
    let iterations = 50;

    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let idx = Arc::clone(&index);
            std::thread::spawn(move || {
                for _ in 0..iterations {
                    for layer in 0..4 {
                        for feat in 0..64 {
                            let meta = idx.feature_meta(layer, feat);
                            // Metadata should always be Some for our test fixture
                            assert!(
                                meta.is_some(),
                                "feature_meta(L{}, F{}) unexpectedly None",
                                layer,
                                feat
                            );
                            let m = meta.unwrap();
                            // Token should match deterministic pattern
                            let expected = format!("token_L{}_F{}", layer, feat);
                            assert_eq!(
                                m.top_token, expected,
                                "metadata inconsistent under concurrent access"
                            );
                        }
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join()
            .expect("thread panicked during concurrent feature_meta");
    }
}

#[test]
fn concurrent_gate_knn_results_are_sorted_by_abs() {
    let index = Arc::new(create_test_index());
    let num_threads = 4;
    let iterations = 200;

    let handles: Vec<_> = (0..num_threads)
        .map(|thread_id| {
            let idx = Arc::clone(&index);
            std::thread::spawn(move || {
                for iter in 0..iterations {
                    let query = Array1::from_vec(
                        (0..32)
                            .map(|d| ((thread_id * 7 + iter * 3 + d) as f32 * 0.05).sin())
                            .collect(),
                    );
                    let layer = (thread_id + iter) % 4;
                    let results = idx.gate_knn(layer, &query, 10);

                    // Results should be sorted by absolute score descending
                    // (gate_knn may return negative dot products)
                    for window in results.windows(2) {
                        assert!(
                            window[0].1.abs() >= window[1].1.abs(),
                            "gate_knn results not sorted by |score|: |{}| < |{}|",
                            window[0].1,
                            window[1].1
                        );
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join()
            .expect("thread panicked during concurrent sorted check");
    }
}
