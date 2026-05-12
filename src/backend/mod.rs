//! Backend abstraction for pg_infer.
//!
//! Two implementations:
//! - [`mmap::MmapBackend`] — loads a vindex directory into memory-mapped
//!   regions and runs gate-KNN and feature lookups in-process.  This is
//!   the path that has existed since day one.
//! - [`remote::RemoteBackend`] — calls a `larql-server` over HTTP/2 for
//!   the high-level operations (`/v1/describe`, `/v1/walk`, `/v1/stats`,
//!   `/v1/relations`, `/v1/infer`).  The server owns the mmap and the
//!   activation cache, so multiple PG backends share one copy.
//!
//! The `Backend` trait exposes a common high-level surface.  Each
//! pg_infer SQL function talks to a `&dyn Backend`; the trait dispatches
//! to the right implementation at runtime.
//!
//! # Why a high-level trait (not low-level primitives)
//!
//! The mmap path naturally exposes `gate_knn(layer, query, top_k)` and
//! `feature_meta(layer, feat)`.  The remote server does not: its API is
//! the *composed* operation (`describe`, `walk`), because that's what
//! the activation cache keys on and what `larql-router` fans out.
//! Forcing the remote backend to emulate per-layer primitives would
//! defeat the cache and blow up round-trip counts.  Instead we let each
//! backend assemble the result its own way.

#![allow(dead_code)] // Some trait methods are only used on the non-active backend at build time.

pub mod grid;
pub mod mmap;
pub mod remote;

use ndarray::Array1;

use crate::error::PgInferError;

/// One edge emitted by `describe()` — a target token the entity gates on,
/// plus the strongest layer + score.
#[derive(Debug, Clone)]
pub struct Edge {
    pub relation: String,
    pub target: String,
    pub gate_score: f64,
    pub layer: i32,
}

/// One feature activation emitted by `walk()` / `nearest_to()`.
#[derive(Debug, Clone)]
pub struct Hit {
    pub layer: i32,
    pub feature: i32,
    pub gate_score: f64,
    pub concept: String,
    /// Optional "also" column — up to three readable secondary tokens,
    /// comma-separated.  Populated by `nearest_to`; empty for the bulk
    /// `walk()` hits.
    pub also: String,
}

/// Richer variant used by `infer_explain_walk()` — adds band + "also"
/// secondary tokens.
#[derive(Debug, Clone)]
pub struct ExplainedHit {
    pub layer: i32,
    pub band: String,
    pub feature: i32,
    pub gate_score: f64,
    pub token: String,
    pub also: String,
}

/// One per-layer activation from `describe_layers()` — not deduplicated.
#[derive(Debug, Clone)]
pub struct LayerHit {
    pub layer: i32,
    pub feature: i32,
    pub target: String,
    pub gate_score: f64,
    pub also: String,
}

/// Per-layer metadata used by `infer_show_layers()`.
#[derive(Debug, Clone)]
pub struct LayerInfo {
    pub layer: i32,
    pub band: String,
    pub num_features: i32,
}

/// One row emitted by `infer_show_relations()`.
#[derive(Debug, Clone)]
pub struct RelationRow {
    pub relation: String,
    pub count: i32,
    pub max_score: f64,
    pub layers: String,
    pub examples: String,
}

/// One feature row from `infer_show_features(layer)`.  Only the mmap
/// backend can enumerate features; remote returns `Unsupported`.
#[derive(Debug, Clone)]
pub struct FeatureRow {
    pub feature: i32,
    pub token: String,
    pub score: f64,
    pub also: String,
}

/// Prediction from `infer(prompt)`.
#[derive(Debug, Clone)]
pub struct Prediction {
    pub token: String,
    pub probability: f64,
    pub rank: i32,
}

/// Common interface every backend implements.
///
/// All methods take `&self` and are expected to be thread-safe.  A
/// cancel-signal parameter is threaded through for the remote backend;
/// mmap ignores it but matches the signature so the caller is uniform.
pub trait Backend: Send + Sync {
    // ── Model metadata ────────────────────────────────────────────────

    fn num_layers(&self) -> usize;

    fn hidden_size(&self) -> usize;

    fn show_layers(&self) -> Result<Vec<LayerInfo>, PgInferError>;

    // ── Query operations ──────────────────────────────────────────────

    fn describe(
        &self,
        entity: &str,
        explicit_threshold: Option<f64>,
    ) -> Result<Vec<Edge>, PgInferError>;

    fn walk(&self, prompt: &str, top_k: usize) -> Result<Vec<Hit>, PgInferError>;

    fn explain_walk(&self, prompt: &str, top_k: usize) -> Result<Vec<ExplainedHit>, PgInferError>;

    fn nearest_to(
        &self,
        entity: &str,
        layer: usize,
        top_k: usize,
    ) -> Result<Vec<Hit>, PgInferError>;

    fn similar_to(&self, a: &str, b: &str) -> Result<f64, PgInferError>;

    /// Compute `similar_to(cand, query)` for each candidate.  Default
    /// implementation just loops; backends that can overlap network
    /// round trips (remote) override this to fan out concurrently.
    fn similar_to_many(&self, candidates: &[String], query: &str) -> Result<Vec<f64>, PgInferError> {
        candidates.iter().map(|c| self.similar_to(c, query)).collect()
    }

    fn implies(&self, subject: &str, object: &str) -> Result<bool, PgInferError>;

    /// Return all per-layer activations without deduplication.
    /// Local-only; remote backends return `RemoteUnsupported`.
    fn describe_layers(
        &self,
        _entity: &str,
        _explicit_threshold: Option<f64>,
    ) -> Result<Vec<LayerHit>, PgInferError> {
        Err(PgInferError::RemoteUnsupported {
            operation: "describe_layers".into(),
        })
    }

    fn infer(&self, prompt: &str, top_k: usize) -> Result<Vec<Prediction>, PgInferError>;

    fn show_relations(&self) -> Result<Vec<RelationRow>, PgInferError>;

    // ── Local-only operations ────────────────────────────────────────
    //
    // These require feature-level enumeration that larql-server doesn't
    // expose.  Remote backends return `PgInferError::RemoteUnsupported`.

    fn show_features(
        &self,
        layer: usize,
        filter: Option<&str>,
        min_score: f32,
        limit: usize,
    ) -> Result<Vec<FeatureRow>, PgInferError>;

    /// Used by `infer_diff()`.  Remote backends return `Unsupported`.
    fn snapshot_features(&self, layer_filter: Option<i32>)
        -> Result<Vec<FeatureSnapshot>, PgInferError>;

    /// Used by `infer_diff()`.  Remote backends return `Unsupported`.
    fn feature_meta_at(&self, layer: usize, feature: usize) -> Option<FeatureMetaLite>;

    // ── Low-level primitives (mmap only, used by internal helpers) ────
    //
    // These are exposed so the existing fn_similar precompute helper
    // keeps working.  Remote backends panic if called; the caller is
    // expected to gate on `is_local()`.

    /// Approximate RSS contribution of this backend (bytes).
    /// Used by the LRU cache for eviction decisions.
    /// Default: 0 (remote backends use negligible local memory).
    fn approx_resident_bytes(&self) -> usize {
        0
    }

    fn is_local(&self) -> bool {
        false
    }

    /// Return the (embedding vector, scale) for a text.  Only implemented
    /// for the mmap backend; the remote path uses server-side walk().
    fn embed(&self, _text: &str) -> Result<Array1<f32>, PgInferError> {
        Err(PgInferError::RemoteUnsupported {
            operation: "embed".into(),
        })
    }

    /// Pre-warm entities in the server-side activation cache.
    /// Only remote/grid backends implement this; local returns (0, 0).
    fn warmup(&self, _entities: &[String]) -> Result<(usize, usize), PgInferError> {
        Ok((0, 0))
    }

    /// Fetch server-side cache stats.  Local backends return `None`.
    fn cache_stats(&self) -> Result<Option<CacheStats>, PgInferError> {
        Ok(None)
    }

    /// Rank candidates by similarity to query, returning top-K sorted
    /// descending by score.  Remote backends can override to use a
    /// server-side `/v1/rank` endpoint with activation cache.
    ///
    /// Default implementation: score all via `similar_to_many`, sort, truncate.
    fn rank(
        &self,
        candidates: &[String],
        query: &str,
        limit: usize,
    ) -> Result<Vec<RankedCandidate>, PgInferError> {
        let scores = self.similar_to_many(candidates, query)?;
        let mut ranked: Vec<RankedCandidate> = scores
            .into_iter()
            .enumerate()
            .map(|(i, score)| RankedCandidate { index: i, score })
            .collect();
        ranked.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if limit > 0 && ranked.len() > limit {
            ranked.truncate(limit);
        }
        Ok(ranked)
    }
}

/// Ranked candidate with position and similarity score.
#[derive(Debug, Clone)]
pub struct RankedCandidate {
    /// Position in the input candidates array.
    pub index: usize,
    /// Similarity score (higher = more similar).
    pub score: f64,
}

/// Server-side activation cache statistics.
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub entries: usize,
    pub hit_count: u64,
    pub miss_count: u64,
    pub eviction_count: u64,
    pub memory_bytes: usize,
}

/// Minimal snapshot used by `infer_diff()`.
#[derive(Debug, Clone)]
pub struct FeatureSnapshot {
    pub layer: usize,
    pub feature: usize,
    pub top_token: String,
    pub c_score: f32,
}

/// Minimal metadata lookup for diff.
#[derive(Debug, Clone)]
pub struct FeatureMetaLite {
    pub top_token: String,
    pub c_score: f32,
}

// ---------------------------------------------------------------------------
// Backend contract tests — validates that ANY Backend implementation
// satisfies the expected behavioral guarantees.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod contract_tests {
    use super::*;

    /// A deterministic mock backend that returns predictable results for
    /// contract verification without needing a real vindex file or server.
    struct MockBackend;

    impl Backend for MockBackend {
        fn num_layers(&self) -> usize {
            4
        }

        fn hidden_size(&self) -> usize {
            128
        }

        fn show_layers(&self) -> Result<Vec<LayerInfo>, PgInferError> {
            Ok(vec![
                LayerInfo { layer: 0, band: "syntax".into(), num_features: 100 },
                LayerInfo { layer: 1, band: "syntax".into(), num_features: 100 },
                LayerInfo { layer: 2, band: "knowledge".into(), num_features: 100 },
                LayerInfo { layer: 3, band: "output".into(), num_features: 100 },
            ])
        }

        fn describe(
            &self,
            entity: &str,
            _explicit_threshold: Option<f64>,
        ) -> Result<Vec<Edge>, PgInferError> {
            // Return edges related to the entity
            Ok(vec![
                Edge {
                    relation: "is_a".into(),
                    target: format!("{entity}_type"),
                    gate_score: 0.9,
                    layer: 2,
                },
                Edge {
                    relation: "has_property".into(),
                    target: format!("{entity}_prop"),
                    gate_score: 0.7,
                    layer: 3,
                },
            ])
        }

        fn walk(&self, _prompt: &str, top_k: usize) -> Result<Vec<Hit>, PgInferError> {
            let mut hits: Vec<Hit> = (0..top_k.min(10))
                .map(|i| Hit {
                    layer: (i % 4) as i32,
                    feature: i as i32,
                    gate_score: 1.0 - (i as f64) * 0.1,
                    concept: format!("concept_{i}"),
                    also: String::new(),
                })
                .collect();
            hits.sort_by(|a, b| b.gate_score.partial_cmp(&a.gate_score).unwrap());
            Ok(hits)
        }

        fn explain_walk(
            &self,
            _prompt: &str,
            top_k: usize,
        ) -> Result<Vec<ExplainedHit>, PgInferError> {
            let hits: Vec<ExplainedHit> = (0..top_k.min(5))
                .map(|i| ExplainedHit {
                    layer: (i % 4) as i32,
                    band: "knowledge".into(),
                    feature: i as i32,
                    gate_score: 1.0 - (i as f64) * 0.1,
                    token: format!("token_{i}"),
                    also: String::new(),
                })
                .collect();
            Ok(hits)
        }

        fn nearest_to(
            &self,
            _entity: &str,
            layer: usize,
            top_k: usize,
        ) -> Result<Vec<Hit>, PgInferError> {
            let hits: Vec<Hit> = (0..top_k.min(5))
                .map(|i| Hit {
                    layer: layer as i32,
                    feature: i as i32,
                    gate_score: 0.95 - (i as f64) * 0.1,
                    concept: format!("near_{i}"),
                    also: "also_a, also_b".into(),
                })
                .collect();
            Ok(hits)
        }

        fn similar_to(&self, a: &str, b: &str) -> Result<f64, PgInferError> {
            // Symmetric: same hash regardless of argument order
            let mut chars_a: Vec<char> = a.chars().collect();
            let mut chars_b: Vec<char> = b.chars().collect();
            chars_a.sort();
            chars_b.sort();
            // Produce a deterministic symmetric score
            let combined: String = if chars_a <= chars_b {
                format!("{}{}", chars_a.iter().collect::<String>(), chars_b.iter().collect::<String>())
            } else {
                format!("{}{}", chars_b.iter().collect::<String>(), chars_a.iter().collect::<String>())
            };
            let hash = combined.bytes().fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
            Ok((hash % 1000) as f64 / 100.0) // Score in [0, 10)
        }

        fn implies(&self, subject: &str, object: &str) -> Result<bool, PgInferError> {
            // Deterministic: implies if subject contains object (case-insensitive)
            Ok(subject.to_lowercase().contains(&object.to_lowercase()))
        }

        fn infer(&self, _prompt: &str, top_k: usize) -> Result<Vec<Prediction>, PgInferError> {
            let predictions: Vec<Prediction> = (0..top_k.min(5))
                .map(|i| Prediction {
                    token: format!("pred_{i}"),
                    probability: 1.0 / (i as f64 + 1.0),
                    rank: i as i32 + 1,
                })
                .collect();
            Ok(predictions)
        }

        fn show_relations(&self) -> Result<Vec<RelationRow>, PgInferError> {
            Ok(vec![
                RelationRow {
                    relation: "is_a".into(),
                    count: 50,
                    max_score: 0.95,
                    layers: "2,3".into(),
                    examples: "cat→animal, dog→animal".into(),
                },
            ])
        }

        fn show_features(
            &self,
            layer: usize,
            _filter: Option<&str>,
            _min_score: f32,
            limit: usize,
        ) -> Result<Vec<FeatureRow>, PgInferError> {
            let rows: Vec<FeatureRow> = (0..limit.min(5))
                .map(|i| FeatureRow {
                    feature: i as i32,
                    token: format!("feat_{layer}_{i}"),
                    score: 0.9 - (i as f64) * 0.1,
                    also: String::new(),
                })
                .collect();
            Ok(rows)
        }

        fn snapshot_features(
            &self,
            _layer_filter: Option<i32>,
        ) -> Result<Vec<FeatureSnapshot>, PgInferError> {
            Ok(vec![FeatureSnapshot {
                layer: 0,
                feature: 0,
                top_token: "test".into(),
                c_score: 0.8,
            }])
        }

        fn feature_meta_at(&self, _layer: usize, _feature: usize) -> Option<FeatureMetaLite> {
            Some(FeatureMetaLite {
                top_token: "meta_token".into(),
                c_score: 0.75,
            })
        }

        fn is_local(&self) -> bool {
            true
        }
    }

    // ── Contract: walk() returns results sorted by gate_score descending ──

    #[test]
    fn walk_returns_sorted_results() {
        let backend = MockBackend;
        let hits = backend.walk("test query", 10).unwrap();

        assert!(!hits.is_empty());
        for window in hits.windows(2) {
            assert!(
                window[0].gate_score >= window[1].gate_score,
                "walk results not sorted: {} before {}",
                window[0].gate_score,
                window[1].gate_score,
            );
        }
    }

    // ── Contract: similar_to() is symmetric ──

    #[test]
    fn similar_to_is_symmetric() {
        let backend = MockBackend;
        let ab = backend.similar_to("cat", "dog").unwrap();
        let ba = backend.similar_to("dog", "cat").unwrap();
        assert!(
            (ab - ba).abs() < 1e-10,
            "similar_to not symmetric: ({}, {}) → {ab}, ({}, {}) → {ba}",
            "cat", "dog", "dog", "cat",
        );
    }

    // ── Contract: similar_to_many() is consistent with similar_to() ──

    #[test]
    fn similar_to_many_matches_individual_calls() {
        let backend = MockBackend;
        let candidates = vec!["cat".to_string(), "dog".to_string(), "fish".to_string()];
        let query = "animal";

        let batch_scores = backend.similar_to_many(&candidates, query).unwrap();
        assert_eq!(batch_scores.len(), candidates.len());

        for (i, candidate) in candidates.iter().enumerate() {
            let individual = backend.similar_to(candidate, query).unwrap();
            assert!(
                (batch_scores[i] - individual).abs() < 1e-10,
                "similar_to_many[{i}] ({}) = {} but similar_to = {}",
                candidate, batch_scores[i], individual,
            );
        }
    }

    // ── Contract: implies() returns bool without panic ──

    #[test]
    fn implies_returns_bool_no_panic() {
        let backend = MockBackend;
        // These should not panic regardless of input
        let _ = backend.implies("Paris", "France").unwrap();
        let _ = backend.implies("", "").unwrap();
        let _ = backend.implies("a very long entity name", "short").unwrap();
    }

    // ── Contract: rank() returns results sorted descending by score ──

    #[test]
    fn rank_returns_sorted_descending() {
        let backend = MockBackend;
        let candidates = vec![
            "alpha".to_string(),
            "beta".to_string(),
            "gamma".to_string(),
            "delta".to_string(),
        ];
        let ranked = backend.rank(&candidates, "query", 3).unwrap();

        assert!(ranked.len() <= 3, "rank exceeded limit");
        for window in ranked.windows(2) {
            assert!(
                window[0].score >= window[1].score,
                "rank not sorted: {} before {}",
                window[0].score,
                window[1].score,
            );
        }

        // All indices should be valid
        for r in &ranked {
            assert!(r.index < candidates.len(), "rank index out of bounds: {}", r.index);
        }
    }

    // ── Contract: rank() limit=0 returns all candidates ──

    #[test]
    fn rank_limit_zero_returns_all() {
        let backend = MockBackend;
        let candidates = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let ranked = backend.rank(&candidates, "q", 0).unwrap();
        assert_eq!(ranked.len(), candidates.len());
    }

    // ── Contract: show_layers() returns num_layers entries ──

    #[test]
    fn show_layers_matches_num_layers() {
        let backend = MockBackend;
        let layers = backend.show_layers().unwrap();
        assert_eq!(layers.len(), backend.num_layers());

        // Layer indices should be sequential from 0
        for (i, layer) in layers.iter().enumerate() {
            assert_eq!(layer.layer, i as i32);
        }
    }

    // ── Contract: infer() respects top_k limit ──

    #[test]
    fn infer_respects_top_k() {
        let backend = MockBackend;
        let preds = backend.infer("test prompt", 3).unwrap();
        assert!(preds.len() <= 3);

        // Ranks should be sequential starting from 1
        for (i, pred) in preds.iter().enumerate() {
            assert_eq!(pred.rank, i as i32 + 1);
        }
    }

    // ── Contract: describe() returns non-empty for known entities ──

    #[test]
    fn describe_returns_edges() {
        let backend = MockBackend;
        let edges = backend.describe("France", None).unwrap();
        assert!(!edges.is_empty());

        // All edges should have valid scores
        for edge in &edges {
            assert!(edge.gate_score >= 0.0);
            assert!(!edge.relation.is_empty());
            assert!(!edge.target.is_empty());
        }
    }

    // ── Contract: show_features() respects limit ──

    #[test]
    fn show_features_respects_limit() {
        let backend = MockBackend;
        let features = backend.show_features(0, None, 0.0, 3).unwrap();
        assert!(features.len() <= 3);
    }

    // ── Contract: metadata accessors are consistent ──

    #[test]
    fn metadata_consistent() {
        let backend = MockBackend;
        assert!(backend.num_layers() > 0);
        assert!(backend.hidden_size() > 0);
        assert!(backend.is_local());
    }

    // ── Contract: empty candidates handled gracefully ──

    #[test]
    fn similar_to_many_empty_candidates() {
        let backend = MockBackend;
        let scores = backend.similar_to_many(&[], "query").unwrap();
        assert!(scores.is_empty());
    }

    #[test]
    fn rank_empty_candidates() {
        let backend = MockBackend;
        let ranked = backend.rank(&[], "query", 10).unwrap();
        assert!(ranked.is_empty());
    }
}
