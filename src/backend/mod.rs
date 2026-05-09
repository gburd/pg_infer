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
