//! Remote `Backend` implementation — stub for Phase C1.
//!
//! Phase C2 wires this up to `infer-client::CancellableClient`.  For now
//! the impl exists as a placeholder so the `Backend` trait is
//! dyn-dispatchable without conditional compilation.

#![allow(dead_code)]

use ndarray::Array1;

use crate::error::PgInferError;

use super::{
    Backend, Edge, ExplainedHit, FeatureMetaLite, FeatureRow, FeatureSnapshot, Hit, LayerInfo,
    Prediction, RelationRow,
};

/// Remote backend pointing at a `larql-server` endpoint.
pub struct RemoteBackend {
    /// Server URL, for diagnostics.
    pub server_url: String,
    /// Cached model metadata fetched from `/v1/stats` at load time.
    pub num_layers: usize,
    pub hidden_size: usize,
}

fn unsupported<T>(op: &str) -> Result<T, PgInferError> {
    Err(PgInferError::RemoteUnsupported {
        operation: op.to_string(),
    })
}

impl Backend for RemoteBackend {
    fn is_local(&self) -> bool {
        false
    }

    fn num_layers(&self) -> usize {
        self.num_layers
    }

    fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    fn show_layers(&self) -> Result<Vec<LayerInfo>, PgInferError> {
        unsupported("show_layers")
    }

    fn describe(
        &self,
        _entity: &str,
        _explicit_threshold: Option<f64>,
    ) -> Result<Vec<Edge>, PgInferError> {
        unsupported("describe")
    }

    fn walk(&self, _prompt: &str, _top_k: usize) -> Result<Vec<Hit>, PgInferError> {
        unsupported("walk")
    }

    fn explain_walk(
        &self,
        _prompt: &str,
        _top_k: usize,
    ) -> Result<Vec<ExplainedHit>, PgInferError> {
        unsupported("explain_walk")
    }

    fn nearest_to(
        &self,
        _entity: &str,
        _layer: usize,
        _top_k: usize,
    ) -> Result<Vec<Hit>, PgInferError> {
        unsupported("nearest_to")
    }

    fn similar_to(&self, _a: &str, _b: &str) -> Result<f64, PgInferError> {
        unsupported("similar_to")
    }

    fn implies(&self, _subject: &str, _object: &str) -> Result<bool, PgInferError> {
        unsupported("implies")
    }

    fn infer(&self, _prompt: &str, _top_k: usize) -> Result<Vec<Prediction>, PgInferError> {
        unsupported("infer")
    }

    fn show_relations(&self) -> Result<Vec<RelationRow>, PgInferError> {
        unsupported("show_relations")
    }

    fn show_features(
        &self,
        _layer: usize,
        _filter: Option<&str>,
        _min_score: f32,
        _limit: usize,
    ) -> Result<Vec<FeatureRow>, PgInferError> {
        unsupported("show_features")
    }

    fn snapshot_features(
        &self,
        _layer_filter: Option<i32>,
    ) -> Result<Vec<FeatureSnapshot>, PgInferError> {
        unsupported("snapshot_features")
    }

    fn feature_meta_at(&self, _layer: usize, _feature: usize) -> Option<FeatureMetaLite> {
        None
    }

    fn embed(&self, _text: &str) -> Result<Array1<f32>, PgInferError> {
        unsupported("embed")
    }
}
