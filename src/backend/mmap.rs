//! Mmap-backed `Backend` implementation.
//!
//! Thin wrapper over the existing `ModelHandle` — each trait method
//! delegates to an implementation function that takes `&ModelHandle`,
//! preserving the existing algorithms in `fn_describe`, `fn_walk`,
//! `fn_similar`, etc.  The trait boundary is purely a dispatch seam.

#![allow(dead_code)] // `mmap_*` helpers are also called directly by fn_*.rs.

use std::sync::Arc;

use ndarray::Array1;

use crate::error::PgInferError;
use crate::registry::ModelHandle;

use super::{
    Backend, Edge, ExplainedHit, FeatureMetaLite, FeatureRow, FeatureSnapshot, Hit, LayerHit,
    LayerInfo, Prediction, RelationRow,
};

/// Mmap-backed backend wrapping a process-local `ModelHandle`.
pub struct MmapBackend {
    pub handle: Arc<ModelHandle>,
}

impl MmapBackend {
    pub fn new(handle: Arc<ModelHandle>) -> Self {
        Self { handle }
    }
}

impl Backend for MmapBackend {
    fn approx_resident_bytes(&self) -> usize {
        self.handle.approx_resident_bytes()
    }

    fn is_local(&self) -> bool {
        true
    }

    fn num_layers(&self) -> usize {
        self.handle.config.num_layers
    }

    fn hidden_size(&self) -> usize {
        self.handle.config.hidden_size
    }

    fn show_layers(&self) -> Result<Vec<LayerInfo>, PgInferError> {
        crate::fn_show::mmap_show_layers(&self.handle)
    }

    fn describe(
        &self,
        entity: &str,
        explicit_threshold: Option<f64>,
    ) -> Result<Vec<Edge>, PgInferError> {
        crate::fn_describe::mmap_describe(&self.handle, entity, explicit_threshold)
    }

    fn walk(&self, prompt: &str, top_k: usize) -> Result<Vec<Hit>, PgInferError> {
        crate::fn_walk::mmap_walk(&self.handle, prompt, top_k)
    }

    fn explain_walk(&self, prompt: &str, top_k: usize) -> Result<Vec<ExplainedHit>, PgInferError> {
        crate::fn_walk::mmap_explain_walk(&self.handle, prompt, top_k)
    }

    fn nearest_to(
        &self,
        entity: &str,
        layer: usize,
        top_k: usize,
    ) -> Result<Vec<Hit>, PgInferError> {
        crate::fn_nearest::mmap_nearest_to(&self.handle, entity, layer, top_k)
    }

    fn similar_to(&self, a: &str, b: &str) -> Result<f64, PgInferError> {
        crate::fn_similar::similar_to_impl(&self.handle, a, b)
    }

    fn implies(&self, subject: &str, object: &str) -> Result<bool, PgInferError> {
        let object_lower = object.to_lowercase();
        let edges = self.describe(subject, None)?;
        Ok(edges.iter().any(|e| e.target.to_lowercase() == object_lower))
    }

    fn describe_layers(
        &self,
        entity: &str,
        explicit_threshold: Option<f64>,
    ) -> Result<Vec<LayerHit>, PgInferError> {
        crate::fn_describe::mmap_describe_layers(&self.handle, entity, explicit_threshold)
    }

    fn infer(&self, prompt: &str, top_k: usize) -> Result<Vec<Prediction>, PgInferError> {
        crate::fn_infer::mmap_infer(&self.handle, prompt, top_k)
    }

    fn show_relations(&self) -> Result<Vec<RelationRow>, PgInferError> {
        crate::fn_show::mmap_show_relations(&self.handle)
    }

    fn show_features(
        &self,
        layer: usize,
        filter: Option<&str>,
        min_score: f32,
        limit: usize,
    ) -> Result<Vec<FeatureRow>, PgInferError> {
        crate::fn_show::mmap_show_features(&self.handle, layer, filter, min_score, limit)
    }

    fn snapshot_features(
        &self,
        layer_filter: Option<i32>,
    ) -> Result<Vec<FeatureSnapshot>, PgInferError> {
        crate::fn_diff::mmap_snapshot_features(&self.handle, layer_filter)
    }

    fn feature_meta_at(&self, layer: usize, feature: usize) -> Option<FeatureMetaLite> {
        self.handle.feature_meta(layer, feature).map(|m| FeatureMetaLite {
            top_token: m.top_token,
            c_score: m.c_score,
        })
    }

    fn embed(&self, text: &str) -> Result<Array1<f32>, PgInferError> {
        crate::fn_similar::embed_text(&self.handle, text)
    }
}
