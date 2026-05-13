//! LayerGraph — pluggable per-layer routing for attention and FFN.
//!
//! The transformer layer loop receives a residual, routes through attention
//! and FFN, and produces the next residual. The mechanism behind each step
//! can vary:
//!
//! - Dense matmul (today's baseline)
//! - Walk/vindex (sparse FFN from mmap)
//! - Template cache (precomputed routing for known templates)
//! - Residual-adaptive graph (cluster-based routing)
//!
//! The `LayerGraph` trait abstracts this: given a residual, produce the
//! layer output. The implementation decides how attention and FFN are computed.

mod dense;
mod walk;
mod cached;
mod template;
pub mod pipeline_layer;
pub mod prefill;
pub mod logits;
pub mod generate;
pub mod grid;
pub mod hybrid;
pub mod predict;

pub use generate::{generate, generate_constrained, GenerateResult, StageTimings};

use ndarray::Array2;

use crate::attention::AttentionWeights;
use crate::model::ModelWeights;

// Re-export everything publicly
pub use dense::*;
pub use walk::*;
pub use cached::*;
pub use template::*;
pub use predict::*;

/// Output of a single layer's computation.
///
/// Contains the post-layer residual that feeds into the next layer, plus
/// optional diagnostic captures (activations and attention weights) used
/// for tracing and analysis.
///
/// # Examples
///
/// ```
/// use ndarray::Array2;
/// use infer_inference::layer_graph::LayerOutput;
///
/// let residual = Array2::<f32>::zeros((4, 128));
/// let output = LayerOutput {
///     residual,
///     activation: None,
///     attention: None,
/// };
/// assert_eq!(output.residual.shape(), &[4, 128]);
/// ```
pub struct LayerOutput {
    /// Post-layer residual (input to next layer).
    pub residual: Array2<f32>,
    /// Optional: FFN activation capture (for tracing/analysis).
    pub activation: Option<Array2<f32>>,
    /// Optional: attention weight capture (for tracing/analysis).
    pub attention: Option<AttentionWeights>,
}

/// Per-layer routing trait. Takes a residual, produces the next residual.
///
/// Implementations control both attention and FFN computation.
/// The residual is always the input. The mechanism changes.
///
/// # Implementations
///
/// - [`DenseLayerGraph`] — standard dense matmul (baseline)
/// - [`WalkLayerGraph`] — dense attention + vindex walk FFN
/// - [`CachedLayerGraph`] — precomputed residuals for template-determined layers
/// - [`PipelinedLayerGraph`] — CPU attention + batched GPU Q4 FFN
/// - [`PerLayerGraph`] — different backend per layer
///
/// # Examples
///
/// ```ignore
/// use infer_inference::layer_graph::{LayerGraph, DenseLayerGraph};
///
/// // Build a dense graph with a FFN backend
/// let graph = DenseLayerGraph {
///     ffn: &weight_ffn,
///     backend: None,
///     capture_activation: false,
///     capture_attention: false,
/// };
/// assert_eq!(graph.name(), "dense");
///
/// // Forward one layer
/// let output = graph.forward_layer(&weights, &residual, 0);
/// ```
pub trait LayerGraph {
    /// Run one transformer layer: attention + FFN + residuals.
    fn forward_layer(
        &self,
        weights: &ModelWeights,
        h: &Array2<f32>,
        layer: usize,
    ) -> Option<LayerOutput>;

    /// Human-readable name for logging.
    fn name(&self) -> &str;
}
