//! JSON DTOs matching the subset of larql-server responses pg_infer consumes.
//!
//! These types are deliberately permissive: `#[serde(default)]` on every
//! optional field so a server upgrade that adds new fields does not break
//! the extension.

use serde::Deserialize;

// ── /v1/stats ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct StatsResponse {
    pub model: String,
    #[serde(default)]
    pub family: String,
    pub layers: usize,
    pub hidden_size: usize,
    #[serde(default)]
    pub vocab_size: usize,
    #[serde(default)]
    pub extract_level: String,
    #[serde(default)]
    pub layer_bands: Option<LayerBands>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LayerBands {
    #[serde(default)]
    pub syntax: [usize; 2],
    #[serde(default)]
    pub knowledge: [usize; 2],
    #[serde(default)]
    pub output: [usize; 2],
}

// ── /v1/describe ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct DescribeResponse {
    pub entity: String,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub edges: Vec<DescribeEdge>,
    #[serde(default)]
    pub latency_ms: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DescribeEdge {
    pub target: String,
    #[serde(default)]
    pub relation: String,
    pub gate_score: f32,
    pub layer: usize,
    #[serde(default)]
    pub also: Vec<String>,
    #[serde(default)]
    pub source: Option<String>,
}

// ── /v1/walk ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct WalkResponse {
    pub prompt: String,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub hits: Vec<WalkHit>,
    #[serde(default)]
    pub latency_ms: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct WalkHit {
    pub layer: usize,
    pub feature: usize,
    pub gate_score: f32,
    #[serde(default)]
    pub target: String,
    #[serde(default)]
    pub relation: Option<String>,
}

// ── /v1/relations ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct RelationsResponse {
    #[serde(default)]
    pub relations: Vec<RelationSummary>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RelationSummary {
    pub token: String,
    pub count: usize,
    #[serde(default)]
    pub max_score: f32,
    #[serde(default)]
    pub layers: Vec<usize>,
    #[serde(default)]
    pub examples: Vec<String>,
}

// ── /v1/infer ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct InferResponse {
    #[serde(default)]
    pub predictions: Vec<InferPrediction>,
    #[serde(default)]
    pub latency_ms: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct InferPrediction {
    pub token: String,
    pub probability: f64,
}
