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
    #[serde(default, deserialize_with = "normalized_relations")]
    pub relations: Vec<RelationSummary>,
}

/// Deserialize the relation list and derive each entry's `layers` from
/// the `min_layer`/`max_layer` span the server actually sends.
fn normalized_relations<'de, D>(d: D) -> Result<Vec<RelationSummary>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let mut v = Vec::<RelationSummary>::deserialize(d)?;
    for r in &mut v {
        r.normalize();
    }
    Ok(v)
}

#[derive(Debug, Clone, Deserialize)]
pub struct RelationSummary {
    /// The relation token.
    ///
    /// larql-server calls this field `name` (see `RelationEntry` in its
    /// OpenAPI spec and `routes/relations.rs`). It has *always* been
    /// `name`; the field was declared here as a bare required `token`,
    /// so every `show_relations()` against a real server failed with
    /// `missing field 'token'`. Only the hand-written mock in
    /// `tests/mock_server_integration.rs` ever sent `token`, which is
    /// why CI stayed green.
    ///
    /// `alias` keeps the mock (and any client that did send `token`)
    /// working while `rename` makes `name` the wire name.
    #[serde(rename = "name", alias = "token")]
    pub token: String,
    pub count: usize,
    #[serde(default)]
    pub max_score: f32,
    /// Layers this relation appears on.
    ///
    /// The server reports the span as two scalars, `min_layer` and
    /// `max_layer`, not a list — so this stayed empty even once the
    /// rename above was fixed. Derived from those two by
    /// [`RelationSummary::normalize`], which the list's deserializer
    /// runs on every entry.
    #[serde(default)]
    pub layers: Vec<usize>,
    #[serde(default)]
    pub examples: Vec<String>,
    /// Lower bound of the layer span, as sent by the server. Kept so
    /// `layers` can be derived and so a caller can tell a one-layer
    /// relation from a two-layer one.
    #[serde(default)]
    pub min_layer: usize,
    #[serde(default)]
    pub max_layer: usize,
}

impl RelationSummary {
    /// Fill `layers` from `min_layer`/`max_layer` when the server did not
    /// send an explicit list (it does not — this is the normal path).
    ///
    /// Called by [`RelationsResponse`]'s deserializer so callers never
    /// observe the un-normalized form.
    fn normalize(&mut self) {
        if !self.layers.is_empty() {
            return;
        }
        if self.max_layer >= self.min_layer {
            self.layers = (self.min_layer..=self.max_layer).collect();
        }
    }
}

// ── /v1/rank ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct RankResponse {
    #[serde(default)]
    pub results: Vec<RankResult>,
    #[serde(default)]
    pub latency_ms: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RankResult {
    pub index: usize,
    pub score: f64,
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

// ── /v1/warmup ───────────────────────────────────────────────────────────────

/// Response from `POST /v1/warmup`.
///
/// The previous shape here was `{warmed, already_cached, latency_ms}`, all
/// `#[serde(default)]`. larql-server sends none of those fields, so the
/// response parsed cleanly and every value came out zero: `infer_warmup()`
/// reported "0 warmed, 0 already cached" no matter how much work the
/// server did. Because the fields defaulted rather than erroring, nothing
/// surfaced the mismatch.
///
/// These are the fields the server actually documents (`WarmupResponse` in
/// its OpenAPI spec). Warmup prefetches *layers* and loads weights; it has
/// no notion of "entities" or a per-entity cache.
#[derive(Debug, Clone, Deserialize)]
pub struct WarmupResponse {
    #[serde(default)]
    pub model: String,
    /// Whether the inference weight load ran (false under `skip_weights`
    /// or on an already-warm server).
    #[serde(default)]
    pub weights_loaded: bool,
    #[serde(default)]
    pub weights_load_ms: u64,
    /// Layers whose pages were faulted in via `madvise(WILLNEED)`.
    #[serde(default)]
    pub layers_prefetched: usize,
    #[serde(default)]
    pub prefetch_ms: u64,
    /// `(layer, expert)` pairs prefetched. Zero for non-MoE models.
    #[serde(default)]
    pub experts_prefetched: usize,
    #[serde(default)]
    pub expert_prefetch_ms: u64,
    #[serde(default)]
    pub hnsw_built: bool,
    #[serde(default)]
    pub hnsw_warmup_ms: u64,
    #[serde(default)]
    pub total_ms: u64,
}

// ── /v1/cache/stats ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct CacheStatsResponse {
    #[serde(default)]
    pub entries: usize,
    #[serde(default)]
    pub hit_count: u64,
    #[serde(default)]
    pub miss_count: u64,
    #[serde(default)]
    pub eviction_count: u64,
    #[serde(default)]
    pub memory_bytes: usize,
}
