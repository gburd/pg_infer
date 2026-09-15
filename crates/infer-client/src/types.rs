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
    /// Server-level counters. Present on current larql-servers; absent on
    /// older ones.
    #[serde(default)]
    pub server: Option<ServerStats>,
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

/// The `server` block on `GET /v1/stats`.
///
/// This is where real larql-servers report cache behaviour. pg_infer's
/// `/v1/cache/stats` has never existed upstream, so before this the only
/// answer `infer_server_stats()` could give against a real server was an
/// empty set.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct ServerStats {
    #[serde(default)]
    pub uptime_secs: u64,
    #[serde(default)]
    pub requests_served: u64,
    /// VINDEX3 KV continuation cache. Absent when the server has V3 KV
    /// disabled, hence `Option`.
    #[serde(default)]
    pub v3_kv: Option<V3KvStats>,
}

/// `server.v3_kv` — the bounded KV continuation cache.
///
/// `hits` counts resident states found; `resumptions` counts the subset
/// that also passed the exact ids-prefix check and so skipped prefill.
/// The gap between them is real information (prefix instability under
/// live request construction), so both are carried rather than collapsed.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct V3KvStats {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub entries: usize,
    #[serde(default)]
    pub capacity: usize,
    #[serde(default)]
    pub hits: u64,
    #[serde(default)]
    pub misses: u64,
    #[serde(default)]
    pub resumptions: u64,
    #[serde(default)]
    pub reused_tokens_total: u64,
}

// ── /v1/capabilities ─────────────────────────────────────────────────────────

/// Response from `GET /v1/capabilities`.
///
/// Lets a client ask what a server does instead of discovering it by
/// probing for 404s. larql-server derives `routes` from the ledger it
/// records *while building its router*, so an advertised route cannot
/// drift from a mounted one.
///
/// Only the fields pg_infer acts on are modelled. The `sources` /
/// `explorer` / `runtime` blocks are generated server-side from a
/// capability table and are deliberately not mirrored here: `routes` is
/// the authority for "can I call this", and duplicating the derived
/// booleans would be a second list to keep in sync.
#[derive(Debug, Clone, Deserialize)]
pub struct CapabilitiesResponse {
    /// Report schema version.
    ///
    /// The server's own contract: *"A client that does not recognise it
    /// must refuse the document rather than read the keys it knows."*
    /// See [`Self::is_schema_understood`].
    #[serde(default)]
    pub schema: u32,
    /// `"public_explorer"` | `"single_model"` | `"multi_model"`.
    #[serde(default)]
    pub profile: String,
    /// Every path this server mounted, sorted.
    #[serde(default)]
    pub routes: Vec<String>,
}

/// The `/v1/capabilities` report schema pg_infer knows how to read.
pub const CAPABILITIES_SCHEMA: u32 = 1;

impl CapabilitiesResponse {
    /// Whether this report's schema is one we can interpret.
    ///
    /// A newer schema may reorganize what `routes` means, so an
    /// unrecognized version is treated as "no information" and the
    /// caller falls back to probing — which still works, just less
    /// efficiently.
    pub fn is_schema_understood(&self) -> bool {
        self.schema == CAPABILITIES_SCHEMA
    }

    /// Whether the server mounted `path`.
    ///
    /// Exact match against the server's own route table. Parameterized
    /// routes appear in their template form (`/v1/{model_id}/describe`),
    /// so callers should ask about the concrete path they intend to
    /// call and accept that a templated equivalent reads as absent —
    /// pg_infer only calls unparameterized paths.
    pub fn serves(&self, path: &str) -> bool {
        self.routes.iter().any(|r| r == path)
    }
}

// ── /v1/embed, /v1/token/encode ──────────────────────────────────────────────

/// Response from `POST /v1/embed`.
///
/// `residual` is row-major `seq_len × hidden_size`. The server has already
/// applied the model's `embed_scale`, so these rows match what a local
/// vindex load produces for the same token ids.
#[derive(Debug, Clone, Deserialize)]
pub struct EmbedResponse {
    #[serde(default)]
    pub residual: Vec<Vec<f32>>,
    #[serde(default)]
    pub seq_len: usize,
    #[serde(default)]
    pub hidden_size: usize,
}

/// Response from `POST /v1/token/encode`.
#[derive(Debug, Clone, Deserialize)]
pub struct TokenEncodeResponse {
    #[serde(default)]
    pub token_ids: Vec<u32>,
}

// ── /v1/select ───────────────────────────────────────────────────────────────

/// Response from `POST /v1/select` — server-side feature scan.
///
/// **The OpenAPI schema and the handler disagree here**, and this follows
/// the handler, because that is what a client receives:
///
/// | | schema says | server sends |
/// |---|---|---|
/// | list key | `rows` | `edges` |
/// | score field | `confidence` | `c_score` |
///
/// Verified against a live larql-server (`routes/select.rs` builds
/// `{"edges": [...]}` with `c_score`, while `openapi.rs`'s `SelectRow`
/// declares `rows`/`confidence`). Believing the schema here would have
/// produced exactly the failure mode this crate already shipped three
/// times: a struct that parses nothing, or silently parses to empty.
///
/// Both spellings are accepted so this keeps working if upstream aligns
/// the handler with its own spec.
#[derive(Debug, Clone, Deserialize)]
pub struct SelectResponse {
    #[serde(default, alias = "rows")]
    pub edges: Vec<SelectRow>,
    /// Matches before `limit` was applied.
    #[serde(default)]
    pub total: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SelectRow {
    #[serde(default)]
    pub layer: usize,
    #[serde(default)]
    pub feature: usize,
    /// Top token for this feature, already trimmed by the server.
    #[serde(default)]
    pub target: String,
    #[serde(default, alias = "confidence")]
    pub c_score: f32,
    /// Probe-confirmed relation label, when the feature has one.
    #[serde(default)]
    pub relation: Option<String>,
}
