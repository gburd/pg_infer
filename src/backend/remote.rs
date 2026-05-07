//! Remote `Backend` implementation.
//!
//! Translates the high-level `Backend` trait into HTTP calls against a
//! `larql-server` endpoint.  Uses `infer_client::CancellableClient` so
//! in-flight requests can be aborted when PostgreSQL signals a query
//! cancellation.
//!
//! The mapping is direct for operations larql-server exposes:
//!
//! | Trait method        | Endpoint                                |
//! |---------------------|-----------------------------------------|
//! | `describe`          | `GET /v1/describe?entity=…`             |
//! | `walk`              | `GET /v1/walk?prompt=…&top=…`           |
//! | `nearest_to`        | `GET /v1/walk?prompt=…&layers=N&top=…`  |
//! | `show_layers`       | `GET /v1/stats` (derive from layers+bands) |
//! | `show_relations`    | `GET /v1/relations`                     |
//! | `infer`             | `POST /v1/infer`                        |
//! | `similar_to`        | two `GET /v1/walk` calls + overlap      |
//! | `implies`           | one `describe` + substring check        |
//!
//! Feature-level introspection (`show_features`, `snapshot_features`,\n//! `feature_meta_at`, `embed`) has no server-side equivalent and returns\n//! `PgInferError::RemoteUnsupported`.  A future `/v1/features` endpoint\n//! on larql-server would let us light these up.

use std::collections::HashMap;
use std::time::Duration;

use infer_client::{CancelToken, CancellableClient, ClientError};
use ndarray::Array1;

use crate::error::PgInferError;

use super::{
    Backend, Edge, ExplainedHit, FeatureMetaLite, FeatureRow, FeatureSnapshot, Hit, LayerInfo,
    Prediction, RelationRow,
};

/// A remote model backed by a `larql-server` endpoint.
pub struct RemoteBackend {
    /// Origin URL (for error messages and diagnostics).
    pub server_url: String,
    /// Layer count cached at load time from `/v1/stats`.
    pub num_layers: usize,
    /// Hidden size cached at load time.
    pub hidden_size: usize,
    /// Layer-band classification from `/v1/stats`; used to label
    /// `show_layers` rows.  Empty when the server didn't report bands.
    pub layer_bands: Option<LayerBandsCached>,
    /// HTTP client.  One dedicated runtime thread per client.
    client: CancellableClient,
}

#[derive(Debug, Clone, Default)]
pub struct LayerBandsCached {
    pub syntax: (usize, usize),
    pub knowledge: (usize, usize),
    pub output: (usize, usize),
}

impl LayerBandsCached {
    fn band_for(&self, layer: usize) -> &'static str {
        if (self.syntax.0..=self.syntax.1).contains(&layer) {
            "syntax"
        } else if (self.knowledge.0..=self.knowledge.1).contains(&layer) {
            "knowledge"
        } else if (self.output.0..=self.output.1).contains(&layer) {
            "output"
        } else {
            ""
        }
    }
}

impl RemoteBackend {
    /// Connect to a `larql-server` and cache its stats.
    pub fn connect(server_url: &str, timeout: Duration) -> Result<Self, PgInferError> {
        let client = CancellableClient::connect(server_url, timeout)
            .map_err(|e| PgInferError::Remote(format!("connect {server_url}: {e}")))?;

        // Pull /v1/stats to learn layer count + hidden size + bands.
        let cancel = CancelToken::new();
        let stats: infer_client::StatsResponse = client
            .get_json("/v1/stats", &cancel)
            .map_err(map_err)?;

        let layer_bands = stats.layer_bands.map(|b| LayerBandsCached {
            syntax: (b.syntax[0], b.syntax[1]),
            knowledge: (b.knowledge[0], b.knowledge[1]),
            output: (b.output[0], b.output[1]),
        });

        Ok(Self {
            server_url: server_url.to_string(),
            num_layers: stats.layers,
            hidden_size: stats.hidden_size,
            layer_bands,
            client,
        })
    }

    fn cancel_for_current_call(&self) -> CancelToken {
        // Phase C3 will tie this to PG's QueryCancelPending.  For now the
        // token is fresh-per-call and only signalled on reqwest-level
        // timeout (which reqwest handles itself via Builder::timeout).
        CancelToken::new()
    }
}

fn map_err(e: ClientError) -> PgInferError {
    match e {
        ClientError::Cancelled => PgInferError::Remote("request cancelled".into()),
        other => PgInferError::Remote(other.to_string()),
    }
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
        // The server doesn't currently expose per-layer feature counts in
        // `/v1/stats` (it reports total + per-layer-0).  Approximate by
        // using features_per_layer as a uniform value until a `/v1/layers`
        // endpoint lands upstream.
        let cancel = self.cancel_for_current_call();
        let stats: serde_json::Value = self
            .client
            .get_json("/v1/stats", &cancel)
            .map_err(map_err)?;

        let uniform = stats
            .get("features_per_layer")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as i32;

        let mut out = Vec::with_capacity(self.num_layers);
        for layer in 0..self.num_layers {
            let band = self
                .layer_bands
                .as_ref()
                .map(|b| b.band_for(layer).to_string())
                .unwrap_or_default();
            out.push(LayerInfo {
                layer: layer as i32,
                band,
                num_features: uniform,
            });
        }
        Ok(out)
    }

    fn describe(
        &self,
        entity: &str,
        explicit_threshold: Option<f64>,
    ) -> Result<Vec<Edge>, PgInferError> {
        let threshold = explicit_threshold
            .filter(|t| *t > 0.0)
            .map(|t| t as f32)
            .unwrap_or(0.0);

        let top_k = crate::gucs::describe_top_k();

        let mut path = format!(
            "/v1/describe?entity={}&limit={}",
            urlencoding(entity),
            top_k
        );
        if threshold > 0.0 {
            path.push_str(&format!("&min_score={threshold}"));
        }

        let cancel = self.cancel_for_current_call();
        let resp: infer_client::DescribeResponse =
            self.client.get_json(&path, &cancel).map_err(map_err)?;

        Ok(resp
            .edges
            .into_iter()
            .map(|e| Edge {
                relation: e.relation,
                target: e.target,
                gate_score: e.gate_score as f64,
                layer: e.layer as i32,
            })
            .collect())
    }

    fn walk(&self, prompt: &str, top_k: usize) -> Result<Vec<Hit>, PgInferError> {
        let path = format!(
            "/v1/walk?prompt={}&top={}",
            urlencoding(prompt),
            top_k
        );
        let cancel = self.cancel_for_current_call();
        let resp: infer_client::WalkResponse =
            self.client.get_json(&path, &cancel).map_err(map_err)?;

        Ok(resp
            .hits
            .into_iter()
            .map(|h| Hit {
                layer: h.layer as i32,
                feature: h.feature as i32,
                gate_score: h.gate_score as f64,
                concept: h.target,
                also: String::new(),
            })
            .collect())
    }

    fn explain_walk(&self, prompt: &str, top_k: usize) -> Result<Vec<ExplainedHit>, PgInferError> {
        // No dedicated endpoint; reuse /v1/walk and annotate bands client-side.
        let base = self.walk(prompt, top_k)?;
        Ok(base
            .into_iter()
            .map(|h| ExplainedHit {
                band: self
                    .layer_bands
                    .as_ref()
                    .map(|b| b.band_for(h.layer as usize).to_string())
                    .unwrap_or_default(),
                layer: h.layer,
                feature: h.feature,
                gate_score: h.gate_score,
                token: h.concept,
                also: h.also,
            })
            .collect())
    }

    fn nearest_to(
        &self,
        entity: &str,
        layer: usize,
        top_k: usize,
    ) -> Result<Vec<Hit>, PgInferError> {
        let path = format!(
            "/v1/walk?prompt={}&top={}&layers={}",
            urlencoding(entity),
            top_k,
            layer
        );
        let cancel = self.cancel_for_current_call();
        let resp: infer_client::WalkResponse =
            self.client.get_json(&path, &cancel).map_err(map_err)?;

        Ok(resp
            .hits
            .into_iter()
            .filter(|h| h.layer == layer)
            .map(|h| Hit {
                layer: h.layer as i32,
                feature: h.feature as i32,
                gate_score: h.gate_score as f64,
                concept: h.target,
                also: String::new(),
            })
            .collect())
    }

    fn similar_to(&self, a: &str, b: &str) -> Result<f64, PgInferError> {
        // Two /v1/walk calls, then overlap by (layer, feature).  The
        // server's L2 activation cache turns recurring queries into
        // sub-millisecond round trips, so this is the right place for
        // the compose.
        let top_k = 50_usize;
        let path_a = format!("/v1/walk?prompt={}&top={}", urlencoding(a), top_k);
        let path_b = format!("/v1/walk?prompt={}&top={}", urlencoding(b), top_k);

        let cancel = self.cancel_for_current_call();
        let resp_a: infer_client::WalkResponse =
            self.client.get_json(&path_a, &cancel).map_err(map_err)?;
        let resp_b: infer_client::WalkResponse =
            self.client.get_json(&path_b, &cancel).map_err(map_err)?;

        // Build (layer, feature) → score for B.
        let mut b_map: HashMap<(usize, usize), f32> = HashMap::new();
        for h in &resp_b.hits {
            b_map.insert((h.layer, h.feature), h.gate_score);
        }

        let mut max_shared = 0.0f32;
        for h in &resp_a.hits {
            if let Some(&score_b) = b_map.get(&(h.layer, h.feature)) {
                let shared = h.gate_score.min(score_b);
                if shared > max_shared {
                    max_shared = shared;
                }
            }
        }
        Ok(max_shared as f64)
    }

    fn implies(&self, subject: &str, object: &str) -> Result<bool, PgInferError> {
        let obj_lower = object.to_lowercase();
        let edges = self.describe(subject, None)?;
        Ok(edges.iter().any(|e| e.target.to_lowercase() == obj_lower))
    }

    fn infer(&self, prompt: &str, top_k: usize) -> Result<Vec<Prediction>, PgInferError> {
        let body = serde_json::json!({
            "prompt": prompt,
            "top": top_k,
        });
        let cancel = self.cancel_for_current_call();
        let resp: infer_client::InferResponse = self
            .client
            .post_json("/v1/infer", body, &cancel)
            .map_err(map_err)?;

        Ok(resp
            .predictions
            .into_iter()
            .enumerate()
            .map(|(i, p)| Prediction {
                token: p.token,
                probability: p.probability,
                rank: (i + 1) as i32,
            })
            .collect())
    }

    fn show_relations(&self) -> Result<Vec<RelationRow>, PgInferError> {
        let cancel = self.cancel_for_current_call();
        let resp: infer_client::RelationsResponse = self
            .client
            .get_json("/v1/relations", &cancel)
            .map_err(map_err)?;

        Ok(resp
            .relations
            .into_iter()
            .map(|r| RelationRow {
                relation: r.token,
                count: r.count as i32,
                max_score: r.max_score as f64,
                layers: r
                    .layers
                    .iter()
                    .map(|l| l.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
                examples: r.examples.join(", "),
            })
            .collect())
    }

    fn show_features(
        &self,
        _layer: usize,
        _filter: Option<&str>,
        _min_score: f32,
        _limit: usize,
    ) -> Result<Vec<FeatureRow>, PgInferError> {
        unsupported("show_features (no larql-server endpoint yet)")
    }

    fn snapshot_features(
        &self,
        _layer_filter: Option<i32>,
    ) -> Result<Vec<FeatureSnapshot>, PgInferError> {
        unsupported("infer_diff (requires local vindex)")
    }

    fn feature_meta_at(&self, _layer: usize, _feature: usize) -> Option<FeatureMetaLite> {
        None
    }

    fn embed(&self, _text: &str) -> Result<Array1<f32>, PgInferError> {
        // Could be lit up via `/v1/embed`, but nothing in pg_infer's hot
        // path currently calls it (precompute_query_gates is dead).
        // Revisit when we need client-side vector arithmetic.
        unsupported("embed (not wired yet)")
    }
}

/// Minimal URL percent-encoding for query-string values.
///
/// Encodes characters outside `A-Za-z0-9_.~-`.  Good enough for prompts
/// and entity names; not a full RFC 3986 implementation.
fn urlencoding(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for byte in s.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char);
            }
            _ => {
                out.push_str(&format!("%{byte:02X}"));
            }
        }
    }
    out
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn urlencoding_basic() {
        assert_eq!(urlencoding("hello"), "hello");
        assert_eq!(urlencoding("a b"), "a%20b");
        assert_eq!(urlencoding("a&b=c"), "a%26b%3Dc");
    }

    #[test]
    fn urlencoding_utf8() {
        // é = C3 A9
        assert_eq!(urlencoding("é"), "%C3%A9");
    }

    #[test]
    fn layer_bands_classification() {
        let b = LayerBandsCached {
            syntax: (0, 6),
            knowledge: (7, 26),
            output: (27, 33),
        };
        assert_eq!(b.band_for(3), "syntax");
        assert_eq!(b.band_for(18), "knowledge");
        assert_eq!(b.band_for(30), "output");
        assert_eq!(b.band_for(99), "");
    }
}
