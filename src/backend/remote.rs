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

use infer_client::{BatchItem, CancelToken, CancellableClient, ClientError, Method};
use ndarray::Array1;

use crate::error::PgInferError;

use super::{
    Backend, CacheStats, Edge, ExplainedHit, FeatureMetaLite, FeatureRow, FeatureSnapshot, Hit,
    LayerInfo, Prediction, RankedCandidate, RelationRow,
};

/// Probe for a colocated larql-server on well-known UDS paths.
/// Returns the first path that exists as a Unix socket.
pub fn detect_local_socket() -> Option<String> {
    let candidates: &[&str] = &["/run/larql.sock", "/tmp/larql.sock"];

    let pgdata_sock = std::env::var("PGDATA")
        .ok()
        .map(|d| format!("{}/larql.sock", d));

    for path in candidates.iter().copied().chain(pgdata_sock.as_deref()) {
        if let Ok(m) = std::fs::metadata(path) {
            use std::os::unix::fs::FileTypeExt;
            if m.file_type().is_socket() {
                return Some(format!("uds://{}", path));
            }
        }
    }
    None
}

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
    /// What the server said it serves, fetched once at connect.
    ///
    /// `None` when the server has no `/v1/capabilities` (predates it, or
    /// reports a schema we don't understand). Callers must then fall
    /// back to calling the endpoint and interpreting a 404 — see
    /// [`Self::serves`].
    capabilities: Option<infer_client::CapabilitiesResponse>,
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
        // At connect time there's no PG backend yet (this runs from
        // infer_create_model_remote), so a plain no-tick GET is fine.
        let cancel = CancelToken::new();
        let stats: infer_client::StatsResponse = client
            .get_json("/v1/stats", &cancel)
            .map_err(|e| PgInferError::Remote(format!("stats {server_url}: {e}")))?;

        let layer_bands = stats.layer_bands.map(|b| LayerBandsCached {
            syntax: (b.syntax[0], b.syntax[1]),
            knowledge: (b.knowledge[0], b.knowledge[1]),
            output: (b.output[0], b.output[1]),
        });

        // Ask the server what it serves, once, here — rather than
        // discovering it per-call from 404s. Entirely optional: a server
        // without the endpoint, or with a schema we don't recognise,
        // leaves this `None` and every caller keeps its probe path.
        //
        // Errors are swallowed on purpose. This is an optimisation and a
        // source of better diagnostics; it must never be the reason a
        // model fails to register.
        let capabilities = client
            .get_json::<infer_client::CapabilitiesResponse>("/v1/capabilities", &cancel)
            .ok()
            .filter(infer_client::CapabilitiesResponse::is_schema_understood);

        Ok(Self {
            server_url: server_url.to_string(),
            num_layers: stats.layers,
            hidden_size: stats.hidden_size,
            layer_bands,
            capabilities,
            client,
        })
    }

    /// Whether the server is known to serve `path`.
    ///
    /// Three-valued on purpose:
    ///
    /// * `Some(true)`  — it said it mounts this route;
    /// * `Some(false)` — it said it does not, so skip the call entirely;
    /// * `None`        — it did not say (no `/v1/capabilities`, or an
    ///   unrecognised report schema); the caller must try the call and
    ///   interpret the result.
    ///
    /// Collapsing the last two into `false` would silently disable
    /// features against any older server, which is the opposite of what
    /// a capability check is for.
    pub fn serves(&self, path: &str) -> Option<bool> {
        self.capabilities.as_ref().map(|c| c.serves(path))
    }

    /// The server's profile (`single_model`, `multi_model`,
    /// `public_explorer`), when it reported one.
    pub fn profile(&self) -> Option<&str> {
        self.capabilities.as_ref().map(|c| c.profile.as_str())
    }

    fn cancel_for_current_call(&self) -> CancelToken {
        // Fresh per-call token.  The tick callback (`pg_interrupt_tick`)
        // observes PG's `InterruptPending` flag and flips this token via
        // the `CancellableClient::*_with_tick` APIs.
        CancelToken::new()
    }

    /// Shorthand for issuing a GET to the server, with PG interrupt
    /// polling wired into the wait loop.
    fn get_json<T: serde::de::DeserializeOwned>(&self, path: &str) -> Result<T, PgInferError> {
        let cancel = self.cancel_for_current_call();
        self.client
            .get_json_with_tick(path, &cancel, crate::interrupt::pg_interrupt_tick)
            .map_err(map_err)
    }

    /// Shorthand for issuing a POST to the server, with PG interrupt
    /// polling wired into the wait loop.
    fn post_json<T: serde::de::DeserializeOwned>(
        &self,
        path: &str,
        body: serde_json::Value,
    ) -> Result<T, PgInferError> {
        let cancel = self.cancel_for_current_call();
        self.client
            .post_json_with_tick(path, body, &cancel, crate::interrupt::pg_interrupt_tick)
            .map_err(map_err)
    }

    /// Pre-warm the server: prefetch layer pages and load inference
    /// weights before the first query pays for it.
    ///
    /// Returns the server's own [`WarmupResponse`]. Returns `None` when
    /// the server does not serve `/v1/warmup`.
    ///
    /// This used to take an `entities: &[String]` and post
    /// `{"entities": [...]}`, which the server ignores entirely — warmup
    /// prefetches *layers*, and has no per-entity cache to populate. The
    /// old return of `(warmed, already_cached)` was always `(0, 0)`
    /// because neither field exists in the response.
    pub fn warmup(
        &self,
        layers: Option<&[usize]>,
    ) -> Result<Option<infer_client::WarmupResponse>, PgInferError> {
        if self.serves("/v1/warmup") == Some(false) {
            return Ok(None);
        }
        // Omit `layers` entirely to warm every owned layer, which is the
        // server's default and the case callers want at registration.
        let body = match layers {
            Some(l) => serde_json::json!({ "layers": l }),
            None => serde_json::json!({}),
        };
        optional_endpoint(self.post_json::<infer_client::WarmupResponse>("/v1/warmup", body))
    }

    /// Fetch server-side cache statistics.
    ///
    /// Tries `/v1/cache/stats` first, then falls back to the `server`
    /// block on `/v1/stats`. The fallback is the path that actually runs:
    /// `/v1/cache/stats` is a pg_infer-local invention no released
    /// larql-server has ever served (see
    /// `docs/src/compatibility/upstream.md`), so before the fallback
    /// existed `infer_server_stats()` returned an empty set against every
    /// real server.
    ///
    /// Returns `None` only when neither source reports anything.
    pub fn cache_stats(&self) -> Result<Option<CacheStats>, PgInferError> {
        if self.serves("/v1/cache/stats") != Some(false) {
            let resp = optional_endpoint(
                self.get_json::<infer_client::CacheStatsResponse>("/v1/cache/stats"),
            )?;
            if let Some(resp) = resp {
                return Ok(Some(CacheStats {
                    entries: resp.entries,
                    hit_count: resp.hit_count,
                    miss_count: resp.miss_count,
                    eviction_count: resp.eviction_count,
                    memory_bytes: resp.memory_bytes,
                }));
            }
        }

        // Fallback: the `server` block on /v1/stats.
        let stats: infer_client::StatsResponse = self.get_json("/v1/stats")?;
        let Some(kv) = stats.server.and_then(|s| s.v3_kv) else {
            return Ok(None);
        };
        if !kv.enabled {
            // Reporting zeros for a disabled cache would look like a cache
            // that is being missed rather than one that is switched off.
            return Ok(None);
        }
        Ok(Some(CacheStats {
            entries: kv.entries,
            hit_count: kv.hits,
            miss_count: kv.misses,
            // The server reports a bounded capacity, not an eviction
            // counter. Left at 0 rather than derived from capacity, which
            // would be a fabricated number.
            eviction_count: 0,
            // Not reported: the KV cache is bounded by entry count.
            memory_bytes: 0,
        }))
    }
}

/// Treat "the server does not have this endpoint" as `Ok(None)` and every
/// other failure as an error.
///
/// One place that inspects error text for a 404, rather than five. The
/// text match is a fallback for servers that do not report capabilities;
/// callers consult [`RemoteBackend::serves`] first and skip the request
/// entirely when the server has already said it does not serve the route.
///
/// Matching on a message substring is fragile — it depends on how the
/// transport formats a status — but it is the only signal available from
/// a server that predates `/v1/capabilities`, and being wrong here fails
/// safe: a misclassified error degrades an optional feature rather than
/// corrupting a result.
fn optional_endpoint<T>(r: Result<T, PgInferError>) -> Result<Option<T>, PgInferError> {
    match r {
        Ok(v) => Ok(Some(v)),
        Err(PgInferError::Remote(ref msg)) if msg.contains("404") || msg.contains("Not Found") => {
            Ok(None)
        }
        Err(e) => Err(e),
    }
}

fn map_err(e: ClientError) -> PgInferError {
    match e {
        ClientError::Cancelled => {
            // PG's InterruptPending was observed; turn that into a
            // proper SQL-level ERROR via ProcessInterrupts.  If the
            // cancel came from somewhere else (unlikely given how the
            // tick callback is wired) this is a no-op and we fall
            // through to the plain Remote error below.
            crate::interrupt::raise_if_pending();
            PgInferError::Remote("request cancelled".into())
        }
        other => PgInferError::Remote(other.to_string()),
    }
}

fn unsupported<T>(op: &str) -> Result<T, PgInferError> {
    Err(PgInferError::RemoteUnsupported {
        operation: op.to_string(),
    })
}

/// Build the JSON body for a `POST /v1/infer` request.
///
/// Always sets `mode: "dense"`.  larql-server defaults to walk-mode
/// when `mode` is omitted, which on a dense-only BitNet vindex
/// (`--keep-quant --dense-only`) has no gate vectors and returns
/// nothing useful.  Dense mode runs the native-ternary forward pass
/// — the next-token prediction pg_infer wants.
fn infer_request_body(prompt: &str, top_k: usize) -> serde_json::Value {
    serde_json::json!({
        "prompt": prompt,
        "top": top_k,
        "mode": "dense",
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
        let stats: serde_json::Value = self.get_json("/v1/stats")?;

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

        let resp: infer_client::DescribeResponse = self.get_json(&path)?;

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

    /// Pipelined `describe` for many entities — one batch, N concurrent
    /// requests over the same connection (HTTP/2 multiplexed, or
    /// concurrent over UDS).
    ///
    /// This is the difference between `describe()` being a per-row cost
    /// and a per-statement one. Sequentially, a 100-entity join pays the
    /// full round trip 100 times; batched it pays roughly one, bounded by
    /// the slowest response rather than their sum.
    fn describe_many(
        &self,
        entities: &[String],
        explicit_threshold: Option<f64>,
    ) -> Result<Vec<Vec<Edge>>, PgInferError> {
        if entities.is_empty() {
            return Ok(Vec::new());
        }
        let threshold = explicit_threshold
            .filter(|t| *t > 0.0)
            .map(|t| t as f32)
            .unwrap_or(0.0);
        let top_k = crate::gucs::describe_top_k();

        let items: Vec<BatchItem> = entities
            .iter()
            .map(|entity| {
                let mut url = format!(
                    "/v1/describe?entity={}&limit={}",
                    urlencoding(entity),
                    top_k
                );
                if threshold > 0.0 {
                    url.push_str(&format!("&min_score={threshold}"));
                }
                BatchItem {
                    url,
                    method: Method::Get,
                    body: None,
                }
            })
            .collect();

        let cancel = self.cancel_for_current_call();
        let results: Vec<Result<infer_client::DescribeResponse, _>> = self
            .client
            .batch_with_tick(items, &cancel, crate::interrupt::pg_interrupt_tick)
            .map_err(map_err)?;

        // Preserve input order and surface the first failure rather than
        // silently returning a short list — a caller joining on this needs
        // to know a row is missing, not infer it from a count.
        let mut out = Vec::with_capacity(entities.len());
        for r in results {
            let resp = r.map_err(map_err)?;
            out.push(
                resp.edges
                    .into_iter()
                    .map(|e| Edge {
                        relation: e.relation,
                        target: e.target,
                        gate_score: e.gate_score as f64,
                        layer: e.layer as i32,
                    })
                    .collect(),
            );
        }
        if out.len() != entities.len() {
            return Err(PgInferError::Remote(format!(
                "describe batch returned {} results for {} entities",
                out.len(),
                entities.len()
            )));
        }
        Ok(out)
    }

    fn walk(&self, prompt: &str, top_k: usize) -> Result<Vec<Hit>, PgInferError> {
        let path = format!("/v1/walk?prompt={}&top={}", urlencoding(prompt), top_k);
        let resp: infer_client::WalkResponse = self.get_json(&path)?;

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
        let resp: infer_client::WalkResponse = self.get_json(&path)?;

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
        // One batch, two walks, one round trip (HTTP/2 multiplexed).
        // The server's L2 activation cache still catches recurring
        // query text — this just halves the cold path.
        let top_k = 50_usize;
        let items = vec![
            BatchItem {
                url: format!("/v1/walk?prompt={}&top={}", urlencoding(a), top_k),
                method: Method::Get,
                body: None,
            },
            BatchItem {
                url: format!("/v1/walk?prompt={}&top={}", urlencoding(b), top_k),
                method: Method::Get,
                body: None,
            },
        ];
        let cancel = self.cancel_for_current_call();
        let results: Vec<Result<infer_client::WalkResponse, _>> = self
            .client
            .batch_with_tick(items, &cancel, crate::interrupt::pg_interrupt_tick)
            .map_err(map_err)?;

        let mut walks = results.into_iter();
        let resp_a = walks
            .next()
            .ok_or_else(|| PgInferError::Remote("batch missing result 0".into()))?
            .map_err(map_err)?;
        let resp_b = walks
            .next()
            .ok_or_else(|| PgInferError::Remote("batch missing result 1".into()))?
            .map_err(map_err)?;

        Ok(overlap_max(&resp_a.hits, &resp_b.hits))
    }

    fn similar_to_many(
        &self,
        candidates: &[String],
        query: &str,
    ) -> Result<Vec<f64>, PgInferError> {
        // Fire one walk for the query + N walks for candidates, all
        // concurrent in a single runtime round trip.
        let top_k = 50_usize;
        let mut items = Vec::with_capacity(candidates.len() + 1);
        items.push(BatchItem {
            url: format!("/v1/walk?prompt={}&top={}", urlencoding(query), top_k),
            method: Method::Get,
            body: None,
        });
        for cand in candidates {
            items.push(BatchItem {
                url: format!("/v1/walk?prompt={}&top={}", urlencoding(cand), top_k),
                method: Method::Get,
                body: None,
            });
        }
        let cancel = self.cancel_for_current_call();
        let results: Vec<Result<infer_client::WalkResponse, _>> = self
            .client
            .batch_with_tick(items, &cancel, crate::interrupt::pg_interrupt_tick)
            .map_err(map_err)?;

        let mut iter = results.into_iter();
        let query_resp = iter
            .next()
            .ok_or_else(|| PgInferError::Remote("batch missing query result".into()))?
            .map_err(map_err)?;

        let mut scores = Vec::with_capacity(candidates.len());
        for r in iter {
            match r {
                Ok(cand_resp) => scores.push(overlap_max(&cand_resp.hits, &query_resp.hits)),
                // Individual row failure: return 0.0 rather than aborting
                // the whole scan.  Mirrors pgvector's NULL-on-dimension-
                // mismatch behaviour for ORDER BY.
                Err(_) => scores.push(0.0),
            }
        }
        Ok(scores)
    }

    fn implies(&self, subject: &str, object: &str) -> Result<bool, PgInferError> {
        let obj_lower = object.to_lowercase();
        let edges = self.describe(subject, None)?;
        Ok(edges.iter().any(|e| e.target.to_lowercase() == obj_lower))
    }

    fn infer(&self, prompt: &str, top_k: usize) -> Result<Vec<Prediction>, PgInferError> {
        let body = infer_request_body(prompt, top_k);
        let resp: infer_client::InferResponse = self.post_json("/v1/infer", body)?;

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
        let resp: infer_client::RelationsResponse = self.get_json("/v1/relations")?;

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
        layer: usize,
        filter: Option<&str>,
        min_score: f32,
        limit: usize,
    ) -> Result<Vec<FeatureRow>, PgInferError> {
        // `/v1/select` is larql's server-side feature scan: it walks a
        // layer's feature metadata and filters on the top token, which is
        // exactly what `infer_show_features` asks for. Note this is *not*
        // gate-KNN like `describe()` — it is a metadata scan, so pushing
        // the filter down avoids shipping every feature in the layer.
        if self.serves("/v1/select") == Some(false) {
            return unsupported("show_features (server serves no /v1/select)");
        }
        let mut body = serde_json::json!({
            "layer": layer,
            "limit": limit,
            // Accepted values are gate_score / confidence / c_score /
            // layer; anything else falls through to the server's default.
            // Naming one explicitly rather than relying on that default.
            "order_by": "c_score",
            "order": "desc",
        });
        // `entity` is a case-insensitive substring match on the feature's
        // top token, which is what the local backend's `filter` does too.
        if let Some(f) = filter {
            body["entity"] = serde_json::json!(f);
        }
        if min_score > 0.0 {
            body["min_confidence"] = serde_json::json!(min_score);
        }

        let Some(resp) =
            optional_endpoint(self.post_json::<infer_client::SelectResponse>("/v1/select", body))?
        else {
            return unsupported("show_features (server serves no /v1/select)");
        };

        Ok(resp
            .edges
            .into_iter()
            .map(|r| FeatureRow {
                feature: r.feature as i32,
                token: r.target,
                score: r.c_score as f64,
                // The local backend fills `also` with the feature's other
                // top tokens; `/v1/select` reports only the top one, so
                // this is empty rather than guessed. The probe relation
                // label is the useful thing the server *does* add.
                also: r.relation.unwrap_or_default(),
            })
            .collect())
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

    fn rank(
        &self,
        candidates: &[String],
        query: &str,
        limit: usize,
    ) -> Result<Vec<RankedCandidate>, PgInferError> {
        // `/v1/rank` is pg_infer-local and has never existed upstream, so
        // a capability-reporting server lets us go straight to the
        // fallback instead of paying a 404 on every call.
        let served = self.serves("/v1/rank") != Some(false);
        let body = serde_json::json!({
            "query": query,
            "candidates": candidates,
            "top_k": limit,
        });
        let remote = if served {
            optional_endpoint(self.post_json::<infer_client::RankResponse>("/v1/rank", body))?
        } else {
            None
        };
        if let Some(resp) = remote {
            return Ok(resp
                .results
                .into_iter()
                .map(|r| RankedCandidate {
                    index: r.index,
                    score: r.score,
                })
                .collect());
        }
        // No server-side ranking: score each candidate with a walk and
        // sort here.
        let scores = self.similar_to_many(candidates, query)?;
        let mut ranked: Vec<RankedCandidate> = scores
            .into_iter()
            .enumerate()
            .map(|(i, s)| RankedCandidate { index: i, score: s })
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

    fn embed(&self, text: &str) -> Result<Array1<f32>, PgInferError> {
        // Two hops, because `/v1/embed` takes token ids: `/v1/token/encode`
        // to tokenize with the *server's* tokenizer, then `/v1/embed` to
        // look up the rows. Doing it server-side matters — a remote model
        // has no local tokenizer or embedding table here, and encoding
        // with a different tokenizer would silently produce vectors from
        // the wrong rows.
        //
        // Mean-pools multi-token input to match the local backend
        // (`fn_similar::embed_text`): same vector for the same text
        // regardless of backend, which `am_scan`'s HNSW query path
        // depends on. The server has already applied `embed_scale`.
        if self.serves("/v1/embed") == Some(false) || self.serves("/v1/token/encode") == Some(false)
        {
            return unsupported("embed (server serves no /v1/embed)");
        }

        // `/v1/token/encode` is a GET with a `text` query param, and the
        // server encodes with `add_special_tokens=false` — the same call
        // `fn_similar::embed_text` makes locally, so the token ids agree.
        let enc: infer_client::TokenEncodeResponse =
            self.get_json(&format!("/v1/token/encode?text={}", urlencoding(text)))?;
        if enc.token_ids.is_empty() {
            return Err(PgInferError::EmptyPrompt);
        }

        let resp: infer_client::EmbedResponse = self.post_json(
            "/v1/embed",
            serde_json::json!({ "token_ids": enc.token_ids }),
        )?;

        let mut rows = resp.residual.into_iter();
        let Some(first) = rows.next() else {
            return Err(PgInferError::Remote(format!(
                "/v1/embed returned no rows for {} token(s)",
                enc.token_ids.len()
            )));
        };
        let hidden = first.len();
        let mut acc = Array1::<f32>::from(first);
        let mut n = 1usize;
        for row in rows {
            if row.len() != hidden {
                return Err(PgInferError::Remote(format!(
                    "/v1/embed rows disagree on width: {hidden} vs {}",
                    row.len()
                )));
            }
            acc += &Array1::from(row);
            n += 1;
        }
        if n > 1 {
            acc /= n as f32;
        }
        Ok(acc)
    }

    fn warmup(
        &self,
        layers: Option<&[usize]>,
    ) -> Result<Option<infer_client::WarmupResponse>, PgInferError> {
        RemoteBackend::warmup(self, layers)
    }

    fn cache_stats(&self) -> Result<Option<super::CacheStats>, PgInferError> {
        RemoteBackend::cache_stats(self)
    }
}

/// Max overlap between two sparse gate-KNN result sets.  Matches the
/// similarity heuristic in fn_similar::similar_to_impl: for every
/// feature that fires in both, the contribution is `min(score_a,
/// score_b)`; the result is the max across all such features.
fn overlap_max(hits_a: &[infer_client::WalkHit], hits_b: &[infer_client::WalkHit]) -> f64 {
    let mut b_map: HashMap<(usize, usize), f32> = HashMap::new();
    for h in hits_b {
        b_map.insert((h.layer, h.feature), h.gate_score);
    }
    let mut max_shared = 0.0f32;
    for h in hits_a {
        if let Some(&score_b) = b_map.get(&(h.layer, h.feature)) {
            let shared = h.gate_score.min(score_b);
            if shared > max_shared {
                max_shared = shared;
            }
        }
    }
    max_shared as f64
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

    #[test]
    fn infer_request_always_sends_dense_mode() {
        // Regression: /v1/infer must carry mode:"dense".  Omitting it
        // lets larql-server default to walk-mode, which returns
        // nothing useful on a dense-only BitNet vindex.
        let body = infer_request_body("the capital of France is", 5);
        assert_eq!(body["prompt"], "the capital of France is");
        assert_eq!(body["top"], 5);
        assert_eq!(body["mode"], "dense");
    }
}
