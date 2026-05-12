//! Grid-routed `Backend` implementation.
//!
//! A `GridBackend` represents a model registered with `backend='grid'`.
//! Rather than pointing at a single fixed server, it discovers available
//! servers from a grid discovery endpoint (`/v1/models`) and maintains a
//! route table that's refreshed in the background.  Queries are dispatched
//! round-robin to the available servers with zero resolution latency.
//!
//! The discovery endpoint is polled at `infer.grid_poll_interval` seconds
//! (default 30).  Each discovered server that hosts the target model gets
//! its own `RemoteBackend` (and thus its own `CancellableClient` + runtime
//! thread).  When a server disappears from the grid, its `RemoteBackend` is
//! dropped (which sends `Cmd::Shutdown` to clean up its runtime thread).

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use ndarray::Array1;

use crate::error::PgInferError;

use super::remote::RemoteBackend;
use super::{
    Backend, CacheStats, Edge, ExplainedHit, FeatureMetaLite, FeatureRow, FeatureSnapshot, Hit,
    LayerInfo, Prediction, RankedCandidate, RelationRow,
};

/// A grid-routed backend: multiple servers behind round-robin dispatch.
pub struct GridBackend {
    /// Model ID used for routing / discovery filtering.
    model_id: String,
    /// Shared route table, updated by the poller thread.
    route_table: Arc<RwLock<RouteTable>>,
    /// Signal to stop the poller.
    stop: Arc<AtomicBool>,
    /// Handle to the background poller (joined on Drop).
    _poller: Option<JoinHandle<()>>,
}

struct RouteTable {
    /// Available servers for this model, with their backends.
    servers: Vec<ServerEntry>,
    /// Round-robin index.
    next: AtomicUsize,
    /// Last successful refresh time.
    last_refresh: Option<Instant>,
}

struct ServerEntry {
    url: String,
    backend: RemoteBackend,
}

/// Response shape for `GET /v1/models` from a larql-router or server.
#[derive(Debug, serde::Deserialize)]
struct ModelsResponse {
    #[serde(default)]
    models: Vec<ModelEntry>,
}

#[derive(Debug, serde::Deserialize)]
struct ModelEntry {
    #[serde(default)]
    model: String,
    #[serde(default, alias = "model_id")]
    id: String,
    #[serde(default)]
    url: String,
    #[serde(default, alias = "server_url")]
    server: String,
}

impl GridBackend {
    /// Connect to the grid and start the background poller.
    ///
    /// Does an initial synchronous discovery so at least one server is
    /// available (or returns an error if the grid has no servers for this
    /// model).
    pub fn connect(
        model_id: &str,
        grid_url: &str,
        poll_interval: Duration,
        timeout: Duration,
    ) -> Result<Self, PgInferError> {
        let route_table = Arc::new(RwLock::new(RouteTable {
            servers: Vec::new(),
            next: AtomicUsize::new(0),
            last_refresh: None,
        }));
        let stop = Arc::new(AtomicBool::new(false));

        // Initial synchronous discovery.
        discover_and_update(model_id, grid_url, timeout, &route_table)?;

        // Verify we got at least one server.
        {
            let table = route_table
                .read()
                .map_err(|e| PgInferError::Internal(format!("route table lock: {e}")))?;
            if table.servers.is_empty() {
                return Err(PgInferError::Remote(format!(
                    "no servers found for model '{}' in grid at {}",
                    model_id, grid_url
                )));
            }
        }

        // Spawn background poller.
        let poller_table = Arc::clone(&route_table);
        let poller_stop = Arc::clone(&stop);
        let poller_model_id = model_id.to_string();
        let poller_grid_url = grid_url.to_string();
        let poller_timeout = timeout;

        let poller = std::thread::Builder::new()
            .name(format!("infer-grid-{}", model_id))
            .spawn(move || {
                poller_loop(
                    &poller_model_id,
                    &poller_grid_url,
                    poller_timeout,
                    poll_interval,
                    &poller_table,
                    &poller_stop,
                );
            })
            .map_err(|e| PgInferError::Internal(format!("spawn grid poller: {e}")))?;

        Ok(Self {
            model_id: model_id.to_string(),
            route_table,
            stop,
            _poller: Some(poller),
        })
    }

    /// Get a reference to the next available backend (round-robin).
    fn with_server<F, R>(&self, f: F) -> Result<R, PgInferError>
    where
        F: FnOnce(&RemoteBackend) -> Result<R, PgInferError>,
    {
        let table = self
            .route_table
            .read()
            .map_err(|e| PgInferError::Internal(format!("route table lock: {e}")))?;

        if table.servers.is_empty() {
            return Err(PgInferError::Remote(format!(
                "no servers available for model '{}' in grid",
                self.model_id
            )));
        }

        let idx = table.next.fetch_add(1, Ordering::Relaxed) % table.servers.len();
        f(&table.servers[idx].backend)
    }
}

impl Drop for GridBackend {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        // Don't join — the poller will exit on its next poll_interval tick.
        // Joining here would block the PG backend if the poller is mid-sleep.
    }
}

// ── Discovery logic ──────────────────────────────────────────────────────────

fn discover_and_update(
    model_id: &str,
    grid_url: &str,
    timeout: Duration,
    route_table: &Arc<RwLock<RouteTable>>,
) -> Result<(), PgInferError> {
    let discovered = discover_servers(model_id, grid_url, timeout)?;

    let mut table = route_table
        .write()
        .map_err(|e| PgInferError::Internal(format!("route table write lock: {e}")))?;

    // Diff: keep existing connections for URLs that are still present,
    // add new ones, remove stale ones.
    let existing_urls: std::collections::HashSet<String> =
        table.servers.iter().map(|s| s.url.clone()).collect();
    let new_urls: std::collections::HashSet<String> =
        discovered.iter().cloned().collect();

    // Remove stale servers.
    table.servers.retain(|s| new_urls.contains(&s.url));

    // Add new servers.
    for url in &discovered {
        if !existing_urls.contains(url) {
            match RemoteBackend::connect(url, timeout) {
                Ok(backend) => {
                    table.servers.push(ServerEntry {
                        url: url.clone(),
                        backend,
                    });
                }
                Err(_) => {
                    // Skip unreachable servers — they'll be retried next poll.
                }
            }
        }
    }

    table.last_refresh = Some(Instant::now());
    Ok(())
}

fn discover_servers(
    model_id: &str,
    grid_url: &str,
    timeout: Duration,
) -> Result<Vec<String>, PgInferError> {
    // Use a temporary CancellableClient to probe the grid's /v1/models.
    use infer_client::{CancelToken, CancellableClient};

    let client = CancellableClient::connect(grid_url, timeout)
        .map_err(|e| PgInferError::Remote(format!("grid connect {grid_url}: {e}")))?;

    let cancel = CancelToken::new();
    let resp: ModelsResponse = client
        .get_json("/v1/models", &cancel)
        .map_err(|e| PgInferError::Remote(format!("grid /v1/models: {e}")))?;

    // Filter for entries that match our model_id and have a reachable URL.
    let servers: Vec<String> = resp
        .models
        .into_iter()
        .filter(|entry| {
            entry.model == model_id
                || entry.id == model_id
                || entry.model.is_empty() // Single-model servers list without model field
        })
        .filter_map(|entry| {
            let url = if !entry.url.is_empty() {
                entry.url
            } else if !entry.server.is_empty() {
                entry.server
            } else {
                // Single-model server: use the grid_url itself.
                grid_url.to_string()
            };
            if url.is_empty() {
                None
            } else {
                Some(url)
            }
        })
        .collect();

    // If /v1/models returned nothing but the grid_url itself serves the
    // model (single-server case), use it directly.
    if servers.is_empty() {
        // Try /v1/health or /v1/stats on the grid_url itself.
        let cancel2 = CancelToken::new();
        if let Ok(stats) = client.get_json::<infer_client::StatsResponse>("/v1/stats", &cancel2) {
            if stats.model == model_id || model_id.is_empty() {
                return Ok(vec![grid_url.to_string()]);
            }
        }
    }

    Ok(servers)
}

fn poller_loop(
    model_id: &str,
    grid_url: &str,
    timeout: Duration,
    poll_interval: Duration,
    route_table: &Arc<RwLock<RouteTable>>,
    stop: &AtomicBool,
) {
    loop {
        std::thread::sleep(poll_interval);
        if stop.load(Ordering::SeqCst) {
            return;
        }
        // Best-effort refresh — errors are silently ignored (stale table
        // continues serving until next successful poll).
        let _ = discover_and_update(model_id, grid_url, timeout, route_table);
        if stop.load(Ordering::SeqCst) {
            return;
        }
    }
}

// ── Backend trait implementation ─────────────────────────────────────────────

impl Backend for GridBackend {
    fn approx_resident_bytes(&self) -> usize {
        // Remote backends use negligible local memory.
        0
    }

    fn is_local(&self) -> bool {
        false
    }

    fn num_layers(&self) -> usize {
        self.with_server(|s| Ok(s.num_layers())).unwrap_or(0)
    }

    fn hidden_size(&self) -> usize {
        self.with_server(|s| Ok(s.hidden_size())).unwrap_or(0)
    }

    fn show_layers(&self) -> Result<Vec<LayerInfo>, PgInferError> {
        self.with_server(|s| s.show_layers())
    }

    fn describe(
        &self,
        entity: &str,
        explicit_threshold: Option<f64>,
    ) -> Result<Vec<Edge>, PgInferError> {
        self.with_server(|s| s.describe(entity, explicit_threshold))
    }

    fn walk(&self, prompt: &str, top_k: usize) -> Result<Vec<Hit>, PgInferError> {
        self.with_server(|s| s.walk(prompt, top_k))
    }

    fn explain_walk(&self, prompt: &str, top_k: usize) -> Result<Vec<ExplainedHit>, PgInferError> {
        self.with_server(|s| s.explain_walk(prompt, top_k))
    }

    fn nearest_to(
        &self,
        entity: &str,
        layer: usize,
        top_k: usize,
    ) -> Result<Vec<Hit>, PgInferError> {
        self.with_server(|s| s.nearest_to(entity, layer, top_k))
    }

    fn similar_to(&self, a: &str, b: &str) -> Result<f64, PgInferError> {
        self.with_server(|s| s.similar_to(a, b))
    }

    fn similar_to_many(&self, candidates: &[String], query: &str) -> Result<Vec<f64>, PgInferError> {
        self.with_server(|s| s.similar_to_many(candidates, query))
    }

    fn implies(&self, subject: &str, object: &str) -> Result<bool, PgInferError> {
        self.with_server(|s| s.implies(subject, object))
    }

    fn infer(&self, prompt: &str, top_k: usize) -> Result<Vec<Prediction>, PgInferError> {
        self.with_server(|s| s.infer(prompt, top_k))
    }

    fn show_relations(&self) -> Result<Vec<RelationRow>, PgInferError> {
        self.with_server(|s| s.show_relations())
    }

    fn show_features(
        &self,
        _layer: usize,
        _filter: Option<&str>,
        _min_score: f32,
        _limit: usize,
    ) -> Result<Vec<FeatureRow>, PgInferError> {
        Err(PgInferError::RemoteUnsupported {
            operation: "show_features (grid backend)".into(),
        })
    }

    fn snapshot_features(
        &self,
        _layer_filter: Option<i32>,
    ) -> Result<Vec<FeatureSnapshot>, PgInferError> {
        Err(PgInferError::RemoteUnsupported {
            operation: "infer_diff (grid backend)".into(),
        })
    }

    fn feature_meta_at(&self, _layer: usize, _feature: usize) -> Option<FeatureMetaLite> {
        None
    }

    fn embed(&self, _text: &str) -> Result<Array1<f32>, PgInferError> {
        Err(PgInferError::RemoteUnsupported {
            operation: "embed (grid backend)".into(),
        })
    }

    fn rank(
        &self,
        candidates: &[String],
        query: &str,
        limit: usize,
    ) -> Result<Vec<RankedCandidate>, PgInferError> {
        self.with_server(|s| s.rank(candidates, query, limit))
    }

    fn warmup(&self, entities: &[String]) -> Result<(usize, usize), PgInferError> {
        self.with_server(|s| s.warmup(entities))
    }

    fn cache_stats(&self) -> Result<Option<CacheStats>, PgInferError> {
        self.with_server(|s| s.cache_stats())
    }
}
