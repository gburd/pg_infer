//! Integration test: drive the `CancellableClient` against a mock HTTP
//! server that implements the subset of larql-server's API pg_infer
//! consumes.  This is the contract test for the remote backend; if it
//! passes, pg_infer's parsing and URL construction match what a real
//! larql-server emits.
//!
//! The mock uses hyper directly to avoid pulling in a full server
//! framework, and covers every endpoint pg_infer touches:
//!
//! | Endpoint        | Verified                                       |
//! |-----------------|------------------------------------------------|
//! | GET /v1/stats   | Layer count, hidden size, layer bands parse    |
//! | GET /v1/describe| entity query-param, edges array parses          |
//! | GET /v1/walk    | prompt + top + optional layers query-params     |
//! | GET /v1/relations | relations summary parses                      |
//! | POST /v1/infer  | JSON body + predictions array parses            |
//!
//! Request routing is asserted by the mock: if pg_infer ever sends a
//! URL that doesn't match an expected pattern, the mock returns 404
//! and the test fails loudly.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::{Request, Response};
use hyper_util::rt::TokioIo;
use infer_client::{
    CancelToken, CancellableClient, DescribeResponse, InferResponse, RelationsResponse,
    StatsResponse, WalkResponse,
};
use tokio::net::TcpListener;

/// Records every request path the mock received, in order.
type RequestLog = Arc<Mutex<Vec<(String, String)>>>;

/// Spin up a mock server on an ephemeral TCP port with a 10-second
/// lifetime.  Returns the base URL and a handle to the request log.
fn spawn_mock_server() -> (String, RequestLog, std::thread::JoinHandle<()>) {
    let log: RequestLog = Arc::new(Mutex::new(Vec::new()));
    let log_server = log.clone();

    // We need the bound port before returning — bind synchronously, then
    // hand the listener to the runtime thread.
    let listener_std = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
    listener_std.set_nonblocking(true).expect("set nonblocking");
    let addr = listener_std.local_addr().expect("local_addr");
    let base = format!("http://{}", addr);

    let handle = std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("rt");
        rt.block_on(async move {
            let listener = TcpListener::from_std(listener_std).expect("from_std");
            let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
            loop {
                tokio::select! {
                    _ = tokio::time::sleep_until(deadline) => return,
                    res = listener.accept() => {
                        let (stream, _) = match res {
                            Ok(v) => v,
                            Err(_) => continue,
                        };
                        let log_conn = log_server.clone();
                        tokio::spawn(async move {
                            let io = TokioIo::new(stream);
                            let svc = hyper::service::service_fn(move |req: Request<Incoming>| {
                                let log_req = log_conn.clone();
                                async move { handle_request(req, log_req).await }
                            });
                            let _ = hyper::server::conn::http1::Builder::new()
                                .serve_connection(io, svc)
                                .await;
                        });
                    }
                }
            }
        });
    });

    (base, log, handle)
}

async fn handle_request(
    req: Request<Incoming>,
    log: RequestLog,
) -> Result<Response<Full<Bytes>>, std::io::Error> {
    let method = req.method().to_string();
    let path_and_query = req
        .uri()
        .path_and_query()
        .map(|pq| pq.to_string())
        .unwrap_or_default();
    log.lock().expect("log lock").push((method.clone(), path_and_query.clone()));

    // Split path and query string.
    let (path, query) = match path_and_query.split_once('?') {
        Some((p, q)) => (p.to_string(), parse_query(q)),
        None => (path_and_query.clone(), HashMap::new()),
    };

    let body_bytes = match req.into_body().collect().await {
        Ok(c) => c.to_bytes(),
        Err(_) => Bytes::new(),
    };

    let json: serde_json::Value = match (method.as_str(), path.as_str()) {
        ("GET", "/v1/stats") => serde_json::json!({
            "model": "mock-model",
            "family": "mock",
            "layers": 33,
            "hidden_size": 2560,
            "vocab_size": 262144,
            "extract_level": "browse",
            "features_per_layer": 16384,
            "layer_bands": {
                "syntax":    [0, 6],
                "knowledge": [7, 26],
                "output":    [27, 32],
            },
        }),

        ("GET", "/v1/describe") => {
            let entity = query.get("entity").cloned().unwrap_or_default();
            serde_json::json!({
                "entity": entity,
                "model": "mock-model",
                "edges": [
                    {"target": "Paris", "gate_score": 42.7, "layer": 18, "relation": "capital"},
                    {"target": "French", "gate_score": 38.1, "layer": 17},
                    {"target": "Europe", "gate_score": 35.4, "layer": 16},
                ],
                "latency_ms": 0.8,
            })
        }

        ("GET", "/v1/walk") => {
            let prompt = query.get("prompt").cloned().unwrap_or_default();
            let layer_filter: Option<usize> = query
                .get("layers")
                .and_then(|s| s.parse().ok());
            let mut hits = vec![
                serde_json::json!({"layer": 5, "feature": 3401, "gate_score": 31.47, "target": "European"}),
                serde_json::json!({"layer": 12, "feature": 467, "gate_score": 44.82, "target": "Paris"}),
                serde_json::json!({"layer": 18, "feature": 2103, "gate_score": 52.19, "target": "geography"}),
            ];
            if let Some(n) = layer_filter {
                hits.retain(|h| h["layer"].as_u64() == Some(n as u64));
            }
            serde_json::json!({
                "prompt": prompt,
                "model": "mock-model",
                "hits": hits,
                "latency_ms": 0.3,
            })
        }

        ("GET", "/v1/relations") => serde_json::json!({
            "relations": [
                {"token": "capital", "count": 42, "max_score": 44.1, "layers": [14, 15, 17], "examples": ["Paris", "Rome", "Tokyo"]},
                {"token": "language", "count": 31, "max_score": 39.8, "layers": [12, 13], "examples": ["French", "Spanish"]},
            ],
        }),

        ("POST", "/v1/infer") => {
            let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap_or_default();
            let top = body.get("top").and_then(|v| v.as_u64()).unwrap_or(5) as usize;
            let mut preds = Vec::new();
            for (i, tok) in ["Paris", "Lyon", "Nice", "Marseille", "Bordeaux"].iter().enumerate() {
                if i >= top {
                    break;
                }
                preds.push(serde_json::json!({
                    "token": tok,
                    "probability": 1.0 - (i as f64 * 0.15),
                }));
            }
            serde_json::json!({
                "predictions": preds,
                "latency_ms": 12.4,
            })
        }

        _ => {
            return Ok(Response::builder()
                .status(404)
                .body(Full::new(Bytes::from_static(b"not found")))
                .expect("resp 404"));
        }
    };

    let bytes = serde_json::to_vec(&json).expect("to_vec");
    Ok(Response::builder()
        .status(200)
        .header("content-type", "application/json")
        .body(Full::new(Bytes::from(bytes)))
        .expect("resp ok"))
}

fn parse_query(q: &str) -> HashMap<String, String> {
    let mut out = HashMap::new();
    for pair in q.split('&') {
        if let Some((k, v)) = pair.split_once('=') {
            out.insert(k.to_string(), percent_decode(v));
        } else if !pair.is_empty() {
            out.insert(pair.to_string(), String::new());
        }
    }
    out
}

fn percent_decode(s: &str) -> String {
    let mut out = Vec::new();
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).unwrap_or("00");
            let v = u8::from_str_radix(hex, 16).unwrap_or(0);
            out.push(v);
            i += 3;
        } else if bytes[i] == b'+' {
            out.push(b' ');
            i += 1;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(out).unwrap_or_default()
}

// ── Test cases ────────────────────────────────────────────────────────────────

#[test]
fn stats_round_trip() {
    let (base, log, _h) = spawn_mock_server();
    let client =
        CancellableClient::connect(&base, Duration::from_secs(5)).expect("client builds");
    let cancel = CancelToken::new();

    let stats: StatsResponse = client.get_json("/v1/stats", &cancel).expect("stats");
    assert_eq!(stats.model, "mock-model");
    assert_eq!(stats.layers, 33);
    assert_eq!(stats.hidden_size, 2560);
    assert_eq!(stats.vocab_size, 262144);
    let bands = stats.layer_bands.expect("bands present");
    assert_eq!(bands.knowledge, [7, 26]);

    let requests = log.lock().expect("lock");
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0], ("GET".into(), "/v1/stats".into()));
}

#[test]
fn describe_with_entity_query_param() {
    let (base, log, _h) = spawn_mock_server();
    let client =
        CancellableClient::connect(&base, Duration::from_secs(5)).expect("client builds");
    let cancel = CancelToken::new();

    // Issue the describe URL pg_infer's RemoteBackend builds.
    let resp: DescribeResponse = client
        .get_json("/v1/describe?entity=France&limit=20", &cancel)
        .expect("describe");
    assert_eq!(resp.entity, "France");
    assert_eq!(resp.edges.len(), 3);
    assert_eq!(resp.edges[0].target, "Paris");
    assert_eq!(resp.edges[0].gate_score, 42.7);
    assert_eq!(resp.edges[0].relation, "capital");
    assert_eq!(resp.edges[0].layer, 18);

    let requests = log.lock().expect("lock");
    assert_eq!(requests[0].0, "GET");
    assert!(
        requests[0].1.starts_with("/v1/describe?entity=France"),
        "unexpected path {}",
        requests[0].1
    );
}

#[test]
fn walk_all_layers_vs_single_layer() {
    let (base, _log, _h) = spawn_mock_server();
    let client =
        CancellableClient::connect(&base, Duration::from_secs(5)).expect("client builds");
    let cancel = CancelToken::new();

    // All layers
    let all: WalkResponse = client
        .get_json("/v1/walk?prompt=hello&top=3", &cancel)
        .expect("walk all");
    assert_eq!(all.hits.len(), 3);

    // Single layer filter (nearest_to style).
    let one: WalkResponse = client
        .get_json("/v1/walk?prompt=hello&top=3&layers=12", &cancel)
        .expect("walk L12");
    assert_eq!(one.hits.len(), 1);
    assert_eq!(one.hits[0].layer, 12);
    assert_eq!(one.hits[0].feature, 467);
}

#[test]
fn relations_parses() {
    let (base, _log, _h) = spawn_mock_server();
    let client =
        CancellableClient::connect(&base, Duration::from_secs(5)).expect("client builds");
    let cancel = CancelToken::new();

    let resp: RelationsResponse = client
        .get_json("/v1/relations", &cancel)
        .expect("relations");
    assert_eq!(resp.relations.len(), 2);
    assert_eq!(resp.relations[0].token, "capital");
    assert_eq!(resp.relations[0].count, 42);
    assert_eq!(resp.relations[0].layers.len(), 3);
    assert_eq!(resp.relations[0].examples.len(), 3);
}

#[test]
fn infer_post_body_and_parse() {
    let (base, log, _h) = spawn_mock_server();
    let client =
        CancellableClient::connect(&base, Duration::from_secs(5)).expect("client builds");
    let cancel = CancelToken::new();

    let body = serde_json::json!({
        "prompt": "The capital of France is",
        "top": 3,
    });
    let resp: InferResponse = client
        .post_json("/v1/infer", body, &cancel)
        .expect("infer");
    assert_eq!(resp.predictions.len(), 3);
    assert_eq!(resp.predictions[0].token, "Paris");
    assert!(resp.predictions[0].probability > 0.99);

    let requests = log.lock().expect("lock");
    assert_eq!(requests[0], ("POST".into(), "/v1/infer".into()));
}

#[test]
fn batch_issues_concurrent_requests() {
    use infer_client::{BatchItem, Method};

    let (base, log, _h) = spawn_mock_server();
    let client =
        CancellableClient::connect(&base, Duration::from_secs(5)).expect("client builds");
    let cancel = CancelToken::new();

    let items = vec![
        BatchItem {
            url: "/v1/walk?prompt=France&top=50".into(),
            method: Method::Get,
            body: None,
        },
        BatchItem {
            url: "/v1/walk?prompt=Paris&top=50".into(),
            method: Method::Get,
            body: None,
        },
    ];
    let results: Vec<Result<WalkResponse, _>> = client
        .batch_with_tick(items, &cancel, || Ok(()))
        .expect("batch");
    assert_eq!(results.len(), 2);
    assert!(results[0].is_ok());
    assert!(results[1].is_ok());

    // Both requests should be logged; order isn't guaranteed with
    // concurrent fanout but both must be present.
    let requests = log.lock().expect("lock");
    let paths: Vec<String> = requests.iter().map(|(_, p)| p.clone()).collect();
    assert!(paths.iter().any(|p| p.contains("France")));
    assert!(paths.iter().any(|p| p.contains("Paris")));
}

#[test]
fn unknown_path_yields_server_error() {
    let (base, _log, _h) = spawn_mock_server();
    let client =
        CancellableClient::connect(&base, Duration::from_secs(5)).expect("client builds");
    let cancel = CancelToken::new();

    let err: Result<serde_json::Value, _> = client.get_json("/v1/bogus", &cancel);
    match err {
        Err(infer_client::ClientError::Server { status, .. }) => {
            assert_eq!(status, 404);
        }
        other => panic!("expected 404 ClientError::Server, got {other:?}"),
    }
}
