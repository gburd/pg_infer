//! Cancellable HTTP / UDS transport.
//!
//! See the crate-level docs for the cancellation model.  This module owns:
//! - [`CancelToken`]: a drop-safe flag the caller sets to abort an in-flight
//!   request.  Cloneable and shareable across threads.
//! - [`CancellableClient`]: the actual client.  One per `RemoteBackend`
//!   instance (i.e. roughly one per loaded remote model per PG backend).
//!   Internally owns a dedicated OS thread running a current-thread tokio
//!   runtime and one of two transports:
//!   - `Transport::Http` — `reqwest::Client` with HTTP/2 + pooling.
//!   - `Transport::Uds`  — raw `hyper_util::client::legacy::Client` over
//!     a `tokio::net::UnixStream` (HTTP/1.1 on the socket file).
//!
//! URL scheme dispatch at [`CancellableClient::connect`] time:
//! - `http://…` / `https://…` → `Http`
//! - `uds:///path/to/sock` or `unix:///path/to/sock` → `Uds`
//!
//! Both transports speak the same JSON API; callers don't need to know
//! which they got.

use std::path::PathBuf;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{sync_channel, RecvTimeoutError, SyncSender};
use std::sync::Arc;
use std::task::{Context, Poll};
use std::thread::JoinHandle;
use std::time::Duration;

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::{Request, Response, Uri};
use hyper_util::client::legacy::Client as HyperClient;
use hyper_util::rt::{TokioExecutor, TokioIo};
use reqwest::Client as ReqwestClient;
use serde::de::DeserializeOwned;
use tokio::net::UnixStream;
use tokio::runtime::Builder;
use tower_service::Service;

/// Maximum request body size the client will build.  64 MiB matches
/// larql-server's `REQUEST_BODY_LIMIT_BYTES`.
const MAX_BODY_BYTES: usize = 64 * 1024 * 1024;

// ── CancelToken ───────────────────────────────────────────────────────────────

/// A cancellation flag shared between the PG foreground thread (which sets
/// it when `QueryCancelPending` is observed) and the background runtime
/// thread (which races its futures against it).
#[derive(Debug, Clone, Default)]
pub struct CancelToken {
    flag: Arc<AtomicBool>,
}

impl CancelToken {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.flag.store(true, Ordering::SeqCst);
    }

    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::SeqCst)
    }

    /// Await cancellation.  Resolves when another holder calls
    /// [`cancel`](Self::cancel).  Used inside `tokio::select!`.
    pub async fn cancelled(&self) {
        // Poll the flag every 20 ms.  This is the worst-case lag between a
        // PG interrupt arriving and the request being unwound; it's below
        // user-perceivable latency and avoids a parking dependency.
        loop {
            if self.is_cancelled() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }
}

// ── ClientError ───────────────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum ClientError {
    #[error("HTTP transport error: {0}")]
    Transport(String),

    #[error("server returned HTTP {status}: {body}")]
    Server { status: u16, body: String },

    #[error("failed to parse response JSON: {0}")]
    Parse(String),

    #[error("request was cancelled")]
    Cancelled,

    #[error("runtime error: {0}")]
    Runtime(String),

    #[error("invalid URL: {0}")]
    InvalidUrl(String),
}

impl From<reqwest::Error> for ClientError {
    fn from(e: reqwest::Error) -> Self {
        ClientError::Transport(e.to_string())
    }
}

// ── UDS connector ─────────────────────────────────────────────────────────────

/// `tower_service::Service<Uri>` that ignores the authority portion of the
/// URI and dials a fixed Unix socket path.
#[derive(Clone, Debug)]
struct UdsConnector {
    path: PathBuf,
}

impl Service<Uri> for UdsConnector {
    type Response = TokioIo<UnixStream>;
    type Error = std::io::Error;
    type Future =
        Pin<Box<dyn std::future::Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, _uri: Uri) -> Self::Future {
        let path = self.path.clone();
        Box::pin(async move {
            let stream = UnixStream::connect(&path).await?;
            Ok(TokioIo::new(stream))
        })
    }
}

// ── Transport ────────────────────────────────────────────────────────────────

enum Transport {
    /// TCP/TLS transport via reqwest.  Handles `http://` and `https://`.
    Http {
        client: ReqwestClient,
        /// `http://host:port` — path is appended at request time.
        base: String,
    },
    /// Unix domain socket transport via raw hyper_util.  The connector
    /// ignores URI authority and always dials the socket path.
    Uds {
        client: HyperClient<UdsConnector, Full<Bytes>>,
        /// Socket path, kept for logging.
        _path: PathBuf,
    },
}

// ── CancellableClient ─────────────────────────────────────────────────────────

/// Command sent from the foreground to the runtime thread.
enum Cmd {
    Request(RequestJob),
    Batch(BatchJob),
    Shutdown,
}

struct RequestJob {
    /// Absolute URL for HTTP, or just the path (`/v1/stats`) for UDS.
    url: String,
    body: Option<serde_json::Value>,
    method: Method,
    cancel: CancelToken,
    /// std::sync::mpsc sender.  Chosen over `tokio::oneshot` so the
    /// foreground thread can call `recv_timeout` and interleave
    /// `tick()` calls for PG's `CHECK_FOR_INTERRUPTS`.
    reply: SyncSender<Result<Vec<u8>, ClientError>>,
}

struct BatchJob {
    items: Vec<BatchItem>,
    cancel: CancelToken,
    reply: SyncSender<Result<Vec<Result<Vec<u8>, ClientError>>, ClientError>>,
}

#[derive(Debug, Clone)]
pub struct BatchItem {
    pub url: String,
    pub method: Method,
    pub body: Option<serde_json::Value>,
}

#[derive(Clone, Copy, Debug)]
pub enum Method {
    Get,
    Post,
}

/// HTTP / UDS client that runs requests on a private tokio runtime thread
/// and supports cooperative cancellation via [`CancelToken`].
pub struct CancellableClient {
    /// For HTTP: origin like `"http://host:port"`.  For UDS: the empty
    /// string — the runtime builds relative URIs against a fake host.
    base_url: String,
    /// Scheme: `"http"`, `"https"`, or `"uds"`.
    scheme: Scheme,
    tx: SyncSender<Cmd>,
    _thread: JoinHandle<()>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Scheme {
    Http,
    Uds,
}

impl CancellableClient {
    /// Build a new client.  `base_url` is either:
    /// - `http://host:port` / `https://host:port` — reqwest with HTTP/2
    ///   prior knowledge plus pooled keepalive, or
    /// - `uds:///path/to/sock` / `unix:///path/to/sock` — raw hyper_util
    ///   over a Unix domain socket.
    ///
    /// Spawns one OS thread owning a current-thread tokio runtime.
    pub fn connect(base_url: impl Into<String>, timeout: Duration) -> Result<Self, ClientError> {
        let base_url_raw = base_url.into();
        let (scheme, normalized, transport_init) = classify(&base_url_raw)?;

        let (tx, rx) = sync_channel::<Cmd>(8);

        let thread = std::thread::Builder::new()
            .name("infer-client-rt".into())
            .spawn(move || runtime_thread(rx, transport_init, timeout))
            .map_err(|e| ClientError::Runtime(format!("spawn thread: {e}")))?;

        Ok(Self {
            base_url: normalized,
            scheme,
            tx,
            _thread: thread,
        })
    }

    pub fn get_json<T: DeserializeOwned>(
        &self,
        path: &str,
        cancel: &CancelToken,
    ) -> Result<T, ClientError> {
        self.run(Method::Get, path, None, cancel, no_tick)
    }

    pub fn post_json<T: DeserializeOwned>(
        &self,
        path: &str,
        body: serde_json::Value,
        cancel: &CancelToken,
    ) -> Result<T, ClientError> {
        self.run(Method::Post, path, Some(body), cancel, no_tick)
    }

    /// Same as [`get_json`] but invokes `tick` between every ~50 ms poll
    /// interval while waiting for the response.  The callback checks any
    /// external cancellation signal (e.g. PG's `QueryCancelPending`) and
    /// returns `Err` to abort.
    pub fn get_json_with_tick<T, F>(
        &self,
        path: &str,
        cancel: &CancelToken,
        tick: F,
    ) -> Result<T, ClientError>
    where
        T: DeserializeOwned,
        F: FnMut() -> Result<(), ClientError>,
    {
        self.run(Method::Get, path, None, cancel, tick)
    }

    pub fn post_json_with_tick<T, F>(
        &self,
        path: &str,
        body: serde_json::Value,
        cancel: &CancelToken,
        tick: F,
    ) -> Result<T, ClientError>
    where
        T: DeserializeOwned,
        F: FnMut() -> Result<(), ClientError>,
    {
        self.run(Method::Post, path, Some(body), cancel, tick)
    }

    /// Run a batch of requests concurrently on the runtime thread.  All
    /// futures are submitted to a single `tokio::join_all`, so they run
    /// in parallel (multiplexed over HTTP/2 on the HTTP transport, or
    /// over a pool of Unix-socket connections on UDS).
    ///
    /// Returns one result per input item, in order.  A failure in one
    /// item does not abort the others — each slot carries its own
    /// `Result`.  A failure at the transport level (e.g. runtime thread
    /// gone) yields an outer `Err`.
    ///
    /// `tick` is called between ~50 ms poll intervals while waiting for
    /// the batch to finish, mirroring the single-request API.
    pub fn batch_with_tick<T, F>(
        &self,
        items: Vec<BatchItem>,
        cancel: &CancelToken,
        mut tick: F,
    ) -> Result<Vec<Result<T, ClientError>>, ClientError>
    where
        T: DeserializeOwned,
        F: FnMut() -> Result<(), ClientError>,
    {
        let items = items
            .into_iter()
            .map(|it| match self.scheme {
                Scheme::Http => BatchItem {
                    url: format!("{}{}", self.base_url, it.url),
                    method: it.method,
                    body: it.body,
                },
                Scheme::Uds => it,
            })
            .collect::<Vec<_>>();

        let (reply_tx, reply_rx) = sync_channel(1);

        self.tx
            .send(Cmd::Batch(BatchJob {
                items,
                cancel: cancel.clone(),
                reply: reply_tx,
            }))
            .map_err(|_| ClientError::Runtime("runtime thread gone".into()))?;

        let bytes_per_item: Vec<Result<Vec<u8>, ClientError>> = loop {
            match reply_rx.recv_timeout(Duration::from_millis(50)) {
                Ok(Ok(batch)) => break batch,
                Ok(Err(e)) => return Err(e),
                Err(RecvTimeoutError::Disconnected) => {
                    return Err(ClientError::Runtime("reply channel dropped".into()));
                }
                Err(RecvTimeoutError::Timeout) => {
                    if let Err(e) = tick() {
                        cancel.cancel();
                        let _ = reply_rx.recv();
                        return Err(e);
                    }
                }
            }
        };

        Ok(bytes_per_item
            .into_iter()
            .map(|r| {
                r.and_then(|b| {
                    serde_json::from_slice::<T>(&b).map_err(|e| ClientError::Parse(e.to_string()))
                })
            })
            .collect())
    }

    fn run<T, F>(
        &self,
        method: Method,
        path: &str,
        body: Option<serde_json::Value>,
        cancel: &CancelToken,
        mut tick: F,
    ) -> Result<T, ClientError>
    where
        T: DeserializeOwned,
        F: FnMut() -> Result<(), ClientError>,
    {
        let url = match self.scheme {
            Scheme::Http => format!("{}{}", self.base_url, path),
            // For UDS the "URL" is just the path; the runtime builds the
            // hyper Request with a synthetic host header.
            Scheme::Uds => path.to_string(),
        };

        let (reply_tx, reply_rx) = sync_channel::<Result<Vec<u8>, ClientError>>(1);

        self.tx
            .send(Cmd::Request(RequestJob {
                url,
                body,
                method,
                cancel: cancel.clone(),
                reply: reply_tx,
            }))
            .map_err(|_| ClientError::Runtime("runtime thread gone".into()))?;

        loop {
            match reply_rx.recv_timeout(Duration::from_millis(50)) {
                Ok(Ok(bytes)) => {
                    return serde_json::from_slice::<T>(&bytes)
                        .map_err(|e| ClientError::Parse(e.to_string()));
                }
                Ok(Err(e)) => return Err(e),
                Err(RecvTimeoutError::Disconnected) => {
                    return Err(ClientError::Runtime("reply channel dropped".into()));
                }
                Err(RecvTimeoutError::Timeout) => {
                    if let Err(e) = tick() {
                        cancel.cancel();
                        let _ = reply_rx.recv();
                        return Err(e);
                    }
                }
            }
        }
    }
}

fn no_tick() -> Result<(), ClientError> {
    Ok(())
}

impl Drop for CancellableClient {
    fn drop(&mut self) {
        let _ = self.tx.send(Cmd::Shutdown);
    }
}

// ── URL classification ───────────────────────────────────────────────────────

/// Pre-built transport handle — the client thread consumes this to avoid
/// re-parsing the URL inside the runtime.
enum TransportInit {
    Http { base: String },
    Uds { path: PathBuf },
}

fn classify(raw: &str) -> Result<(Scheme, String, TransportInit), ClientError> {
    if let Some(rest) = raw
        .strip_prefix("uds://")
        .or_else(|| raw.strip_prefix("unix://"))
    {
        // Accept `uds:///path`, `uds://localhost/path`, or `uds:/path`.
        let path = if let Some(stripped) = rest.strip_prefix('/') {
            PathBuf::from(format!("/{stripped}"))
        } else if rest.is_empty() {
            return Err(ClientError::InvalidUrl(format!(
                "empty uds path in '{raw}'"
            )));
        } else {
            // Form like `uds://localhost/path` — take the path component.
            match rest.find('/') {
                Some(idx) => PathBuf::from(&rest[idx..]),
                None => return Err(ClientError::InvalidUrl(format!("no socket path in '{raw}'"))),
            }
        };
        // UDS "base_url" stays empty; per-request paths are used directly.
        return Ok((Scheme::Uds, String::new(), TransportInit::Uds { path }));
    }

    if raw.starts_with("uds:/") && !raw.starts_with("uds:///") {
        // Accept single-slash abbreviation: `uds:/path/to/sock`.
        let path = PathBuf::from(raw.trim_start_matches("uds:"));
        return Ok((Scheme::Uds, String::new(), TransportInit::Uds { path }));
    }

    // Anything else must be a valid http/https URL.
    let parsed =
        url::Url::parse(raw).map_err(|e| ClientError::InvalidUrl(format!("{e} in '{raw}'")))?;
    match parsed.scheme() {
        "http" | "https" => {
            let base = raw.trim_end_matches('/').to_string();
            Ok((
                Scheme::Http,
                base.clone(),
                TransportInit::Http { base },
            ))
        }
        other => Err(ClientError::InvalidUrl(format!(
            "unsupported URL scheme '{other}' in '{raw}'"
        ))),
    }
}

// ── Runtime thread ────────────────────────────────────────────────────────────

fn runtime_thread(
    rx: std::sync::mpsc::Receiver<Cmd>,
    init: TransportInit,
    timeout: Duration,
) {
    let runtime = match Builder::new_current_thread().enable_all().build() {
        Ok(rt) => rt,
        Err(e) => {
            tracing::error!("infer-client: failed to build tokio runtime: {e}");
            return;
        }
    };

    let transport = match build_transport(init, timeout) {
        Ok(t) => t,
        Err(e) => {
            tracing::error!("infer-client: failed to build transport: {e:?}");
            return;
        }
    };

    while let Ok(cmd) = rx.recv() {
        match cmd {
            Cmd::Shutdown => break,
            Cmd::Request(job) => {
                runtime.block_on(async {
                    let result = match &transport {
                        Transport::Http { client, base } => {
                            execute_http(client, base, &job).await
                        }
                        Transport::Uds { client, .. } => execute_uds(client, &job, timeout).await,
                    };
                    let _ = job.reply.send(result);
                });
            }
            Cmd::Batch(batch) => {
                runtime.block_on(async {
                    // Capture &transport in a local so each future grabs a
                    // reference (Copy), not the transport itself.
                    let transport_ref = &transport;
                    let mut futures = Vec::with_capacity(batch.items.len());
                    for item in &batch.items {
                        let item_url = item.url.clone();
                        let item_body = item.body.clone();
                        let item_method = item.method;
                        let item_cancel = batch.cancel.clone();
                        let fut = async move {
                            let job = RequestJob {
                                url: item_url,
                                body: item_body,
                                method: item_method,
                                cancel: item_cancel,
                                reply: sync_channel::<Result<Vec<u8>, ClientError>>(1).0,
                            };
                            match transport_ref {
                                Transport::Http { client, base } => {
                                    execute_http(client, base, &job).await
                                }
                                Transport::Uds { client, .. } => {
                                    execute_uds(client, &job, timeout).await
                                }
                            }
                        };
                        futures.push(fut);
                    }
                    let results: Vec<Result<Vec<u8>, ClientError>> =
                        join_all_preserving_order(futures).await;
                    let _ = batch.reply.send(Ok(results));
                });
            }
        }
    }

    drop(runtime);
    drop(transport);
}

/// Run all futures concurrently on the current runtime and return their
/// results in input order.  A thin wrapper around `futures::join_all` via
/// `FuturesOrdered` — implemented here so we don't pull the `futures`
/// crate for one helper.
async fn join_all_preserving_order<F, T>(futures: Vec<F>) -> Vec<T>
where
    F: std::future::Future<Output = T>,
{
    // Spawn via local tasks on the current-thread runtime: spawn_local
    // isn't available on the basic runtime, so we use tokio::spawn with
    // a multi-thread runtime.  Since the runtime here is current-thread,
    // we poll cooperatively via a simple sequential `select!` + index.
    //
    // Simpler + correct: collect via a join set that preserves order.
    use std::pin::Pin;
    use std::task::Poll;
    let mut pinned: Vec<(usize, Pin<Box<F>>)> =
        futures.into_iter().enumerate().map(|(i, f)| (i, Box::pin(f))).collect();
    let mut out: Vec<Option<T>> = Vec::new();
    out.resize_with(pinned.len(), || None);

    std::future::poll_fn(|cx| {
        let mut i = 0;
        while i < pinned.len() {
            let (idx, fut) = &mut pinned[i];
            match fut.as_mut().poll(cx) {
                Poll::Ready(v) => {
                    out[*idx] = Some(v);
                    pinned.swap_remove(i);
                }
                Poll::Pending => {
                    i += 1;
                }
            }
        }
        if pinned.is_empty() {
            Poll::Ready(())
        } else {
            Poll::Pending
        }
    })
    .await;

    out.into_iter().map(|v| v.expect("slot filled")).collect()
}

fn build_transport(init: TransportInit, timeout: Duration) -> Result<Transport, ClientError> {
    match init {
        TransportInit::Http { base } => {
            let client = ReqwestClient::builder()
                .timeout(timeout)
                .connect_timeout(Duration::from_secs(5))
                .tcp_keepalive(Duration::from_secs(30))
                .pool_idle_timeout(Duration::from_secs(90))
                .pool_max_idle_per_host(16)
                .http2_prior_knowledge()
                .http2_adaptive_window(true)
                .build()?;
            Ok(Transport::Http { client, base })
        }
        TransportInit::Uds { path } => {
            let connector = UdsConnector { path: path.clone() };
            let client = HyperClient::builder(TokioExecutor::new())
                .pool_idle_timeout(Duration::from_secs(90))
                .pool_max_idle_per_host(16)
                .build::<_, Full<Bytes>>(connector);
            Ok(Transport::Uds { client, _path: path })
        }
    }
}

async fn execute_http(
    client: &ReqwestClient,
    base: &str,
    job: &RequestJob,
) -> Result<Vec<u8>, ClientError> {
    // `job.url` for the HTTP path is already `{base}{path}`, but we keep
    // base around for debugging and potential future use.
    let _ = base;
    let builder = match job.method {
        Method::Get => client.get(&job.url),
        Method::Post => {
            let b = client.post(&job.url);
            match &job.body {
                Some(v) => b.json(v),
                None => b,
            }
        }
    };

    let fut = async move {
        let resp = builder.send().await?;
        let status = resp.status();
        let bytes = resp.bytes().await?;
        if !status.is_success() {
            return Err(ClientError::Server {
                status: status.as_u16(),
                body: String::from_utf8_lossy(&bytes).into_owned(),
            });
        }
        if bytes.len() > MAX_BODY_BYTES {
            return Err(ClientError::Transport(format!(
                "response body too large: {} > {}",
                bytes.len(),
                MAX_BODY_BYTES
            )));
        }
        Ok::<Vec<u8>, ClientError>(bytes.to_vec())
    };

    tokio::select! {
        r = fut => r,
        _ = job.cancel.cancelled() => Err(ClientError::Cancelled),
    }
}

async fn execute_uds(
    client: &HyperClient<UdsConnector, Full<Bytes>>,
    job: &RequestJob,
    timeout: Duration,
) -> Result<Vec<u8>, ClientError> {
    // `job.url` for UDS is just the path (e.g. `/v1/stats`).  hyper
    // requires a scheme + authority in the URI, so we synthesise
    // `http://larql-uds{path}`.  The UdsConnector ignores the authority.
    let uri_str = format!("http://larql-uds{}", job.url);
    let uri: Uri = uri_str
        .parse()
        .map_err(|e: hyper::http::uri::InvalidUri| ClientError::InvalidUrl(e.to_string()))?;

    let (method_str, body_bytes) = match job.method {
        Method::Get => ("GET", Bytes::new()),
        Method::Post => {
            let body = match &job.body {
                Some(v) => serde_json::to_vec(v)
                    .map_err(|e| ClientError::Transport(format!("serialize body: {e}")))?,
                None => Vec::new(),
            };
            ("POST", Bytes::from(body))
        }
    };

    let mut req_builder = Request::builder()
        .method(method_str)
        .uri(uri)
        .header("host", "larql-uds");
    if matches!(job.method, Method::Post) && !body_bytes.is_empty() {
        req_builder = req_builder.header("content-type", "application/json");
    }
    let req = req_builder
        .body(Full::new(body_bytes))
        .map_err(|e| ClientError::Transport(format!("build request: {e}")))?;

    let send_fut = async move {
        let resp: Response<Incoming> = client
            .request(req)
            .await
            .map_err(|e| ClientError::Transport(format!("uds request: {e}")))?;
        let status = resp.status();
        let collected = resp
            .into_body()
            .collect()
            .await
            .map_err(|e| ClientError::Transport(format!("uds body: {e}")))?;
        let bytes = collected.to_bytes();
        if !status.is_success() {
            return Err(ClientError::Server {
                status: status.as_u16(),
                body: String::from_utf8_lossy(&bytes).into_owned(),
            });
        }
        if bytes.len() > MAX_BODY_BYTES {
            return Err(ClientError::Transport(format!(
                "response body too large: {} > {}",
                bytes.len(),
                MAX_BODY_BYTES
            )));
        }
        Ok::<Vec<u8>, ClientError>(bytes.to_vec())
    };

    tokio::select! {
        r = tokio::time::timeout(timeout, send_fut) => {
            match r {
                Ok(inner) => inner,
                Err(_) => Err(ClientError::Transport("uds request timed out".into())),
            }
        }
        _ = job.cancel.cancelled() => Err(ClientError::Cancelled),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cancel_token_flips_once() {
        let t = CancelToken::new();
        assert!(!t.is_cancelled());
        t.cancel();
        assert!(t.is_cancelled());
        let c = t.clone();
        assert!(c.is_cancelled());
    }

    #[test]
    fn invalid_url_errors_on_connect() {
        let err = CancellableClient::connect("not a url", Duration::from_secs(1));
        assert!(matches!(err, Err(ClientError::InvalidUrl(_))));
    }

    #[test]
    fn unsupported_scheme_rejected() {
        let err = CancellableClient::connect("ftp://example.com/", Duration::from_secs(1));
        assert!(matches!(err, Err(ClientError::InvalidUrl(_))));
    }

    #[test]
    fn uds_url_parses() {
        let (scheme, base, init) =
            classify("uds:///tmp/larql.sock").expect("uds:/// form parses");
        assert_eq!(scheme, Scheme::Uds);
        assert_eq!(base, "");
        match init {
            TransportInit::Uds { path } => assert_eq!(path.as_os_str(), "/tmp/larql.sock"),
            _ => panic!("expected uds init"),
        }

        let (_, _, init) = classify("unix:///var/run/larql.sock").expect("unix:/// form parses");
        match init {
            TransportInit::Uds { path } => assert_eq!(path.as_os_str(), "/var/run/larql.sock"),
            _ => panic!("expected uds init"),
        }

        let (_, _, init) = classify("uds:/tmp/short.sock").expect("uds:/ abbreviation parses");
        match init {
            TransportInit::Uds { path } => assert_eq!(path.as_os_str(), "/tmp/short.sock"),
            _ => panic!("expected uds init"),
        }
    }

    #[test]
    fn http_url_parses() {
        let (scheme, base, _) =
            classify("http://localhost:8080").expect("http parses");
        assert_eq!(scheme, Scheme::Http);
        assert_eq!(base, "http://localhost:8080");

        let (_, base, _) =
            classify("https://example.com:9999/").expect("trailing slash is stripped");
        assert_eq!(base, "https://example.com:9999");
    }

    #[test]
    fn cancelled_request_returns_quickly() {
        let c = CancellableClient::connect("http://198.51.100.1:9999", Duration::from_secs(30))
            .expect("client builds");
        let cancel = CancelToken::new();
        let cancel_bg = cancel.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(100));
            cancel_bg.cancel();
        });
        let start = std::time::Instant::now();
        let r: Result<serde_json::Value, _> = c.get_json("/v1/health", &cancel);
        let elapsed = start.elapsed();
        assert!(matches!(r, Err(ClientError::Cancelled)), "got {:?}", r);
        assert!(
            elapsed < Duration::from_secs(2),
            "cancellation took {:?}",
            elapsed
        );
    }

    #[test]
    fn batch_concurrent_over_uds() {
        use std::sync::{Arc, Mutex};

        let tmpdir = std::env::temp_dir().join(format!(
            "infer-client-batch-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&tmpdir).expect("mkdir");
        let sock_path = tmpdir.join("batch.sock");
        let _ = std::fs::remove_file(&sock_path);

        let sock_server = sock_path.clone();
        let ready = Arc::new(Mutex::new(false));
        let ready_server = ready.clone();

        let handle = std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("rt");
            rt.block_on(async move {
                let listener = match tokio::net::UnixListener::bind(&sock_server) {
                    Ok(l) => l,
                    Err(_) => return,
                };
                *ready_server.lock().expect("lock") = true;
                let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
                loop {
                    tokio::select! {
                        _ = tokio::time::sleep_until(deadline) => return,
                        res = listener.accept() => {
                            let (stream, _) = match res {
                                Ok(v) => v,
                                Err(_) => return,
                            };
                            tokio::spawn(async move {
                                let io = TokioIo::new(stream);
                                let svc = hyper::service::service_fn(
                                    |req: Request<Incoming>| async move {
                                        // Artificial 30 ms delay per request
                                        // so batched items actually overlap.
                                        tokio::time::sleep(Duration::from_millis(30)).await;
                                        let path = req.uri().path().to_string();
                                        let body = serde_json::json!({"path": path});
                                        let bytes = serde_json::to_vec(&body).expect("vec");
                                        Ok::<Response<Full<Bytes>>, std::io::Error>(
                                            Response::builder()
                                                .status(200)
                                                .header("content-type", "application/json")
                                                .body(Full::new(Bytes::from(bytes)))
                                                .expect("resp"),
                                        )
                                    },
                                );
                                let _ = hyper::server::conn::http1::Builder::new()
                                    .serve_connection(io, svc)
                                    .await;
                            });
                        }
                    }
                }
            });
        });

        for _ in 0..100 {
            if *ready.lock().expect("ready") {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(*ready.lock().expect("ready final"), "server did not start");

        let url = format!("uds://{}", sock_path.display());
        let client =
            CancellableClient::connect(&url, Duration::from_secs(5)).expect("uds client");
        let cancel = CancelToken::new();

        // 10 concurrent requests.  Serial would be 10 * 30 ms = 300 ms;
        // batched should be ~30–60 ms (transport overhead + one delay).
        let items: Vec<BatchItem> = (0..10)
            .map(|i| BatchItem {
                url: format!("/item/{i}"),
                method: Method::Get,
                body: None,
            })
            .collect();

        let start = std::time::Instant::now();
        let results: Vec<Result<serde_json::Value, _>> = client
            .batch_with_tick(items, &cancel, no_tick)
            .expect("batch");
        let elapsed = start.elapsed();

        assert_eq!(results.len(), 10);
        for (i, r) in results.iter().enumerate() {
            let v = r.as_ref().expect("ok slot");
            assert_eq!(v["path"], format!("/item/{i}"));
        }
        // A generous upper bound: well under 10× serial latency.
        assert!(
            elapsed < Duration::from_millis(200),
            "batch took {:?} (expected <200ms for 10 x 30ms concurrent)",
            elapsed
        );

        drop(client);
        let _ = std::fs::remove_file(&sock_path);
        let _ = std::fs::remove_dir(&tmpdir);
        let _ = handle.join();
    }

    #[test]
    fn tick_driven_cancellation() {
        let c = CancellableClient::connect("http://198.51.100.1:9999", Duration::from_secs(30))
            .expect("client builds");
        let cancel = CancelToken::new();
        let mut ticks = 0;
        let start = std::time::Instant::now();
        let tick = || {
            ticks += 1;
            if ticks >= 3 {
                Err(ClientError::Cancelled)
            } else {
                Ok(())
            }
        };
        let r: Result<serde_json::Value, _> = c.get_json_with_tick("/v1/health", &cancel, tick);
        let elapsed = start.elapsed();
        assert!(matches!(r, Err(ClientError::Cancelled)), "got {:?}", r);
        assert!(
            elapsed < Duration::from_secs(2),
            "tick cancel took {:?}",
            elapsed
        );
        assert!(cancel.is_cancelled(), "token should have been signalled");
    }

    /// End-to-end UDS test: spin up a tiny hyper echo server on a
    /// Unix socket in TMPDIR, hit it through CancellableClient, verify
    /// the round trip.
    #[test]
    fn uds_round_trip() {
        use std::sync::{Arc, Mutex};

        let tmpdir = std::env::temp_dir().join(format!(
            "infer-client-uds-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&tmpdir).expect("mkdir tmp");
        let sock_path = tmpdir.join("test.sock");
        let _ = std::fs::remove_file(&sock_path);

        let sock_server = sock_path.clone();
        let ready = Arc::new(Mutex::new(false));
        let ready_server = ready.clone();

        // Server runs inside a 5-second timeout so the test always
        // terminates regardless of how many connections the client
        // makes.  Each accepted connection is serviced concurrently
        // with a bounded per-request lifetime.
        let handle = std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("build test runtime");
            rt.block_on(async move {
                let listener = match tokio::net::UnixListener::bind(&sock_server) {
                    Ok(l) => l,
                    Err(_) => return,
                };
                *ready_server.lock().expect("ready lock") = true;
                let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
                loop {
                    let accept_fut = listener.accept();
                    tokio::select! {
                        _ = tokio::time::sleep_until(deadline) => return,
                        res = accept_fut => {
                            let (stream, _) = match res {
                                Ok(v) => v,
                                Err(_) => return,
                            };
                            tokio::spawn(async move {
                                let io = TokioIo::new(stream);
                                let svc = hyper::service::service_fn(
                                    |req: Request<Incoming>| async move {
                                        let path = req.uri().path().to_string();
                                        let body = serde_json::json!({
                                            "echo_path": path,
                                            "method": req.method().to_string(),
                                        });
                                        let bytes =
                                            serde_json::to_vec(&body).expect("to_vec");
                                        Ok::<Response<Full<Bytes>>, std::io::Error>(
                                            Response::builder()
                                                .status(200)
                                                .header("content-type", "application/json")
                                                .body(Full::new(Bytes::from(bytes)))
                                                .expect("build response"),
                                        )
                                    },
                                );
                                let _ = hyper::server::conn::http1::Builder::new()
                                    .serve_connection(io, svc)
                                    .await;
                            });
                        }
                    }
                }
            });
        });

        // Wait for the server to bind.
        for _ in 0..100 {
            if *ready.lock().expect("ready poll") {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(*ready.lock().expect("ready final"), "server did not start");

        let url = format!("uds://{}", sock_path.display());
        let client = CancellableClient::connect(&url, Duration::from_secs(5))
            .expect("uds client builds");
        let cancel = CancelToken::new();
        let resp: serde_json::Value = client
            .get_json("/v1/health", &cancel)
            .expect("uds round trip");
        assert_eq!(resp["echo_path"], "/v1/health");
        assert_eq!(resp["method"], "GET");

        let body = serde_json::json!({"x": 1});
        let resp: serde_json::Value = client
            .post_json("/v1/infer", body, &cancel)
            .expect("uds post");
        assert_eq!(resp["method"], "POST");
        assert_eq!(resp["echo_path"], "/v1/infer");

        drop(client);
        let _ = std::fs::remove_file(&sock_path);
        let _ = std::fs::remove_dir(&tmpdir);
        // Server self-terminates at the 5s deadline; join it now.
        let _ = handle.join();
    }
}
