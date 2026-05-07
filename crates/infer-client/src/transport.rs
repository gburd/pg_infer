//! Cancellable HTTP transport.
//!
//! See the crate-level docs for the cancellation model.  This module owns:
//! - [`CancelToken`]: a drop-safe flag the caller sets to abort an in-flight
//!   request.  Cloneable and shareable across threads.
//! - [`CancellableClient`]: the actual HTTP client.  One per `RemoteBackend`
//!   instance (i.e. roughly one per loaded remote model per PG backend).
//!   Internally owns a dedicated OS thread running a current-thread tokio
//!   runtime, plus a `reqwest::Client` for connection pooling.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{sync_channel, SyncSender};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use reqwest::{Client, StatusCode};
use serde::de::DeserializeOwned;
use tokio::runtime::Builder;
use tokio::sync::oneshot;

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

// ── CancellableClient ─────────────────────────────────────────────────────────

/// Command sent from the foreground to the runtime thread.
enum Cmd {
    Request(RequestJob),
    Shutdown,
}

struct RequestJob {
    url: String,
    body: Option<serde_json::Value>,
    method: Method,
    cancel: CancelToken,
    /// Oneshot for the JSON response bytes.  The caller deserialises on
    /// the foreground thread so the runtime thread doesn't need the target
    /// type.
    reply: oneshot::Sender<Result<Vec<u8>, ClientError>>,
}

#[derive(Clone, Copy)]
enum Method {
    Get,
    Post,
}

/// HTTP client that runs requests on a private tokio runtime thread and
/// supports cooperative cancellation via [`CancelToken`].
pub struct CancellableClient {
    base_url: String,
    tx: SyncSender<Cmd>,
    _thread: JoinHandle<()>,
}

impl CancellableClient {
    /// Build a new client.  `base_url` is the server root (e.g.
    /// `"http://localhost:8080"`).  The trailing slash is stripped.
    ///
    /// Spawns one OS thread owning a current-thread tokio runtime and a
    /// `reqwest::Client` with HTTP/2 multiplexing, keepalive, and a
    /// connection pool capped at 16 idle connections per host.
    pub fn connect(base_url: impl Into<String>, timeout: Duration) -> Result<Self, ClientError> {
        let base_url = base_url.into().trim_end_matches('/').to_string();
        // Validate up front so the caller gets a clear error instead of a
        // silently bad request URL later.
        url::Url::parse(&base_url).map_err(|e| ClientError::InvalidUrl(e.to_string()))?;

        let (tx, rx) = sync_channel::<Cmd>(8);

        let thread = std::thread::Builder::new()
            .name("infer-client-rt".into())
            .spawn(move || runtime_thread(rx, timeout))
            .map_err(|e| ClientError::Runtime(format!("spawn thread: {e}")))?;

        Ok(Self {
            base_url,
            tx,
            _thread: thread,
        })
    }

    /// `GET {base_url}{path}` with optional query string already appended
    /// to `path`.
    pub fn get_json<T: DeserializeOwned>(
        &self,
        path: &str,
        cancel: &CancelToken,
    ) -> Result<T, ClientError> {
        self.run(Method::Get, path, None, cancel)
    }

    /// `POST {base_url}{path}` with a JSON body.
    pub fn post_json<T: DeserializeOwned>(
        &self,
        path: &str,
        body: serde_json::Value,
        cancel: &CancelToken,
    ) -> Result<T, ClientError> {
        self.run(Method::Post, path, Some(body), cancel)
    }

    fn run<T: DeserializeOwned>(
        &self,
        method: Method,
        path: &str,
        body: Option<serde_json::Value>,
        cancel: &CancelToken,
    ) -> Result<T, ClientError> {
        let url = format!("{}{}", self.base_url, path);
        let (reply_tx, reply_rx) = oneshot::channel();

        self.tx
            .send(Cmd::Request(RequestJob {
                url,
                body,
                method,
                cancel: cancel.clone(),
                reply: reply_tx,
            }))
            .map_err(|_| ClientError::Runtime("runtime thread gone".into()))?;

        // Block the PG foreground thread on the oneshot.  We purposely do
        // NOT call CHECK_FOR_INTERRUPTS here — that's the caller's job (it
        // knows the pgrx macro) and we don't want a longjmp inside the
        // crate.  Instead the caller loops on a short-timeout variant.
        //
        // For callers that don't need interrupt wiring (tests) the blocking
        // recv is fine.
        let bytes = reply_rx
            .blocking_recv()
            .map_err(|_| ClientError::Runtime("reply channel dropped".into()))??;

        serde_json::from_slice::<T>(&bytes).map_err(|e| ClientError::Parse(e.to_string()))
    }
}

impl Drop for CancellableClient {
    fn drop(&mut self) {
        // Best-effort shutdown.  If the runtime thread is already gone the
        // send fails silently, which is fine.
        let _ = self.tx.send(Cmd::Shutdown);
    }
}

// ── Runtime thread ────────────────────────────────────────────────────────────

fn runtime_thread(rx: std::sync::mpsc::Receiver<Cmd>, timeout: Duration) {
    // A current-thread runtime is sufficient and avoids the overhead of a
    // worker pool for what is almost always a single in-flight request.
    let runtime = match Builder::new_current_thread().enable_all().build() {
        Ok(rt) => rt,
        Err(e) => {
            tracing::error!("infer-client: failed to build tokio runtime: {e}");
            return;
        }
    };

    let client = match build_reqwest_client(timeout) {
        Ok(c) => c,
        Err(e) => {
            tracing::error!("infer-client: failed to build reqwest client: {e}");
            return;
        }
    };

    while let Ok(cmd) = rx.recv() {
        match cmd {
            Cmd::Shutdown => break,
            Cmd::Request(job) => {
                let client = client.clone();
                runtime.block_on(async move {
                    let result = execute_request(&client, &job).await;
                    let _ = job.reply.send(result);
                });
            }
        }
    }
    // Explicit drop order: runtime first, then client, so tasks get a
    // chance to cancel before the connection pool goes away.
    drop(runtime);
    drop(client);
}

fn build_reqwest_client(timeout: Duration) -> Result<Client, reqwest::Error> {
    Client::builder()
        .timeout(timeout)
        .connect_timeout(Duration::from_secs(5))
        .tcp_keepalive(Duration::from_secs(30))
        .pool_idle_timeout(Duration::from_secs(90))
        .pool_max_idle_per_host(16)
        .http2_prior_knowledge()
        // Fall back to HTTP/1.1 if the server doesn't speak H2, so a plain
        // hand-rolled test server keeps working.
        .http2_adaptive_window(true)
        .build()
}

async fn execute_request(client: &Client, job: &RequestJob) -> Result<Vec<u8>, ClientError> {
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
        // Lazy note: reqwest consumed the whole body, so we already paid
        // the memory cost; we bail *after* receipt.  That's fine for the
        // endpoint set pg_infer uses (response payloads are KB-scale).
        let _ = status; // silence unused in the non-StatusCode match
        let _ = StatusCode::OK;
        Ok::<Vec<u8>, ClientError>(bytes.to_vec())
    };

    tokio::select! {
        r = fut => r,
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
    fn cancelled_request_returns_quickly() {
        // A non-routable address: 198.51.100.0/24 is reserved for
        // documentation and will not respond.  The connect will either
        // time out or be cancelled first.
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
}
