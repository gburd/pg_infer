//! HTTP client for larql-server.
//!
//! Speaks the JSON subset of larql-server's API that pg_infer needs:
//! `/v1/describe`, `/v1/walk`, `/v1/stats`, `/v1/relations`, `/v1/embed`,
//! and `/v1/infer`.  Each call is run on a per-client tokio runtime and
//! is cooperatively cancellable — callers pass a [`CancelToken`] that
//! is polled on the server-side future via `tokio::select!`.
//!
//! The client is thread-safe (`Send + Sync`) and cheap to clone; internal
//! state is an `Arc` over the runtime + `reqwest::Client`.
//!
//! # Cancellation model
//!
//! pg_infer runs inside a PostgreSQL backend.  When a user issues
//! `pg_cancel_backend(...)` or hits ^C, PG sets `QueryCancelPending` and
//! the backend is expected to notice at the next `CHECK_FOR_INTERRUPTS`
//! call.  To propagate that into in-flight HTTP requests we:
//!
//! 1. Run the HTTP future on a current-thread tokio runtime, spawned on
//!    a dedicated OS thread so the PG backend thread stays free to check
//!    for interrupts.
//! 2. Race the future against a [`CancelToken`] in `tokio::select!`;
//!    dropping the future cleanly cancels the TCP connect / TLS handshake
//!    / response read at whatever `await` point it was at.
//! 3. The caller (pg_infer) polls [`CancelToken::is_cancelled`] in a tight
//!    loop while the background thread runs, invoking
//!    `CHECK_FOR_INTERRUPTS` in between.  If the flag gets set, it signals
//!    the token and the request unwinds within milliseconds.

pub mod transport;
pub mod types;

pub use transport::{BatchItem, CancelToken, CancellableClient, ClientError, Method};
pub use types::{
    CacheStatsResponse, DescribeEdge, DescribeResponse, InferPrediction, InferResponse,
    RankResponse, RankResult, RelationSummary, RelationsResponse, StatsResponse, WalkHit,
    WalkResponse, WarmupResponse,
};
