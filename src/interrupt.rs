//! Bridge PostgreSQL's query-cancel machinery into `infer-client` calls.
//!
//! PostgreSQL signals a cancellation by setting `QueryCancelPending` (SIGINT)
//! or `ProcDiePending` (SIGTERM) and flipping `InterruptPending`.  These are
//! `sig_atomic_t` globals set from the signal handler; reading them is safe
//! without `HOLD_INTERRUPTS`.  Acting on them — turning them into a SQL-level
//! `ERROR` — is the job of `pg_sys::ProcessInterrupts()`, which may
//! `ereport(ERROR, ...)` and longjmp out.
//!
//! This module provides a [`pg_interrupt_tick`] function suitable for passing
//! to [`infer_client::CancellableClient::get_json_with_tick`] et al.  It is
//! called roughly every 50 ms while the HTTP future is in flight:
//!
//! 1. If `InterruptPending` is clear, return `Ok(())` so the client keeps
//!    waiting.
//! 2. If it is set, return `Err(ClientError::Cancelled)`.  The client will
//!    flip the `CancelToken`, the in-flight tokio future will unwind at
//!    its next `await`, and control returns to pg_infer.
//! 3. pg_infer then calls [`raise_if_pending`], which invokes
//!    `pg_sys::ProcessInterrupts()` — that's the safe point where PG's
//!    longjmp happens, tearing the transaction down with a proper SQL
//!    error rather than from inside a reqwest state machine.
//!
//! The 50 ms cadence matches the cancel-token poll interval inside the
//! runtime thread; total end-to-end cancellation latency is therefore
//! bounded by ~100 ms, well below interactive expectations.

use infer_client::ClientError;

/// Non-longjmping check: `Ok(())` if no interrupt is pending, else
/// `Err(Cancelled)`.  Safe to call from any context where reading a
/// `sig_atomic_t` is safe (i.e., anywhere).
pub fn pg_interrupt_tick() -> Result<(), ClientError> {
    // SAFETY: `InterruptPending` is a `sig_atomic_t` set from the signal
    // handler.  A plain volatile read is the entire protocol.
    let pending = unsafe { pgrx::pg_sys::InterruptPending };
    if pending != 0 {
        Err(ClientError::Cancelled)
    } else {
        Ok(())
    }
}

/// Invoke `pg_sys::ProcessInterrupts()` if a PG interrupt is pending.
/// This is the function that actually converts a pending cancel into a
/// SQL-level `ERROR` via `ereport` + longjmp.
///
/// Call this *after* an HTTP request returns `ClientError::Cancelled`,
/// at a point where it is safe to unwind.  If no interrupt is actually
/// pending (e.g. the cancel came from a server-side timeout), this is a
/// no-op.
pub fn raise_if_pending() {
    // SAFETY: `ProcessInterrupts` expects to be called from a backend's
    // main thread with no held spinlocks.  We're on the PG main thread
    // (inside a pg_extern function body) and we hold no spinlocks of our
    // own.  `ProcessInterrupts` internally checks the pending flags and
    // ereports; if none are set it returns immediately.
    unsafe {
        if pgrx::pg_sys::InterruptPending != 0 {
            pgrx::pg_sys::ProcessInterrupts();
        }
    }
}
