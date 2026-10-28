//! Inference engines — pluggable decode strategies.
//!
//! The primary engine is [`MarkovResidualEngine`] which replaces traditional
//! KV caching with stored pre-layer residuals. K/V are recomputed from
//! stored residuals at decode time (validated KL=0.0 vs full-KV baseline).

pub mod markov_residual;

pub use markov_residual::{kv_memory_bytes_for_seq, recompute_kv, rs_decode_step, rs_prefill};
pub use markov_residual::{MarkovResidualEngine, RsPrefillResult, RsStore};
