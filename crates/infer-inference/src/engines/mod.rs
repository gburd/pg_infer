//! Inference engines — pluggable decode strategies.
//!
//! The primary engine is [`MarkovResidualEngine`] which replaces traditional
//! KV caching with stored pre-layer residuals. K/V are recomputed from
//! stored residuals at decode time (validated KL=0.0 vs full-KV baseline).

pub mod markov_residual;

pub use markov_residual::{MarkovResidualEngine, RsStore, RsPrefillResult};
pub use markov_residual::{rs_prefill, rs_decode_step, recompute_kv, kv_memory_bytes_for_seq};
