//! Build pipeline — extract model weights into vindex format.

use std::path::Path;

use crate::error::VindexError;
use crate::storage::residual_cache::ResidualCacheBuilder;

pub mod build;
pub mod build_from_vectors;
pub mod build_helpers;
pub mod callbacks;
#[cfg(feature = "moe-svd")]
pub mod moe_svd;
pub mod streaming;

pub use build::build_vindex;
pub use build::build_vindex_resume;
pub use build::resolve_gate_dtype;
pub use build_from_vectors::build_vindex_from_vectors;
pub use streaming::build_vindex_streaming;
pub use streaming::build_vindex_resume as build_vindex_streaming_resume;
pub use callbacks::{IndexBuildCallbacks, SilentBuildCallbacks};

/// Generate residual cache for template-fixed layers.
///
/// Runs one forward pass per unique template through the first `num_cached_layers`
/// layers and stores the output residuals. Called after the main vindex extraction
/// completes.
///
/// # Arguments
///
/// * `output_dir` — Directory to write `residual_cache.bin` into.
/// * `hidden_size` — Model hidden dimension (must match the model's config).
/// * `templates` — Slice of `(template_hash, token_ids)` pairs representing
///   unique prompt templates whose prefix layers are invariant.
/// * `num_cached_layers` — Number of initial layers to cache (e.g. 13 for L0–L12).
/// * `forward_fn` — Closure that runs a partial forward pass: given token IDs and
///   a layer index, returns the residual stream output as a flat `Vec<f32>` of
///   length `seq_len * hidden_size`.
pub fn extract_residual_cache(
    output_dir: &Path,
    hidden_size: usize,
    templates: &[(u64, Vec<u32>)],
    num_cached_layers: usize,
    forward_fn: impl Fn(&[u32], usize) -> Vec<f32>,
) -> Result<(), VindexError> {
    let mut builder = ResidualCacheBuilder::new(hidden_size, num_cached_layers);

    for (template_hash, token_ids) in templates {
        for layer in 0..num_cached_layers {
            let residual = forward_fn(token_ids, layer);
            builder.add(*template_hash, layer, &residual);
        }
    }

    let cache_path = output_dir.join("residual_cache.bin");
    builder.write(&cache_path)?;
    Ok(())
}
