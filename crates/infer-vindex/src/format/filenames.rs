//! Canonical filenames within a `.vindex` directory.
//!
//! Centralising these as constants keeps the writer and loader in
//! agreement and lets tools introspect a vindex without hard-coding
//! string literals.

/// Top-level manifest read at load time.
pub const INDEX_JSON: &str = "index.json";

/// Subdirectory holding BitNet b1.58 native-ternary weights, written
/// only when a vindex is built with `--keep-quant` from an I2_S GGUF.
pub const BITNET_DIR: &str = "bitnet";

/// Sidecar copy of the BitNet layout, also stored in `index.json`'s
/// `bitnet_layout` field. Kept under a conventional name so tools that
/// don't yet parse `index.json` can still introspect the layout.
pub const BITNET_LAYOUT_JSON: &str = "bitnet_layout.json";

/// Concatenated per-channel f32 scales for every BitLinear tensor.
pub const BITNET_SCALES_BIN: &str = "bitnet/scales.f32";

/// Path (relative to the vindex root) of one BitLinear tensor's raw
/// I2_S bytes, e.g. `bitnet/blk.0.attn_q.weight.i2s`.
pub fn bitnet_tensor_filename(tensor_name: &str) -> String {
    format!("{BITNET_DIR}/{tensor_name}.i2s")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bitnet_tensor_filename_is_under_bitnet_dir() {
        assert_eq!(
            bitnet_tensor_filename("blk.0.attn_q.weight"),
            "bitnet/blk.0.attn_q.weight.i2s"
        );
        assert_eq!(
            bitnet_tensor_filename("blk.29.ffn_down.weight"),
            "bitnet/blk.29.ffn_down.weight.i2s"
        );
    }
}
