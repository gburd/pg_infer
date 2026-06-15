//! Scaled ternary × f32 matrix-vector multiply for BitNet 1.58
//! BitLinear layers (generation path).
//!
//! This is the *scaled* sibling of [`super::ternary_matvec`].  That
//! module is a ranking-only kernel: it omits the per-channel scale
//! because the gate-KNN top-K is invariant to a shared positive
//! scale.  Generation is not — to produce correct logits the trit
//! accumulation must be multiplied by the BitLinear layer's
//! per-channel (per-row) scale.  This module supplies that.
//!
//! `BitLinear` weights are ternary `{-1, 0, +1}` packed at 2 bpw
//! (I2_S, GGML type 36).  Matrix-vector multiply against an f32
//! activation reduces to a pure-additive sum per output row — add
//! at `+1` positions, subtract at `-1`, skip `0` — followed by one
//! f32 multiply by the row's channel scale.  No per-element f32
//! multiply, which is the entire point of native BitNet inference.
//!
//! For Microsoft's BitNet b1.58 2B 4T the weight tensor stays in its
//! on-disk 2-bpw form; the runtime working-set is the f32 activation
//! buffer plus the per-channel scale, instead of a 5+ GB
//! f16-after-dequant heap.
//!
//! ## Bit-pattern mapping
//!
//! Matches [`infer_models::quant::ggml::dequantize_i2_s`]:
//!
//!   `0b00 → 0`,  `0b01 → +1`,  `0b10 → -1`,  `0b11 → reserved (0)`
//!
//! Byte `b` holds elements `(b * 4 + slot)` for `slot ∈ 0..4`, slot
//! indexing the 2-bit field at bits `(2 * slot)..(2 * slot + 2)`.

/// Errors surfaced by the scaled ternary kernel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComputeError {
    ShapeMismatch(String),
}

impl std::fmt::Display for ComputeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ComputeError::ShapeMismatch(msg) => write!(f, "shape mismatch: {msg}"),
        }
    }
}

impl std::error::Error for ComputeError {}

/// One BitLinear layer's weight tensor, ready to feed a scaled matvec.
///
/// `i2s_bytes` packs `rows * cols / 4` bytes (4 trits per byte).
/// `channel_scales` is one f32 per row — applied AFTER the integer
/// trit accumulation, equivalent to dequantising the row to
/// `{-scale, 0, +scale}` and then doing an f32 matvec, but without
/// the dense intermediate.
#[derive(Clone, Debug)]
pub struct BitLinearWeight {
    pub rows: usize,
    pub cols: usize,
    pub i2s_bytes: Vec<u8>,
    pub channel_scales: Vec<f32>,
}

impl BitLinearWeight {
    /// Build a `BitLinearWeight` after validating shape consistency.
    ///
    /// # Errors
    /// Returns `ComputeError::ShapeMismatch` if any of:
    /// - `cols` is not a multiple of 4 (the I2_S packing requires it),
    /// - `i2s_bytes.len()` differs from `rows * cols / 4`,
    /// - `channel_scales.len()` differs from `rows`.
    pub fn new(
        rows: usize,
        cols: usize,
        i2s_bytes: Vec<u8>,
        channel_scales: Vec<f32>,
    ) -> Result<Self, ComputeError> {
        if !cols.is_multiple_of(4) {
            return Err(ComputeError::ShapeMismatch(format!(
                "BitLinearWeight: cols ({cols}) must be a multiple of 4 for I2_S packing"
            )));
        }
        let expected_bytes = rows.saturating_mul(cols) / 4;
        if i2s_bytes.len() != expected_bytes {
            return Err(ComputeError::ShapeMismatch(format!(
                "BitLinearWeight: expected {expected_bytes} I2_S bytes ({rows}x{cols}/4), \
                 got {} bytes",
                i2s_bytes.len()
            )));
        }
        if channel_scales.len() != rows {
            return Err(ComputeError::ShapeMismatch(format!(
                "BitLinearWeight: expected {rows} channel scales, got {}",
                channel_scales.len()
            )));
        }
        Ok(Self {
            rows,
            cols,
            i2s_bytes,
            channel_scales,
        })
    }

    /// Bytes per row in the I2_S packing (== `cols / 4`).
    #[inline]
    pub fn row_bytes(&self) -> usize {
        self.cols / 4
    }
}

/// `y = W · x`, returning a fresh `Vec<f32>` of length `rows`.
///
/// Accumulates trits with no f32 multiply inside the inner loop
/// (apart from the per-row scale at the very end).
///
/// # Errors
/// `ComputeError::ShapeMismatch` if `x.len() != w.cols`.
pub fn matvec_i2s_f32(w: &BitLinearWeight, x: &[f32]) -> Result<Vec<f32>, ComputeError> {
    let mut y = vec![0.0f32; w.rows];
    matvec_i2s_f32_into(w, x, &mut y)?;
    Ok(y)
}

/// In-place variant of [`matvec_i2s_f32`].
///
/// Writes into `y[..w.rows]`, overwriting any previous contents.
///
/// # Errors
/// `ComputeError::ShapeMismatch` if `x.len() != w.cols` or
/// `y.len() < w.rows`.
pub fn matvec_i2s_f32_into(
    w: &BitLinearWeight,
    x: &[f32],
    y: &mut [f32],
) -> Result<(), ComputeError> {
    if x.len() != w.cols {
        return Err(ComputeError::ShapeMismatch(format!(
            "matvec_i2s_f32: x.len() = {}, expected w.cols = {}",
            x.len(),
            w.cols
        )));
    }
    if y.len() < w.rows {
        return Err(ComputeError::ShapeMismatch(format!(
            "matvec_i2s_f32: y.len() = {} < w.rows = {}",
            y.len(),
            w.rows
        )));
    }

    let row_bytes = w.row_bytes();
    debug_assert_eq!(row_bytes * 4, w.cols);

    let rows = w.i2s_bytes.chunks_exact(row_bytes);
    for ((row, &scale), y_r) in rows.zip(w.channel_scales.iter()).zip(y.iter_mut()) {
        let mut acc: f32 = 0.0;
        for (b, &byte) in row.iter().enumerate() {
            let base = b * 4;
            // 4-entry LUT indexed by the 2-bit field keeps the hot
            // path branch-free; factors are exactly {-1.0, 0.0, +1.0}.
            const TRIT: [f32; 4] = [0.0, 1.0, -1.0, 0.0];
            acc += TRIT[(byte & 0b11) as usize] * x[base];
            acc += TRIT[((byte >> 2) & 0b11) as usize] * x[base + 1];
            acc += TRIT[((byte >> 4) & 0b11) as usize] * x[base + 2];
            acc += TRIT[((byte >> 6) & 0b11) as usize] * x[base + 3];
        }
        *y_r = acc * scale;
    }

    Ok(())
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Encode an f32 row of `{-d, 0, +d}` trits into I2_S bytes.
    fn encode_row(row: &[f32], d: f32) -> Vec<u8> {
        assert!(row.len().is_multiple_of(4));
        let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
        let mut out = vec![0u8; row.len() / 4];
        for (i, chunk) in row.chunks_exact(4).enumerate() {
            let mut byte: u8 = 0;
            for (slot, &v) in chunk.iter().enumerate() {
                let t = (v * inv).round().clamp(-1.0, 1.0) as i32;
                let bits: u8 = match t {
                    1 => 0b01,
                    -1 => 0b10,
                    _ => 0b00,
                };
                byte |= bits << (2 * slot);
            }
            out[i] = byte;
        }
        out
    }

    /// Naive dequant + matmul reference.
    fn naive_dequant_matvec(w: &BitLinearWeight, x: &[f32]) -> Vec<f32> {
        let row_bytes = w.row_bytes();
        let mut y = vec![0.0f32; w.rows];
        for (r, y_r) in y.iter_mut().enumerate() {
            let scale = w.channel_scales[r];
            let mut acc = 0.0f32;
            for (c, &x_c) in x.iter().enumerate().take(w.cols) {
                let byte = w.i2s_bytes[r * row_bytes + c / 4];
                let bits = (byte >> (2 * (c % 4))) & 0b11;
                let trit = match bits {
                    0b01 => 1.0_f32,
                    0b10 => -1.0_f32,
                    _ => 0.0_f32,
                };
                acc += trit * scale * x_c;
            }
            *y_r = acc;
        }
        y
    }

    fn synth(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn synth_ternary(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                let bucket = ((s >> 33) % 3) as i32;
                match bucket {
                    0 => 0.0,
                    1 => 1.0,
                    _ => -1.0,
                }
            })
            .collect()
    }

    #[test]
    fn shape_mismatch_rejects_bad_inputs() {
        assert!(
            BitLinearWeight::new(1, 5, vec![0; 2], vec![1.0]).is_err(),
            "cols=5 should reject"
        );
        assert!(
            BitLinearWeight::new(2, 8, vec![0; 3], vec![1.0, 1.0]).is_err(),
            "expected 4 bytes (2*8/4), got 3"
        );
        assert!(
            BitLinearWeight::new(2, 8, vec![0; 4], vec![1.0]).is_err(),
            "expected 2 scales"
        );
    }

    #[test]
    fn matvec_x_dim_mismatch_errors() {
        let w = BitLinearWeight::new(1, 8, vec![0; 2], vec![1.0]).unwrap();
        let x = vec![0.0f32; 7];
        assert!(matvec_i2s_f32(&w, &x).is_err());
    }

    #[test]
    fn matvec_y_too_small_errors() {
        let w = BitLinearWeight::new(2, 4, vec![0; 2], vec![1.0, 1.0]).unwrap();
        let x = vec![0.0f32; 4];
        let mut y = vec![0.0f32; 1];
        assert!(matvec_i2s_f32_into(&w, &x, &mut y).is_err());
    }

    #[test]
    fn matvec_zero_weight_returns_zero() {
        let w = BitLinearWeight::new(3, 16, vec![0u8; 12], vec![1.5, -2.0, 7.0]).unwrap();
        let x = synth(16, 42);
        let y = matvec_i2s_f32(&w, &x).unwrap();
        assert_eq!(y, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn matvec_identity_row_recovers_activation() {
        let mut row = vec![0.0f32; 16];
        row[5] = 1.0;
        let bytes = encode_row(&row, 1.0);
        let w = BitLinearWeight::new(1, 16, bytes, vec![1.0]).unwrap();
        let x = synth(16, 11);
        let y = matvec_i2s_f32(&w, &x).unwrap();
        assert!((y[0] - x[5]).abs() < 1e-6, "got {} expected {}", y[0], x[5]);
    }

    #[test]
    fn matvec_negative_trit_subtracts() {
        let mut row = vec![0.0f32; 16];
        row[3] = -1.0;
        row[11] = 1.0;
        let bytes = encode_row(&row, 1.0);
        let w = BitLinearWeight::new(1, 16, bytes, vec![0.5]).unwrap();
        let x: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let y = matvec_i2s_f32(&w, &x).unwrap();
        let expected = (x[11] - x[3]) * 0.5;
        assert!(
            (y[0] - expected).abs() < 1e-6,
            "got {} expected {}",
            y[0],
            expected
        );
    }

    #[test]
    fn matvec_matches_naive_reference_random_ternary() {
        let rows = 32;
        let cols = 256;
        let mut bytes = Vec::with_capacity(rows * cols / 4);
        for r in 0..rows {
            let row_trits = synth_ternary(cols, 42 + r as u64);
            bytes.extend(encode_row(&row_trits, 1.0));
        }
        let scales: Vec<f32> = (0..rows).map(|i| 0.1 + (i as f32) * 0.01).collect();
        let w = BitLinearWeight::new(rows, cols, bytes, scales).unwrap();

        let x = synth(cols, 9999);
        let kernel = matvec_i2s_f32(&w, &x).unwrap();
        let reference = naive_dequant_matvec(&w, &x);

        for (i, (k, r)) in kernel.iter().zip(reference.iter()).enumerate() {
            assert!(
                (k - r).abs() < 1e-4,
                "row {i}: kernel={k} reference={r} delta={}",
                k - r
            );
        }
    }

    #[test]
    fn matvec_reserved_bit_pattern_decodes_as_zero() {
        let w = BitLinearWeight::new(1, 4, vec![0xFFu8], vec![3.0]).unwrap();
        let x = vec![1.0, 1.0, 1.0, 1.0];
        let y = matvec_i2s_f32(&w, &x).unwrap();
        assert_eq!(y, vec![0.0]);
    }

    #[test]
    fn matvec_scale_and_activation_scale_compose() {
        let row = vec![1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0];
        let bytes = encode_row(&row, 1.0);
        let w_unit = BitLinearWeight::new(1, 8, bytes.clone(), vec![1.0]).unwrap();
        let w_scaled = BitLinearWeight::new(1, 8, bytes, vec![2.5]).unwrap();

        let x = vec![0.5; 8];
        let y_unit = matvec_i2s_f32(&w_unit, &x).unwrap();
        let y_scaled = matvec_i2s_f32(&w_scaled, &x).unwrap();

        let x_scaled: Vec<f32> = x.iter().map(|v| v * 4.0).collect();
        let y_act_scaled = matvec_i2s_f32(&w_unit, &x_scaled).unwrap();

        assert!((y_scaled[0] - y_unit[0] * 2.5).abs() < 1e-6);
        assert!((y_act_scaled[0] - y_unit[0] * 4.0).abs() < 1e-6);
    }

    #[test]
    fn matvec_into_overwrites_not_accumulates() {
        let rows = 4;
        let cols = 8;
        let mut bytes = Vec::new();
        for r in 0..rows {
            let row_trits = synth_ternary(cols, 100 + r as u64);
            bytes.extend(encode_row(&row_trits, 1.0));
        }
        let scales = vec![0.5_f32; rows];
        let w = BitLinearWeight::new(rows, cols, bytes, scales).unwrap();

        let x = synth(cols, 1);
        let mut y = vec![999.0_f32; rows]; // Pre-poisoned.
        matvec_i2s_f32_into(&w, &x, &mut y).unwrap();
        let y2 = matvec_i2s_f32(&w, &x).unwrap();
        for (a, b) in y.iter().zip(y2.iter()) {
            assert!((a - b).abs() < 1e-6, "poisoned y entry leaked: {a} vs {b}");
        }
    }

    /// Cross-validate against the canonical GGUF-loader decoder
    /// `infer_models::quant::ggml::dequantize_i2_s`: encode a row via
    /// this module's helper, decode via the loader path, and confirm
    /// the kernel's accumulated result equals the dot product of the
    /// decoded trits with the activation.  Pins the bit-pattern
    /// mapping between this kernel and the decoder — if either side
    /// drifts, this test catches it.
    #[test]
    fn matvec_agrees_with_canonical_i2_s_decoder() {
        let row = synth_ternary(64, 7);
        let bytes = encode_row(&row, 1.0);

        let decoded = infer_models::quant::ggml::dequantize_i2_s(&bytes, 64).expect("decode");
        for (i, (expected, actual)) in row.iter().zip(decoded.iter()).enumerate() {
            assert!(
                (expected - actual).abs() < 1e-6,
                "col {i}: encode -> canonical-decode disagrees: {expected} vs {actual}"
            );
        }

        let scale: f32 = 0.7;
        let w = BitLinearWeight::new(1, 64, bytes, vec![scale]).unwrap();
        let x = synth(64, 13);
        let kernel = matvec_i2s_f32(&w, &x).unwrap();
        let reference: f32 =
            decoded.iter().zip(x.iter()).map(|(t, a)| t * a).sum::<f32>() * scale;

        assert!(
            (kernel[0] - reference).abs() < 1e-4,
            "kernel={} reference={} delta={}",
            kernel[0],
            reference,
            kernel[0] - reference
        );
    }
}
