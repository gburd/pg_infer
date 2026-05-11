//! Ternary GEMV: packed_ternary[N, hidden/4 bytes] × x[hidden] → scores[N].
//!
//! Scalar reference implementation for BitNet b1.58 gate-KNN scoring.
//! Each byte encodes 4 ternary values using I2_S packing (2 bits each):
//!   `0b00` → 0 (skip), `0b01` → +1 (add), `0b10` → -1 (subtract)
//!
//! No scale factor is applied — BitNet's AbsMean quantization uses a
//! per-tensor scale which cancels out in top-K ranking (all rows share
//! the same scale, so relative ordering is preserved).
//!
//! Performance characteristics:
//! - No FP multiply — pure f32 add/subtract/skip
//! - 8× smaller than f32 storage (2 bits vs 32 bits per weight)
//! - ~10× faster than f32 BLAS for large feature counts (memory-bound)

/// Ternary GEMV dispatch: scores[N] = packed[N, hidden/4] × x[hidden].
///
/// # Arguments
/// - `packed`: contiguous ternary-packed bytes, `num_rows * (hidden / 4)` long
/// - `x`: input vector (residual), `hidden` f32 values
/// - `num_rows`: number of feature rows (N)
/// - `hidden`: hidden dimension (must be multiple of 4)
///
/// # Returns
/// Score vector of length `num_rows`.
pub fn dispatch(packed: &[u8], x: &[f32], num_rows: usize, hidden: usize) -> Vec<f32> {
    debug_assert!(hidden % 4 == 0, "hidden must be multiple of 4");
    let bytes_per_row = hidden / 4;
    debug_assert_eq!(packed.len(), num_rows * bytes_per_row,
        "packed length mismatch: expected {}, got {}",
        num_rows * bytes_per_row, packed.len());

    let mut scores = vec![0.0f32; num_rows];

    for row in 0..num_rows {
        let row_data = &packed[row * bytes_per_row..(row + 1) * bytes_per_row];
        let mut acc = 0.0f32;

        for (byte_idx, &byte) in row_data.iter().enumerate() {
            let base = byte_idx * 4;
            let v0 = byte & 0x03;
            let v1 = (byte >> 2) & 0x03;
            let v2 = (byte >> 4) & 0x03;
            let v3 = (byte >> 6) & 0x03;

            // 0b01 → +1 (add), 0b10 → -1 (subtract), 0b00 → 0 (skip)
            if v0 == 1 { acc += x[base]; } else if v0 == 2 { acc -= x[base]; }
            if v1 == 1 { acc += x[base + 1]; } else if v1 == 2 { acc -= x[base + 1]; }
            if v2 == 1 { acc += x[base + 2]; } else if v2 == 2 { acc -= x[base + 2]; }
            if v3 == 1 { acc += x[base + 3]; } else if v3 == 2 { acc -= x[base + 3]; }
        }
        scores[row] = acc;
    }
    scores
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: pack i8 ternary values into I2_S bytes.
    fn pack_ternary(data: &[i8]) -> Vec<u8> {
        assert!(data.len() % 4 == 0);
        data.chunks_exact(4)
            .map(|chunk| {
                let e = |v: i8| -> u8 {
                    match v { 0 => 0b00, 1 => 0b01, -1 => 0b10, _ => 0b00 }
                };
                e(chunk[0]) | (e(chunk[1]) << 2) | (e(chunk[2]) << 4) | (e(chunk[3]) << 6)
            })
            .collect()
    }

    #[test]
    fn known_output_single_row() {
        // Row: [+1, -1, +1, 0], x: [1.0, 2.0, 3.0, 4.0]
        // Expected: 1*1.0 + (-1)*2.0 + 1*3.0 + 0*4.0 = 1 - 2 + 3 + 0 = 2.0
        let weights: Vec<i8> = vec![1, -1, 1, 0];
        let packed = pack_ternary(&weights);
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let scores = dispatch(&packed, &x, 1, 4);
        assert_eq!(scores.len(), 1);
        assert!((scores[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn known_output_multi_row() {
        // 2 rows, hidden=8
        // Row 0: [+1, +1, +1, +1, -1, -1, -1, -1]
        // Row 1: [-1, -1, -1, -1, +1, +1, +1, +1]
        // x: [1, 2, 3, 4, 5, 6, 7, 8]
        // Row 0: (1+2+3+4) - (5+6+7+8) = 10 - 26 = -16
        // Row 1: -(1+2+3+4) + (5+6+7+8) = -10 + 26 = 16
        let weights: Vec<i8> = vec![
            1, 1, 1, 1, -1, -1, -1, -1,
            -1, -1, -1, -1, 1, 1, 1, 1,
        ];
        let packed = pack_ternary(&weights);
        let x: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        let scores = dispatch(&packed, &x, 2, 8);
        assert_eq!(scores.len(), 2);
        assert!((scores[0] - (-16.0)).abs() < 1e-6);
        assert!((scores[1] - 16.0).abs() < 1e-6);
    }

    #[test]
    fn zero_weights_give_zero_scores() {
        let packed = vec![0u8; 4]; // 16 zeros
        let x = vec![99.0f32; 16];
        let scores = dispatch(&packed, &x, 1, 16);
        assert!((scores[0]).abs() < 1e-6);
    }

    #[test]
    fn all_positive_ones() {
        // All +1 weights: score = sum(x)
        let hidden = 16;
        let packed = vec![0x55u8; hidden / 4]; // 0b01_01_01_01
        let x: Vec<f32> = (1..=hidden).map(|i| i as f32).collect();
        let expected: f32 = x.iter().sum();
        let scores = dispatch(&packed, &x, 1, hidden);
        assert!((scores[0] - expected).abs() < 1e-6);
    }

    #[test]
    fn all_negative_ones() {
        // All -1 weights: score = -sum(x)
        let hidden = 16;
        let packed = vec![0xAAu8; hidden / 4]; // 0b10_10_10_10
        let x: Vec<f32> = (1..=hidden).map(|i| i as f32).collect();
        let expected: f32 = -x.iter().sum::<f32>();
        let scores = dispatch(&packed, &x, 1, hidden);
        assert!((scores[0] - expected).abs() < 1e-6);
    }

    #[test]
    fn zero_input_gives_zero_scores() {
        let weights: Vec<i8> = vec![1, -1, 1, -1, 0, 1, -1, 0];
        let packed = pack_ternary(&weights);
        let x = vec![0.0f32; 8];
        let scores = dispatch(&packed, &x, 1, 8);
        assert!((scores[0]).abs() < 1e-6);
    }

    #[test]
    fn many_rows() {
        // 4 rows, hidden=4, all +1 weights
        let hidden = 4;
        let num_rows = 4;
        let packed = vec![0x55u8; num_rows]; // each byte = 4× +1
        let x = vec![1.0f32; hidden];
        let scores = dispatch(&packed, &x, num_rows, hidden);
        assert_eq!(scores.len(), num_rows);
        for &s in &scores {
            assert!((s - 4.0).abs() < 1e-6);
        }
    }

    #[test]
    fn sparse_weights() {
        // Mostly zeros with a few ±1: simulates typical BitNet sparsity
        // hidden=8: [0, 0, +1, 0, 0, -1, 0, 0]
        let weights: Vec<i8> = vec![0, 0, 1, 0, 0, -1, 0, 0];
        let packed = pack_ternary(&weights);
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        // Expected: 0 + 0 + 3.0 + 0 + 0 - 6.0 + 0 + 0 = -3.0
        let scores = dispatch(&packed, &x, 1, 8);
        assert!((scores[0] - (-3.0)).abs() < 1e-6);
    }
}
