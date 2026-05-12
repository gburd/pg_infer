//! AVX2+FMA optimized Q6_K matrix-vector multiply for x86_64.
//!
//! Processes 8 float elements per iteration using 256-bit SIMD.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::q6k_matvec::{Q6K_BLOCK_SIZE, f16_to_f32};

/// AVX2-accelerated Q6_K matvec dispatch.
///
/// This function is only called after a runtime check confirms AVX2+FMA.
#[cfg(target_arch = "x86_64")]
pub fn dispatch_avx2(q6k_data: &[u8], x: &[f32], num_rows: usize, hidden: usize) -> Vec<f32> {
    let superblocks = hidden / 256;
    let bytes_per_row = superblocks * Q6K_BLOCK_SIZE;
    let mut out = vec![0.0f32; num_rows];

    for (row, out_val) in out.iter_mut().enumerate().take(num_rows) {
        let row_start = row * bytes_per_row;
        // SAFETY: caller verified AVX2+FMA via is_x86_feature_detected!
        *out_val = unsafe {
            q6k_row_dot_avx2(
                &q6k_data[row_start..row_start + bytes_per_row],
                x,
                superblocks,
            )
        };
    }
    out
}

/// Compute a single Q6_K row dot product using AVX2+FMA.
///
/// # Safety
///
/// Caller must ensure AVX2 and FMA are available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn q6k_row_dot_avx2(row_data: &[u8], x: &[f32], superblocks: usize) -> f32 {
    let mut acc = _mm256_setzero_ps();

    for sb in 0..superblocks {
        let block = &row_data[sb * Q6K_BLOCK_SIZE..(sb + 1) * Q6K_BLOCK_SIZE];

        // Layout: ql[128] | qh[64] | scales[16] | d(f16)[2]
        let ql = &block[0..128];
        let qh = &block[128..192];
        let scales_bytes = &block[192..208];
        let d_bits = u16::from_le_bytes([block[208], block[209]]);
        let d = f16_to_f32(d_bits);

        let x_base = sb * 256;

        // 16 sub-blocks of 16 values each = 256 values total
        for (j, &scale_byte) in scales_bytes.iter().enumerate() {
            let sc = d * (scale_byte as i8) as f32;
            let sc_v = _mm256_set1_ps(sc);
            let bias_v = _mm256_set1_ps(32.0);
            let sub_base = j * 16;

            // Process 16 values in 2 chunks of 8
            for chunk in 0..2 {
                let i_start = chunk * 8;
                let mut vals = [0.0f32; 8];

                for i in 0..8 {
                    let qi = sub_base + i_start + i;
                    let byte_idx = qi / 2;
                    let lo_byte = ql[byte_idx];

                    let hi_byte_idx = qi / 4;
                    let hi_byte = qh[hi_byte_idx];

                    // Extract 4-bit low value
                    let lo4 = if qi % 2 == 0 {
                        (lo_byte & 0x0F) as f32
                    } else {
                        ((lo_byte >> 4) & 0x0F) as f32
                    };

                    // Extract 2-bit high value
                    let bit_offset = (qi % 4) * 2;
                    let hi2 = ((hi_byte >> bit_offset) & 0x03) as f32;

                    // Full 6-bit value: lo4 + hi2*16 - 32
                    vals[i] = lo4 + hi2 * 16.0;
                }

                let raw_v = _mm256_loadu_ps(vals.as_ptr());
                // dequant = sc * (raw - 32)
                let centered = _mm256_sub_ps(raw_v, bias_v);
                let dequant = _mm256_mul_ps(sc_v, centered);

                // Load corresponding x values
                let x_v = _mm256_loadu_ps(x.as_ptr().add(x_base + sub_base + i_start));

                // acc += dequant * x
                acc = _mm256_fmadd_ps(dequant, x_v, acc);
            }
        }
    }

    hsum_avx2(acc)
}

/// Horizontal sum of all 8 f32 lanes in a __m256.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_avx2(v: __m256) -> f32 {
    let hi128 = _mm256_extractf128_ps(v, 1);
    let lo128 = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo128, hi128);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let hi64 = _mm_movehl_ps(sums, sums);
    let total = _mm_add_ss(sums, hi64);
    _mm_cvtss_f32(total)
}
