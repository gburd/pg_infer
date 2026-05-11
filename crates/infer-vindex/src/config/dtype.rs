//! Data type conversion utilities for vindex storage.
//!
//! Supports f32 (default), f16 (half precision), and ternary (2-bit packed)
//! storage. Half-precision conversion functions are in `infer_models::quant::half`.

use serde::{Deserialize, Serialize};

/// Storage precision for vindex binary files.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[derive(Default)]
pub enum StorageDtype {
    #[default]
    F32,
    F16,
    /// 2 bits per ternary weight {-1, 0, +1}, 4 values per byte (I2_S packing).
    /// Used for BitNet b1.58 models where gate weights are inherently ternary.
    Ternary,
}


impl std::fmt::Display for StorageDtype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::F32 => write!(f, "f32"),
            Self::F16 => write!(f, "f16"),
            Self::Ternary => write!(f, "ternary"),
        }
    }
}

/// Write `data` to `w`, encoded according to `dtype`. Returns bytes written.
///
/// Convenience wrapper around `encode_floats` for the binary writers in
/// `extract::build`, `extract::streaming`, and `format::weights::write` —
/// they all need the same f32→bytes encode + write + length-tracking
/// pattern.
///
/// Note: for Ternary dtype, use `write_ternary` instead — this function
/// panics on Ternary since f32 data cannot be losslessly round-tripped
/// through 2-bit encoding without explicit quantization.
pub fn write_floats(
    w: &mut impl std::io::Write,
    data: &[f32],
    dtype: StorageDtype,
) -> std::io::Result<u64> {
    let bytes = encode_floats(data, dtype);
    w.write_all(&bytes)?;
    Ok(bytes.len() as u64)
}

/// Encode f32 data as either f32 or f16 bytes.
///
/// Panics on `StorageDtype::Ternary` — use `encode_ternary` for ternary data.
pub fn encode_floats(data: &[f32], dtype: StorageDtype) -> Vec<u8> {
    match dtype {
        StorageDtype::F32 => {
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
            };
            bytes.to_vec()
        }
        StorageDtype::F16 => infer_models::quant::half::encode_f16(data),
        StorageDtype::Ternary => {
            panic!("encode_floats does not support Ternary — use encode_ternary()")
        }
    }
}

/// Decode bytes back to f32, handling dtype.
///
/// For Ternary: each byte unpacks to 4 f32 values (-1.0, 0.0, or +1.0).
pub fn decode_floats(data: &[u8], dtype: StorageDtype) -> Vec<f32> {
    match dtype {
        StorageDtype::F32 => {
            let floats: &[f32] = unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const f32, data.len() / 4)
            };
            floats.to_vec()
        }
        StorageDtype::F16 => infer_models::quant::half::decode_f16(data),
        StorageDtype::Ternary => {
            // Each byte → 4 f32 values
            let count = data.len() * 4;
            let decoded = decode_ternary(data, count);
            decoded.into_iter().map(|v| v as f32).collect()
        }
    }
}

/// Bytes per float for a given dtype.
///
/// For Ternary this returns 0 — callers must use `ternary_bytes_per_row`
/// instead since ternary packs 4 values per byte (not a whole number of
/// bytes per value).
pub fn bytes_per_float(dtype: StorageDtype) -> usize {
    match dtype {
        StorageDtype::F32 => 4,
        StorageDtype::F16 => 2,
        // Ternary: 4 values per byte, so there's no integer bytes-per-float.
        // Callers must use ternary_bytes_per_row() for offset math.
        // Return 1 as sentinel — only used in F32/F16 offset calculations.
        StorageDtype::Ternary => 1,
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Ternary encoding/decoding (I2_S: 2 bits per value, 4 values per byte)
// ═══════════════════════════════════════════════════════════════════════

/// Pack i8 ternary values {-1, 0, +1} into 2-bit I2_S format.
///
/// Encoding per value: `0b00` = 0 (skip), `0b01` = +1 (add), `0b10` = -1 (subtract).
/// Four values are packed per byte, LSB-first:
///   byte = v0 | (v1 << 2) | (v2 << 4) | (v3 << 6)
///
/// Input length must be a multiple of 4. Values outside {-1, 0, +1} are
/// clamped (positive → +1, negative → -1).
pub fn encode_ternary(data: &[i8]) -> Vec<u8> {
    debug_assert!(data.len() % 4 == 0, "ternary encode requires len % 4 == 0");
    let num_bytes = data.len() / 4;
    let mut packed = vec![0u8; num_bytes];

    for (byte_idx, chunk) in data.chunks_exact(4).enumerate() {
        let e0 = ternary_to_bits(chunk[0]);
        let e1 = ternary_to_bits(chunk[1]);
        let e2 = ternary_to_bits(chunk[2]);
        let e3 = ternary_to_bits(chunk[3]);
        packed[byte_idx] = e0 | (e1 << 2) | (e2 << 4) | (e3 << 6);
    }
    packed
}

/// Unpack 2-bit I2_S bytes back into i8 ternary values.
///
/// `count` is the number of i8 values to produce (must be ≤ data.len() * 4).
pub fn decode_ternary(data: &[u8], count: usize) -> Vec<i8> {
    debug_assert!(count <= data.len() * 4);
    let mut out = Vec::with_capacity(count);

    for &byte in data.iter() {
        if out.len() >= count { break; }
        out.push(bits_to_ternary(byte & 0x03));
        if out.len() >= count { break; }
        out.push(bits_to_ternary((byte >> 2) & 0x03));
        if out.len() >= count { break; }
        out.push(bits_to_ternary((byte >> 4) & 0x03));
        if out.len() >= count { break; }
        out.push(bits_to_ternary((byte >> 6) & 0x03));
    }
    out.truncate(count);
    out
}

/// Number of packed bytes needed for one feature row of `hidden_size` ternary values.
pub fn ternary_bytes_per_row(hidden_size: usize) -> usize {
    // 4 values per byte, hidden_size must be multiple of 4
    debug_assert!(hidden_size % 4 == 0, "hidden_size must be multiple of 4 for ternary packing");
    hidden_size / 4
}

/// Map an i8 ternary value to its 2-bit encoding.
#[inline(always)]
fn ternary_to_bits(v: i8) -> u8 {
    match v {
        0 => 0b00,
        1 => 0b01,
        -1 => 0b10,
        x if x > 0 => 0b01, // clamp positive
        _ => 0b10,           // clamp negative
    }
}

/// Map a 2-bit encoding back to i8 ternary value.
#[inline(always)]
fn bits_to_ternary(bits: u8) -> i8 {
    match bits & 0x03 {
        0b00 => 0,
        0b01 => 1,
        0b10 => -1,
        _ => 0, // 0b11 reserved, treat as zero
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_decode_f32() {
        let data = vec![1.0f32, 2.0, 3.0];
        let encoded = encode_floats(&data, StorageDtype::F32);
        assert_eq!(encoded.len(), 12);
        let decoded = decode_floats(&encoded, StorageDtype::F32);
        assert_eq!(decoded, data);
    }

    #[test]
    fn encode_decode_f16() {
        let data = vec![1.0f32, 2.0, 3.0];
        let encoded = encode_floats(&data, StorageDtype::F16);
        assert_eq!(encoded.len(), 6);
        let decoded = decode_floats(&encoded, StorageDtype::F16);
        for (orig, dec) in data.iter().zip(decoded.iter()) {
            assert!((orig - dec).abs() < 0.01);
        }
    }

    #[test]
    fn encode_decode_ternary_basic() {
        // 8 values = 2 bytes
        let data: Vec<i8> = vec![1, -1, 0, 1, -1, -1, 1, 0];
        let packed = encode_ternary(&data);
        assert_eq!(packed.len(), 2);
        let decoded = decode_ternary(&packed, 8);
        assert_eq!(decoded, data);
    }

    #[test]
    fn encode_decode_ternary_all_zeros() {
        let data: Vec<i8> = vec![0; 16];
        let packed = encode_ternary(&data);
        assert_eq!(packed.len(), 4);
        assert!(packed.iter().all(|&b| b == 0));
        let decoded = decode_ternary(&packed, 16);
        assert_eq!(decoded, data);
    }

    #[test]
    fn encode_decode_ternary_all_positive() {
        let data: Vec<i8> = vec![1; 8];
        let packed = encode_ternary(&data);
        // Each byte: 01 | 01<<2 | 01<<4 | 01<<6 = 0b01_01_01_01 = 0x55
        assert!(packed.iter().all(|&b| b == 0x55));
        let decoded = decode_ternary(&packed, 8);
        assert_eq!(decoded, data);
    }

    #[test]
    fn encode_decode_ternary_all_negative() {
        let data: Vec<i8> = vec![-1; 8];
        let packed = encode_ternary(&data);
        // Each byte: 10 | 10<<2 | 10<<4 | 10<<6 = 0b10_10_10_10 = 0xAA
        assert!(packed.iter().all(|&b| b == 0xAA));
        let decoded = decode_ternary(&packed, 8);
        assert_eq!(decoded, data);
    }

    #[test]
    fn ternary_bytes_per_row_basic() {
        assert_eq!(ternary_bytes_per_row(16), 4);
        assert_eq!(ternary_bytes_per_row(2560), 640);
        assert_eq!(ternary_bytes_per_row(4096), 1024);
    }

    #[test]
    fn decode_floats_ternary() {
        let data: Vec<i8> = vec![1, -1, 0, 1];
        let packed = encode_ternary(&data);
        let floats = decode_floats(&packed, StorageDtype::Ternary);
        assert_eq!(floats, vec![1.0, -1.0, 0.0, 1.0]);
    }
}
