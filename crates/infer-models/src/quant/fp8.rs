//! FP8 E4M3 ↔ f32 conversion.
//!
//! FP8 E4M3 per the OCP FP8 specification v1.0:
//! 1 sign bit, 4 exponent bits (bias 7), 3 mantissa bits.
//! Range ≈ ±448, min positive normal 2⁻⁶, min positive subnormal 2⁻⁹.
//! `0x7F` and `0xFF` are NaN; there is no Inf.
//!
//! Used by the LARQL FP4 vindex format as both the per-sub-block scale
//! format and the per-block scale format.

/// Convert one E4M3 byte to f32.
///
/// Uses a 256-entry precomputed lookup table for speed; the table is
/// materialised once per thread via `thread_local!`.
#[inline]
pub fn e4m3_to_f32(byte: u8) -> f32 {
    E4M3_TABLE.with(|t| t[byte as usize])
}

thread_local! {
    static E4M3_TABLE: [f32; 256] = build_e4m3_table();
}

fn build_e4m3_table() -> [f32; 256] {
    let mut t = [0.0f32; 256];
    for i in 0..256u32 {
        t[i as usize] = e4m3_bits_to_f32_compute(i as u8);
    }
    t
}

fn e4m3_bits_to_f32_compute(byte: u8) -> f32 {
    let sign = (byte >> 7) & 1;
    let exp = (byte >> 3) & 0x0F;
    let mant = byte & 0x07;

    // NaN encoding: exp = 1111, mant = 111 (both signs).
    if exp == 0x0F && mant == 0x07 {
        return f32::NAN;
    }

    let mag = if exp == 0 {
        // Subnormal: value = mant / 8 × 2⁻⁶.
        (mant as f32) * (1.0 / 8.0) * (2.0_f32).powi(-6)
    } else {
        // Normal: value = (1 + mant/8) × 2^(exp - 7).
        let frac = 1.0 + (mant as f32) / 8.0;
        frac * (2.0_f32).powi(exp as i32 - 7)
    };

    if sign == 1 { -mag } else { mag }
}

/// Convert f32 to E4M3 byte with round-to-nearest-even.
///
/// Saturates to ±448 on overflow (no Inf in E4M3). NaN inputs produce
/// the canonical E4M3 NaN (`0x7F` for positive, `0xFF` for negative).
#[inline]
pub fn f32_to_e4m3(value: f32) -> u8 {
    if value.is_nan() {
        return if value.is_sign_negative() { 0xFF } else { 0x7F };
    }

    let sign_bit: u8 = if value.is_sign_negative() { 0x80 } else { 0x00 };
    let mag = value.abs();

    if mag == 0.0 {
        return sign_bit;
    }

    // E4M3 max normal: exp=14, mant=6 → (1 + 6/8) × 2^(14-7) = 1.75 × 128 = 224?
    // Actually exp=15 mant<7 are valid normals in E4M3 (no Inf representation).
    // Max normal is exp=15 mant=6: (1 + 6/8) × 2^8 = 1.75 × 256 = 448.
    const E4M3_MAX: f32 = 448.0;
    if mag >= E4M3_MAX {
        return sign_bit | 0x7E;
    }

    let bits = mag.to_bits();
    let f32_exp = ((bits >> 23) & 0xFF) as i32 - 127;

    if f32_exp < -9 {
        // Below smallest subnormal — flush to zero.
        return sign_bit;
    }

    if f32_exp < -6 {
        // Subnormal in E4M3. Value = 2^-6 × (mant/8).
        // mant = mag × 2^9.
        let scaled = mag * (2.0_f32).powi(9);
        let rounded = round_ties_to_even(scaled);
        let m = rounded.clamp(0.0, 7.0) as u32;
        return sign_bit | (m as u8);
    }

    // Normal in E4M3.
    let e4m3_exp = (f32_exp + 7) as u32;
    if e4m3_exp > 15 {
        return sign_bit | 0x7E;
    }

    // f32 mantissa: 23 bits; E4M3 keeps 3 bits.
    let f32_mant_full = bits & 0x007F_FFFF;
    let keep = f32_mant_full >> 20;
    let rem = f32_mant_full & 0x000F_FFFF;
    let half = 0x0008_0000;
    let rounded_up = rem > half || (rem == half && (keep & 1) == 1);

    let (mut e, mut m) = (e4m3_exp, keep);
    if rounded_up {
        m += 1;
        if m == 8 {
            m = 0;
            e += 1;
        }
    }

    if e >= 15 && m >= 7 {
        return sign_bit | 0x7E;
    }
    if e > 15 {
        return sign_bit | 0x7E;
    }

    sign_bit | ((e as u8) << 3) | (m as u8)
}

fn round_ties_to_even(x: f32) -> f32 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 {
        if (r as i32) % 2 != 0 {
            r - r.signum()
        } else {
            r
        }
    } else {
        r
    }
}

/// Encode a slice of f32 values to E4M3 bytes.
pub fn encode_e4m3(data: &[f32]) -> Vec<u8> {
    data.iter().map(|&v| f32_to_e4m3(v)).collect()
}

/// Decode an E4M3 byte slice to f32.
pub fn decode_e4m3(bytes: &[u8]) -> Vec<f32> {
    bytes.iter().map(|&b| e4m3_to_f32(b)).collect()
}

/// Decode F8 E4M3 tensor data (1 byte per value) to f32 vec.
/// This is the safetensors dtype decoder entry point.
pub fn decode_f8_e4m3(data: &[u8]) -> Vec<f32> {
    decode_e4m3(data)
}

/// Decode I8 (signed int8) tensor data to f32 vec.
pub fn decode_i8(data: &[u8]) -> Vec<f32> {
    data.iter().map(|&b| (b as i8) as f32).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e4m3_canonical_values() {
        assert_eq!(e4m3_to_f32(0x00), 0.0);
        assert_eq!(e4m3_to_f32(0x80).to_bits(), (-0.0f32).to_bits());
        // Smallest positive subnormal: 2^-9
        assert!((e4m3_to_f32(0x01) - 1.0 / 512.0).abs() < 1e-7);
        // Smallest positive normal: 2^-6
        assert!((e4m3_to_f32(0x08) - 1.0 / 64.0).abs() < 1e-7);
        // Max normal: 448
        assert_eq!(e4m3_to_f32(0x7E), 448.0);
        assert_eq!(e4m3_to_f32(0xFE), -448.0);
        // NaN
        assert!(e4m3_to_f32(0x7F).is_nan());
        assert!(e4m3_to_f32(0xFF).is_nan());
    }

    #[test]
    fn e4m3_round_trip_representable() {
        for byte in 0..=255u8 {
            let f = e4m3_to_f32(byte);
            if f.is_nan() { continue; }
            let back = f32_to_e4m3(f);
            if f == 0.0 {
                assert!(back == 0x00 || back == 0x80);
                continue;
            }
            assert_eq!(back, byte, "roundtrip {byte:#x} → {f} → {back:#x}");
        }
    }

    #[test]
    fn e4m3_saturation() {
        assert_eq!(f32_to_e4m3(1000.0), 0x7E);
        assert_eq!(f32_to_e4m3(-1000.0), 0xFE);
        assert_eq!(f32_to_e4m3(448.0), 0x7E);
        assert_eq!(f32_to_e4m3(-448.0), 0xFE);
    }

    #[test]
    fn e4m3_tiny_flush_to_zero() {
        assert_eq!(f32_to_e4m3(1e-10), 0x00);
        assert_eq!(f32_to_e4m3(-1e-10), 0x80);
    }

    #[test]
    fn e4m3_subnormal_sweep() {
        for m in 1..=7u8 {
            let expected = (m as f32 / 8.0) * (2.0_f32).powi(-6);
            let decoded = e4m3_to_f32(m);
            assert!((decoded - expected).abs() < 1e-12, "m={m}: expected {expected}, got {decoded}");
        }
        for m in 1..=7u8 {
            let expected = -(m as f32 / 8.0) * (2.0_f32).powi(-6);
            let decoded = e4m3_to_f32(0x80 | m);
            assert!((decoded - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn e4m3_infinity_saturates() {
        assert_eq!(f32_to_e4m3(f32::INFINITY), 0x7E);
        assert_eq!(f32_to_e4m3(f32::NEG_INFINITY), 0xFE);
    }

    #[test]
    fn decode_i8_basic() {
        let data: Vec<u8> = vec![0, 1, 127, 128, 255];
        let result = decode_i8(&data);
        assert_eq!(result, vec![0.0, 1.0, 127.0, -128.0, -1.0]);
    }

    #[test]
    fn decode_f8_e4m3_batch() {
        let data = vec![0x38, 0x00, 0x7E]; // 1.0, 0.0, 448.0
        let result = decode_f8_e4m3(&data);
        assert_eq!(result[0], 1.0);
        assert_eq!(result[1], 0.0);
        assert_eq!(result[2], 448.0);
    }
}
