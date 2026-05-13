//! Scalar quantization (SQ8): compress f32 embeddings to u8 with linear scaling.
//!
//! SQ8 reduces memory 4x while preserving distance ordering for ANN search.
//! The scheme: linearly map [min, max] of each vector to [0, 255].
//!
//! Distance computation uses "asymmetric" comparison: the query stays in f32
//! while stored vectors are u8, avoiding double quantization error.

/// Quantize an f32 embedding to u8 with linear scaling.
///
/// Returns `(quantized_bytes, min, max)` where min/max are needed for
/// dequantization.
pub fn quantize_sq8(embedding: &[f32]) -> (Vec<u8>, f32, f32) {
    if embedding.is_empty() {
        return (Vec::new(), 0.0, 0.0);
    }

    let min = embedding.iter().copied().fold(f32::INFINITY, f32::min);
    let max = embedding.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let range = max - min;
    let scale = if range > 0.0 { 255.0 / range } else { 0.0 };

    let quantized: Vec<u8> = embedding
        .iter()
        .map(|&v| ((v - min) * scale).round().clamp(0.0, 255.0) as u8)
        .collect();

    (quantized, min, max)
}

/// Dequantize u8 bytes back to f32 using stored min/max.
pub fn dequantize_sq8(quantized: &[u8], min: f32, max: f32) -> Vec<f32> {
    let range = max - min;
    let scale = if range > 0.0 { range / 255.0 } else { 0.0 };
    quantized
        .iter()
        .map(|&v| min + (v as f32) * scale)
        .collect()
}

/// Asymmetric L2 distance: query stays f32, stored vector is u8.
///
/// This avoids double quantization error (query is never quantized).
/// Returns the Euclidean distance (sqrt of sum of squared differences).
pub fn asymmetric_distance_sq8(query_f32: &[f32], stored_u8: &[u8], min: f32, max: f32) -> f32 {
    debug_assert_eq!(
        query_f32.len(),
        stored_u8.len(),
        "dimension mismatch in asymmetric_distance_sq8"
    );

    let range = max - min;
    let scale = if range > 0.0 { range / 255.0 } else { 0.0 };

    let sum_sq: f32 = query_f32
        .iter()
        .zip(stored_u8.iter())
        .map(|(&q, &s)| {
            let s_f32 = min + (s as f32) * scale;
            let diff = q - s_f32;
            diff * diff
        })
        .sum();

    sum_sq.sqrt()
}

/// Asymmetric squared L2 distance (no sqrt for faster comparisons during search).
pub fn asymmetric_distance_sq8_squared(
    query_f32: &[f32],
    stored_u8: &[u8],
    min: f32,
    max: f32,
) -> f32 {
    debug_assert_eq!(
        query_f32.len(),
        stored_u8.len(),
        "dimension mismatch in asymmetric_distance_sq8_squared"
    );

    let range = max - min;
    let scale = if range > 0.0 { range / 255.0 } else { 0.0 };

    query_f32
        .iter()
        .zip(stored_u8.iter())
        .map(|(&q, &s)| {
            let s_f32 = min + (s as f32) * scale;
            let diff = q - s_f32;
            diff * diff
        })
        .sum()
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        let original = vec![0.0, 0.5, 1.0, -1.0, 0.25, 0.75];
        let (quantized, min, max) = quantize_sq8(&original);

        assert_eq!(quantized.len(), original.len());
        assert_eq!(min, -1.0);
        assert_eq!(max, 1.0);

        // Check boundary values.
        assert_eq!(quantized[3], 0); // min value maps to 0
        assert_eq!(quantized[2], 255); // max value maps to 255

        let restored = dequantize_sq8(&quantized, min, max);
        assert_eq!(restored.len(), original.len());

        // SQ8 error should be at most range/255 per element.
        let max_error = (max - min) / 255.0;
        for (orig, rest) in original.iter().zip(restored.iter()) {
            assert!(
                (orig - rest).abs() <= max_error + f32::EPSILON,
                "orig={orig}, restored={rest}, max_error={max_error}"
            );
        }
    }

    #[test]
    fn test_quantize_constant_vector() {
        // All same values => scale = 0, all quantize to 0.
        let constant = vec![3.14; 128];
        let (quantized, min, max) = quantize_sq8(&constant);

        assert_eq!(min, 3.14);
        assert_eq!(max, 3.14);
        assert!(quantized.iter().all(|&v| v == 0));

        // Dequantize should return min for all elements.
        let restored = dequantize_sq8(&quantized, min, max);
        for v in &restored {
            assert_eq!(*v, 3.14);
        }
    }

    #[test]
    fn test_quantize_empty() {
        let (quantized, min, max) = quantize_sq8(&[]);
        assert!(quantized.is_empty());
        assert_eq!(min, 0.0);
        assert_eq!(max, 0.0);
    }

    #[test]
    fn test_asymmetric_distance_exact() {
        // When the stored vector dequantizes exactly back to query,
        // distance should be ~0.
        let query = vec![0.0, 0.5, 1.0];
        let (stored, min, max) = quantize_sq8(&query);
        let dist = asymmetric_distance_sq8(&query, &stored, min, max);
        // Allow small quantization error.
        assert!(dist < 0.02, "distance should be near zero, got {dist}");
    }

    #[test]
    fn test_asymmetric_distance_known() {
        // Query = [0, 0, 0], stored reconstructs to [1, 1, 1].
        // Distance should be sqrt(3).
        let stored_orig = vec![1.0, 1.0, 1.0];
        let (stored, min, max) = quantize_sq8(&stored_orig);
        let query = vec![0.0, 0.0, 0.0];
        let dist = asymmetric_distance_sq8(&query, &stored, min, max);
        let expected = 3.0_f32.sqrt();
        assert!(
            (dist - expected).abs() < 0.05,
            "expected ~{expected}, got {dist}"
        );
    }

    #[test]
    fn test_sq8_preserves_distance_ordering() {
        // Verify that SQ8 preserves the relative ordering of distances.
        let query = vec![0.5; 64];
        let close = vec![0.6; 64];
        let far = vec![2.0; 64];

        let (close_q, close_min, close_max) = quantize_sq8(&close);
        let (far_q, far_min, far_max) = quantize_sq8(&far);

        let dist_close = asymmetric_distance_sq8(&query, &close_q, close_min, close_max);
        let dist_far = asymmetric_distance_sq8(&query, &far_q, far_min, far_max);

        assert!(
            dist_close < dist_far,
            "close={dist_close}, far={dist_far}: ordering violated"
        );
    }

    #[test]
    fn test_sq8_max_quantization_error() {
        // For a uniform random vector, max per-element error is range/255.
        // Total L2 error for dim D is at most sqrt(D) * (range/255).
        let dim = 768;
        let original: Vec<f32> = (0..dim).map(|i| (i as f32) / (dim as f32)).collect();
        let (quantized, min, max) = quantize_sq8(&original);
        let dist = asymmetric_distance_sq8(&original, &quantized, min, max);

        let range = max - min;
        let max_element_error = range / 255.0;
        let max_total_error = (dim as f32).sqrt() * max_element_error;

        assert!(
            dist <= max_total_error + 0.01,
            "dist={dist}, bound={max_total_error}"
        );
    }
}
