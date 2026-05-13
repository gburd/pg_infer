//! GGUF skip-dequant loader — native quantized mmap loading without f32 expansion.
//!
//! The standard `load_gguf()` dequantizes all tensors to f32 during load, expanding
//! a 4-bit model to 8x its on-disk size. This module keeps 2D weight tensors in their
//! native quantized format as raw byte references into a memory-mapped file, providing
//! ~6x memory savings for large models.
//!
//! 1D tensors (norms, biases) are still dequantized to f32 since they are small and
//! needed as-is by normalization routines.
//!
//! The `QuantizedTensor` struct provides zero-copy access to quantized weight data
//! through `Arc<Mmap>`. Callers in `infer-inference` can construct
//! `infer_compute::QuantWeight` from the raw data slice and format tag.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use memmap2::Mmap;

use crate::config::{ModelArchitecture, ModelConfig};
use crate::detect::ModelError;
use crate::quant::ggml;

use super::gguf::GgufFile;

/// Quantization format for a GGUF tensor — mirrors GGML type IDs but as a Rust enum.
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(non_camel_case_types)]
pub enum GgufQuantFormat {
    F32,
    F16,
    BF16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
}

impl GgufQuantFormat {
    /// Convert a GGML type ID to our enum, or None if unsupported.
    pub fn from_ggml_type(ty: u32) -> Option<Self> {
        match ty {
            ggml::TYPE_F32 => Some(Self::F32),
            ggml::TYPE_F16 => Some(Self::F16),
            ggml::TYPE_BF16 => Some(Self::BF16),
            ggml::TYPE_Q4_0 => Some(Self::Q4_0),
            ggml::TYPE_Q4_1 => Some(Self::Q4_1),
            ggml::TYPE_Q5_0 => Some(Self::Q5_0),
            ggml::TYPE_Q5_1 => Some(Self::Q5_1),
            ggml::TYPE_Q8_0 => Some(Self::Q8_0),
            ggml::TYPE_Q2_K => Some(Self::Q2_K),
            ggml::TYPE_Q3_K => Some(Self::Q3_K),
            ggml::TYPE_Q4_K => Some(Self::Q4_K),
            ggml::TYPE_Q5_K => Some(Self::Q5_K),
            ggml::TYPE_Q6_K => Some(Self::Q6_K),
            _ => None,
        }
    }

    /// Whether this format has a compute kernel for quantized matvec.
    ///
    /// Formats with kernels: Q4_0, Q4_K, Q6_K, Q8_0.
    /// Callers in infer-inference use this to decide between quantized matvec
    /// and dequantize-then-multiply paths.
    pub fn has_compute_kernel(self) -> bool {
        matches!(self, Self::Q4_0 | Self::Q4_K | Self::Q6_K | Self::Q8_0)
    }

    /// Convert back to the GGML type ID constant.
    pub fn as_ggml_type(self) -> u32 {
        match self {
            Self::F32 => ggml::TYPE_F32,
            Self::F16 => ggml::TYPE_F16,
            Self::BF16 => ggml::TYPE_BF16,
            Self::Q4_0 => ggml::TYPE_Q4_0,
            Self::Q4_1 => ggml::TYPE_Q4_1,
            Self::Q5_0 => ggml::TYPE_Q5_0,
            Self::Q5_1 => ggml::TYPE_Q5_1,
            Self::Q8_0 => ggml::TYPE_Q8_0,
            Self::Q2_K => ggml::TYPE_Q2_K,
            Self::Q3_K => ggml::TYPE_Q3_K,
            Self::Q4_K => ggml::TYPE_Q4_K,
            Self::Q5_K => ggml::TYPE_Q5_K,
            Self::Q6_K => ggml::TYPE_Q6_K,
        }
    }
}

/// A tensor stored as a reference into mmap'd GGUF file data — NO f32 expansion.
pub struct QuantizedTensor {
    /// Shared reference to the memory-mapped file. Kept alive via Arc so the
    /// mmap is not unmapped while any QuantizedTensor exists.
    pub mmap: Arc<Mmap>,
    /// Byte offset into the mmap where this tensor's data begins.
    pub offset: usize,
    /// Length in bytes of this tensor's data.
    pub length: usize,
    /// Native quantization format.
    pub format: GgufQuantFormat,
    /// Number of rows (dim 1 in GGML column-major = outer dimension in row-major).
    pub rows: usize,
    /// Number of columns (dim 0 in GGML column-major = inner dimension in row-major).
    pub cols: usize,
}

impl QuantizedTensor {
    /// Get a byte slice reference into the mmap for this tensor's data.
    #[inline]
    pub fn data(&self) -> &[u8] {
        &self.mmap[self.offset..self.offset + self.length]
    }

    /// Total number of elements in this tensor.
    #[inline]
    pub fn n_elements(&self) -> usize {
        self.rows * self.cols
    }

    /// Get a byte slice for a single row of this quantized tensor.
    ///
    /// Returns `None` if `row` is out of bounds or the byte-size computation
    /// fails for this format.
    pub fn row_data(&self, row: usize) -> Option<&[u8]> {
        if row >= self.rows {
            return None;
        }
        let row_bytes = ggml::tensor_data_size(
            self.format.as_ggml_type(),
            self.cols,
        ).ok()?;
        let start = self.offset + row * row_bytes;
        let end = start + row_bytes;
        if end <= self.offset + self.length {
            Some(&self.mmap[start..end])
        } else {
            None
        }
    }
}

/// Model weights kept in native quantized form — ~6x memory savings vs f32.
///
/// 2D weight tensors stay as raw byte references into the mmap. 1D tensors
/// (norms, biases) are dequantized to f32 since they are small and needed
/// directly by normalization routines.
pub struct QuantizedModelWeights {
    /// The underlying memory-mapped GGUF file.
    pub mmap: Arc<Mmap>,
    /// 2D weight tensors in native quantized format.
    pub tensors: HashMap<String, QuantizedTensor>,
    /// 1D tensors (norms, biases) dequantized to f32.
    pub vectors: HashMap<String, Vec<f32>>,
    /// Model configuration derived from GGUF metadata.
    pub config: ModelConfig,
    /// Architecture detection result for key mapping.
    pub arch: Box<dyn ModelArchitecture>,
}

impl QuantizedModelWeights {
    /// Get a quantized tensor by normalized key name.
    pub fn get_tensor(&self, key: &str) -> Option<&QuantizedTensor> {
        self.tensors.get(key)
    }

    /// Get a vector (1D f32) by normalized key name.
    pub fn get_vector(&self, key: &str) -> Option<&[f32]> {
        self.vectors.get(key).map(|v| v.as_slice())
    }

    /// Number of layers in the model.
    pub fn num_layers(&self) -> usize {
        self.config.num_layers
    }

    /// Hidden dimension.
    pub fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }
}

/// Load a GGUF file without dequantizing 2D weight tensors.
///
/// - 2D weight tensors: stored as `QuantizedTensor` (raw mmap reference, no f32 expansion)
/// - 1D tensors (norms, biases): dequantized to f32 (small, needed as-is)
///
/// Memory savings example: a 7B Q4_K model is ~4GB on disk. Standard loading would
/// expand to ~28GB of f32. Skip-dequant keeps the 4GB mmap and adds only ~10MB for
/// 1D tensors.
pub fn load_gguf_quantized(path: &Path) -> Result<QuantizedModelWeights, ModelError> {
    // 1. Parse GGUF header
    let gguf = GgufFile::open(path)?;

    // 2. Detect architecture from GGUF metadata
    let config_json = gguf.to_config_json();
    let arch = crate::detect_from_json(&config_json);
    let prefixes = arch.key_prefixes_to_strip();
    let config = arch.config().clone();

    // 3. Open file and create shared mmap
    let file = std::fs::File::open(path)?;
    let mmap = Arc::new(unsafe { Mmap::map(&file)? });

    // 4. Process each tensor
    let mut tensors = HashMap::new();
    let mut vectors = HashMap::new();

    for info in &gguf.tensor_infos {
        // Compute absolute offset in the mmap
        let abs_offset = gguf
            .data_offset
            .checked_add(info.offset)
            .ok_or_else(|| {
                ModelError::Parse(format!(
                    "tensor {}: data_offset {} + tensor offset {} overflows u64",
                    info.name, gguf.data_offset, info.offset,
                ))
            })?;

        let n_elements: u64 = info.dims.iter().product();
        let data_size = ggml::tensor_data_size(info.tensor_type, n_elements as usize)?;

        let abs_offset_usize = usize::try_from(abs_offset).map_err(|_| {
            ModelError::Parse(format!(
                "tensor {}: absolute offset {} exceeds usize on this platform",
                info.name, abs_offset,
            ))
        })?;
        let end = abs_offset_usize.checked_add(data_size).ok_or_else(|| {
            ModelError::Parse(format!(
                "tensor {}: offset {} + size {} overflows usize",
                info.name, abs_offset_usize, data_size,
            ))
        })?;
        if end > mmap.len() {
            return Err(ModelError::Parse(format!(
                "tensor {} data out of bounds (offset {} + size {} > file {})",
                info.name, abs_offset, data_size, mmap.len()
            )));
        }

        // Normalize key name (strip GGUF prefixes)
        let raw_key = super::gguf::normalize_gguf_key(&info.name);
        let key = super::safetensors::normalize_key_pub(&raw_key, prefixes);

        let format = GgufQuantFormat::from_ggml_type(info.tensor_type).ok_or_else(|| {
            ModelError::UnsupportedDtype(format!(
                "GGML type {} for tensor {}",
                info.tensor_type, info.name,
            ))
        })?;

        match info.n_dims {
            2 => {
                // GGUF/GGML column-major: dims[0] = cols, dims[1] = rows
                let cols = info.dims[0] as usize;
                let rows = info.dims[1] as usize;

                tensors.insert(
                    key,
                    QuantizedTensor {
                        mmap: Arc::clone(&mmap),
                        offset: abs_offset_usize,
                        length: data_size,
                        format,
                        rows,
                        cols,
                    },
                );
            }
            1 => {
                // 1D tensors: dequantize to f32 (norms, biases — small)
                let raw = &mmap[abs_offset_usize..end];
                let floats = ggml::dequantize(raw, info.tensor_type, n_elements as usize)?;
                vectors.insert(key, floats);
            }
            _ => {
                // Skip higher-dim tensors (rare in transformer models)
            }
        }
    }

    Ok(QuantizedModelWeights {
        mmap,
        tensors,
        vectors,
        config,
        arch,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Seek, Write};

    /// Build a minimal GGUF file with one 2D Q4_0 tensor and one 1D F32 tensor.
    fn write_test_gguf(path: &Path) {
        let mut file = std::fs::File::create(path).expect("create test gguf");

        // Header
        let magic: u32 = 0x46554747;
        file.write_all(&magic.to_le_bytes()).expect("write magic");
        file.write_all(&3u32.to_le_bytes()).expect("write version");
        file.write_all(&2u64.to_le_bytes()).expect("write n_tensors");
        file.write_all(&0u64.to_le_bytes()).expect("write n_metadata");

        // Tensor 1: 2D Q4_0 weight [4 rows, 32 cols] = 128 elements
        // Q4_0: 18 bytes per 32 elements → 4 rows × 18 bytes/row = 72 bytes
        let name1 = b"blk.0.ffn_gate.weight";
        file.write_all(&(name1.len() as u64).to_le_bytes()).expect("write name1 len");
        file.write_all(name1).expect("write name1");
        file.write_all(&2u32.to_le_bytes()).expect("write n_dims 1");
        file.write_all(&32u64.to_le_bytes()).expect("write dim0 (cols)");
        file.write_all(&4u64.to_le_bytes()).expect("write dim1 (rows)");
        file.write_all(&ggml::TYPE_Q4_0.to_le_bytes()).expect("write type Q4_0");
        file.write_all(&0u64.to_le_bytes()).expect("write offset 0");

        // Tensor 2: 1D F32 norm [16 elements] = 64 bytes, starts after tensor 1
        let name2 = b"blk.0.attn_norm.weight";
        file.write_all(&(name2.len() as u64).to_le_bytes()).expect("write name2 len");
        file.write_all(name2).expect("write name2");
        file.write_all(&1u32.to_le_bytes()).expect("write n_dims 2");
        file.write_all(&16u64.to_le_bytes()).expect("write dim0");
        file.write_all(&ggml::TYPE_F32.to_le_bytes()).expect("write type F32");
        // Tensor 2 offset = 72 (after tensor 1's 72 bytes)
        file.write_all(&72u64.to_le_bytes()).expect("write offset 72");

        // Pad to 32-byte alignment for data section
        let pos = file.stream_position().expect("get position");
        let aligned = pos.div_ceil(32) * 32;
        let padding = (aligned - pos) as usize;
        file.write_all(&vec![0u8; padding]).expect("write padding");

        // Tensor 1 data: 4 rows × 18 bytes = 72 bytes of Q4_0 data
        // Each row: 2 bytes f16 scale + 16 bytes quants
        for row in 0..4u8 {
            // f16 scale = 1.0 (0x3C00)
            file.write_all(&[0x00, 0x3C]).expect("write scale");
            // 16 bytes of quant data (fill with row index for verification)
            file.write_all(&[row | (row << 4); 16]).expect("write quants");
        }

        // Tensor 2 data: 16 f32 values
        for i in 0..16u32 {
            let val = (i as f32) * 0.1;
            file.write_all(&val.to_le_bytes()).expect("write f32");
        }

        file.flush().expect("flush");
    }

    #[test]
    fn test_quantized_tensor_references_mmap_data() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("test_skip_dequant.gguf");
        write_test_gguf(&path);

        let weights = load_gguf_quantized(&path).expect("load_gguf_quantized");

        // 2D tensor should be in tensors map (not dequantized)
        let gate = weights
            .tensors
            .get("layers.0.mlp.gate_proj.weight")
            .expect("gate_proj tensor missing");

        assert_eq!(gate.rows, 4);
        assert_eq!(gate.cols, 32);
        assert_eq!(gate.format, GgufQuantFormat::Q4_0);
        // Q4_0: 18 bytes per 32 elements, 4 rows = 72 bytes
        assert_eq!(gate.length, 72);

        // Verify the data slice references actual mmap content
        let data = gate.data();
        assert_eq!(data.len(), 72);
        // First row: scale bytes 0x00 0x3C (f16 1.0)
        assert_eq!(data[0], 0x00);
        assert_eq!(data[1], 0x3C);

        // Per-row access works
        let row0 = gate.row_data(0).expect("row 0");
        assert_eq!(row0.len(), 18);
        assert_eq!(row0[0], 0x00); // scale low
        assert_eq!(row0[1], 0x3C); // scale high
        // Quant bytes for row 0: 0x00 | (0x00 << 4) = 0x00
        assert_eq!(row0[2], 0x00);

        let row3 = gate.row_data(3).expect("row 3");
        assert_eq!(row3.len(), 18);
        // Quant bytes for row 3: 3 | (3 << 4) = 0x33
        assert_eq!(row3[2], 0x33);

        // Out of bounds row returns None
        assert!(gate.row_data(4).is_none());
    }

    #[test]
    fn test_1d_tensors_dequantized_to_f32() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("test_vectors.gguf");
        write_test_gguf(&path);

        let weights = load_gguf_quantized(&path).expect("load_gguf_quantized");

        // 1D tensor should be in vectors map (dequantized)
        let norm = weights
            .vectors
            .get("layers.0.input_layernorm.weight")
            .expect("norm vector missing");

        assert_eq!(norm.len(), 16);
        // Values should be i * 0.1
        for (i, &val) in norm.iter().enumerate() {
            let expected = i as f32 * 0.1;
            assert!(
                (val - expected).abs() < 1e-6,
                "norm[{i}] = {val}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_has_compute_kernel() {
        // Formats with compute kernels
        assert!(GgufQuantFormat::Q4_0.has_compute_kernel());
        assert!(GgufQuantFormat::Q4_K.has_compute_kernel());
        assert!(GgufQuantFormat::Q6_K.has_compute_kernel());
        assert!(GgufQuantFormat::Q8_0.has_compute_kernel());

        // Formats without compute kernels
        assert!(!GgufQuantFormat::F16.has_compute_kernel());
        assert!(!GgufQuantFormat::BF16.has_compute_kernel());
        assert!(!GgufQuantFormat::F32.has_compute_kernel());
        assert!(!GgufQuantFormat::Q5_0.has_compute_kernel());
        assert!(!GgufQuantFormat::Q2_K.has_compute_kernel());
    }

    #[test]
    fn test_ggml_type_roundtrip() {
        let formats = [
            (GgufQuantFormat::F32, ggml::TYPE_F32),
            (GgufQuantFormat::F16, ggml::TYPE_F16),
            (GgufQuantFormat::BF16, ggml::TYPE_BF16),
            (GgufQuantFormat::Q4_0, ggml::TYPE_Q4_0),
            (GgufQuantFormat::Q4_K, ggml::TYPE_Q4_K),
            (GgufQuantFormat::Q6_K, ggml::TYPE_Q6_K),
            (GgufQuantFormat::Q8_0, ggml::TYPE_Q8_0),
        ];
        for (fmt, expected_type) in formats {
            assert_eq!(fmt.as_ggml_type(), expected_type);
            assert_eq!(
                GgufQuantFormat::from_ggml_type(expected_type),
                Some(fmt),
            );
        }
    }

    #[test]
    fn test_mmap_shared_across_tensors() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("test_shared_mmap.gguf");
        write_test_gguf(&path);

        let weights = load_gguf_quantized(&path).expect("load");

        // All tensors should share the same Arc<Mmap>
        let gate = weights
            .tensors
            .get("layers.0.mlp.gate_proj.weight")
            .expect("gate");
        assert!(Arc::ptr_eq(&gate.mmap, &weights.mmap));
    }

    #[test]
    fn test_rejects_truncated_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("truncated.gguf");

        // Write a valid header but truncate tensor data
        let mut file = std::fs::File::create(&path).expect("create");
        let magic: u32 = 0x46554747;
        file.write_all(&magic.to_le_bytes()).expect("magic");
        file.write_all(&3u32.to_le_bytes()).expect("version");
        file.write_all(&1u64.to_le_bytes()).expect("n_tensors");
        file.write_all(&0u64.to_le_bytes()).expect("n_metadata");

        let name = b"blk.0.ffn_gate.weight";
        file.write_all(&(name.len() as u64).to_le_bytes()).expect("len");
        file.write_all(name).expect("name");
        file.write_all(&2u32.to_le_bytes()).expect("ndims");
        file.write_all(&32u64.to_le_bytes()).expect("cols");
        file.write_all(&4u64.to_le_bytes()).expect("rows");
        file.write_all(&ggml::TYPE_Q4_0.to_le_bytes()).expect("type");
        file.write_all(&0u64.to_le_bytes()).expect("offset");

        // Pad to alignment, write only 10 bytes (need 72)
        let pos = file.stream_position().expect("pos");
        let aligned = pos.div_ceil(32) * 32;
        file.write_all(&vec![0u8; (aligned - pos) as usize]).expect("pad");
        file.write_all(&[0u8; 10]).expect("short data");
        file.flush().expect("flush");

        match load_gguf_quantized(&path) {
            Err(ModelError::Parse(msg)) => {
                assert!(
                    msg.contains("out of bounds"),
                    "expected 'out of bounds' error, got: {msg}"
                );
            }
            Err(other) => panic!("expected Parse error, got {other:?}"),
            Ok(_) => panic!("expected error, got Ok"),
        }
    }
}
