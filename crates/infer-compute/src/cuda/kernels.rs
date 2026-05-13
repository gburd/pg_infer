//! Custom CUDA kernel dispatch for Q4_K and Q6_K dequant+matvec.
//!
//! Kernels are compiled from embedded CUDA C source at runtime via cudarc's
//! PTX compilation facility. Each kernel processes one output row per thread
//! block, with threads cooperating on the dot product over super-blocks.
//!
//! ## Q4_K Block Layout (144 bytes per 256 values)
//!
//! - Bytes 0..2: f16 `d` (super-block scale)
//! - Bytes 2..4: f16 `dmin` (super-block minimum offset)
//! - Bytes 4..16: packed sub-block scales + mins (12 bytes → 8 scales + 8 mins)
//! - Bytes 16..144: 128 packed nibbles (256 × 4 bits)
//!
//! ## Q6_K Block Layout (210 bytes per 256 values)
//!
//! - Bytes 0..128: `ql` — lower 4 bits of each value (128 bytes for 256 nibbles)
//! - Bytes 128..192: `qh` — upper 2 bits (64 bytes, 4 values per byte)
//! - Bytes 192..208: 16 × int8 sub-block scales
//! - Bytes 208..210: f16 `d` (super-block scale)

use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaFunction, CudaStream, LaunchAsync, LaunchConfig};

use super::buffers::BufferPool;
use super::CudaError;

/// Q4_K super-block size in bytes (same as CPU reference).
const Q4K_BLOCK_SIZE: usize = 144;

/// Q6_K super-block size in bytes (same as CPU reference).
const Q6K_BLOCK_SIZE: usize = 210;

/// Number of threads per block for quantized matvec kernels.
/// Each block handles one output row; threads cooperate on the dot product.
const THREADS_PER_BLOCK: u32 = 256;

/// CUDA C source for the Q4_K dequantize + matvec kernel.
///
/// Each thread block handles one output row. Threads partition the super-blocks
/// and perform local dot products, then reduce via shared memory.
const Q4K_KERNEL_SRC: &str = r#"
extern "C" __global__ void q4k_matvec(
    const unsigned char* __restrict__ q4k_data,
    const float* __restrict__ x,
    float* __restrict__ out,
    int num_rows,
    int hidden,
    int superblocks,
    int bytes_per_row
) {
    int row = blockIdx.x;
    if (row >= num_rows) return;

    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    const unsigned char* row_data = q4k_data + (long long)row * bytes_per_row;

    float acc = 0.0f;

    // Each thread processes a subset of super-blocks
    for (int sb = tid; sb < superblocks; sb += nthreads) {
        const unsigned char* block = row_data + sb * 144;

        // Decode f16 d and dmin
        unsigned short d_bits = (unsigned short)block[0] | ((unsigned short)block[1] << 8);
        unsigned short dmin_bits = (unsigned short)block[2] | ((unsigned short)block[3] << 8);
        float d = __half2float(*reinterpret_cast<const __half*>(&d_bits));
        float dmin = __half2float(*reinterpret_cast<const __half*>(&dmin_bits));

        // Unpack 12 bytes → 8 scales + 8 mins
        unsigned char scales[8], mins[8];
        const unsigned char* sb_bytes = block + 4;
        for (int j = 0; j < 4; j++) {
            scales[j] = sb_bytes[j] & 0x3F;
            mins[j] = sb_bytes[j + 4] & 0x3F;
        }
        for (int j = 4; j < 8; j++) {
            scales[j] = (sb_bytes[j + 4] & 0x0F) | ((sb_bytes[j - 4] >> 6) << 4);
            mins[j] = (sb_bytes[j + 4] >> 4) | ((sb_bytes[j] >> 6) << 4);
        }

        const unsigned char* qs = block + 16;
        int x_base = sb * 256;

        // Four groups x 32 bytes
        for (int g = 0; g < 4; g++) {
            int sb_lo = 2 * g;
            int sb_hi = 2 * g + 1;
            float sc_lo = d * (float)scales[sb_lo];
            float sc_hi = d * (float)scales[sb_hi];
            float mn_lo = dmin * (float)mins[sb_lo];
            float mn_hi = dmin * (float)mins[sb_hi];
            int qs_off = g * 32;
            int base_lo = x_base + sb_lo * 32;
            int base_hi = x_base + sb_hi * 32;

            for (int l = 0; l < 32; l++) {
                unsigned char byte = qs[qs_off + l];
                float lo = (float)(byte & 0x0F);
                float hi = (float)(byte >> 4);
                acc += (sc_lo * lo - mn_lo) * x[base_lo + l];
                acc += (sc_hi * hi - mn_hi) * x[base_hi + l];
            }
        }
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    // Block reduction via shared memory
    __shared__ float shared[32];
    int warp_id = tid / 32;
    int lane = tid % 32;

    if (lane == 0) shared[warp_id] = acc;
    __syncthreads();

    if (warp_id == 0) {
        acc = (lane < (nthreads / 32)) ? shared[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
        }
        if (lane == 0) {
            out[row] = acc;
        }
    }
}
"#;

/// CUDA C source for the Q6_K dequantize + matvec kernel.
const Q6K_KERNEL_SRC: &str = r#"
extern "C" __global__ void q6k_matvec(
    const unsigned char* __restrict__ q6k_data,
    const float* __restrict__ x,
    float* __restrict__ out,
    int num_rows,
    int hidden,
    int superblocks,
    int bytes_per_row
) {
    int row = blockIdx.x;
    if (row >= num_rows) return;

    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    const unsigned char* row_data = q6k_data + (long long)row * bytes_per_row;

    float acc = 0.0f;

    for (int sb = tid; sb < superblocks; sb += nthreads) {
        const unsigned char* block = row_data + sb * 210;

        // Layout: ql[128] | qh[64] | scales[16] | d[2]
        const unsigned char* ql = block;
        const unsigned char* qh = block + 128;
        const signed char* scales = (const signed char*)(block + 192);
        unsigned short d_bits = (unsigned short)block[208] | ((unsigned short)block[209] << 8);
        float d = __half2float(*reinterpret_cast<const __half*>(&d_bits));

        int x_base = sb * 256;

        // 16 sub-blocks of 16 values each
        for (int j = 0; j < 16; j++) {
            float sc = d * (float)scales[j];
            int sub_base = j * 16;

            for (int i = 0; i < 16; i++) {
                int qi = sub_base + i;
                int byte_idx = qi / 2;
                unsigned char lo_byte = ql[byte_idx];

                // Extract 4-bit lower portion
                int lo_val;
                if (qi % 2 == 0) {
                    lo_val = lo_byte & 0x0F;
                } else {
                    lo_val = lo_byte >> 4;
                }

                // Extract 2-bit upper portion from qh
                int qh_byte_idx = qi / 4;
                int qh_shift = (qi % 4) * 2;
                int hi_val = (qh[qh_byte_idx] >> qh_shift) & 0x03;

                // Combine: 6-bit value (0..63) → signed (-32..31)
                int val = lo_val | (hi_val << 4);
                float dequant = sc * (float)(val - 32);

                acc += dequant * x[x_base + qi];
            }
        }
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    // Block reduction via shared memory
    __shared__ float shared[32];
    int warp_id = tid / 32;
    int lane = tid % 32;

    if (lane == 0) shared[warp_id] = acc;
    __syncthreads();

    if (warp_id == 0) {
        acc = (lane < (nthreads / 32)) ? shared[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
        }
        if (lane == 0) {
            out[row] = acc;
        }
    }
}
"#;

/// Module name for the Q4_K kernel in cudarc's module registry.
const Q4K_MODULE: &str = "q4k_matvec_mod";
/// Module name for the Q6_K kernel.
const Q6K_MODULE: &str = "q6k_matvec_mod";

/// Ensure the Q4_K kernel module is loaded, compiling from source if needed.
fn ensure_q4k_module(device: &Arc<CudaDevice>) -> Result<(), CudaError> {
    if device.get_func(Q4K_MODULE, "q4k_matvec").is_some() {
        return Ok(());
    }
    let ptx = cudarc::nvrtc::compile_ptx(Q4K_KERNEL_SRC)
        .map_err(|e| CudaError::KernelLaunch(format!("Q4_K compile: {e}")))?;
    device.load_ptx(ptx, Q4K_MODULE, &["q4k_matvec"])
        .map_err(|e| CudaError::KernelLaunch(format!("Q4_K load: {e}")))?;
    Ok(())
}

/// Ensure the Q6_K kernel module is loaded, compiling from source if needed.
fn ensure_q6k_module(device: &Arc<CudaDevice>) -> Result<(), CudaError> {
    if device.get_func(Q6K_MODULE, "q6k_matvec").is_some() {
        return Ok(());
    }
    let ptx = cudarc::nvrtc::compile_ptx(Q6K_KERNEL_SRC)
        .map_err(|e| CudaError::KernelLaunch(format!("Q6_K compile: {e}")))?;
    device.load_ptx(ptx, Q6K_MODULE, &["q6k_matvec"])
        .map_err(|e| CudaError::KernelLaunch(format!("Q6_K load: {e}")))?;
    Ok(())
}

/// Dispatch the Q4_K dequant+matvec CUDA kernel.
///
/// Computes `out[N] = Q4_K[N, hidden] @ x[hidden]`.
pub fn q4k_matvec_cuda(
    device: &Arc<CudaDevice>,
    stream: &CudaStream,
    pool: &BufferPool,
    q4k_data: &[u8],
    x: &[f32],
    num_rows: usize,
    hidden: usize,
) -> Result<Vec<f32>, CudaError> {
    ensure_q4k_module(device)?;

    let superblocks = hidden / 256;
    let bytes_per_row = superblocks * Q4K_BLOCK_SIZE;

    // Upload data to device
    let d_q4k = pool.upload_bytes(q4k_data)?;
    let d_x = pool.upload_f32(x)?;
    let d_out = pool.alloc_f32(num_rows)?;

    let func: CudaFunction = device
        .get_func(Q4K_MODULE, "q4k_matvec")
        .ok_or_else(|| CudaError::KernelLaunch("q4k_matvec function not found".into()))?;

    let cfg = LaunchConfig {
        grid_dim: (num_rows as u32, 1, 1),
        block_dim: (THREADS_PER_BLOCK, 1, 1),
        shared_mem_bytes: 128, // 32 floats for warp reduction
    };

    unsafe {
        func.launch_on_stream(
            stream,
            cfg,
            (
                &d_q4k.slice,
                &d_x.slice,
                &d_out.slice,
                num_rows as i32,
                hidden as i32,
                superblocks as i32,
                bytes_per_row as i32,
            ),
        ).map_err(|e| CudaError::KernelLaunch(format!("q4k launch: {e}")))?;
    }

    // Synchronize and download
    device.synchronize().map_err(|e| CudaError::KernelLaunch(format!("sync: {e}")))?;
    pool.download_f32(&d_out)
}

/// Dispatch the Q6_K dequant+matvec CUDA kernel.
///
/// Computes `out[N] = Q6_K[N, hidden] @ x[hidden]`.
pub fn q6k_matvec_cuda(
    device: &Arc<CudaDevice>,
    stream: &CudaStream,
    pool: &BufferPool,
    q6k_data: &[u8],
    x: &[f32],
    num_rows: usize,
    hidden: usize,
) -> Result<Vec<f32>, CudaError> {
    ensure_q6k_module(device)?;

    let superblocks = hidden / 256;
    let bytes_per_row = superblocks * Q6K_BLOCK_SIZE;

    // Upload data to device
    let d_q6k = pool.upload_bytes(q6k_data)?;
    let d_x = pool.upload_f32(x)?;
    let d_out = pool.alloc_f32(num_rows)?;

    let func: CudaFunction = device
        .get_func(Q6K_MODULE, "q6k_matvec")
        .ok_or_else(|| CudaError::KernelLaunch("q6k_matvec function not found".into()))?;

    let cfg = LaunchConfig {
        grid_dim: (num_rows as u32, 1, 1),
        block_dim: (THREADS_PER_BLOCK, 1, 1),
        shared_mem_bytes: 128, // 32 floats for warp reduction
    };

    unsafe {
        func.launch_on_stream(
            stream,
            cfg,
            (
                &d_q6k.slice,
                &d_x.slice,
                &d_out.slice,
                num_rows as i32,
                hidden as i32,
                superblocks as i32,
                bytes_per_row as i32,
            ),
        ).map_err(|e| CudaError::KernelLaunch(format!("q6k launch: {e}")))?;
    }

    // Synchronize and download
    device.synchronize().map_err(|e| CudaError::KernelLaunch(format!("sync: {e}")))?;
    pool.download_f32(&d_out)
}
