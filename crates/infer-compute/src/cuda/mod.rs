//! CUDA GPU compute backend — NVIDIA GPUs via cudarc.
//!
//! All operations go through the [`ComputeBackend`] trait. CUDA-specific
//! optimisations: cuBLAS sgemm/gemv for f32, custom PTX kernels for Q4_K/Q6_K
//! dequant+matvec, device buffer pooling for zero-alloc hot paths.
//!
//! ## Modules
//!
//! - `ops`:      cuBLAS wrapper functions (sgemm, gemv)
//! - `kernels`:  Custom Q4_K/Q6_K dequant+matvec kernel dispatch
//! - `buffers`:  GPU buffer pool and host<->device transfer utilities

mod ops;
mod kernels;
mod buffers;

use std::sync::Arc;

use cudarc::cublas::CudaBlas;
use cudarc::driver::{CudaDevice, CudaStream};
use ndarray::{Array2, ArrayView2};

use crate::backend::{ComputeBackend, MatMulOp};
use crate::cpu::CpuBackend;
use buffers::BufferPool;

/// CUDA GPU compute backend.
///
/// Uses cuBLAS for f32 matrix operations and custom PTX kernels for quantized
/// Q4_K/Q6_K matrix-vector products. Operations not yet GPU-accelerated fall
/// back to the CPU backend.
pub struct CudaBackend {
    device: Arc<CudaDevice>,
    blas: CudaBlas,
    stream: CudaStream,
    buffer_pool: BufferPool,
    cpu_fallback: CpuBackend,
}

/// Errors from CUDA backend initialization or operation.
#[derive(Debug)]
pub enum CudaError {
    /// Failed to initialize CUDA device.
    DeviceInit(String),
    /// cuBLAS initialization failed.
    BlasInit(String),
    /// Stream creation failed.
    StreamCreate(String),
    /// Kernel launch failed.
    KernelLaunch(String),
    /// Memory allocation failed.
    MemAlloc(String),
    /// Host-device transfer failed.
    Transfer(String),
}

impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DeviceInit(e) => write!(f, "CUDA device init: {e}"),
            Self::BlasInit(e) => write!(f, "cuBLAS init: {e}"),
            Self::StreamCreate(e) => write!(f, "CUDA stream: {e}"),
            Self::KernelLaunch(e) => write!(f, "kernel launch: {e}"),
            Self::MemAlloc(e) => write!(f, "GPU alloc: {e}"),
            Self::Transfer(e) => write!(f, "H2D/D2H transfer: {e}"),
        }
    }
}

impl std::error::Error for CudaError {}

impl CudaBackend {
    /// Create a new CUDA backend on the given device ordinal.
    ///
    /// Returns `Err` if the device cannot be initialized or cuBLAS fails.
    pub fn new(device_id: usize) -> Result<Self, CudaError> {
        let device = CudaDevice::new(device_id)
            .map_err(|e| CudaError::DeviceInit(e.to_string()))?;

        let blas = CudaBlas::new(device.clone())
            .map_err(|e| CudaError::BlasInit(e.to_string()))?;

        let stream = device
            .fork_default_stream()
            .map_err(|e| CudaError::StreamCreate(e.to_string()))?;

        let buffer_pool = BufferPool::new(device.clone());

        Ok(Self {
            device,
            blas,
            stream,
            buffer_pool,
            cpu_fallback: CpuBackend,
        })
    }

    /// Device name from CUDA properties.
    pub fn device_name(&self) -> String {
        format!("CUDA device {}", self.device.ordinal())
    }
}

impl ComputeBackend for CudaBackend {
    // ── f32 matrix operations (cuBLAS) ──

    fn matmul(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
        match ops::sgemm(&self.device, &self.blas, &self.stream, &self.buffer_pool, a, b, false) {
            Ok(result) => result,
            Err(_) => self.cpu_fallback.matmul(a, b),
        }
    }

    fn matmul_transb(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
        match ops::sgemm(&self.device, &self.blas, &self.stream, &self.buffer_pool, a, b, true) {
            Ok(result) => result,
            Err(_) => self.cpu_fallback.matmul_transb(a, b),
        }
    }

    fn f32_gemv(&self, w: ArrayView2<f32>, x: &[f32]) -> Option<Vec<f32>> {
        let (n, k) = (w.shape()[0], w.shape()[1]);
        if x.len() != k {
            return None;
        }
        match ops::gemv(&self.device, &self.blas, &self.stream, &self.buffer_pool, w, x) {
            Ok(result) => Some(result),
            Err(_) => self.cpu_fallback.f32_gemv(w, x),
        }
    }

    fn f32_gemv_force(&self, w: ArrayView2<f32>, x: &[f32]) -> Option<Vec<f32>> {
        self.f32_gemv(w, x)
    }

    fn matmul_batch(&self, ops: &[MatMulOp]) -> Vec<Array2<f32>> {
        ops.iter().map(|op| {
            if op.transpose_b {
                self.matmul_transb(op.a.view(), op.b.view())
            } else {
                self.matmul(op.a.view(), op.b.view())
            }
        }).collect()
    }

    // ── Q4 quantized operations (CPU fallback) ──

    fn q4_matvec(
        &self, q4_data: &[u8], q8_x: &[i8], q8_scales: &[f32],
        num_rows: usize, hidden: usize,
    ) -> Option<Vec<f32>> {
        self.cpu_fallback.q4_matvec(q4_data, q8_x, q8_scales, num_rows, hidden)
    }

    fn q4_vecmat(
        &self, activation: &[f32], q4_data: &[u8],
        intermediate: usize, hidden: usize,
    ) -> Option<Vec<f32>> {
        self.cpu_fallback.q4_vecmat(activation, q4_data, intermediate, hidden)
    }

    // ── Q4_K / Q6_K quantized operations (CUDA kernels) ──

    fn q4k_matvec(
        &self, q4k_data: &[u8], x: &[f32], num_rows: usize, hidden: usize,
    ) -> Option<Vec<f32>> {
        match kernels::q4k_matvec_cuda(
            &self.device, &self.stream, &self.buffer_pool,
            q4k_data, x, num_rows, hidden,
        ) {
            Ok(result) => Some(result),
            Err(_) => self.cpu_fallback.q4k_matvec(q4k_data, x, num_rows, hidden),
        }
    }

    fn q6k_matvec(
        &self, q6k_data: &[u8], x: &[f32], num_rows: usize, hidden: usize,
    ) -> Option<Vec<f32>> {
        match kernels::q6k_matvec_cuda(
            &self.device, &self.stream, &self.buffer_pool,
            q6k_data, x, num_rows, hidden,
        ) {
            Ok(result) => Some(result),
            Err(_) => self.cpu_fallback.q6k_matvec(q6k_data, x, num_rows, hidden),
        }
    }

    // ── Ternary (CPU fallback) ──

    fn ternary_matvec(
        &self, packed: &[u8], x: &[f32], num_rows: usize, hidden: usize,
    ) -> Option<Vec<f32>> {
        self.cpu_fallback.ternary_matvec(packed, x, num_rows, hidden)
    }

    // ── Capabilities ──

    fn has_q4(&self) -> bool { true }

    fn name(&self) -> &str {
        "cuda (cuBLAS + Q4_K/Q6_K kernels)"
    }

    fn device_info(&self) -> String {
        self.device_name()
    }
}
