//! # infer-compute
//!
//! Hardware-accelerated compute backends for infer.
//!
//! Provides the [`ComputeBackend`] trait that abstracts all hardware-specific
//! matrix operations. Every infer crate (inference, vindex) uses this trait —
//! the caller never knows whether the operation runs on CPU or GPU.
//!
//! ## Backends
//!
//! | Backend | Feature | Operations |
//! |---------|---------|------------|
//! | CPU | (always) | BLAS f32, C kernel Q4 (ARM vdotq_s32), vector ops |
//! | Metal | `metal` | Tiled f32, simdgroup Q4, multi-layer pipeline |
//! | CUDA | (planned) | — |
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use infer_compute::{ComputeBackend, default_backend, cpu_backend, dot, norm, cosine};
//!
//! let backend = default_backend();
//! println!("Using: {}", backend.name());
//! ```
//!
//! ## Feature flags
//!
//! - `metal`: Metal GPU backend (macOS only). Adds optimised Q4 shaders,
//!   multi-layer pipeline, zero-copy mmap buffers.
//! - `cuda`: (planned) CUDA GPU backend.

extern crate blas_src;

pub mod backend;
pub mod cpu;
pub mod pipeline;

#[cfg(feature = "metal")]
pub mod metal;

#[cfg(feature = "cuda")]
pub mod cuda;

// ── Re-exports: pipeline types ──

pub use pipeline::{
    QuantFormat, QuantWeight,
    NormType, FfnType, Activation,
    FullPipelineLayer, MoeLayerWeights,
};

// ── Re-exports: backend ──

pub use backend::{ComputeBackend, MatMulOp, dot_proj_gpu, matmul_gpu};
pub use cpu::CpuBackend;
pub use cpu::ops::vector::{dot, norm, cosine};
pub use cpu::ops::linalg::{cholesky, cholesky_solve, cholesky_inverse, ridge_decomposition_solve};

#[cfg(feature = "metal")]
pub use metal::MetalBackend;

#[cfg(feature = "cuda")]
pub use cuda::CudaBackend;

/// Create the best available backend.
///
/// With `--features metal`: tries Metal GPU first, auto-calibrates the
/// FLOP threshold for hybrid CPU/GPU dispatch, falls back to CPU.
/// Without: returns CPU (Accelerate BLAS on macOS, OpenBLAS on Linux).
///
/// # Example
/// ```rust,no_run
/// let backend = infer_compute::default_backend();
/// println!("{} ({})", backend.name(), backend.device_info());
/// ```
pub fn default_backend() -> Box<dyn ComputeBackend> {
    #[cfg(feature = "metal")]
    {
        if let Some(m) = metal::MetalBackend::new() {
            m.calibrate();
            return Box::new(m);
        }
        eprintln!("[compute] Metal not available, falling back to CPU");
    }
    #[cfg(feature = "cuda")]
    {
        match cuda::CudaBackend::new(0) {
            Ok(c) => return Box::new(c),
            Err(e) => eprintln!("[compute] CUDA not available ({e}), falling back to CPU"),
        }
    }
    Box::new(cpu::CpuBackend)
}

/// Force CPU-only backend. No GPU, no calibration overhead.
///
/// Use when you want deterministic CPU execution or to benchmark
/// CPU vs GPU paths.
pub fn cpu_backend() -> Box<dyn ComputeBackend> {
    Box::new(cpu::CpuBackend)
}
