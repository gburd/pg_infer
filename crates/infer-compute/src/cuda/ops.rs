//! cuBLAS wrapper operations — sgemm and gemv.
//!
//! All operations use column-major layout as required by cuBLAS, with
//! appropriate transposition flags to handle ndarray's row-major storage.

use std::sync::Arc;

use cudarc::cublas::{CudaBlas, GemmConfig};
use cudarc::cublas::sys::cublasOperation_t;
use cudarc::driver::{CudaDevice, CudaSlice, CudaStream};
use ndarray::{Array2, ArrayView2};

use super::buffers::BufferPool;
use super::CudaError;

/// Single-precision general matrix multiply via cuBLAS.
///
/// Computes C = A * B (if `transpose_b` is false) or C = A * B^T (if true).
/// Handles the row-major to column-major translation required by cuBLAS.
///
/// cuBLAS expects column-major, ndarray is row-major. The standard row-major
/// trick: A_row[m,k] stored in memory is the same as A^T_col[k,m]. So to
/// compute C_row = A_row * B_row we compute C^T_col = B^T_col * A^T_col.
pub fn sgemm(
    device: &Arc<CudaDevice>,
    blas: &CudaBlas,
    _stream: &CudaStream,
    pool: &BufferPool,
    a: ArrayView2<f32>,
    b: ArrayView2<f32>,
    transpose_b: bool,
) -> Result<Array2<f32>, CudaError> {
    let (m, k_a) = (a.shape()[0], a.shape()[1]);
    let (n, k_check) = if transpose_b {
        // B is [n, k], we want A[m,k] * B^T => C[m, n]
        (b.shape()[0], b.shape()[1])
    } else {
        // B is [k, n], we want A[m,k] * B => C[m, n]
        (b.shape()[1], b.shape()[0])
    };

    assert_eq!(k_a, k_check, "matmul dimension mismatch: A cols {} != B rows {}", k_a, k_check);

    let k = k_a;

    // Ensure contiguous data for upload
    let a_data: Vec<f32> = if a.is_standard_layout() {
        a.as_slice().expect("contiguous a").to_vec()
    } else {
        a.iter().copied().collect()
    };

    let b_data: Vec<f32> = if b.is_standard_layout() {
        b.as_slice().expect("contiguous b").to_vec()
    } else {
        b.iter().copied().collect()
    };

    // Upload to device
    let d_a = pool.upload_f32(&a_data)?;
    let d_b = pool.upload_f32(&b_data)?;
    let mut d_c = pool.alloc_f32(m * n)?;

    // Row-major to col-major: C^T = B_col * A_col
    //
    // Case 1: C = A * B (transpose_b = false)
    //   A_row[m,k] in memory = A^T_col[k,m], lda=k
    //   B_row[k,n] in memory = B^T_col[n,k], ldb=n
    //   C_row[m,n] in memory = C^T_col[n,m], ldc=n
    //   Compute: C^T[n,m] = B^T[n,k] * A^T[k,m]
    //   cuBLAS: sgemm(N, N, n, m, k, 1, B, n, A, k, 0, C, n)
    //
    // Case 2: C = A * B^T (transpose_b = true)
    //   A_row[m,k] in memory = A^T_col[k,m], lda=k
    //   B_row[n,k] in memory = B^T_col[k,n], ldb=k
    //   C_row[m,n] in memory = C^T_col[n,m], ldc=n
    //   Compute: C^T[n,m] = T(B^T_col)[n,k] * A^T_col[k,m]
    //   cuBLAS: sgemm(T, N, n, m, k, 1, B, k, A, k, 0, C, n)

    let transa = if transpose_b {
        cublasOperation_t::CUBLAS_OP_T
    } else {
        cublasOperation_t::CUBLAS_OP_N
    };
    let transb = cublasOperation_t::CUBLAS_OP_N;
    let lda = if transpose_b { k as i32 } else { n as i32 };
    let ldb = k as i32;
    let ldc = n as i32;

    let cfg = GemmConfig {
        transa,
        transb,
        m: n as i32,
        n: m as i32,
        k: k as i32,
        alpha: 1.0f32,
        lda,
        ldb,
        beta: 0.0f32,
        ldc,
    };

    unsafe {
        blas.gemm(cfg, &d_b.slice, &d_a.slice, &mut d_c.slice)
            .map_err(|e| CudaError::KernelLaunch(format!("cuBLAS sgemm: {e}")))?;
    }

    // Download result
    let c_data = pool.download_f32(&d_c)?;

    let result = Array2::from_shape_vec((m, n), c_data)
        .map_err(|e| CudaError::Transfer(format!("reshape result: {e}")))?;

    Ok(result)
}

/// Single-precision general matrix-vector multiply via cuBLAS.
///
/// Computes y = W * x where W is [n, k] and x is [k].
/// Returns y as [n] f32 values.
///
/// Implemented as sgemm with the vector treated as a [k, 1] matrix.
/// cuBLAS internally dispatches to its gemv kernel for M=1 or N=1 cases,
/// so this is equivalent in performance to a direct gemv call.
pub fn gemv(
    device: &Arc<CudaDevice>,
    blas: &CudaBlas,
    stream: &CudaStream,
    pool: &BufferPool,
    w: ArrayView2<f32>,
    x: &[f32],
) -> Result<Vec<f32>, CudaError> {
    let (n, k) = (w.shape()[0], w.shape()[1]);
    assert_eq!(x.len(), k, "gemv dimension mismatch: x len {} != W cols {}", x.len(), k);

    // Treat x as a [k, 1] matrix and compute W[n,k] * x[k,1] = y[n,1]
    // Using the row-major sgemm path.
    let x_view = ArrayView2::from_shape((k, 1), x)
        .map_err(|e| CudaError::Transfer(format!("reshape x: {e}")))?;

    let result = sgemm(device, blas, stream, pool, w, x_view, false)?;

    // Result is [n, 1], flatten to Vec<f32>
    Ok(result.into_raw_vec())
}
