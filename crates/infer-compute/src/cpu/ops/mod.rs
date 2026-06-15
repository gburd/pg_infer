//! CPU operation dispatch — one file per operation type.
//!
//! Mirrors the Metal ops/ structure for consistent API across backends.
//! Each module handles dispatch for one category of compute operation.

pub mod f32_matmul;
pub mod q4_matvec;
pub mod q4_vecmat;
pub mod q4_common;
pub mod q4k_matvec;
#[cfg(target_arch = "x86_64")]
pub mod q4k_matvec_avx2;
pub mod q6k_matvec;
#[cfg(target_arch = "x86_64")]
pub mod q6k_matvec_avx2;
pub mod q8_matvec;
pub mod ternary_matvec;
pub mod bitlinear_matvec;
pub mod vector;
pub mod attention;
pub mod geglu;
pub mod linalg;
pub mod moe;
