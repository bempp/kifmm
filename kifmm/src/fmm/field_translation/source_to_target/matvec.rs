//! Implementations of 8x8 matrix vector product operation during Hadamard product in FFT based M2L operations.
use rlst::{RlstScalar, c32, c64};

use crate::fmm::types::Isa;

/// The 8x8 gemv operation, always inlined. Implemented via a fully unrolled inner loop, and partially unrolled outer loop.
///
/// # Arguments
/// * - `matrix` - 8x8 matrix.
/// * - `signal` - 8x1 vector.
/// * - `result` - Save buffer.
/// * - `scale` - Scalar scaling factor.
#[inline(always)]
pub fn gemv8x8<U>(matrix: &[U], vector: &[U], result: &mut [U], scale: U)
where
    U: RlstScalar,
{
    let s1 = vector[0];
    let s2 = vector[1];
    let s3 = vector[2];
    let s4 = vector[3];
    let s5 = vector[4];
    let s6 = vector[5];
    let s7 = vector[6];
    let s8 = vector[7];

    for i in 0..4 {
        let mut sum1 = U::zero();
        let mut sum2 = U::zero();
        let i1 = 2 * i;
        let i2 = 2 * i + 1;

        sum1 += matrix[i1 * 8] * s1;
        sum1 += matrix[i1 * 8 + 1] * s2;
        sum1 += matrix[i1 * 8 + 2] * s3;
        sum1 += matrix[i1 * 8 + 3] * s4;
        sum1 += matrix[i1 * 8 + 4] * s5;
        sum1 += matrix[i1 * 8 + 5] * s6;
        sum1 += matrix[i1 * 8 + 6] * s7;
        sum1 += matrix[i1 * 8 + 7] * s8;

        sum2 += matrix[i2 * 8] * s1;
        sum2 += matrix[i2 * 8 + 1] * s2;
        sum2 += matrix[i2 * 8 + 2] * s3;
        sum2 += matrix[i2 * 8 + 3] * s4;
        sum2 += matrix[i2 * 8 + 4] * s5;
        sum2 += matrix[i2 * 8 + 5] * s6;
        sum2 += matrix[i2 * 8 + 6] * s7;
        sum2 += matrix[i2 * 8 + 7] * s8;

        result[i1] += sum1 * scale;
        result[i2] += sum2 * scale;
    }
}

/// Implement vectorised 8x8 gemv,
pub trait Gemv8x8 {

    type Scalar: RlstScalar;

    /// 8x8 GEMV
    fn gemv8x8(
        simd: Isa, matrix: &[Self::Scalar], vector: &[Self::Scalar], result: &mut [Self::Scalar], scale: Self::Scalar
    );
}


impl Gemv8x8 for c32 {
    type Scalar = Self;

    #[inline(always)]
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    fn gemv8x8(
            isa: Isa, matrix: &[Self::Scalar], vector: &[Self::Scalar], result: &mut [Self::Scalar], scale: Self::Scalar
        ) {

    }

    #[inline(always)]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
    fn gemv8x8(
        isa: Isa, matrix: &[Self::Scalar], vector: &[Self::Scalar], result: &mut [Self::Scalar], scale: Self::Scalar
        ) {

    }

    #[inline(always)]
    #[cfg(not(any(
        all(target_arch = "aarch64", target_feature = "neon"),
        all(target_arch = "x86_64", target_feature = "avx")
    )))]
    fn gemv8x8(
            isa: Isa, matrix: &[Self::Scalar], vector: &[Self::Scalar], result: &mut [Self::Scalar], scale: Self::Scalar
        ) {

        gemv8x8(matrix, vector, result, scale)
    }
}

impl Gemv8x8 for c64 {
    type Scalar = Self;
    #[inline(always)]
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    fn gemv8x8(
            isa: Isa, matrix: &[Self::Scalar], vector: &[Self::Scalar], result: &mut [Self::Scalar], scale: Self::Scalar
        ) {

    }

    #[inline(always)]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
    fn gemv8x8(
        isa: Isa, matrix: &[Self::Scalar], vector: &[Self::Scalar], result: &mut [Self::Scalar], scale: Self::Scalar
        ) {

    }

    #[inline(always)]
    #[cfg(not(any(
        all(target_arch = "aarch64", target_feature = "neon"),
        all(target_arch = "x86_64", target_feature = "avx")
    )))]
    fn gemv8x8(
            isa: Isa, matrix: &[Self::Scalar], vector: &[Self::Scalar], result: &mut [Self::Scalar], scale: Self::Scalar
        ) {

        gemv8x8(matrix, vector, result, scale)
    }
}