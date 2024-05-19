//! Implementations of 8x8 matrix vector product operation during Hadamard product in FFT based M2L operations.
use pulp::{f32x4, f64x2, Simd};
use rlst::{c32, c64, RlstScalar};

/// The 8x8 matvec operation, always inlined. Implemented via a fully unrolled inner loop, and partially unrolled outer loop.
///
/// # Arguments
/// * - `matrix` - 8x8 matrix.
/// * - `signal` - 8x1 vector.
/// * `result` - Save buffer.
/// * `scale` - Scalar scaling factor.
#[inline(always)]
pub fn matvec8x8_auto<U>(matrix: &[U; 64], vector: &[U; 8], result: &mut [U; 8], scale: U)
where
    U: RlstScalar,
{
    for i in 0..8 {
        let mut sum = U::zero();
        let row = &matrix[i * 8..(i + 1) * 8];
        for j in 0..8 {
            sum += row[j] * vector[j]
        }

        result[i] += sum * scale;
    }
}

#[derive(Debug)]
pub struct Matvec8x8<'a, T: RlstScalar, S> {
    pub simd: S,
    matrix: &'a [T; 64],
    vector: &'a [T; 8],
    result: &'a mut [T; 8],
    scale: T::Real,
}

macro_rules! matvec_trait {
    ($simd:ty) => {
        pub trait Matvec {
            type Scalar: RlstScalar;

            fn matvec8x8(
                _simd: $simd,
                matrix: &[Self::Scalar; 64],
                vector: &[Self::Scalar; 8],
                save_buffer: &mut [Self::Scalar; 8],
                scale: Self::Scalar,
            ) {
                matvec8x8_auto(matrix, vector, save_buffer, scale)
            }
        }
    };
}

#[cfg(target_arch = "aarch64")]
pub mod aarch64 {
    use super::*;
    use pulp::aarch64::NeonFcma;
    use std::{
        arch::aarch64::{float32x4_t, float64x2_t},
    };

    matvec_trait!(NeonFcma);

    impl<'a> pulp::NullaryFnOnce for Matvec8x8<'a, c32, NeonFcma> {
        type Output = ();

        #[inline(always)]
        fn call(self) -> Self::Output {
            let Self {
                simd,
                scale,
                matrix,
                vector,
                result,
            } = self;

            let mut a1 = f32x4(0., 0., 0., 0.);
            let mut a2 = f32x4(0., 0., 0., 0.);
            let mut a3 = f32x4(0., 0., 0., 0.);
            let mut a4 = f32x4(0., 0., 0., 0.);
            let [r1, r2, r3, r4]: [f32x4; 4] = pulp::cast(*result);
            let scale = simd.f32s_splat(scale);

            let (matrix, _) = pulp::as_arrays::<8, _>(matrix);
            let [v1, v2, v3, v4]: [f32x4; 4] = pulp::cast(*vector);

            let v01 = f32x4(v1.0, v1.1, v1.0, v1.1);
            let v02 = f32x4(v1.2, v1.3, v1.2, v1.3);
            let v03 = f32x4(v2.0, v2.1, v2.0, v2.1);
            let v04 = f32x4(v2.2, v2.3, v2.2, v2.3);
            let v05 = f32x4(v3.0, v3.1, v3.0, v3.1);
            let v06 = f32x4(v3.2, v3.3, v3.2, v3.3);
            let v07 = f32x4(v4.0, v4.1, v4.0, v4.1);
            let v08 = f32x4(v4.2, v4.3, v4.2, v4.3);

            // Unroll loop
            let [m1, m2, m3, m4]: [f32x4; 4] = pulp::cast(*&matrix[0]);
            a1 = simd.c32s_mul_add_e(m1, v01, a1);
            a2 = simd.c32s_mul_add_e(m2, v01, a2);
            a3 = simd.c32s_mul_add_e(m3, v01, a3);
            a4 = simd.c32s_mul_add_e(m4, v01, a4);

            let [m1, m2, m3, m4]: [f32x4; 4] = pulp::cast(*&matrix[1]);
            a1 = simd.c32s_mul_add_e(m1, v02, a1);
            a2 = simd.c32s_mul_add_e(m2, v02, a2);
            a3 = simd.c32s_mul_add_e(m3, v02, a3);
            a4 = simd.c32s_mul_add_e(m4, v02, a4);

            let [m1, m2, m3, m4]: [f32x4; 4] = pulp::cast(*&matrix[2]);
            a1 = simd.c32s_mul_add_e(m1, v03, a1);
            a2 = simd.c32s_mul_add_e(m2, v03, a2);
            a3 = simd.c32s_mul_add_e(m3, v03, a3);
            a4 = simd.c32s_mul_add_e(m4, v03, a4);

            let [m1, m2, m3, m4]: [f32x4; 4] = pulp::cast(*&matrix[3]);
            a1 = simd.c32s_mul_add_e(m1, v04, a1);
            a2 = simd.c32s_mul_add_e(m2, v04, a2);
            a3 = simd.c32s_mul_add_e(m3, v04, a3);
            a4 = simd.c32s_mul_add_e(m4, v04, a4);

            let [m1, m2, m3, m4]: [f32x4; 4] = pulp::cast(*&matrix[4]);
            a1 = simd.c32s_mul_add_e(m1, v05, a1);
            a2 = simd.c32s_mul_add_e(m2, v05, a2);
            a3 = simd.c32s_mul_add_e(m3, v05, a3);
            a4 = simd.c32s_mul_add_e(m4, v05, a4);

            let [m1, m2, m3, m4]: [f32x4; 4] = pulp::cast(*&matrix[5]);
            a1 = simd.c32s_mul_add_e(m1, v06, a1);
            a2 = simd.c32s_mul_add_e(m2, v06, a2);
            a3 = simd.c32s_mul_add_e(m3, v06, a3);
            a4 = simd.c32s_mul_add_e(m4, v06, a4);

            let [m1, m2, m3, m4]: [f32x4; 4] = pulp::cast(*&matrix[6]);
            a1 = simd.c32s_mul_add_e(m1, v07, a1);
            a2 = simd.c32s_mul_add_e(m2, v07, a2);
            a3 = simd.c32s_mul_add_e(m3, v07, a3);
            a4 = simd.c32s_mul_add_e(m4, v07, a4);

            let [m1, m2, m3, m4]: [f32x4; 4] = pulp::cast(*&matrix[7]);
            a1 = simd.c32s_mul_add_e(m1, v08, a1);
            a2 = simd.c32s_mul_add_e(m2, v08, a2);
            a3 = simd.c32s_mul_add_e(m3, v08, a3);
            a4 = simd.c32s_mul_add_e(m4, v08, a4);

            a1 = simd.mul_add_f32x4(a1, scale, r1);
            a2 = simd.mul_add_f32x4(a2, scale, r2);
            a3 = simd.mul_add_f32x4(a3, scale, r3);
            a4 = simd.mul_add_f32x4(a4, scale, r4);

            let a1: float32x4_t = unsafe { std::mem::transmute(a1) };
            let a2: float32x4_t = unsafe { std::mem::transmute(a2) };
            let a3: float32x4_t = unsafe { std::mem::transmute(a3) };
            let a4: float32x4_t = unsafe { std::mem::transmute(a4) };

            let ptr = result.as_ptr() as *mut f32;
            unsafe { simd.neon.vst1q_f32(ptr, a1) };
            unsafe { simd.neon.vst1q_f32(ptr.add(4), a2) };
            unsafe { simd.neon.vst1q_f32(ptr.add(8), a3) };
            unsafe { simd.neon.vst1q_f32(ptr.add(12), a4) };
        }
    }

    impl<'a> pulp::NullaryFnOnce for Matvec8x8<'a, c64, NeonFcma> {
        type Output = ();

        #[inline(always)]
        fn call(self) -> Self::Output {
            let Self {
                simd,
                scale,
                matrix,
                vector,
                result,
            } = self;
            let mut a1 = f64x2(0., 0.);
            let mut a2 = f64x2(0., 0.);
            let mut a3 = f64x2(0., 0.);
            let mut a4 = f64x2(0., 0.);
            let mut a5 = f64x2(0., 0.);
            let mut a6 = f64x2(0., 0.);
            let mut a7 = f64x2(0., 0.);
            let mut a8 = f64x2(0., 0.);

            let [r1, r2, r3, r4, r5, r6, r7, r8]: [f64x2; 8] = pulp::cast(*result);
            let scale = simd.f64s_splat(scale);

            let (matrix, _) = pulp::as_arrays::<8, _>(matrix);
            let [v1, v2, v3, v4, v5, v6, v7, v8]: [f64x2; 8] = pulp::cast(*vector);

            // Unroll loop
            let [m1, m2, m3, m4, m5, m6, m7, m8]: [f64x2; 8] = pulp::cast(*&matrix[0]);
            a1 = simd.c64s_mul_add_e(m1, v1, a1);
            a2 = simd.c64s_mul_add_e(m2, v1, a2);
            a3 = simd.c64s_mul_add_e(m3, v1, a3);
            a4 = simd.c64s_mul_add_e(m4, v1, a4);
            a5 = simd.c64s_mul_add_e(m5, v1, a5);
            a6 = simd.c64s_mul_add_e(m6, v1, a6);
            a7 = simd.c64s_mul_add_e(m7, v1, a7);
            a8 = simd.c64s_mul_add_e(m8, v1, a8);

            let [m1, m2, m3, m4, m5, m6, m7, m8]: [f64x2; 8] = pulp::cast(*&matrix[1]);
            a1 = simd.c64s_mul_add_e(m1, v2, a1);
            a2 = simd.c64s_mul_add_e(m2, v2, a2);
            a3 = simd.c64s_mul_add_e(m3, v2, a3);
            a4 = simd.c64s_mul_add_e(m4, v2, a4);
            a5 = simd.c64s_mul_add_e(m5, v2, a5);
            a6 = simd.c64s_mul_add_e(m6, v2, a6);
            a7 = simd.c64s_mul_add_e(m7, v2, a7);
            a8 = simd.c64s_mul_add_e(m8, v2, a8);

            let [m1, m2, m3, m4, m5, m6, m7, m8]: [f64x2; 8] = pulp::cast(*&matrix[2]);
            a1 = simd.c64s_mul_add_e(m1, v3, a1);
            a2 = simd.c64s_mul_add_e(m2, v3, a2);
            a3 = simd.c64s_mul_add_e(m3, v3, a3);
            a4 = simd.c64s_mul_add_e(m4, v3, a4);
            a5 = simd.c64s_mul_add_e(m5, v3, a5);
            a6 = simd.c64s_mul_add_e(m6, v3, a6);
            a7 = simd.c64s_mul_add_e(m7, v3, a7);
            a8 = simd.c64s_mul_add_e(m8, v3, a8);

            let [m1, m2, m3, m4, m5, m6, m7, m8]: [f64x2; 8] = pulp::cast(*&matrix[3]);
            a1 = simd.c64s_mul_add_e(m1, v4, a1);
            a2 = simd.c64s_mul_add_e(m2, v4, a2);
            a3 = simd.c64s_mul_add_e(m3, v4, a3);
            a4 = simd.c64s_mul_add_e(m4, v4, a4);
            a5 = simd.c64s_mul_add_e(m5, v4, a5);
            a6 = simd.c64s_mul_add_e(m6, v4, a6);
            a7 = simd.c64s_mul_add_e(m7, v4, a7);
            a8 = simd.c64s_mul_add_e(m8, v4, a8);

            let [m1, m2, m3, m4, m5, m6, m7, m8]: [f64x2; 8] = pulp::cast(*&matrix[4]);
            a1 = simd.c64s_mul_add_e(m1, v5, a1);
            a2 = simd.c64s_mul_add_e(m2, v5, a2);
            a3 = simd.c64s_mul_add_e(m3, v5, a3);
            a4 = simd.c64s_mul_add_e(m4, v5, a4);
            a5 = simd.c64s_mul_add_e(m5, v5, a5);
            a6 = simd.c64s_mul_add_e(m6, v5, a6);
            a7 = simd.c64s_mul_add_e(m7, v5, a7);
            a8 = simd.c64s_mul_add_e(m8, v5, a8);

            let [m1, m2, m3, m4, m5, m6, m7, m8]: [f64x2; 8] = pulp::cast(*&matrix[5]);
            a1 = simd.c64s_mul_add_e(m1, v6, a1);
            a2 = simd.c64s_mul_add_e(m2, v6, a2);
            a3 = simd.c64s_mul_add_e(m3, v6, a3);
            a4 = simd.c64s_mul_add_e(m4, v6, a4);
            a5 = simd.c64s_mul_add_e(m5, v6, a5);
            a6 = simd.c64s_mul_add_e(m6, v6, a6);
            a7 = simd.c64s_mul_add_e(m7, v6, a7);
            a8 = simd.c64s_mul_add_e(m8, v6, a8);

            let [m1, m2, m3, m4, m5, m6, m7, m8]: [f64x2; 8] = pulp::cast(*&matrix[6]);
            a1 = simd.c64s_mul_add_e(m1, v7, a1);
            a2 = simd.c64s_mul_add_e(m2, v7, a2);
            a3 = simd.c64s_mul_add_e(m3, v7, a3);
            a4 = simd.c64s_mul_add_e(m4, v7, a4);
            a5 = simd.c64s_mul_add_e(m5, v7, a5);
            a6 = simd.c64s_mul_add_e(m6, v7, a6);
            a7 = simd.c64s_mul_add_e(m7, v7, a7);
            a8 = simd.c64s_mul_add_e(m8, v7, a8);

            let [m1, m2, m3, m4, m5, m6, m7, m8]: [f64x2; 8] = pulp::cast(*&matrix[7]);
            a1 = simd.c64s_mul_add_e(m1, v8, a1);
            a2 = simd.c64s_mul_add_e(m2, v8, a2);
            a3 = simd.c64s_mul_add_e(m3, v8, a3);
            a4 = simd.c64s_mul_add_e(m4, v8, a4);
            a5 = simd.c64s_mul_add_e(m5, v8, a5);
            a6 = simd.c64s_mul_add_e(m6, v8, a6);
            a7 = simd.c64s_mul_add_e(m7, v8, a7);
            a8 = simd.c64s_mul_add_e(m8, v8, a8);

            a1 = simd.mul_add_f64x2(a1, scale, r1);
            a2 = simd.mul_add_f64x2(a2, scale, r2);
            a3 = simd.mul_add_f64x2(a3, scale, r3);
            a4 = simd.mul_add_f64x2(a4, scale, r4);
            a5 = simd.mul_add_f64x2(a5, scale, r5);
            a6 = simd.mul_add_f64x2(a6, scale, r6);
            a7 = simd.mul_add_f64x2(a7, scale, r7);
            a8 = simd.mul_add_f64x2(a8, scale, r8);

            let a1: float64x2_t = unsafe { std::mem::transmute(a1) };
            let a2: float64x2_t = unsafe { std::mem::transmute(a2) };
            let a3: float64x2_t = unsafe { std::mem::transmute(a3) };
            let a4: float64x2_t = unsafe { std::mem::transmute(a4) };
            let a5: float64x2_t = unsafe { std::mem::transmute(a5) };
            let a6: float64x2_t = unsafe { std::mem::transmute(a6) };
            let a7: float64x2_t = unsafe { std::mem::transmute(a7) };
            let a8: float64x2_t = unsafe { std::mem::transmute(a8) };

            let ptr = result.as_ptr() as *mut f64;
            unsafe { simd.neon.vst1q_f64(ptr, a1) };
            unsafe { simd.neon.vst1q_f64(ptr.add(2), a2) };
            unsafe { simd.neon.vst1q_f64(ptr.add(4), a3) };
            unsafe { simd.neon.vst1q_f64(ptr.add(6), a4) };
            unsafe { simd.neon.vst1q_f64(ptr.add(8), a5) };
            unsafe { simd.neon.vst1q_f64(ptr.add(10), a6) };
            unsafe { simd.neon.vst1q_f64(ptr.add(12), a7) };
            unsafe { simd.neon.vst1q_f64(ptr.add(14), a8) };
        }
    }

    impl Matvec for c32 {
        type Scalar = c32;

        #[inline(always)]
        fn matvec8x8(
            simd: NeonFcma,
            matrix: &[Self::Scalar; 64],
            vector: &[Self::Scalar; 8],
            save_buffer: &mut [Self::Scalar; 8],
            alpha: Self::Scalar,
        ) {
            simd.vectorize(Matvec8x8 {
                simd,
                scale: alpha.re(),
                matrix: matrix,
                vector: vector,
                result: save_buffer,
            });
        }
    }

    impl Matvec for c64 {
        type Scalar = c64;

        #[inline(always)]
        fn matvec8x8(
            simd: NeonFcma,
            matrix: &[Self::Scalar; 64],
            vector: &[Self::Scalar; 8],
            save_buffer: &mut [Self::Scalar; 8],
            alpha: Self::Scalar,
        ) {
            simd.vectorize(Matvec8x8 {
                simd,
                scale: alpha.re(),
                matrix: matrix,
                vector: vector,
                result: save_buffer,
            })
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub use aarch64::Matvec;

#[cfg(target_arch = "x86_64")]
pub mod x86 {
    use super::*;
    use pulp::{f32x8, f64x2, f64x4};
    use pulp::x86::V3;

    matvec_trait!(V3);

    macro_rules! generate_deinterleave_fn {
        ($fn_name:ident, $simd_type:ty, $array_type:ty, $splat_method:ident, $scalar_type:ty) => {
            fn $fn_name(simd: $simd_type, value: [$array_type; 2]) -> [$array_type; 2] {
                let mut out = [simd.$splat_method(0.); 2];

                {
                    let n =
                        std::mem::size_of::<$array_type>() / std::mem::size_of::<$scalar_type>();
                    let out: &mut [$scalar_type] =
                        bytemuck::cast_slice_mut(std::slice::from_mut(&mut out));
                    let x: &[$scalar_type] = bytemuck::cast_slice(std::slice::from_ref(&value));

                    for i in 0..n {
                        out[i] = x[2 * i];
                        out[n + i] = x[2 * i + 1];
                    }
                }

                out
            }
        };
    }

    generate_deinterleave_fn!(deinterleave_avx_f32, V3, f32x8, splat_f32x8, f32);
    generate_deinterleave_fn!(deinterleave_avx_f64, V3, f64x4, splat_f64x4, f64);

    impl<'a> pulp::NullaryFnOnce for Matvec8x8<'a, c32, V3> {
        type Output = ();

        #[inline(always)]
        fn call(self) -> Self::Output {

            let Self {
                simd,
                scale,
                matrix,
                vector,
                result,
            } = self;

            let mut a1 = simd.splat_f32x8(0.);
            let mut a2 = simd.splat_f32x8(0.);

            let (matrix, _) = pulp::as_arrays::<8, _>(matrix);
            let (vectors, _) = pulp::as_arrays::<8, _>(vector);
            let [v_re, v_im] = deinterleave_avx_f32(simd, pulp::cast(vectors[0]));

            {
                let [m1, m2]: [f32x8; 2] = pulp::cast(*&matrix[0]); // 9 registers
                let v1_re = simd.splat_f32x8(v_re.0);
                let v1_im = simd.splat_f32x8(v_im.0);

                let prod1 = simd.mul_f32x8(m1, v1_re);
                let m1 = simd
                    .avx
                    ._mm256_shuffle_ps::<0b10110001>(pulp::cast(m1), pulp::cast(m1));
                let prod2 = simd.mul_f32x8(pulp::cast(m1), v1_im);
                let r1 = simd
                    .avx
                    ._mm256_addsub_ps(pulp::cast(prod1), pulp::cast(prod2));
                a1 = simd.add_f32x8(a1, pulp::cast(r1));

                let prod1 = simd.mul_f32x8(m2, v1_re);
                let m2 = simd
                    .avx
                    ._mm256_shuffle_ps::<0b10110001>(pulp::cast(m2), pulp::cast(m2));
                let prod2 = simd.mul_f32x8(pulp::cast(m2), v1_im);
                let r2 = simd
                    .avx
                    ._mm256_addsub_ps(pulp::cast(prod1), pulp::cast(prod2));
                a2 = simd.add_f32x8(a2, pulp::cast(r2));
            }

            {
                let [m1, m2]: [f32x8; 2] = pulp::cast(*&matrix[1]); // 9 registers
                let v1_re = simd.splat_f32x8(v_re.1);
                let v1_im = simd.splat_f32x8(v_im.1);

                let prod1 = simd.mul_f32x8(m1, v1_re);
                let m1 = simd
                    .avx
                    ._mm256_shuffle_ps::<0b10110001>(pulp::cast(m1), pulp::cast(m1));
                let prod2 = simd.mul_f32x8(pulp::cast(m1), v1_im);
                let r1 = simd
                    .avx
                    ._mm256_addsub_ps(pulp::cast(prod1), pulp::cast(prod2));
                a1 = simd.add_f32x8(a1, pulp::cast(r1));

                let prod1 = simd.mul_f32x8(m2, v1_re);
                let m2 = simd
                    .avx
                    ._mm256_shuffle_ps::<0b10110001>(pulp::cast(m2), pulp::cast(m2));
                let prod2 = simd.mul_f32x8(pulp::cast(m2), v1_im);
                let r2 = simd
                    .avx
                    ._mm256_addsub_ps(pulp::cast(prod1), pulp::cast(prod2));
                a2 = simd.add_f32x8(a2, pulp::cast(r2));
            }

            {
                let [m1, m2]: [f32x8; 2] = pulp::cast(*&matrix[2]); // 9 registers
                let v1_re = simd.splat_f32x8(v_re.2);
                let v1_im = simd.splat_f32x8(v_im.2);

                let prod1 = simd.mul_f32x8(m1, v1_re);
                let m1 = simd
                    .avx
                    ._mm256_shuffle_ps::<0b10110001>(pulp::cast(m1), pulp::cast(m1));
                let prod2 = simd.mul_f32x8(pulp::cast(m1), v1_im);
                let r1 = simd
                    .avx
                    ._mm256_addsub_ps(pulp::cast(prod1), pulp::cast(prod2));
                a1 = simd.add_f32x8(a1, pulp::cast(r1));

                let prod1 = simd.mul_f32x8(m2, v1_re);
                let m2 = simd
                    .avx
                    ._mm256_shuffle_ps::<0b10110001>(pulp::cast(m2), pulp::cast(m2));
                let prod2 = simd.mul_f32x8(pulp::cast(m2), v1_im);
                let r2 = simd
                    .avx
                    ._mm256_addsub_ps(pulp::cast(prod1), pulp::cast(prod2));
                a2 = simd.add_f32x8(a2, pulp::cast(r2));
            }

            {
                let [m1, m2]: [f32x8; 2] = pulp::cast(*&matrix[3]); // 9 registers
                let v1_re = simd.splat_f32x8(v_re.3);
                let v1_im = simd.splat_f32x8(v_im.3);

                let prod1 = simd.mul_f32x8(m1, v1_re);
                let m1 = simd
                    .avx
                    ._mm256_shuffle_ps::<0b10110001>(pulp::cast(m1), pulp::cast(m1));
                let prod2 = simd.mul_f32x8(pulp::cast(m1), v1_im);
                let r1 = simd
                    .avx
                    ._mm256_addsub_ps(pulp::cast(prod1), pulp::cast(prod2));
                a1 = simd.add_f32x8(a1, pulp::cast(r1));

                let prod1 = simd.mul_f32x8(m2, v1_re);
                let m2 = simd
                    .avx
                    ._mm256_shuffle_ps::<0b10110001>(pulp::cast(m2), pulp::cast(m2));
                let prod2 = simd.mul_f32x8(pulp::cast(m2), v1_im);
                let r2 = simd
                    .avx
                    ._mm256_addsub_ps(pulp::cast(prod1), pulp::cast(prod2));
                a2 = simd.add_f32x8(a2, pulp::cast(r2));
            }

            {
                let [m1, m2]: [f32x8; 2] = pulp::cast(*&matrix[4]); // 9 registers
                let v1_re = simd.splat_f32x8(v_re.4);
                let v1_im = simd.splat_f32x8(v_im.4);

                let prod1 = simd.mul_f32x8(m1, v1_re);
                let m1 = simd
                    .avx
                    ._mm256_shuffle_ps::<0b10110001>(pulp::cast(m1), pulp::cast(m1));
                let prod2 = simd.mul_f32x8(pulp::cast(m1), v1_im);
                let r1 = simd
                    .avx
                    ._mm256_addsub_ps(pulp::cast(prod1), pulp::cast(prod2));
                a1 = simd.add_f32x8(a1, pulp::cast(r1));

                let prod1 = simd.mul_f32x8(m2, v1_re);
                let m2 = simd
                    .avx
                    ._mm256_shuffle_ps::<0b10110001>(pulp::cast(m2), pulp::cast(m2));
                let prod2 = simd.mul_f32x8(pulp::cast(m2), v1_im);
                let r2 = simd
                    .avx
                    ._mm256_addsub_ps(pulp::cast(prod1), pulp::cast(prod2));
                a2 = simd.add_f32x8(a2, pulp::cast(r2));
            }

            {
                let [m1, m2]: [f32x8; 2] = pulp::cast(*&matrix[5]); // 9 registers
                let v1_re = simd.splat_f32x8(v_re.5);
                let v1_im = simd.splat_f32x8(v_im.5);

                let prod1 = simd.mul_f32x8(m1, v1_re);
                let m1 = simd
                    .avx
                    ._mm256_shuffle_ps::<0b10110001>(pulp::cast(m1), pulp::cast(m1));
                let prod2 = simd.mul_f32x8(pulp::cast(m1), v1_im);
                let r1 = simd
                    .avx
                    ._mm256_addsub_ps(pulp::cast(prod1), pulp::cast(prod2));
                a1 = simd.add_f32x8(a1, pulp::cast(r1));

                let prod1 = simd.mul_f32x8(m2, v1_re);
                let m2 = simd
                    .avx
                    ._mm256_shuffle_ps::<0b10110001>(pulp::cast(m2), pulp::cast(m2));
                let prod2 = simd.mul_f32x8(pulp::cast(m2), v1_im);
                let r2 = simd
                    .avx
                    ._mm256_addsub_ps(pulp::cast(prod1), pulp::cast(prod2));
                a2 = simd.add_f32x8(a2, pulp::cast(r2));
            }

            {
                let [m1, m2]: [f32x8; 2] = pulp::cast(*&matrix[6]); // 9 registers
                let v1_re = simd.splat_f32x8(v_re.6);
                let v1_im = simd.splat_f32x8(v_im.6);

                let prod1 = simd.mul_f32x8(m1, v1_re);
                let m1 = simd
                    .avx
                    ._mm256_shuffle_ps::<0b10110001>(pulp::cast(m1), pulp::cast(m1));
                let prod2 = simd.mul_f32x8(pulp::cast(m1), v1_im);
                let r1 = simd
                    .avx
                    ._mm256_addsub_ps(pulp::cast(prod1), pulp::cast(prod2));
                a1 = simd.add_f32x8(a1, pulp::cast(r1));

                let prod1 = simd.mul_f32x8(m2, v1_re);
                let m2 = simd
                    .avx
                    ._mm256_shuffle_ps::<0b10110001>(pulp::cast(m2), pulp::cast(m2));
                let prod2 = simd.mul_f32x8(pulp::cast(m2), v1_im);
                let r2 = simd
                    .avx
                    ._mm256_addsub_ps(pulp::cast(prod1), pulp::cast(prod2));
                a2 = simd.add_f32x8(a2, pulp::cast(r2));
            }

            {
                let [m1, m2]: [f32x8; 2] = pulp::cast(*&matrix[7]); // 9 registers
                let v1_re = simd.splat_f32x8(v_re.7);
                let v1_im = simd.splat_f32x8(v_im.7);

                let prod1 = simd.mul_f32x8(m1, v1_re);
                let m1 = simd
                    .avx
                    ._mm256_shuffle_ps::<0b10110001>(pulp::cast(m1), pulp::cast(m1));
                let prod2 = simd.mul_f32x8(pulp::cast(m1), v1_im);
                let r1 = simd
                    .avx
                    ._mm256_addsub_ps(pulp::cast(prod1), pulp::cast(prod2));
                a1 = simd.add_f32x8(a1, pulp::cast(r1));

                let prod1 = simd.mul_f32x8(m2, v1_re);
                let m2 = simd
                    .avx
                    ._mm256_shuffle_ps::<0b10110001>(pulp::cast(m2), pulp::cast(m2));
                let prod2 = simd.mul_f32x8(pulp::cast(m2), v1_im);
                let r2 = simd
                    .avx
                    ._mm256_addsub_ps(pulp::cast(prod1), pulp::cast(prod2));
                a2 = simd.add_f32x8(a2, pulp::cast(r2));
            }

            let scale = simd.splat_f32x8(scale);
            a1 = simd.mul_f32x8(scale, a1);
            a2 = simd.mul_f32x8(scale, a2);

            // Store results
            {
                let ptr = result.as_mut_ptr() as *mut f32;
                unsafe { simd.avx._mm256_storeu_ps(ptr, std::mem::transmute(a1)) }
                unsafe {
                    simd.avx
                        ._mm256_storeu_ps(ptr.add(8), std::mem::transmute(a2))
                }
            }
        }
    }

    impl<'a> pulp::NullaryFnOnce for Matvec8x8<'a, c64, V3> {
        type Output = ();

        #[inline(always)]
        fn call(self) -> Self::Output {
            let Self {
                simd,
                scale,
                matrix,
                vector,
                result,
            } = self;
            let mut a1 = simd.splat_f64x4(0.);
            let mut a2 = simd.splat_f64x4(0.);
            let mut a3 = simd.splat_f64x4(0.);
            let mut a4 = simd.splat_f64x4(0.);
            let scale = simd.splat_f64x4(scale);

            let (matrix, _) = pulp::as_arrays::<8, _>(matrix);
            let (vectors, _) = pulp::as_arrays::<4, _>(vector);

            let [v1_re, v1_im] = deinterleave_avx_f64(simd, pulp::cast(vectors[0]));
            let [v2_re, v2_im] = deinterleave_avx_f64(simd, pulp::cast(vectors[1]));

            {
                let [m1, m2, m3, m4]: [f64x4; 4] = pulp::cast(*&matrix[0]);
                let v1_re = simd.splat_f64x4(v1_re.0);
                let v1_im = simd.splat_f64x4(v1_im.0);

                let prod1 = simd.mul_f64x4(m1, v1_re);
                let m1 = simd
                    .avx
                    ._mm256_shuffle_pd::<0b0101>(pulp::cast(m1), pulp::cast(m1));
                let prod2 = simd.mul_f64x4(pulp::cast(m1), v1_im);
                let r1 = simd
                    .avx
                    ._mm256_addsub_pd(pulp::cast(prod1), pulp::cast(prod2));

                a1 = simd.add_f64x4(a1, pulp::cast(r1));

                let prod1 = simd.mul_f64x4(m2, v1_re);
                let m2 = simd
                    .avx
                    ._mm256_shuffle_pd::<0b0101>(pulp::cast(m2), pulp::cast(m2));
                let prod2 = simd.mul_f64x4(pulp::cast(m2), v1_im);
                let r2 = simd
                    .avx
                    ._mm256_addsub_pd(pulp::cast(prod1), pulp::cast(prod2));
                a2 = simd.add_f64x4(a2, pulp::cast(r2));

                let prod1 = simd.mul_f64x4(m3, v1_re);
                let m3 = simd
                    .avx
                    ._mm256_shuffle_pd::<0b0101>(pulp::cast(m3), pulp::cast(m3));
                let prod2 = simd.mul_f64x4(pulp::cast(m3), v1_im);
                let r3 = simd
                    .avx
                    ._mm256_addsub_pd(pulp::cast(prod1), pulp::cast(prod2));
                a3 = simd.add_f64x4(a3, pulp::cast(r3));

                let prod1 = simd.mul_f64x4(m4, v1_re);
                let m4 = simd
                    .avx
                    ._mm256_shuffle_pd::<0b0101>(pulp::cast(m4), pulp::cast(m4));
                let prod2 = simd.mul_f64x4(pulp::cast(m4), v1_im);
                let r4 = simd
                    .avx
                    ._mm256_addsub_pd(pulp::cast(prod1), pulp::cast(prod2));
                a4 = simd.add_f64x4(a4, pulp::cast(r4));
            }

            {
                let [m1, m2, m3, m4]: [f64x4; 4] = pulp::cast(*&matrix[1]);
                let v1_re = simd.splat_f64x4(v1_re.1);
                let v1_im = simd.splat_f64x4(v1_im.1);

                let prod1 = simd.mul_f64x4(m1, v1_re);
                let m1 = simd
                    .avx
                    ._mm256_shuffle_pd::<0b0101>(pulp::cast(m1), pulp::cast(m1));
                let prod2 = simd.mul_f64x4(pulp::cast(m1), v1_im);
                let r1 = simd
                    .avx
                    ._mm256_addsub_pd(pulp::cast(prod1), pulp::cast(prod2));
                a1 = simd.add_f64x4(a1, pulp::cast(r1));

                let prod1 = simd.mul_f64x4(m2, v1_re);
                let m2 = simd
                    .avx
                    ._mm256_shuffle_pd::<0b0101>(pulp::cast(m2), pulp::cast(m2));
                let prod2 = simd.mul_f64x4(pulp::cast(m2), v1_im);
                let r2 = simd
                    .avx
                    ._mm256_addsub_pd(pulp::cast(prod1), pulp::cast(prod2));
                a2 = simd.add_f64x4(a2, pulp::cast(r2));

                let prod1 = simd.mul_f64x4(m3, v1_re);
                let m3 = simd
                    .avx
                    ._mm256_shuffle_pd::<0b0101>(pulp::cast(m3), pulp::cast(m3));
                let prod2 = simd.mul_f64x4(pulp::cast(m3), v1_im);
                let r3 = simd
                    .avx
                    ._mm256_addsub_pd(pulp::cast(prod1), pulp::cast(prod2));
                a3 = simd.add_f64x4(a3, pulp::cast(r3));

                let prod1 = simd.mul_f64x4(m4, v1_re);
                let m4 = simd
                    .avx
                    ._mm256_shuffle_pd::<0b0101>(pulp::cast(m4), pulp::cast(m4));
                let prod2 = simd.mul_f64x4(pulp::cast(m4), v1_im);
                let r4 = simd
                    .avx
                    ._mm256_addsub_pd(pulp::cast(prod1), pulp::cast(prod2));
                a4 = simd.add_f64x4(a4, pulp::cast(r4));
            }

            {
                let [m1, m2, m3, m4]: [f64x4; 4] = pulp::cast(*&matrix[2]);
                let v1_re = simd.splat_f64x4(v1_re.2);
                let v1_im = simd.splat_f64x4(v1_im.2);

                let prod1 = simd.mul_f64x4(m1, v1_re);
                let m1 = simd
                    .avx
                    ._mm256_shuffle_pd::<0b0101>(pulp::cast(m1), pulp::cast(m1));
                let prod2 = simd.mul_f64x4(pulp::cast(m1), v1_im);
                let r1 = simd
                    .avx
                    ._mm256_addsub_pd(pulp::cast(prod1), pulp::cast(prod2));
                a1 = simd.add_f64x4(a1, pulp::cast(r1));

                let prod1 = simd.mul_f64x4(m2, v1_re);
                let m2 = simd
                    .avx
                    ._mm256_shuffle_pd::<0b0101>(pulp::cast(m2), pulp::cast(m2));
                let prod2 = simd.mul_f64x4(pulp::cast(m2), v1_im);
                let r2 = simd
                    .avx
                    ._mm256_addsub_pd(pulp::cast(prod1), pulp::cast(prod2));
                a2 = simd.add_f64x4(a2, pulp::cast(r2));

                let prod1 = simd.mul_f64x4(m3, v1_re);
                let m3 = simd
                    .avx
                    ._mm256_shuffle_pd::<0b0101>(pulp::cast(m3), pulp::cast(m3));
                let prod2 = simd.mul_f64x4(pulp::cast(m3), v1_im);
                let r3 = simd
                    .avx
                    ._mm256_addsub_pd(pulp::cast(prod1), pulp::cast(prod2));
                a3 = simd.add_f64x4(a3, pulp::cast(r3));

                let prod1 = simd.mul_f64x4(m4, v1_re);
                let m4 = simd
                    .avx
                    ._mm256_shuffle_pd::<0b0101>(pulp::cast(m4), pulp::cast(m4));
                let prod2 = simd.mul_f64x4(pulp::cast(m4), v1_im);
                let r4 = simd
                    .avx
                    ._mm256_addsub_pd(pulp::cast(prod1), pulp::cast(prod2));
                a4 = simd.add_f64x4(a4, pulp::cast(r4));
            }

            {
                let [m1, m2, m3, m4]: [f64x4; 4] = pulp::cast(*&matrix[3]);
                let v1_re = simd.splat_f64x4(v1_re.3);
                let v1_im = simd.splat_f64x4(v1_im.3);

                let prod1 = simd.mul_f64x4(m1, v1_re);
                let m1 = simd
                    .avx
                    ._mm256_shuffle_pd::<0b0101>(pulp::cast(m1), pulp::cast(m1));
                let prod2 = simd.mul_f64x4(pulp::cast(m1), v1_im);
                let r1 = simd
                    .avx
                    ._mm256_addsub_pd(pulp::cast(prod1), pulp::cast(prod2));
                a1 = simd.add_f64x4(a1, pulp::cast(r1));

                let prod1 = simd.mul_f64x4(m2, v1_re);
                let m2 = simd
                    .avx
                    ._mm256_shuffle_pd::<0b0101>(pulp::cast(m2), pulp::cast(m2));
                let prod2 = simd.mul_f64x4(pulp::cast(m2), v1_im);
                let r2 = simd
                    .avx
                    ._mm256_addsub_pd(pulp::cast(prod1), pulp::cast(prod2));
                a2 = simd.add_f64x4(a2, pulp::cast(r2));

                let prod1 = simd.mul_f64x4(m3, v1_re);
                let m3 = simd
                    .avx
                    ._mm256_shuffle_pd::<0b0101>(pulp::cast(m3), pulp::cast(m3));
                let prod2 = simd.mul_f64x4(pulp::cast(m3), v1_im);
                let r3 = simd
                    .avx
                    ._mm256_addsub_pd(pulp::cast(prod1), pulp::cast(prod2));
                a3 = simd.add_f64x4(a3, pulp::cast(r3));

                let prod1 = simd.mul_f64x4(m4, v1_re);
                let m4 = simd
                    .avx
                    ._mm256_shuffle_pd::<0b0101>(pulp::cast(m4), pulp::cast(m4));
                let prod2 = simd.mul_f64x4(pulp::cast(m4), v1_im);
                let r4 = simd
                    .avx
                    ._mm256_addsub_pd(pulp::cast(prod1), pulp::cast(prod2));
                a4 = simd.add_f64x4(a4, pulp::cast(r4));
            }

            {
                let [m1, m2, m3, m4]: [f64x4; 4] = pulp::cast(*&matrix[4]);
                let v1_re = simd.splat_f64x4(v2_re.0);
                let v1_im = simd.splat_f64x4(v2_im.0);

                let prod1 = simd.mul_f64x4(m1, v1_re);
                let m1 = simd
                    .avx
                    ._mm256_shuffle_pd::<0b0101>(pulp::cast(m1), pulp::cast(m1));
                let prod2 = simd.mul_f64x4(pulp::cast(m1), v1_im);
                let r1 = simd
                    .avx
                    ._mm256_addsub_pd(pulp::cast(prod1), pulp::cast(prod2));
                a1 = simd.add_f64x4(a1, pulp::cast(r1));

                let prod1 = simd.mul_f64x4(m2, v1_re);
                let m2 = simd
                    .avx
                    ._mm256_shuffle_pd::<0b0101>(pulp::cast(m2), pulp::cast(m2));
                let prod2 = simd.mul_f64x4(pulp::cast(m2), v1_im);
                let r2 = simd
                    .avx
                    ._mm256_addsub_pd(pulp::cast(prod1), pulp::cast(prod2));
                a2 = simd.add_f64x4(a2, pulp::cast(r2));

                let prod1 = simd.mul_f64x4(m3, v1_re);
                let m3 = simd
                    .avx
                    ._mm256_shuffle_pd::<0b0101>(pulp::cast(m3), pulp::cast(m3));
                let prod2 = simd.mul_f64x4(pulp::cast(m3), v1_im);
                let r3 = simd
                    .avx
                    ._mm256_addsub_pd(pulp::cast(prod1), pulp::cast(prod2));
                a3 = simd.add_f64x4(a3, pulp::cast(r3));

                let prod1 = simd.mul_f64x4(m4, v1_re);
                let m4 = simd
                    .avx
                    ._mm256_shuffle_pd::<0b0101>(pulp::cast(m4), pulp::cast(m4));
                let prod2 = simd.mul_f64x4(pulp::cast(m4), v1_im);
                let r4 = simd
                    .avx
                    ._mm256_addsub_pd(pulp::cast(prod1), pulp::cast(prod2));
                a4 = simd.add_f64x4(a4, pulp::cast(r4));
            }

            {
                let [m1, m2, m3, m4]: [f64x4; 4] = pulp::cast(*&matrix[5]);
                let v1_re = simd.splat_f64x4(v2_re.1);
                let v1_im = simd.splat_f64x4(v2_im.1);

                let prod1 = simd.mul_f64x4(m1, v1_re);
                let m1 = simd
                    .avx
                    ._mm256_shuffle_pd::<0b0101>(pulp::cast(m1), pulp::cast(m1));
                let prod2 = simd.mul_f64x4(pulp::cast(m1), v1_im);
                let r1 = simd
                    .avx
                    ._mm256_addsub_pd(pulp::cast(prod1), pulp::cast(prod2));
                a1 = simd.add_f64x4(a1, pulp::cast(r1));

                let prod1 = simd.mul_f64x4(m2, v1_re);
                let m2 = simd
                    .avx
                    ._mm256_shuffle_pd::<0b0101>(pulp::cast(m2), pulp::cast(m2));
                let prod2 = simd.mul_f64x4(pulp::cast(m2), v1_im);
                let r2 = simd
                    .avx
                    ._mm256_addsub_pd(pulp::cast(prod1), pulp::cast(prod2));
                a2 = simd.add_f64x4(a2, pulp::cast(r2));

                let prod1 = simd.mul_f64x4(m3, v1_re);
                let m3 = simd
                    .avx
                    ._mm256_shuffle_pd::<0b0101>(pulp::cast(m3), pulp::cast(m3));
                let prod2 = simd.mul_f64x4(pulp::cast(m3), v1_im);
                let r3 = simd
                    .avx
                    ._mm256_addsub_pd(pulp::cast(prod1), pulp::cast(prod2));
                a3 = simd.add_f64x4(a3, pulp::cast(r3));

                let prod1 = simd.mul_f64x4(m4, v1_re);
                let m4 = simd
                    .avx
                    ._mm256_shuffle_pd::<0b0101>(pulp::cast(m4), pulp::cast(m4));
                let prod2 = simd.mul_f64x4(pulp::cast(m4), v1_im);
                let r4 = simd
                    .avx
                    ._mm256_addsub_pd(pulp::cast(prod1), pulp::cast(prod2));
                a4 = simd.add_f64x4(a4, pulp::cast(r4));
            }

            {
                let [m1, m2, m3, m4]: [f64x4; 4] = pulp::cast(*&matrix[6]);
                let v1_re = simd.splat_f64x4(v2_re.2);
                let v1_im = simd.splat_f64x4(v2_im.2);

                let prod1 = simd.mul_f64x4(m1, v1_re);
                let m1 = simd
                    .avx
                    ._mm256_shuffle_pd::<0b0101>(pulp::cast(m1), pulp::cast(m1));
                let prod2 = simd.mul_f64x4(pulp::cast(m1), v1_im);
                let r1 = simd
                    .avx
                    ._mm256_addsub_pd(pulp::cast(prod1), pulp::cast(prod2));
                a1 = simd.add_f64x4(a1, pulp::cast(r1));

                let prod1 = simd.mul_f64x4(m2, v1_re);
                let m2 = simd
                    .avx
                    ._mm256_shuffle_pd::<0b0101>(pulp::cast(m2), pulp::cast(m2));
                let prod2 = simd.mul_f64x4(pulp::cast(m2), v1_im);
                let r2 = simd
                    .avx
                    ._mm256_addsub_pd(pulp::cast(prod1), pulp::cast(prod2));
                a2 = simd.add_f64x4(a2, pulp::cast(r2));

                let prod1 = simd.mul_f64x4(m3, v1_re);
                let m3 = simd
                    .avx
                    ._mm256_shuffle_pd::<0b0101>(pulp::cast(m3), pulp::cast(m3));
                let prod2 = simd.mul_f64x4(pulp::cast(m3), v1_im);
                let r3 = simd
                    .avx
                    ._mm256_addsub_pd(pulp::cast(prod1), pulp::cast(prod2));
                a3 = simd.add_f64x4(a3, pulp::cast(r3));

                let prod1 = simd.mul_f64x4(m4, v1_re);
                let m4 = simd
                    .avx
                    ._mm256_shuffle_pd::<0b0101>(pulp::cast(m4), pulp::cast(m4));
                let prod2 = simd.mul_f64x4(pulp::cast(m4), v1_im);
                let r4 = simd
                    .avx
                    ._mm256_addsub_pd(pulp::cast(prod1), pulp::cast(prod2));
                a4 = simd.add_f64x4(a4, pulp::cast(r4));
            }

            {
                let [m1, m2, m3, m4]: [f64x4; 4] = pulp::cast(*&matrix[7]);
                let v1_re = simd.splat_f64x4(v2_re.3);
                let v1_im = simd.splat_f64x4(v2_im.3);

                let prod1 = simd.mul_f64x4(m1, v1_re);
                let m1 = simd
                    .avx
                    ._mm256_shuffle_pd::<0b0101>(pulp::cast(m1), pulp::cast(m1));
                let prod2 = simd.mul_f64x4(pulp::cast(m1), v1_im);
                let r1 = simd
                    .avx
                    ._mm256_addsub_pd(pulp::cast(prod1), pulp::cast(prod2));
                a1 = simd.add_f64x4(a1, pulp::cast(r1));

                let prod1 = simd.mul_f64x4(m2, v1_re);
                let m2 = simd
                    .avx
                    ._mm256_shuffle_pd::<0b0101>(pulp::cast(m2), pulp::cast(m2));
                let prod2 = simd.mul_f64x4(pulp::cast(m2), v1_im);
                let r2 = simd
                    .avx
                    ._mm256_addsub_pd(pulp::cast(prod1), pulp::cast(prod2));
                a2 = simd.add_f64x4(a2, pulp::cast(r2));

                let prod1 = simd.mul_f64x4(m3, v1_re);
                let m3 = simd
                    .avx
                    ._mm256_shuffle_pd::<0b0101>(pulp::cast(m3), pulp::cast(m3));
                let prod2 = simd.mul_f64x4(pulp::cast(m3), v1_im);
                let r3 = simd
                    .avx
                    ._mm256_addsub_pd(pulp::cast(prod1), pulp::cast(prod2));
                a3 = simd.add_f64x4(a3, pulp::cast(r3));

                let prod1 = simd.mul_f64x4(m4, v1_re);
                let m4 = simd
                    .avx
                    ._mm256_shuffle_pd::<0b0101>(pulp::cast(m4), pulp::cast(m4));
                let prod2 = simd.mul_f64x4(pulp::cast(m4), v1_im);
                let r4 = simd
                    .avx
                    ._mm256_addsub_pd(pulp::cast(prod1), pulp::cast(prod2));
                a4 = simd.add_f64x4(a4, pulp::cast(r4));
            }

            a1 = simd.mul_f64x4(scale, a1);
            a2 = simd.mul_f64x4(scale, a2);
            a3 = simd.mul_f64x4(scale, a3);
            a4 = simd.mul_f64x4(scale, a4);

            let ptr = result.as_mut_ptr() as *mut f64;

            unsafe { simd.avx._mm256_storeu_pd(ptr, std::mem::transmute(a1)) }
            unsafe {
                simd.avx
                    ._mm256_storeu_pd(ptr.add(4), std::mem::transmute(a2))
            }
            unsafe {
                simd.avx
                    ._mm256_storeu_pd(ptr.add(8), std::mem::transmute(a3))
            }
            unsafe {
                simd.avx
                    ._mm256_storeu_pd(ptr.add(12), std::mem::transmute(a4))
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
pub use x86::Matvec;
