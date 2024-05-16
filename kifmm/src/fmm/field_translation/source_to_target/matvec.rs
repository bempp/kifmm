//! Implementations of 8x8 matrix vector product operation during Hadamard product in FFT based M2L operations.
use pulp::{aarch64::NeonFcma, f32x4, f64x2, Simd};
use rlst::{c32, c64, RlstScalar};
use std::arch::aarch64::{float32x4_t, float64x2_t};

/// The 8x8 matvec operation, always inlined. Implemented via a fully unrolled inner loop, and partially unrolled outer loop.
///
/// # Arguments
/// * - `matrix` - 8x8 matrix.
/// * - `signal` - 8x1 vector.
/// * `save_locations` - Save buffer.
/// * `scale` - Scalar scaling factor.
#[inline(always)]
pub fn matvec8x8_auto<U>(matrix: &[U; 64], vector: &[U; 8], save_buffer: &mut [U; 8], alpha: U)
where
    U: RlstScalar,
{
    // let s1 = vector[0];
    // let s2 = vector[1];
    // let s3 = vector[2];
    // let s4 = vector[3];
    // let s5 = vector[4];
    // let s6 = vector[5];
    // let s7 = vector[6];
    // let s8 = vector[7];

    // for i in 0..4 {
    //     let mut sum1 = U::zero();
    //     let mut sum2 = U::zero();
    //     let i1 = 2 * i;
    //     let i2 = 2 * i + 1;

    //     sum1 += matrix[i1 * 8] * s1;
    //     sum1 += matrix[i1 * 8 + 1] * s2;
    //     sum1 += matrix[i1 * 8 + 2] * s3;
    //     sum1 += matrix[i1 * 8 + 3] * s4;
    //     sum1 += matrix[i1 * 8 + 4] * s5;
    //     sum1 += matrix[i1 * 8 + 5] * s6;
    //     sum1 += matrix[i1 * 8 + 6] * s7;
    //     sum1 += matrix[i1 * 8 + 7] * s8;

    //     sum2 += matrix[i2 * 8] * s1;
    //     sum2 += matrix[i2 * 8 + 1] * s2;
    //     sum2 += matrix[i2 * 8 + 2] * s3;
    //     sum2 += matrix[i2 * 8 + 3] * s4;
    //     sum2 += matrix[i2 * 8 + 4] * s5;
    //     sum2 += matrix[i2 * 8 + 5] * s6;
    //     sum2 += matrix[i2 * 8 + 6] * s7;
    //     sum2 += matrix[i2 * 8 + 7] * s8;

    //     save_buffer[i1] += sum1 * alpha;
    //     save_buffer[i2] += sum2 * alpha;
    // }

    for i in 0..8 {

        let mut sum = U::zero();
        let row = &matrix[i*8..(i+1)*8];
        for j in 0..8 {
            sum += row[j]*vector[j]
        }

        save_buffer[i] += sum * alpha;
    }
}

pub struct ComplexMul8x8NeonFcma32<'a> {
    pub simd: NeonFcma,
    pub alpha: f32,
    pub matrix: &'a [c32; 64],
    pub vector: &'a [c32; 8],
    pub result: &'a mut [c32; 8],
}

pub struct ComplexMul8x8NeonFcma64<'a> {
    pub simd: NeonFcma,
    pub alpha: f64,
    pub matrix: &'a [c64; 64],
    pub vector: &'a [c64; 8],
    pub result: &'a mut [c64; 8],
}

impl pulp::NullaryFnOnce for ComplexMul8x8NeonFcma32<'_> {
    type Output = ();

    #[inline(always)]
    fn call(self) -> Self::Output {
        let Self {
            simd,
            alpha,
            matrix,
            vector,
            result,
        } = self;

        let mut a1 = f32x4(0., 0., 0., 0.);
        let mut a2 = f32x4(0., 0., 0., 0.);
        let mut a3 = f32x4(0., 0., 0., 0.);
        let mut a4 = f32x4(0., 0., 0., 0.);
        let [r1, r2, r3, r4]: [f32x4; 4] = pulp::cast(*result);
        let alpha = simd.f32s_splat(alpha);

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

        a1 = simd.mul_f32x4(a1, alpha);
        a2 = simd.mul_f32x4(a2, alpha);
        a3 = simd.mul_f32x4(a3, alpha);
        a4 = simd.mul_f32x4(a4, alpha);

        a1 = simd.add_f32x4(r1, a1);
        a2 = simd.add_f32x4(r2, a2);
        a3 = simd.add_f32x4(r3, a3);
        a4 = simd.add_f32x4(r4, a4);

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


impl pulp::NullaryFnOnce for ComplexMul8x8NeonFcma64<'_> {
    type Output = ();

    #[inline(always)]
    fn call(self) -> Self::Output {
        let Self {
            simd,
            alpha,
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
        let alpha = simd.f64s_splat(alpha);

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

        a1 = simd.mul_f64x2(a1, alpha);
        a2 = simd.mul_f64x2(a2, alpha);
        a3 = simd.mul_f64x2(a3, alpha);
        a4 = simd.mul_f64x2(a4, alpha);
        a5 = simd.mul_f64x2(a5, alpha);
        a6 = simd.mul_f64x2(a6, alpha);
        a7 = simd.mul_f64x2(a7, alpha);
        a8 = simd.mul_f64x2(a8, alpha);

        a1 = simd.add_f64x2(r1, a1);
        a2 = simd.add_f64x2(r2, a2);
        a3 = simd.add_f64x2(r3, a3);
        a4 = simd.add_f64x2(r4, a4);
        a5 = simd.add_f64x2(r5, a5);
        a6 = simd.add_f64x2(r6, a6);
        a7 = simd.add_f64x2(r7, a7);
        a8 = simd.add_f64x2(r8, a8);

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
pub trait Matvec {
    type Scalar: RlstScalar;

    fn matvec8x8(
        simd: NeonFcma,
        matrix: &[Self::Scalar; 64],
        vector: &[Self::Scalar; 8],
        save_buffer: &mut [Self::Scalar; 8],
        alpha: Self::Scalar,
    );
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
        simd.vectorize(ComplexMul8x8NeonFcma32 {
            simd,
            alpha: alpha.re(),
            matrix: matrix,
            vector: vector,
            result: save_buffer,
        });
    }
}

impl Matvec for c64 {
    type Scalar = c64;

    // #[inline(always)]
    // // fn matvec8x8(simd: NeonFcma, matrix: &[Self::Scalar], vector: &[Self::Scalar], save_buffer: &mut [Self::Scalar], alpha: Self::Scalar) {
    // fn matvec8x8(simd: NeonFcma, matrix: &[Self::Scalar; 64], vector: &[Self::Scalar; 8], save_buffer: &mut [Self::Scalar; 8], alpha: Self::Scalar) {
    //     matvec8x8_auto(matrix, vector, save_buffer, alpha)
    // }
    #[inline(always)]
    fn matvec8x8(
        simd: NeonFcma,
        matrix: &[Self::Scalar; 64],
        vector: &[Self::Scalar; 8],
        save_buffer: &mut [Self::Scalar; 8],
        alpha: Self::Scalar,
    ) {
        simd.vectorize(ComplexMul8x8NeonFcma64 {
            simd,
            alpha: alpha.re(),
            matrix: matrix,
            vector: vector,
            result: save_buffer,
        });
    }
}
