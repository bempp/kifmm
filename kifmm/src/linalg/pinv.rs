//! Implementation of Moore-Penrose PseudoInverse
use crate::{
    linalg::aca::aca_plus,
    traits::general::single_node::{ArgmaxValue, Cast, Epsilon, Upcast},
};
use coe::{is_same, Coerce};
use green_kernels::traits::Kernel;
use num::Zero;
use rlst::{
    c32, c64, empty_array, rlst_dynamic_array2, Array, BaseArray, MatrixQr, MatrixSvd, MultInto,
    MultIntoResize, QrDecomposition, RawAccess, RawAccessMut, RlstError, RlstResult, RlstScalar,
    Shape, SvdMode, VectorContainer,
};

/// Matrix type
pub type PinvMatrix<T> = Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>;

type PinvReturnType<T> = RlstResult<(Vec<<T as RlstScalar>::Real>, PinvMatrix<T>, PinvMatrix<T>)>;

/// Compute the (Moore-Penrose) pseudo-inverse of a matrix.
///
/// Calculate a generalised inverse using its singular value decomposition `U @ S @ V*`.
/// If `s` is the maximum singular value, then the signifance cut-off value is determined by
/// `atol + rtol * s`. Any singular value below this is assumed insignificant.
///
/// # Arguments
/// * `mat` - (M, N) matrix to be inverted.
/// * `atol` - Absolute threshold term, default is 0.
/// * `rtol` - Relative threshold term, default value is max(M, N) * eps
pub(crate) fn pinv<T>(
    mat: &PinvMatrix<T>,
    atol: Option<T::Real>,
    rtol: Option<T::Real>,
) -> PinvReturnType<T>
where
    T: RlstScalar + Epsilon + MatrixSvd + Epsilon,
    <T as RlstScalar>::Real: Epsilon,
{
    let shape = mat.shape();

    if shape[0] == 0 || shape[1] == 0 {
        return Err(RlstError::MatrixIsEmpty((shape[0], shape[1])));
    }

    if shape[0] == 1 || shape[1] == 1 {
        // If we have a vector

        let eps = T::real(T::epsilon());
        let max_dim = T::real(std::cmp::max(shape[0], shape[1]));

        let atol = atol.unwrap_or(T::zero().re());
        let rtol = rtol.unwrap_or(max_dim * eps);

        if shape[0] == 1 {
            // Row vector
            let l2_norm = mat
                .data()
                .iter()
                .map(|&x| (x * x.conj()).re())
                .sum::<T::Real>()
                .sqrt();

            if l2_norm <= <T::Real as Epsilon>::epsilon() {
                // Zero vector, so pseudo-inverse is zero
                let zero_s = vec![T::Real::zero()];
                let v = rlst_dynamic_array2!(T, [shape[1], 1]);
                let ut = rlst_dynamic_array2!(T, [1, 1]);
                return Ok((zero_s, ut, v));
            }

            // Compute SVD of row vector
            let mut s = vec![l2_norm];
            let mut u = rlst_dynamic_array2!(T, [1, 1]);
            u[[0, 0]] = T::one();
            let mut vt = rlst_dynamic_array2!(T, [1, shape[1]]);
            vt.data_mut()
                .iter_mut()
                .zip(mat.data().iter())
                .for_each(|(v, x)| *v = *x / T::from_real(l2_norm));

            // Compute pseudo inverse
            let max_s = s[0];
            let threshold = T::real(atol + rtol) * T::real(max_s);

            // Filter singular values below this threshold
            for s in s.iter_mut() {
                if *s > threshold {
                    *s = T::real(1.0) / T::real(*s);
                } else {
                    *s = T::real(0.)
                }
            }

            let mut v = rlst_dynamic_array2!(T, [vt.shape()[1], vt.shape()[0]]);
            let mut ut = rlst_dynamic_array2!(T, [u.shape()[1], u.shape()[0]]);
            v.fill_from(vt.conj().transpose());
            ut.fill_from(u.conj().transpose());

            Ok((s, ut, v))
        } else if shape[1] == 1 {
            // Column vector (only one singular value)
            let l2_norm = mat
                .data()
                .iter()
                .map(|&x| (x * x.conj()).re())
                .sum::<T::Real>()
                .sqrt();

            if l2_norm <= <T::Real as Epsilon>::epsilon() {
                // Zero vector, so pseudo-inverse is zero
                let zero_s = vec![T::Real::zero()];
                let v = rlst_dynamic_array2!(T, [1, 1]);
                let ut = rlst_dynamic_array2!(T, [1, shape[0]]);
                return Ok((zero_s, ut, v));
            }

            // Compute SVD of column vector
            let mut s = vec![l2_norm];
            let mut u = rlst_dynamic_array2!(T, [shape[0], 1]);
            u.data_mut()
                .iter_mut()
                .zip(mat.data().iter())
                .for_each(|(u, x)| *u = *x / T::from_real(l2_norm));
            let mut vt = rlst_dynamic_array2!(T, [1, 1]);
            vt[[0, 0]] = T::one();

            // Compute pseudo inverse
            let max_s = s[0];
            let threshold = T::real(atol + rtol) * T::real(max_s);

            // Filter singular values below this threshold
            for s in s.iter_mut() {
                if *s > threshold {
                    *s = T::real(1.0) / T::real(*s);
                } else {
                    *s = T::real(0.)
                }
            }

            let mut v = rlst_dynamic_array2!(T, [vt.shape()[1], vt.shape()[0]]);
            let mut ut = rlst_dynamic_array2!(T, [u.shape()[1], u.shape()[0]]);
            v.fill_from(vt.conj().transpose());
            ut.fill_from(u.conj().transpose());

            Ok((s, ut, v))
        } else {
            Err(RlstError::SingleDimensionError {
                expected: 2,
                actual: 1,
            })
        }
    } else {
        // For matrices compute the full SVD
        let k = std::cmp::min(shape[0], shape[1]);
        let mut u = rlst_dynamic_array2!(T, [shape[0], k]);
        let mut s = vec![T::zero().re(); k];
        let mut vt = rlst_dynamic_array2!(T, [k, shape[1]]);

        let mut mat_copy = rlst_dynamic_array2!(T, shape);
        mat_copy.fill_from(mat.r());
        mat_copy
            .into_svd_alloc(u.r_mut(), vt.r_mut(), &mut s[..], SvdMode::Reduced)
            .unwrap();

        let eps = T::real(T::epsilon());
        let max_dim = T::real(std::cmp::max(shape[0], shape[1]));

        let atol = atol.unwrap_or(T::zero().re());
        let rtol = rtol.unwrap_or(max_dim * eps);

        let max_s = s[0];
        let threshold = T::real(atol + rtol) * T::real(max_s);

        // Filter singular values below this threshold
        for s in s.iter_mut() {
            if *s > threshold {
                *s = T::real(1.0) / T::real(*s);
            } else {
                *s = T::real(0.)
            }
        }

        // Return pseudo-inverse in component form
        let mut v = rlst_dynamic_array2!(T, [vt.shape()[1], vt.shape()[0]]);
        let mut ut = rlst_dynamic_array2!(T, [u.shape()[1], u.shape()[0]]);
        v.fill_from(vt.conj().transpose());
        ut.fill_from(u.conj().transpose());

        Ok((s, ut, v))
    }
}

/// Compute the (Moore-Penrose) pseudo-inverse of a 3D Greens fct matrix based on ACA+ for finding factored matrix
/// The matrix is not computed directly, but required entries are computed using a specified kernel function
/// and source/target terms. This is a useful alternative to using a pseudo-inverse based on the SVD for numerically
/// low-rank kernels, where many singular values are near machine epsilon.
///
/// Calculate a generalised inverse using ACA decomposition `U @ V^T*`.
///
/// # Arguments
/// * `eps` - Convergence criteria for decomposition
pub(crate) fn pinv_aca_plus<T, K>(
    sources: &[T::Real],
    targets: &[T::Real],
    kernel: K,
    eps: Option<T::Real>,
    max_iter: Option<usize>,
    local_radius: Option<usize>,
    verbose: bool,
    multithreaded: bool,
    test: bool,
) -> PinvReturnType<T>
where
    T: RlstScalar
        + Epsilon
        + MatrixSvd
        + MatrixQr
        + ArgmaxValue<T>
        + Upcast
        + Cast<<T as Upcast>::Higher>,
    <T as RlstScalar>::Real: ArgmaxValue<<T as RlstScalar>::Real>
        + Epsilon
        + Upcast
        + Cast<<<T as Upcast>::Higher as RlstScalar>::Real>,
    <T as Upcast>::Higher: RlstScalar + MatrixSvd + Epsilon + Cast<T>,
    <<T as Upcast>::Higher as RlstScalar>::Real: Epsilon + Cast<T::Real>,
    K: Kernel<T = T>,
{
    let dim = 3;

    if sources.len() == 0 || targets.len() == 0 {
        return Err(RlstError::MatrixIsEmpty((targets.len(), sources.len())));
    }

    // TODO: If we have a vector can compute directly at low cost
    if targets.len() == dim || sources.len() == dim {
        Err(RlstError::SingleDimensionError {
            expected: 2,
            actual: 1,
        })
    } else {
        // Compute ACA decomposition
        let (u_aca, v_aca_t) = aca_plus::<K, T>(
            sources,
            targets,
            kernel,
            eps,
            max_iter,
            local_radius,
            local_radius,
            verbose,
            multithreaded,
        );

        // v_aca := v_aca^H
        let [m1, n1] = u_aca.shape();
        let [m2, n2] = v_aca_t.shape();
        let mut v_aca = rlst_dynamic_array2!(T, [n2, m2]);
        v_aca.fill_from(v_aca_t.r().conj().transpose());
        let [m3, n3] = v_aca.shape();

        // Compute QR decomposition of result (HACK, because of RLST API)
        let mut qu: Array<T, BaseArray<T, VectorContainer<T>, 2>, 2> =
            rlst_dynamic_array2!(T, [m1, n1]);
        let mut ru = rlst_dynamic_array2!(T, [n1, n1]);

        let mut qv = rlst_dynamic_array2!(T, [m3, n3]);
        let mut rv = rlst_dynamic_array2!(T, [n3, n3]);

        let qr_u_aca = u_aca.into_qr_alloc()?;
        let qr_v_aca = v_aca.into_qr_alloc()?;

        if is_same::<f64, T>() {
            let qr_u_aca: &QrDecomposition<f64, BaseArray<f64, VectorContainer<f64>, 2>> =
                qr_u_aca.coerce();
            let qu: &mut Array<f64, BaseArray<f64, VectorContainer<f64>, 2>, 2> =
                (&mut qu).coerce();
            let ru: &mut Array<f64, BaseArray<f64, VectorContainer<f64>, 2>, 2> =
                (&mut ru).coerce();
            qr_u_aca.get_r(ru.r_mut());
            qr_u_aca.get_q_alloc(qu.r_mut())?;

            let qr_v_aca: &QrDecomposition<f64, BaseArray<f64, VectorContainer<f64>, 2>> =
                qr_v_aca.coerce();
            let qv: &mut Array<f64, BaseArray<f64, VectorContainer<f64>, 2>, 2> =
                (&mut qv).coerce();
            let rv: &mut Array<f64, BaseArray<f64, VectorContainer<f64>, 2>, 2> =
                (&mut rv).coerce();
            qr_v_aca.get_r(rv.r_mut());
            qr_v_aca.get_q_alloc(qv.r_mut())?;
        } else if is_same::<f32, T>() {
            let qr_u_aca: &QrDecomposition<f32, BaseArray<f32, VectorContainer<f32>, 2>> =
                qr_u_aca.coerce();
            let qu: &mut Array<f32, BaseArray<f32, VectorContainer<f32>, 2>, 2> =
                (&mut qu).coerce();
            let ru: &mut Array<f32, BaseArray<f32, VectorContainer<f32>, 2>, 2> =
                (&mut ru).coerce();
            qr_u_aca.get_r(ru.r_mut());
            qr_u_aca.get_q_alloc(qu.r_mut())?;

            let qr_v_aca: &QrDecomposition<f32, BaseArray<f32, VectorContainer<f32>, 2>> =
                qr_v_aca.coerce();
            let qv: &mut Array<f32, BaseArray<f32, VectorContainer<f32>, 2>, 2> =
                (&mut qv).coerce();
            let rv: &mut Array<f32, BaseArray<f32, VectorContainer<f32>, 2>, 2> =
                (&mut rv).coerce();
            qr_v_aca.get_r(rv.r_mut());
            qr_v_aca.get_q_alloc(qv.r_mut())?;
        } else if is_same::<c32, T>() {
            let qr_u_aca: &QrDecomposition<f32, BaseArray<f32, VectorContainer<f32>, 2>> =
                qr_u_aca.coerce();
            let qu: &mut Array<f32, BaseArray<f32, VectorContainer<f32>, 2>, 2> =
                (&mut qu).coerce();
            let ru: &mut Array<f32, BaseArray<f32, VectorContainer<f32>, 2>, 2> =
                (&mut ru).coerce();
            qr_u_aca.get_r(ru.r_mut());
            qr_u_aca.get_q_alloc(qu.r_mut())?;

            let qr_v_aca: &QrDecomposition<c32, BaseArray<c32, VectorContainer<c32>, 2>> =
                qr_v_aca.coerce();
            let qv: &mut Array<c32, BaseArray<c32, VectorContainer<c32>, 2>, 2> =
                (&mut qv).coerce();
            let rv: &mut Array<c32, BaseArray<c32, VectorContainer<c32>, 2>, 2> =
                (&mut rv).coerce();
            qr_v_aca.get_r(rv.r_mut());
            qr_v_aca.get_q_alloc(qv.r_mut())?;
        } else if is_same::<c64, T>() {
            let qr_u_aca: &QrDecomposition<c64, BaseArray<c64, VectorContainer<c64>, 2>> =
                qr_u_aca.coerce();
            let qu: &mut Array<c64, BaseArray<c64, VectorContainer<c64>, 2>, 2> =
                (&mut qu).coerce();
            let ru: &mut Array<c64, BaseArray<c64, VectorContainer<c64>, 2>, 2> =
                (&mut ru).coerce();
            qr_u_aca.get_r(ru.r_mut());
            qr_u_aca.get_q_alloc(qu.r_mut())?;

            let qr_v_aca: &QrDecomposition<c64, BaseArray<c64, VectorContainer<c64>, 2>> =
                qr_v_aca.coerce();
            let qv: &mut Array<c64, BaseArray<c64, VectorContainer<c64>, 2>, 2> =
                (&mut qv).coerce();
            let rv: &mut Array<c64, BaseArray<c64, VectorContainer<c64>, 2>, 2> =
                (&mut rv).coerce();
            qr_v_aca.get_r(rv.r_mut());
            qr_v_aca.get_q_alloc(qv.r_mut())?;
        } else {
            return Err(RlstError::NotImplemented(
                "Unsupported scalar type for this decomposition".to_string(),
            ));
        }

        // rvt := rv^H
        let [m, n] = rv.shape();
        let mut rvt = rlst_dynamic_array2!(T, [n, m]);
        rvt.fill_from(rv.conj().transpose());

        // c := ru * rvt (single pre-allocated multiply)
        let [ru_m, ru_n] = ru.shape();
        let [rvt_m, rvt_n] = rvt.shape();
        debug_assert_eq!(ru_n, rvt_m);
        let mut c = rlst_dynamic_array2!(T, [ru_m, rvt_n]);
        c.r_mut().simple_mult_into(ru.r(), rvt.r());

        // Compute SVD based pseudo-inverse on tiny core matrix formed from R factors

        // qv_t := qv^H
        let [qv_m, qv_n] = qv.shape();
        let mut qv_t = rlst_dynamic_array2!(T, [qv_n, qv_m]);
        qv_t.fill_from(qv.r().conj().transpose());

        // Upcast to V and SVD pinv on c
        let [c_m, c_n] = c.shape();
        let mut c_64 = rlst_dynamic_array2!(<T as Upcast>::Higher, [c_m, c_n]);
        for (dst, src) in c_64.data_mut().iter_mut().zip(c.data().iter()) {
            *dst = src.cast();
        }
        let (s_c_v, ut_c_v, v_c_v) = pinv::<<T as Upcast>::Higher>(&c_64, None, None)?;

        // Downcast back down to T
        let mut ut_c = rlst_dynamic_array2!(T, ut_c_v.shape());
        for (dst, src) in ut_c.data_mut().iter_mut().zip(ut_c_v.data().iter()) {
            *dst = src.cast();
        }

        let mut v_c = rlst_dynamic_array2!(T, v_c_v.shape());
        for (dst, src) in v_c.data_mut().iter_mut().zip(v_c_v.data().iter()) {
            *dst = src.cast();
        }
        let mut s_c: Vec<T::Real> = vec![T::Real::zero(); s_c_v.len()];
        for (dst, src) in s_c.iter_mut().zip(s_c_v.iter()) {
            *dst = src.cast();
        }

        // Form factors of pseudo inverse
        let mut left = rlst_dynamic_array2!(T, [qv.shape()[0], v_c.shape()[1]]);
        left.r_mut().simple_mult_into(qv.r(), v_c.r());

        let [qu_m, qu_n] = qu.shape();
        let mut qu_t = rlst_dynamic_array2!(T, [qu_n, qu_m]);
        qu_t.r_mut().fill_from(qu.r().conj().transpose());

        let mut right = rlst_dynamic_array2!(T, [ut_c.shape()[0], qu_t.shape()[1]]);
        right.r_mut().simple_mult_into(ut_c.r(), qu_t.r());

        if test {
            let mut mat_s = rlst_dynamic_array2!(T, [s_c.len(), s_c.len()]);
            for i in 0..s_c.len() {
                mat_s[[i, i]] = T::from(s_c[i]).unwrap();
            }

            // Test Moore-Penrose residuals
            let aca = empty_array::<T, 2>().simple_mult_into_resize(
                qu.r(),
                empty_array::<T, 2>().simple_mult_into_resize(c.r(), qv_t.r()),
            );

            // Compute pseudo inverse
            let aca_pinv = empty_array::<T, 2>().simple_mult_into_resize(
                left.r(),
                empty_array::<T, 2>().simple_mult_into_resize(mat_s.r(), right.r()),
            );

            // test in Frobenius norm
            let t1 = empty_array::<T, 2>().simple_mult_into_resize(
                empty_array::<T, 2>().simple_mult_into_resize(aca.r(), aca_pinv.r()),
                aca.r(),
            );
            let e1 = (t1.r() - aca.r()).norm_fro() / aca.r().norm_fro();

            let t2 = empty_array::<T, 2>().simple_mult_into_resize(
                empty_array::<T, 2>().simple_mult_into_resize(aca_pinv.r(), aca.r()),
                aca_pinv.r(),
            );
            let e2 = (t2.r() - aca_pinv.r()).norm_fro() / aca_pinv.r().norm_fro();

            let aca_aca_pinv = empty_array::<T, 2>().simple_mult_into_resize(aca.r(), aca_pinv.r());
            let aca_pinv_aca = empty_array::<T, 2>().simple_mult_into_resize(aca_pinv.r(), aca.r());

            let e3 = (aca_aca_pinv.r() - aca_aca_pinv.r().conj().transpose()).norm_fro()
                / aca_aca_pinv.r().norm_fro();
            let e4 = (aca_pinv_aca.r() - aca_pinv_aca.r().conj().transpose()).norm_fro()
                / aca_pinv_aca.r().norm_fro();

            assert!(e1 < eps.unwrap());
            assert!(e2 < eps.unwrap());
            assert!(e3 < eps.unwrap());
            assert!(e4 < eps.unwrap());

            if verbose {
                println!("E1 = ||A A^+ A - A|| / ||A||             = {:?}", e1);
                println!("E2 = ||A^+ A A^+ - A^+|| / ||A^+||       = {:?}", e2);
                println!("E3 = ||(A A^+) - (A A^+)^T|| / ||A A^+|| = {:?}", e3);
                println!("E4 = ||(A^+ A) - (A^+ A)^T|| / ||A^+ A|| = {:?}", e4);
            }
        }

        return Ok((s_c, right, left));
    }
}

#[cfg(test)]
mod test {

    use crate::tree::helpers::points_fixture;

    use super::*;
    use approx::assert_relative_eq;
    use green_kernels::{helmholtz_3d::Helmholtz3dKernel, laplace_3d::Laplace3dKernel};
    use rlst::{empty_array, rlst_dynamic_array2, MultIntoResize, RandomAccessByRef, RawAccess};

    #[test]
    fn test_pinv_square() {
        let dim: usize = 5;
        let mut mat = rlst_dynamic_array2!(f64, [dim, dim]);
        mat.fill_from_seed_equally_distributed(0);

        let (s, ut, v) = pinv::<f64>(&mat, None, None).unwrap();

        let mut mat_s = rlst_dynamic_array2!(f64, [s.len(), s.len()]);
        for i in 0..s.len() {
            mat_s[[i, i]] = s[i];
        }

        let inv = empty_array::<f64, 2>().simple_mult_into_resize(
            v.r(),
            empty_array::<f64, 2>().simple_mult_into_resize(mat_s.r(), ut.r()),
        );

        let actual = empty_array::<f64, 2>().simple_mult_into_resize(inv.r(), mat.r());

        // Expect the identity matrix
        let mut expected = rlst_dynamic_array2!(f64, actual.shape());
        for i in 0..dim {
            expected[[i, i]] = 1.0
        }

        for i in 0..actual.shape()[0] {
            for j in 0..actual.shape()[1] {
                assert_relative_eq!(
                    *actual.get([i, j]).unwrap(),
                    *expected.get([i, j]).unwrap(),
                    epsilon = 1E-13
                );
            }
        }
    }

    #[test]
    fn test_pinv_rectangle() {
        let dim: usize = 5;
        let mut mat = rlst_dynamic_array2!(f64, [dim, dim + 1]);
        mat.fill_from_seed_equally_distributed(0);

        let (s, ut, v) = pinv::<f64>(&mat, None, None).unwrap();

        let mut mat_s = rlst_dynamic_array2!(f64, [s.len(), s.len()]);
        for i in 0..s.len() {
            mat_s[[i, i]] = s[i];
        }

        let inv = empty_array::<f64, 2>().simple_mult_into_resize(
            v.r(),
            empty_array::<f64, 2>().simple_mult_into_resize(mat_s.r(), ut.r()),
        );

        let actual = empty_array::<f64, 2>().simple_mult_into_resize(mat.r(), inv.r());

        // Expect the identity matrix
        let mut expected = rlst_dynamic_array2!(f64, actual.shape());
        for i in 0..dim {
            expected[[i, i]] = 1.0
        }

        for i in 0..actual.shape()[0] {
            for j in 0..actual.shape()[1] {
                assert_relative_eq!(
                    *actual.get([i, j]).unwrap(),
                    *expected.get([i, j]).unwrap(),
                    epsilon = 1E-13
                );
            }
        }
    }

    #[test]
    fn test_pinv_aca_plus() {
        let n = 100;
        let sources = points_fixture::<f64>(n, Some(0.1), Some(0.4), None);
        let targets = points_fixture::<f64>(n, Some(1.1), Some(2.2), None);

        // Test Laplace
        let kernel = Laplace3dKernel::<f64>::new();
        let eps = 1e-6;

        let (_s, _ut, _v) = pinv_aca_plus(
            sources.data(),
            targets.data(),
            kernel.clone(),
            Some(eps),
            None,
            None,
            true,
            true,
            true,
        )
        .unwrap();

        // Test Helmholtz (low wavenumber)
        let kernel = Helmholtz3dKernel::<c64>::new(1.0);
        let eps = 1e-6;
        let (_s, _ut, _v) = pinv_aca_plus(
            sources.data(),
            targets.data(),
            kernel.clone(),
            Some(eps),
            None,
            None,
            true,
            true,
            true,
        )
        .unwrap();
    }
}
