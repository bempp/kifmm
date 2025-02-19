//! Implementation of Moore-Penrose PseudoInverse
use crate::traits::general::single_node::Epsilon;
use rlst::{
    rlst_dynamic_array2, Array, BaseArray, MatrixSvd, RlstError, RlstResult, RlstScalar, Shape,
    SvdMode, VectorContainer,
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
pub fn pinv<T>(
    mat: &PinvMatrix<T>,
    atol: Option<T::Real>,
    rtol: Option<T::Real>,
) -> PinvReturnType<T>
where
    T: RlstScalar + Epsilon + MatrixSvd,
{
    let shape = mat.shape();

    if shape[0] == 0 || shape[1] == 0 {
        return Err(RlstError::MatrixIsEmpty((shape[0], shape[1])));
    }

    // If we have a vector return error
    if shape[0] == 1 || shape[1] == 1 {
        Err(RlstError::SingleDimensionError {
            expected: 2,
            actual: 1,
        })
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

#[cfg(test)]
mod test {

    use super::*;
    use approx::assert_relative_eq;
    use rlst::{empty_array, rlst_dynamic_array2, MultIntoResize, RandomAccessByRef};

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
}
