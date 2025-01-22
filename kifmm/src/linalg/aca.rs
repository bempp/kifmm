//! Implementation of Adaptive Cross Approximation With Partial Pivoting
use std::collections::HashSet;

use crate::traits::general::single_node::Epsilon;
use rlst::{
    rlst_dynamic_array1, rlst_dynamic_array2, Array, BaseArray, DefaultIterator, MatrixSvd, RawAccess, RlstError, RlstResult, RlstScalar, Shape, SvdMode, VectorContainer
};
use rand::{self, Rng};

/// Matrix type
pub type AcaMatrix<T> = Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>;

// type AcaReturnType<T> = RlstResult<(Vec<<T as RlstScalar>::Real>, AcaMatrix<T>, AcaMatrix<T>)>;
type AcaReturnType = RlstResult<()>;




/// Compute the Adaptive Cross Approximation With Partial Pivoting (ACA+)
pub fn aca<T>(
    mat: &AcaMatrix<T>,
    max_iter: Option<usize>,
) -> AcaReturnType
where
    T: RlstScalar + Epsilon + MatrixSvd,
{
    let [m, n] = mat.shape();

    if m == 0 || n == 0 {
        return Err(RlstError::MatrixIsEmpty((m, n)));
    }

    // If we have a vector return error
    if m == 1 || n == 1 {
        Err(RlstError::SingleDimensionError {
            expected: 2,
            actual: 1,
        })
    } else {

        // Previously used i* pivots
        // let mut prev_i_star = HashSet::new();
        // Previously used j* pivots
        // let mut prev_j_star = HashSet::new();

        let max_iter = if let Some(value) = max_iter {
            let tmp = m.min(n);
            value.min(tmp)
        } else {
            m.min(n)
        };

        // Buffer to store R^i*j and R^ij*
        let mut ri_star = rlst_dynamic_array1!(T, [n]);
        let mut rj_star = rlst_dynamic_array1!(T, [m]);

        // Choose random reference row and column as starting point
        let mut rng = rand::thread_rng();
        let i_ref = rng.gen_range(0..m);
        let j_ref = rng.gen_range(0..n);



        let ri_j_ref = mat.r().slice(1, j_ref);
        let ri_ref_j = mat.r().slice(0, i_ref);

        for d in ri_ref_j.iter() {
            println!("{:?}", d)
        }
        println!("");
        for d in ri_j_ref.iter() {
            println!("{:?}", d)
        }

        // println!("{:?} {:?} {:?}", m, i_ref, ri_ref_j.data());

        // Find pivot
        // let ri_ref_j_max = ri_ref_j.data();

        // Collect corresponding blocks of rows/columns


        // // For matrices compute the full SVD
        // let k = std::cmp::min(shape[0], shape[1]);
        // let mut u = rlst_dynamic_array2!(T, [shape[0], k]);
        // let mut s = vec![T::zero().re(); k];
        // let mut vt = rlst_dynamic_array2!(T, [k, shape[1]]);

        // let mut mat_copy = rlst_dynamic_array2!(T, shape);
        // mat_copy.fill_from(mat.r());
        // mat_copy
        //     .into_svd_alloc(u.r_mut(), vt.r_mut(), &mut s[..], SvdMode::Reduced)
        //     .unwrap();

        // let eps = T::real(T::epsilon());
        // let max_dim = T::real(std::cmp::max(shape[0], shape[1]));

        // let atol = atol.unwrap_or(T::zero().re());
        // let rtol = rtol.unwrap_or(max_dim * eps);

        // let max_s = s[0];
        // let threshold = T::real(atol + rtol) * T::real(max_s);

        // // Filter singular values below this threshold
        // for s in s.iter_mut() {
        //     if *s > threshold {
        //         *s = T::real(1.0) / T::real(*s);
        //     } else {
        //         *s = T::real(0.)
        //     }
        // }

        // // Return pseudo-inverse in component form
        // let mut v = rlst_dynamic_array2!(T, [vt.shape()[1], vt.shape()[0]]);
        // let mut ut = rlst_dynamic_array2!(T, [u.shape()[1], u.shape()[0]]);
        // v.fill_from(vt.conj().transpose());
        // ut.fill_from(u.conj().transpose());

        // Ok((s, ut, v))
        Ok(())
    }
}

#[cfg(test)]
mod test {

    use super::*;
    use approx::assert_relative_eq;
    use rlst::{empty_array, rlst_array_from_slice2, rlst_dynamic_array2, MultIntoResize, RandomAccessByRef, RawAccessMut};

    #[test]
    fn test_aca() {

        let m = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
        let  mut mat = rlst_dynamic_array2!(f32, [3, 4]);
        mat.data_mut().copy_from_slice(&m);

        aca(&mat, None);
        assert!(false);
    }
}
