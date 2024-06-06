use crate::traits::general::Epsilon;
use rand::SeedableRng;
use rlst::{
    dense::{linalg::lu, tools::RandScalar}, empty_array, rlst_dynamic_array2, Array, BaseArray, LuDecomposition, MatrixLuDecomposition, MatrixQrDecomposition, MatrixSvd, MultIntoResize, QrDecomposition, RawAccess, RawAccessMut, RlstError, RlstResult, RlstScalar, Shape, VectorContainer
};
use rand_chacha::{self, ChaCha8Rng};
use rand_distr::{StandardNormal, Standard};

/// Matrix type
pub type RsvdMatrix<T> = Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>;

type RsvdReturnType<T> = RlstResult<(Vec<<T as RlstScalar>::Real>, RsvdMatrix<T>, RsvdMatrix<T>)>;


/// Foo
pub fn rsvd(
    mat: &RsvdMatrix<f32>,
    n_components: usize, // Number of singular values and vectors to extract
    n_oversamples: Option<usize>, // Additional number of random vectors to sample the range of M for proper conditioning.
    n_iter: Option<usize>, // number of power iterations
    power_iteration_normaliser: Option<Normalizer>,
    random_state: Option<usize>
) -> RsvdReturnType<f32>
where
    Array<f32, BaseArray<f32, VectorContainer<f32>, 2>, 2>: MatrixSvd<Item = f32>
{

    let n_oversamples = n_oversamples.unwrap_or(10);
    let n_random = n_components + n_oversamples;
    let [n_samples, n_features] = mat.shape();

    let n_random = n_components + n_oversamples;
    let [n_samples, n_features] = mat.shape();

    let n_iter = n_iter.unwrap_or(2);

    let q = randomized_range_finder_f32(mat, n_random, n_iter, power_iteration_normaliser, random_state);
    let mut q_transpose = rlst_dynamic_array2!(f32, [q.shape()[1], q.shape()[0]]);
    q_transpose.fill_from(q.view().transpose());

    // Project matrix to (k+p) dimensional space using orthonormal basis
    let b = empty_array::<f32, 2>().simple_mult_into_resize(
        q_transpose.view(), mat.view()
    );

    // Compute svd on thin matrix (k+p) wide
    let k = std::cmp::min(b.shape()[0], b.shape()[1]);
    let mut uhat = rlst_dynamic_array2!(f32, [b.shape()[0], k]);
    let mut s = vec![0f32; k];
    let mut vt = rlst_dynamic_array2!(f32, [k, b.shape()[1]]);

    let mut b_copy = rlst_dynamic_array2!(f32, b.shape());
    b_copy.fill_from(b.view());
    b_copy
        .into_svd_alloc(uhat.view_mut(), vt.view_mut(), &mut s[..], rlst::SvdMode::Reduced)
        .unwrap();

    let u = empty_array::<f32, 2>().simple_mult_into_resize(
        q.view(), uhat.view()
    );

    Ok((s, u, vt))
}

/// Choose a normalisation method for the orthonormal subspace
pub enum Normalizer {
    /// Expensive, more accurate
    Qr,

    /// Cheaper less accurate
    Lu
}

/// Foo
pub fn randomized_range_finder_f32(
    mat: &RsvdMatrix<f32>,
    size: usize,
    n_iter: usize,
    power_iteration_normaliser: Option<Normalizer>,
    random_state: Option<usize>
) -> RsvdMatrix<f32>
where
    StandardNormal: rand_distr::Distribution<f32>,
    Standard: rand_distr::Distribution<f32>
{

    let mut mat_transpose = rlst_dynamic_array2!(f32, [mat.shape()[1], mat.shape()[0]]);
    mat_transpose.fill_from(mat.view().transpose());

    // Input matrix of size [m, n]. Draw Gaussian matrix of size [n, size].
    let mut omega = rlst_dynamic_array2!(f32, [mat.shape()[1], size]);
    let random_state = random_state.unwrap_or(0);
    omega.fill_from_seed_normally_distributed(random_state);

    let mut q1 = rlst_dynamic_array2!(f32, [mat.shape()[0], size]);

    // Compute sample matrix of size [m, size]
    let y = empty_array::<f32, 2>().simple_mult_into_resize(
        mat.view(),
        omega.view()
    );

    // Ortho-normalise columns using QR
    let qr = QrDecomposition::<f32, _>::new(y).expect("QR Decomposition failed");
    qr.get_q_alloc(q1.view_mut()).unwrap();

    let mut q2 = rlst_dynamic_array2!(f32, [mat.shape()[1], size]);

    if let Some(normaliser) = power_iteration_normaliser {
        match normaliser {
            Normalizer::Lu => {
                // Perform power iterations
                for _ in 0..n_iter {
                    let atq = empty_array::<f32, 2>().simple_mult_into_resize(
                        mat_transpose.view(), q1.view()
                    );
                    let lu = LuDecomposition::<f32, _>::new(atq).unwrap();
                    lu.get_l(q2.view_mut());

                    let aq = empty_array::<f32, 2>().simple_mult_into_resize(
                        mat.view(), q2.view()
                    );

                    let lu = LuDecomposition::<f32, _>::new(aq).unwrap();
                    lu.get_l(q1.view_mut());
                }
            }

            Normalizer::Qr => {
                // Perform power iterations
                for _ in 0..n_iter {

                    let atq = empty_array::<f32, 2>().simple_mult_into_resize(
                        mat_transpose.view(), q1.view()
                    );
                    let qr = QrDecomposition::<f32, _>::new(atq).unwrap();
                    qr.get_q_alloc(q2.view_mut()).unwrap();

                    let aq = empty_array::<f32, 2>().simple_mult_into_resize(
                        mat.view(), q2.view()
                    );

                    let qr = QrDecomposition::<f32, _>::new(aq).unwrap();
                    qr.get_q_alloc(q1.view_mut()).unwrap();
                }

                let atq = empty_array::<f32, 2>().simple_mult_into_resize(
                    mat_transpose.view(), q1.view()
                );

                let qr = QrDecomposition::<f32, _>::new(atq).unwrap();
                qr.get_q_alloc(q2.view_mut()).unwrap();
                let aq = empty_array::<f32, 2>().simple_mult_into_resize(
                    mat.view(), q2.view()
                );

                let qr = QrDecomposition::<f32, _>::new(aq).unwrap();
                qr.get_q_alloc(q1.view_mut()).unwrap();

                return q1
            }
        }
    }

    // compute m x l sample matrix
    let y = empty_array::<f32, 2>().simple_mult_into_resize(
        mat.view(),
        omega.view()
    );

    // orthonormalise columns using QR
    let mut q = rlst_dynamic_array2!(f32, y.shape());
    let qr = QrDecomposition::<f32, _>::new(y).expect("QR Decomposition failed");
    qr.get_q_alloc(q.view_mut()).unwrap();

    q
}


/// Foo
pub fn randomized_range_finder<T>(
    mat: &RsvdMatrix<T>,
    size: usize,
    n_iter: usize,
    power_iteration_normaliser: Option<Normalizer>,
    random_state: Option<usize>
) -> RsvdMatrix<T>
where
    T: RlstScalar + RandScalar,
    Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>: MatrixQrDecomposition<Item = T>,
    StandardNormal: rand_distr::Distribution<T::Real>,
    Standard: rand_distr::Distribution<T::Real>,
{

    let mut mat_transpose = rlst_dynamic_array2!(T, [mat.shape()[1], mat.shape()[0]]);
    mat_transpose.fill_from(mat.view().transpose());

    // Input matrix of size [m, n]. Draw Gaussian matrix of size [n, size].
    let mut omega = rlst_dynamic_array2!(T, [mat.shape()[1], size]);
    let random_state = random_state.unwrap_or(0);
    omega.fill_from_seed_normally_distributed(random_state);

    let mut q1 = rlst_dynamic_array2!(T, [mat.shape()[0], size]);

    // Compute sample matrix of size [m, size]
    let y = empty_array::<T, 2>().simple_mult_into_resize(
        mat.view(),
        omega.view()
    );

    // Ortho-normalise columns using QR
    // let qr = QrDecomposition::<T, _>::new(y).unwrap();

    // qr.get_q_alloc(q1.view_mut()).unwrap();
    // y.new();

    // let mut q2 = rlst_dynamic_array2!(T, [mat.shape()[1], size]);

    // if let Some(normaliser) = power_iteration_normaliser {
    //     match normaliser {
    //         Normalizer::Lu => {
    //             // Perform power iterations
    //             for _ in 0..n_iter {
    //                 let atq = empty_array::<f32, 2>().simple_mult_into_resize(
    //                     mat_transpose.view(), q1.view()
    //                 );
    //                 let lu = LuDecomposition::<f32, _>::new(atq).unwrap();
    //                 lu.get_l(q2.view_mut());

    //                 let aq = empty_array::<f32, 2>().simple_mult_into_resize(
    //                     mat.view(), q2.view()
    //                 );

    //                 let lu = LuDecomposition::<f32, _>::new(aq).unwrap();
    //                 lu.get_l(q1.view_mut());
    //             }
    //         }

    //         Normalizer::Qr => {
    //             // Perform power iterations
    //             for _ in 0..n_iter {

    //                 let atq = empty_array::<f32, 2>().simple_mult_into_resize(
    //                     mat_transpose.view(), q1.view()
    //                 );
    //                 let qr = QrDecomposition::<f32, _>::new(atq).unwrap();
    //                 qr.get_q_alloc(q2.view_mut()).unwrap();

    //                 let aq = empty_array::<f32, 2>().simple_mult_into_resize(
    //                     mat.view(), q2.view()
    //                 );

    //                 let qr = QrDecomposition::<f32, _>::new(aq).unwrap();
    //                 qr.get_q_alloc(q1.view_mut()).unwrap();
    //             }

    //             let atq = empty_array::<f32, 2>().simple_mult_into_resize(
    //                 mat_transpose.view(), q1.view()
    //             );

    //             let qr = QrDecomposition::<f32, _>::new(atq).unwrap();
    //             qr.get_q_alloc(q2.view_mut()).unwrap();
    //             let aq = empty_array::<f32, 2>().simple_mult_into_resize(
    //                 mat.view(), q2.view()
    //             );

    //             let qr = QrDecomposition::<f32, _>::new(aq).unwrap();
    //             qr.get_q_alloc(q1.view_mut()).unwrap();

    //             return q1
    //         }
    //     }
    // }


    // // compute m x l sample matrix
    // let y = empty_array::<f32, 2>().simple_mult_into_resize(
    //     mat.view(),
    //     omega.view()
    // );

    // // orthonormalise columns using QR
    // let mut q = rlst_dynamic_array2!(f32, y.shape());
    // let qr = QrDecomposition::<f32, _>::new(y).expect("QR Decomposition failed");
    // qr.get_q_alloc(q.view_mut()).unwrap();


    q1
}

#[cfg(test)]
mod test {
    use rlst::{assert_array_abs_diff_eq, empty_array, rlst_dynamic_array2, DefaultIterator, MultIntoResize, RawAccessMut, Shape};

    use crate::fmm::linalg::rsvd::{rsvd, Normalizer};

    #[test]
    fn test_rsvd() {
        let mut mat = rlst_dynamic_array2!(f32, [5, 6]);

        mat.data_mut().iter_mut().enumerate().for_each(|(i, v)| *v += (i+1) as f32);
        let (s, u, vt) = rsvd(&mat, 2, None, Some(2), Some(Normalizer::Qr), None).unwrap();

        let mut mat_s = rlst_dynamic_array2!(f32, [s.len(), s.len()]);
        for i in 0..s.len() {
            mat_s[[i, i]] = s[i];
        }

        let mat_rec = empty_array::<f32, 2>().simple_mult_into_resize(
            u.view(),
            empty_array::<f32, 2>().simple_mult_into_resize(
                mat_s.view(),
                vt.view()
            )
        );

        assert_array_abs_diff_eq!(mat_rec.view(), mat.view(), 1e-5);
    }
}