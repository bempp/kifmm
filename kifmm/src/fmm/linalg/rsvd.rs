use rlst::{
    dense::tools::RandScalar, empty_array, rlst_dynamic_array2, Array, BaseArray, MatrixQrDecomposition, MatrixSvd, MultIntoResize, QrDecomposition, RlstResult, RlstScalar, Shape, VectorContainer
};
use rand_distr::{StandardNormal, Standard};

/// Matrix type
pub type RsvdMatrix<T> = Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>;

type RsvdReturnType<T> = RlstResult<(Vec<<T as RlstScalar>::Real>, RsvdMatrix<T>, RsvdMatrix<T>)>;

/// Choose a normalisation method for the orthonormal subspace
pub enum Normaliser {
    /// Expensive, more accurate
    Qr(usize),
}

/// Foo
pub trait MatrixRsvd
where
    Self: RlstScalar + RandScalar,
    Array<Self, BaseArray<Self, VectorContainer<Self>, 2>, 2>: MatrixSvd<Item = Self>
{
    /// Compute randomised SVD when target rank is known.
    fn rsvd_fixed_rank(
        mat: &RsvdMatrix<Self>,
        n_components: usize, // Number of singular values and vectors to extract
        n_oversamples: Option<usize>, // Additional number of random vectors to sample the range of M for proper conditioning.
        power_iteration_normaliser: Option<Normaliser>,
        random_state: Option<usize>
    ) -> RsvdReturnType<Self>;

    /// Compute randomised SVD when target rank is unknown, and a given tolerance is required
    fn rsvd_fixed_error(
        mat: &RsvdMatrix<Self>,
        tol: Self, // Error tolerance for final matrix
        n_oversamples: Option<usize>, // Additional number of random vectors to sample the range of M for proper conditioning.
        power_iteration_normaliser: Option<Normaliser>,
        random_state: Option<usize>
    ) -> RsvdReturnType<Self>;
}

macro_rules! impl_matrix_rsvd {
    ($ty:ty, $func:ident) => {
        impl MatrixRsvd for $ty
        where
            $ty: RlstScalar + RandScalar,
            Array<$ty, BaseArray<$ty, VectorContainer<$ty>, 2>, 2>: MatrixSvd<Item = $ty>,
        {
            fn rsvd_fixed_rank(
                mat: &RsvdMatrix<Self>,
                n_components: usize,
                n_oversamples: Option<usize>,
                power_iteration_normaliser: Option<Normaliser>,
                random_state: Option<usize>
            ) -> RsvdReturnType<Self> {
                $func(mat, n_components, n_oversamples, power_iteration_normaliser, random_state)
            }

            fn rsvd_fixed_error(
                mat: &RsvdMatrix<Self>,
                tol: Self, // Error tolerance for final matrix
                n_oversamples: Option<usize>, // Additional number of random vectors to sample the range of M for proper conditioning.
                power_iteration_normaliser: Option<Normaliser>,
                random_state: Option<usize>
            ) -> RsvdReturnType<Self> {
                panic!("Unimplemented")
            }
        }
    };
}

impl_matrix_rsvd!(f32, rsvd_fixed_rank_f32);
impl_matrix_rsvd!(f64, rsvd_fixed_rank_f64);

macro_rules! generate_randomized_range_finder_fixed_rank {
    ($ty:ty, $name:ident) => {
        /// Foo
        pub fn $name(
            mat: &RsvdMatrix<$ty>,
            size: usize,
            power_iteration_normaliser: Option<Normaliser>,
            random_state: Option<usize>
        ) -> RsvdMatrix<$ty>
        where
            StandardNormal: rand_distr::Distribution<$ty>,
            Standard: rand_distr::Distribution<$ty>
        {

            let mut mat_transpose = rlst_dynamic_array2!($ty, [mat.shape()[1], mat.shape()[0]]);
            mat_transpose.fill_from(mat.view().transpose());

            // Input matrix of size [m, n]. Draw Gaussian matrix of size [n, size].
            let mut omega = rlst_dynamic_array2!($ty, [mat.shape()[1], size]);
            let random_state = random_state.unwrap_or(0);
            omega.fill_from_seed_normally_distributed(random_state);

            let mut q1 = rlst_dynamic_array2!($ty, [mat.shape()[0], size]);

            // Compute sample matrix of size [m, size]
            let y = empty_array::<$ty, 2>().simple_mult_into_resize(
                mat.view(),
                omega.view()
            );

            // Ortho-normalise columns using QR
            let qr = QrDecomposition::<$ty, _>::new(y).expect("QR Decomposition failed");
            qr.get_q_alloc(q1.view_mut()).unwrap();

            let mut q2 = rlst_dynamic_array2!($ty, [mat.shape()[1], size]);

            if let Some(normaliser) = power_iteration_normaliser {
                match normaliser {

                    Normaliser::Qr(n_iter) => {
                        // Perform power iterations
                        for _ in 0..n_iter {

                            let atq = empty_array::<$ty, 2>().simple_mult_into_resize(
                                mat_transpose.view(), q1.view()
                            );
                            let qr = QrDecomposition::<$ty, _>::new(atq).unwrap();
                            qr.get_q_alloc(q2.view_mut()).unwrap();

                            let aq = empty_array::<$ty, 2>().simple_mult_into_resize(
                                mat.view(), q2.view()
                            );

                            let qr = QrDecomposition::<$ty, _>::new(aq).unwrap();
                            qr.get_q_alloc(q1.view_mut()).unwrap();
                        }

                        let atq = empty_array::<$ty, 2>().simple_mult_into_resize(
                            mat_transpose.view(), q1.view()
                        );

                        let qr = QrDecomposition::<$ty, _>::new(atq).unwrap();
                        qr.get_q_alloc(q2.view_mut()).unwrap();
                        let aq = empty_array::<$ty, 2>().simple_mult_into_resize(
                            mat.view(), q2.view()
                        );

                        let qr = QrDecomposition::<$ty, _>::new(aq).unwrap();
                        qr.get_q_alloc(q1.view_mut()).unwrap();

                        return q1
                    }
                }
            }

            // compute m x l sample matrix
            let y = empty_array::<$ty, 2>().simple_mult_into_resize(
                mat.view(),
                omega.view()
            );

            // Ortho-normalise columns using QR
            let mut q = rlst_dynamic_array2!($ty, y.shape());
            let qr = QrDecomposition::<$ty, _>::new(y).expect("QR Decomposition failed");
            qr.get_q_alloc(q.view_mut()).unwrap();

            q
        }
    };
}

generate_randomized_range_finder_fixed_rank!(f32, randomized_range_finder_fixed_rank_f32);
generate_randomized_range_finder_fixed_rank!(f64, randomized_range_finder_fixed_rank_f64);

macro_rules! generate_rsvd {
    ($ty:ty, $randomized_range_finder:ident, $name:ident) => {
        /// Foo
        pub fn $name(
            mat: &RsvdMatrix<$ty>,
            n_components: usize, // Number of singular values and vectors to extract
            n_oversamples: Option<usize>, // Additional number of random vectors to sample the range of M for proper conditioning.
            power_iteration_normaliser: Option<Normaliser>,
            random_state: Option<usize>
        ) -> RsvdReturnType<$ty>
        where
            Array<$ty, BaseArray<$ty, VectorContainer<$ty>, 2>, 2>: MatrixSvd<Item = $ty>
        {
            let n_oversamples = n_oversamples.unwrap_or(10);
            let n_random = n_components + n_oversamples;
            let [n_samples, n_features] = mat.shape();

            let q = $randomized_range_finder(mat, n_random, power_iteration_normaliser, random_state);
            let mut q_transpose = rlst_dynamic_array2!($ty, [q.shape()[1], q.shape()[0]]);
            q_transpose.fill_from(q.view().transpose());

            // Project matrix to (k+p) dimensional space using orthonormal basis
            let b = empty_array::<$ty, 2>().simple_mult_into_resize(
                q_transpose.view(), mat.view()
            );

            // Compute svd on thin matrix (k+p) wide
            let k = std::cmp::min(b.shape()[0], b.shape()[1]);
            let mut uhat = rlst_dynamic_array2!($ty, [b.shape()[0], k]);
            let mut s = vec![0 as $ty; k];
            let mut vt = rlst_dynamic_array2!($ty, [k, b.shape()[1]]);

            let mut b_copy = rlst_dynamic_array2!($ty, b.shape());
            b_copy.fill_from(b.view());
            b_copy
                .into_svd_alloc(uhat.view_mut(), vt.view_mut(), &mut s[..], rlst::SvdMode::Reduced)
                .unwrap();

            let u = empty_array::<$ty, 2>().simple_mult_into_resize(
                q.view(), uhat.view()
            );

            Ok((s, u, vt))
        }
    };
}

generate_rsvd!(f32, randomized_range_finder_fixed_rank_f32, rsvd_fixed_rank_f32);
generate_rsvd!(f64, randomized_range_finder_fixed_rank_f64, rsvd_fixed_rank_f64);


#[cfg(test)]
mod test {
    use rlst::{assert_array_abs_diff_eq, empty_array, rlst_dynamic_array2, DefaultIterator, MultIntoResize, RawAccessMut, Shape};

    use crate::fmm::linalg::rsvd::Normaliser;
    use super::MatrixRsvd;

    #[test]
    fn test_rsvd_f32() {
        let mut mat = rlst_dynamic_array2!(f32, [5, 6]);

        mat.data_mut().iter_mut().enumerate().for_each(|(i, v)| *v += (i+1) as f32);
        let n_components = 2;
        let n_oversamples = None;
        let power_iteration_normaliser = Some(Normaliser::Qr(2));
        let random_state = None;

        let (s, u, vt) = f32::rsvd_fixed_rank(&mat, n_components, n_oversamples, power_iteration_normaliser, random_state).unwrap();

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


    #[test]
    fn test_rsvd_f64() {
        let mut mat = rlst_dynamic_array2!(f64, [5, 6]);

        mat.data_mut().iter_mut().enumerate().for_each(|(i, v)| *v += (i+1) as f64);
        let n_components = 2;
        let n_oversamples = None;
        let power_iteration_normaliser = Some(Normaliser::Qr(2));
        let random_state = None;

        let (s, u, vt) = f64::rsvd_fixed_rank(&mat, n_components, n_oversamples, power_iteration_normaliser, random_state).unwrap();

        let mut mat_s = rlst_dynamic_array2!(f64, [s.len(), s.len()]);
        for i in 0..s.len() {
            mat_s[[i, i]] = s[i];
        }

        let mat_rec = empty_array::<f64, 2>().simple_mult_into_resize(
            u.view(),
            empty_array::<f64, 2>().simple_mult_into_resize(
                mat_s.view(),
                vt.view()
            )
        );

        assert_array_abs_diff_eq!(mat_rec.view(), mat.view(), 1e-13);
    }
}