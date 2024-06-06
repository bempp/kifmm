use crate::traits::general::Epsilon;
use rand::SeedableRng;
use rlst::{
    dense::tools::RandScalar, rlst_dynamic_array2, Array, BaseArray, RlstError, RlstResult, RlstScalar, Shape, VectorContainer
};
use rand_chacha::{self, ChaCha8Rng};
use rand_distr::{StandardNormal, Standard};

/// Matrix type
pub type RsvdMatrix<T> = Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>;

type RsvdReturnType<T> = RlstResult<(Vec<<T as RlstScalar>::Real>, RsvdMatrix<T>, RsvdMatrix<T>)>;


/// Foo
pub fn rsvd<T>(
    mat: &RsvdMatrix<T>,
    n_components: usize, // Number of singular values and vectors to extract
    n_oversamples: Option<usize>, // Additional number of random vectors to sample the range of M for proper conditioning.
    n_iter: Option<usize>, // number of power iterations

)
where
    T: RlstScalar + Epsilon
{

    let n_oversamples = n_oversamples.unwrap_or(10);

    let random_state = 0;
    let n_random = n_components + n_oversamples;
    let [n_samples, n_features] = mat.shape();



}

/// Foo
pub fn randomized_range_finder<T>(
    mat: &RsvdMatrix<T>,
    size: usize,
)
where
    T: RlstScalar + Epsilon + RandScalar,
    StandardNormal: rand_distr::Distribution<<T as rlst::RlstScalar>::Real>,
    Standard: rand_distr::Distribution<<T as RlstScalar>::Real>
{

    let mut rng = rand::thread_rng();
    let mut omega = rlst_dynamic_array2!(T, [mat.shape()[1], size]);
    omega.fill_from_seed_normally_distributed(0);




}

#[cfg(test)]
mod test {
    use rlst::{rlst_dynamic_array2, DefaultIteratorMut, RawAccessMut};


    fn test_randomized_range_finder() {
        let mut mat = rlst_dynamic_array2!(f64, [5, 6]);

        mat.data_mut().iter_mut().enumerate().for_each(|(i, v)| *v += (i+1) as f64)
    }
}