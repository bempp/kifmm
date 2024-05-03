use green_kernels::{helmholtz_3d::Helmholtz3dKernel, laplace_3d::Laplace3dKernel};
use rlst::RlstScalar;

use crate::traits::fmm::HomogenousKernel;

impl<T> HomogenousKernel for Laplace3dKernel<T>
where
    T: RlstScalar,
{
    fn is_homogenous(&self) -> bool {
        true
    }
}

impl<T> HomogenousKernel for Helmholtz3dKernel<T>
where
    T: RlstScalar<Complex = T>,
{
    fn is_homogenous(&self) -> bool {
        false
    }
}
