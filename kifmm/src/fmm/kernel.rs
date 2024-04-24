use green_kernels::{helmholtz_3d::Helmholtz3dKernel, laplace_3d::Laplace3dKernel};
use rlst::RlstScalar;

use crate::traits::fmm::FmmKernel;

impl<T> FmmKernel for Laplace3dKernel<T>
where
    T: RlstScalar,
{
    fn homogenous(&self) -> bool {
        true
    }

    fn p2m_operator_index(&self, _level: u64) -> usize {
        0
    }

    fn m2m_operator_index(&self, _level: u64) -> usize {
        0
    }
}

impl<T> FmmKernel for Helmholtz3dKernel<T>
where
    T: RlstScalar<Complex = T>,
{
    fn homogenous(&self) -> bool {
        false
    }

    fn p2m_operator_index(&self, level: u64) -> usize {
        level as usize
    }
    fn m2m_operator_index(&self, level: u64) -> usize {
        (level - 1) as usize
    }
}
