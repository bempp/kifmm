use green_kernels::{helmholtz_3d::Helmholtz3dKernel, laplace_3d::Laplace3dKernel};
use rlst::RlstScalar;

use crate::traits::fmm::FmmOperator;

impl<T> FmmOperator for Laplace3dKernel<T>
where
    T: RlstScalar,
{
    fn is_kernel_homogenous(&self) -> bool {
        true
    }

    fn c2e_operator_index(&self, _level: u64) -> usize {
        0
    }

    fn m2m_operator_index(&self, _level: u64) -> usize {
        0
    }

    fn l2l_operator_index(&self, _level: u64) -> usize {
        0
    }

    fn m2l_operator_index(&self, _level: u64) -> usize {
        0
    }
}

impl<T> FmmOperator for Helmholtz3dKernel<T>
where
    T: RlstScalar<Complex = T>,
{
    fn is_kernel_homogenous(&self) -> bool {
        false
    }

    fn c2e_operator_index(&self, level: u64) -> usize {
        level as usize
    }

    fn m2m_operator_index(&self, level: u64) -> usize {
        (level - 1) as usize
    }

    fn l2l_operator_index(&self, level: u64) -> usize {
        (level - 1) as usize
    }

    fn m2l_operator_index(&self, level: u64) -> usize {
        (level - 2) as usize
    }
}
