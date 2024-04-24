use green_kernels::{helmholtz_3d::Helmholtz3dKernel, laplace_3d::Laplace3dKernel};
use rlst::RlstScalar;

use crate::traits::fmm::FmmKernel;

use super::helpers::homogenous_kernel_scale;


impl <T> FmmKernel for Laplace3dKernel<T>
where
    T: RlstScalar
{
    fn p2m_operator_index(&self, level: u64) -> usize {
        0
    }

    fn m2m_operator_index(&self, level: u64) -> usize {
        0
    }

    fn scale<U: RlstScalar>(&self, level: u64) -> U {
        homogenous_kernel_scale::<U>(level)
    }
}


impl <T> FmmKernel for Helmholtz3dKernel<T>
where
    T: RlstScalar<Complex = T>
{
    fn p2m_operator_index(&self, level: u64) -> usize {
        level as usize
    }
    fn m2m_operator_index(&self, level: u64) -> usize {
        (level - 1) as usize
    }

    fn scale<U: RlstScalar>(&self, _level: u64) -> U {
        U::one()
    }
}
