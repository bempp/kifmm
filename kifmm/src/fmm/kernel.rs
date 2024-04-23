use green_kernels::{helmholtz_3d::Helmholtz3dKernel, laplace_3d::Laplace3dKernel};
use rlst::RlstScalar;

use crate::traits::field::{HomogenousKernel, InhomogenousKernel};

impl<T: RlstScalar> HomogenousKernel for Laplace3dKernel<T> {}

impl<T: RlstScalar<Complex = T>> InhomogenousKernel for Helmholtz3dKernel<T> {}
