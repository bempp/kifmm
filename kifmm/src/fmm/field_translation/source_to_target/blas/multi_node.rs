//! Multipole to local field translation trait implementation using BLAS.

use mpi::traits::Equivalence;
use num::Float;
use rlst::RlstScalar;

use green_kernels::traits::Kernel as KernelTrait;

use crate::{
    fmm::types::KiFmmMulti,
    traits::{
        fmm::{HomogenousKernel, SourceToTargetTranslation},
        types::FmmError,
    },
    BlasFieldTranslationSaRcmp,
};

impl<Scalar, Kernel> SourceToTargetTranslation
    for KiFmmMulti<Scalar, Kernel, BlasFieldTranslationSaRcmp<Scalar>>
where
    Scalar: RlstScalar + Default + Equivalence + Float,
    <Scalar as RlstScalar>::Real: Default + Equivalence + Float,
    Kernel: KernelTrait<T = Scalar> + HomogenousKernel + Default + Send + Sync,
{
    fn m2l(&self, _level: u64) -> Result<(), FmmError> {
        Ok(())
    }

    fn p2l(&self, _level: u64) -> Result<(), FmmError> {
        Ok(())
    }
}
