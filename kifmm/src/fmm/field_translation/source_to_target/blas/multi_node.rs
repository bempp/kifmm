//! Multipole to local field translation trait implementation using BLAS.

use mpi::traits::Equivalence;
use num::Float;
use rlst::RlstScalar;

use green_kernels::traits::Kernel as KernelTrait;

use crate::{
    fmm::types::KiFmmMulti,
    traits::{
        field::SourceToTargetTranslation,
        fmm::{DataAccessMulti, HomogenousKernel},
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
    Self: DataAccessMulti<Scalar = Scalar, Kernel = Kernel>,
{
    fn m2l(&self, level: u64) -> Result<(), FmmError> {
        let _multipoles = self.multipoles(level);
        Ok(())
    }

    fn p2l(&self, _level: u64) -> Result<(), FmmError> {
        Ok(())
    }
}
