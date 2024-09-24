use crate::{
    fmm::types::KiFmmMulti,
    traits::{
        field::SourceToTargetData as SourceToTargetDataTrait,
        fmm::{HomogenousKernel, MultiFmm, TargetTranslation},
    },
};
use green_kernels::traits::Kernel as KernelTrait;
use mpi::traits::Equivalence;
use num::Float;
use rlst::RlstScalar;

impl<Scalar, Kernel, SourceToTargetData> TargetTranslation
    for KiFmmMulti<Scalar, Kernel, SourceToTargetData>
where
    Scalar: RlstScalar + Default + Equivalence + Float,
    <Scalar as RlstScalar>::Real: Default + Equivalence + Float,
    SourceToTargetData: SourceToTargetDataTrait,
    Kernel: KernelTrait<T = Scalar> + HomogenousKernel + Default + Send + Sync,
    // KiFmmMulti<Scalar, Kernel, SourceToTargetData>: FmmDataAccess
{
    fn l2l(&self, _level: u64) -> Result<(), crate::traits::types::FmmError> {
        Ok(())
    }

    fn l2p(&self) -> Result<(), crate::traits::types::FmmError> {
        Ok(())
    }

    fn m2p(&self) -> Result<(), crate::traits::types::FmmError> {
        Ok(())
    }

    fn p2p(&self) -> Result<(), crate::traits::types::FmmError> {
        Ok(())
    }
}