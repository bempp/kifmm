//! Local Translations

use rlst::RlstScalar;

use green_kernels::{traits::Kernel as KernelTrait, types::EvalType};

use crate::{
    new_fmm::KiFmm,
    traits::{field::SourceToTargetData as SourceToTargetDataTrait, fmm::TargetTranslation},
};

impl<Scalar, Kernel, SourceToTargetData> TargetTranslation
    for KiFmm<Scalar, Kernel, SourceToTargetData>
where
    Scalar: RlstScalar,
    Kernel: KernelTrait<T = Scalar>,
    SourceToTargetData: SourceToTargetDataTrait + Send + Sync,
    <Scalar as RlstScalar>::Real: Default,
{
    fn l2l(&self, level: u64) {}

    fn l2p(&self) {}

    fn p2p(&self) {}

    fn m2p(&self) {}
}
