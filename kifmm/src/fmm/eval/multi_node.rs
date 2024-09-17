use crate::traits::fmm::{FmmOperatorData, HomogenousKernel, MultiFmm};
use crate::traits::parallel::GhostExchange;
use crate::traits::{
    field::SourceToTargetData as SourceToTargetDataTrait, fmm::SourceToTargetTranslation,
};
use green_kernels::traits::Kernel as KernelTrait;
use mpi::topology::SimpleCommunicator;
use mpi::traits::Equivalence;
use num::Float;
use rlst::RlstScalar;

use crate::fmm::types::KiFmmMulti;
use crate::{MultiNodeFmmTree, SingleFmm};

impl<Scalar, Kernel, SourceToTargetData> MultiFmm for KiFmmMulti<Scalar, Kernel, SourceToTargetData>
where
    Scalar: RlstScalar + Default + Equivalence + Float,
    <Scalar as RlstScalar>::Real: Default + Float + Equivalence,
    Kernel: KernelTrait<T = Scalar> + HomogenousKernel + Default + Send + Sync,
    SourceToTargetData: SourceToTargetDataTrait + Send + Sync,
    Self: SourceToTargetTranslation + GhostExchange,
{
    type Scalar = Scalar;
    type Kernel = Kernel;
    type Tree = MultiNodeFmmTree<Scalar::Real, SimpleCommunicator>;

    fn dim(&self) -> usize {
        3
    }

    fn evaluate(&mut self, timed: bool) -> Result<(), crate::traits::types::FmmError> {
        self.v_list_exchange();
        Ok(())
    }

    fn kernel<'a>(&'a self) -> &'a Self::Kernel {
        &self.kernel
    }

    fn tree<'a>(&'a self) -> &'a Self::Tree {
        &self.tree
    }
}
