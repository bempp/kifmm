//! Multi Node FMM
//! Single Node FMM
use std::time::Instant;

use green_kernels::traits::Kernel as KernelTrait;

use mpi::traits::{Communicator, Equivalence};
use num::Float;
use rlst::RlstScalar;

use crate::{
    fmm::types::{FmmEvalType, KiFmm},
    traits::{
        field::SourceToTargetData as SourceToTargetDataTrait,
        fmm::{
            FmmOperatorData, HomogenousKernel, MultiNodeFmm, SourceToTargetTranslation,
            SourceTranslation, TargetTranslation,
        },
        tree::{
            MultiNodeFmmTreeTrait, MultiNodeTreeTrait, SingleNodeFmmTreeTrait, SingleNodeTreeTrait,
        },
        types::{FmmError, FmmOperatorTime, FmmOperatorType},
    },
    Fmm, MultiNodeFmmTree,
};

use super::{
    helpers::{leaf_expansion_pointers, level_expansion_pointers, map_charges, potential_pointers},
    types::KiFmmMultiNode,
};

impl<Scalar, Kernel, SourceToTargetData, C> MultiNodeFmm
    for KiFmmMultiNode<Scalar, Kernel, SourceToTargetData, C>
where
    Scalar: RlstScalar + Default + Equivalence + Float,
    Kernel: KernelTrait<T = Scalar> + HomogenousKernel + Default + Send + Sync,
    SourceToTargetData: SourceToTargetDataTrait + Send + Sync,
    <Scalar as RlstScalar>::Real: Default + Equivalence + Float,
    KiFmm<Scalar, Kernel, SourceToTargetData>: SourceToTargetTranslation + FmmOperatorData,
    C: Communicator + Default,
    MultiNodeFmmTree<Scalar, C>: Default,
{
    type Scalar = Scalar;
    type Kernel = Kernel;
    type Tree = MultiNodeFmmTree<Scalar, C>;

    fn dim(&self) -> usize {
        3
    }

    fn evaluate(&mut self, timed: bool) -> Result<(), FmmError> {
        // Run upward pass on local trees
        for fmm in self.fmms.iter() {
            fmm.p2m()?;
        }

        for level in (self.tree.source_tree.global_depth
            ..(self.tree.source_tree.global_depth + self.tree.source_tree.local_depth))
            .rev()
        {
            for fmm in self.fmms.iter() {
                fmm.m2m(level)?;
            }
        }

        // At this point the exchange needs to happen of multipole data
        {}

        // Now can proceed with remainder of the upward pass on chosen node
        {}

        // Now send multipole data to all local trees
        {}

        // Now remainder of downward pass can happen in parallel
        {}

        Ok(())
    }

    fn kernel(&self) -> &Self::Kernel {
        &self.kernel
    }
}
