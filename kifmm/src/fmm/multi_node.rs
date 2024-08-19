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
        {
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
        }

        // At this point the exchange needs to happen of multipole data
        {
            // 1. Gather ranges on other processes

            // 2. Form packets

            // 3. Exchange packets (point to point)

            // 4. Pass all root multipole data to root node so that final part of upward pass can occur on root node
        }

        // Now can proceed with remainder of the upward pass on chosen node, and some of the downward pass
        {
            if self.communicator.rank() == 0 {
                // Global upward pass
                for level in (1..self.tree.source_tree.global_depth).rev() {
                    for fmm in self.fmms.iter() {
                        fmm.m2m(level)?;
                    }
                }

                // Global downward pass
                for level in 2..=self.tree.target_tree.global_depth {
                    if level > 2 {
                        for fmm in self.fmms.iter() {
                            fmm.l2l(level)?;
                        }
                    }
                    for fmm in self.fmms.iter() {
                        fmm.m2l(level)?;
                    }
                }
            }

            // Exchange root multipole data back to required MPI processes
        }

        // Now remainder of downward pass can happen in parallel on each process
        {}

        Ok(())
    }

    fn kernel(&self) -> &Self::Kernel {
        &self.kernel
    }
}
