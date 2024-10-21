use std::time::Instant;

use crate::fmm::KiFmm;
use crate::traits::field::{
    FieldTranslation as FieldTranslationTrait, SourceToTargetTranslation, SourceTranslation,
    TargetTranslation,
};
use crate::traits::fmm::{DataAccessMulti, EvaluateMulti, HomogenousKernel};
use crate::traits::general::multi_node::GhostExchange;
use crate::traits::tree::{MultiFmmTree, MultiTree};
use crate::traits::types::{FmmOperatorTime, FmmOperatorType};
use green_kernels::traits::Kernel as KernelTrait;
use mpi::traits::{Communicator, Equivalence};
use num::Float;
use rlst::RlstScalar;

use crate::fmm::types::KiFmmMulti;
use crate::Evaluate;

impl<Scalar, Kernel, FieldTranslation> EvaluateMulti
    for KiFmmMulti<Scalar, Kernel, FieldTranslation>
where
    Scalar: RlstScalar + Default + Equivalence + Float,
    <Scalar as RlstScalar>::Real: Default + Float + Equivalence,
    Kernel: KernelTrait<T = Scalar> + HomogenousKernel + Default + Send + Sync,
    FieldTranslation: FieldTranslationTrait + Send + Sync,
    Self: SourceToTargetTranslation
        + SourceTranslation
        + TargetTranslation
        + DataAccessMulti
        + GhostExchange,
    KiFmm<Scalar, Kernel, FieldTranslation>: Evaluate,
{
    #[inline(always)]
    fn evaluate_leaf_sources(&mut self, timed: bool) -> Result<(), crate::traits::types::FmmError> {
        if timed {
            let s = Instant::now();
            self.p2m()?;
            self.times
                .push(FmmOperatorTime::from_instant(FmmOperatorType::P2M, s));
        } else {
            self.p2m()?;
        }

        Ok(())
    }

    #[inline(always)]
    fn evaluate_upward_pass(&mut self, timed: bool) -> Result<(), crate::traits::types::FmmError> {
        let global_depth = self.tree.source_tree().global_depth();
        let total_depth = self.tree.source_tree().total_depth();

        if timed {
            for level in ((global_depth + 1)..=total_depth).rev() {
                let s = Instant::now();
                self.m2m(level)?;
                self.times.push(FmmOperatorTime::from_instant(
                    FmmOperatorType::M2M(level),
                    s,
                ));
            }
        } else {
            for level in ((global_depth + 1)..=total_depth).rev() {
                self.m2m(level)?;
            }
        }

        Ok(())
    }

    #[inline(always)]
    fn evaluate_downward_pass(
        &mut self,
        timed: bool,
    ) -> Result<(), crate::traits::types::FmmError> {
        let global_depth = self.tree.source_tree().global_depth();
        let total_depth = self.tree.source_tree().total_depth();
        let start_level = std::cmp::max(2, global_depth);

        if timed {
            for level in (start_level + 1)..=total_depth {
                let s = Instant::now();
                self.l2l(level)?;
                self.times.push(FmmOperatorTime::from_instant(
                    FmmOperatorType::L2L(level),
                    s,
                ));

                let s = Instant::now();
                self.m2l(level)?;
                self.times.push(FmmOperatorTime::from_instant(
                    FmmOperatorType::M2L(level),
                    s,
                ));
            }
        } else {
            if global_depth >= 2 {
                for level in (start_level + 1)..=total_depth {
                    self.l2l(level)?;
                    self.m2l(level)?;
                }
            } else {
                for level in start_level..=total_depth {
                    if level > 2 {
                        self.l2l(level)?;
                    }

                    self.m2l(level)?;
                }
            }
        }

        Ok(())
    }

    #[inline(always)]
    fn evaluate_leaf_targets(&mut self, timed: bool) -> Result<(), crate::traits::types::FmmError> {
        if timed {
            // Leaf level computations
            let s = Instant::now();
            self.p2p()?;
            self.times
                .push(FmmOperatorTime::from_instant(FmmOperatorType::P2P, s));

            let s = Instant::now();
            self.l2p()?;
            self.times
                .push(FmmOperatorTime::from_instant(FmmOperatorType::L2P, s));
        } else {
            // Leaf level computations
            self.p2p()?;
            self.l2p()?;
        }

        Ok(())
    }

    fn evaluate(&mut self, timed: bool) -> Result<(), crate::traits::types::FmmError> {
        // Perform partial upward pass
        self.evaluate_leaf_sources(timed)?;
        self.evaluate_upward_pass(timed)?;

        // Exchange ghost multipole data
        self.v_list_exchange();

        // Gather global FMM
        self.gather_global_fmm_at_root();

        // Execute FMM on global root
        if self.communicator.rank() == 0 {
            self.global_fmm.evaluate_upward_pass(timed)?;
            self.global_fmm.evaluate_downward_pass(timed)?;
        }

        // Scatter root locals back to local tree
        self.scatter_global_fmm_from_root();

        // Perform remainder of downward pass
        self.evaluate_downward_pass(timed)?;
        self.evaluate_leaf_targets(timed)?;

        Ok(())
    }
}
