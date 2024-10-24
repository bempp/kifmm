use crate::fmm::helpers::single_node::optionally_time;
use crate::fmm::KiFmm;
use crate::traits::field::{
    FieldTranslation as FieldTranslationTrait, SourceToTargetTranslation, SourceTranslation,
    TargetTranslation,
};
use crate::traits::fmm::{DataAccessMulti, EvaluateMulti, HomogenousKernel};
use crate::traits::general::multi_node::GhostExchange;
use crate::traits::tree::{MultiFmmTree, MultiTree};
use crate::traits::types::{
    CommunicationTime, CommunicationType, FmmOperatorTime, FmmOperatorType,
};
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
    fn evaluate_leaf_sources(&mut self) -> Result<(), crate::traits::types::FmmError> {
        let (result, duration) = optionally_time(self.timed, || self.p2m());

        result?;

        if let Some(d) = duration {
            self.operator_times
                .push(FmmOperatorTime::from_duration(FmmOperatorType::P2M, d));
        }

        Ok(())
    }

    #[inline(always)]
    fn evaluate_upward_pass(&mut self) -> Result<(), crate::traits::types::FmmError> {
        let global_depth = self.tree.source_tree().global_depth();
        let total_depth = self.tree.source_tree().total_depth();

        for level in ((global_depth + 1)..=total_depth).rev() {
            let (result, duration) = optionally_time(self.timed, || self.m2m(level));

            result?;

            if let Some(d) = duration {
                self.operator_times.push(FmmOperatorTime::from_duration(
                    FmmOperatorType::M2M(level),
                    d,
                ));
            }
        }

        Ok(())
    }

    #[inline(always)]
    fn evaluate_downward_pass(&mut self) -> Result<(), crate::traits::types::FmmError> {
        let global_depth = self.tree.source_tree().global_depth();
        let total_depth = self.tree.source_tree().total_depth();
        let start_level = std::cmp::max(2, global_depth);

        if global_depth >= 2 {
            for level in (start_level + 1)..=total_depth {
                let (result, duration) = optionally_time(self.timed, || self.l2l(level));

                result?;

                if let Some(d) = duration {
                    self.operator_times.push(FmmOperatorTime::from_duration(
                        FmmOperatorType::L2L(level),
                        d,
                    ));
                }

                let (result, duration) = optionally_time(self.timed, || self.m2l(level));

                result?;

                if let Some(d) = duration {
                    self.operator_times.push(FmmOperatorTime::from_duration(
                        FmmOperatorType::M2L(level),
                        d,
                    ));
                }
            }
        } else {
            // Already handled M2L at level 2 during global FMM
            for level in start_level..=total_depth {
                if level > 2 {
                    let (result, duration) = optionally_time(self.timed, || self.l2l(level));

                    result?;

                    if let Some(d) = duration {
                        self.operator_times.push(FmmOperatorTime::from_duration(
                            FmmOperatorType::L2L(level),
                            d,
                        ));
                    }
                }

                let (result, duration) = optionally_time(self.timed, || self.m2l(level));

                result?;

                if let Some(d) = duration {
                    self.operator_times.push(FmmOperatorTime::from_duration(
                        FmmOperatorType::M2L(level),
                        d,
                    ));
                }
            }
        }

        Ok(())
    }

    #[inline(always)]
    fn evaluate_leaf_targets(&mut self) -> Result<(), crate::traits::types::FmmError> {
        let (result, duration) = optionally_time(self.timed, || self.p2p());

        result?;

        if let Some(d) = duration {
            self.operator_times
                .push(FmmOperatorTime::from_duration(FmmOperatorType::P2P, d));
        }

        let (result, duration) = optionally_time(self.timed, || self.l2p());

        result?;

        if let Some(d) = duration {
            self.operator_times
                .push(FmmOperatorTime::from_duration(FmmOperatorType::L2P, d));
        }

        Ok(())
    }

    fn evaluate(&mut self) -> Result<(), crate::traits::types::FmmError> {
        // Perform partial upward pass
        self.evaluate_leaf_sources()?;
        self.evaluate_upward_pass()?;

        // Exchange ghost multipole data
        let (_, d) = optionally_time(self.timed, || {
            self.v_list_exchange();
        });

        if let Some(d) = d {
            self.communication_times
                .push(CommunicationTime::from_duration(
                    CommunicationType::GhostExchangeV,
                    d,
                ))
        }

        // Gather global FMM
        let (_, d) = optionally_time(self.timed, || {
            self.gather_global_fmm_at_root();
        });

        if let Some(d) = d {
            self.communication_times
                .push(CommunicationTime::from_duration(
                    CommunicationType::GatherGlobalFmm,
                    d,
                ))
        }

        // Execute FMM on global root
        if self.communicator.rank() == 0 {
            self.global_fmm.evaluate_upward_pass()?;
            self.global_fmm.evaluate_downward_pass()?;
        }

        // Scatter root locals back to local tree
        let (_, d) = optionally_time(self.timed, || {
            self.scatter_global_fmm_from_root();
        });

        if let Some(d) = d {
            self.communication_times
                .push(CommunicationTime::from_duration(
                    CommunicationType::ScatterGlobalFmm,
                    d,
                ))
        }

        // Perform remainder of downward pass
        self.evaluate_downward_pass()?;
        self.evaluate_leaf_targets()?;

        Ok(())
    }
}
