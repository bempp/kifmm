use crate::{
    fmm::{helpers::single_node::optionally_time, types::KiFmmMulti},
    traits::{
        field::{
            FieldTranslation as FieldTranslationTrait, SourceToTargetTranslation,
            SourceTranslation, TargetTranslation,
        },
        fmm::{DataAccessMulti, EvaluateMulti, HomogenousKernel},
        general::multi_node::GhostExchange,
        tree::{MultiFmmTree, MultiTree},
        types::{CommunicationType, FmmOperatorType, OperatorTime},
    },
    Evaluate, KiFmm,
};

use green_kernels::traits::Kernel as KernelTrait;
use mpi::traits::{Communicator, Equivalence};
use num::Float;
use rlst::RlstScalar;

impl<Scalar, Kernel, FieldTranslation> EvaluateMulti
    for KiFmmMulti<Scalar, Kernel, FieldTranslation>
where
    Scalar: RlstScalar + Default + Equivalence,
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
                .insert(FmmOperatorType::P2M, OperatorTime::from_duration(d));
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
                self.operator_times
                    .insert(FmmOperatorType::M2M(level), OperatorTime::from_duration(d));
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
                    self.operator_times
                        .insert(FmmOperatorType::L2L(level), OperatorTime::from_duration(d));
                }

                let (result, duration) = optionally_time(self.timed, || self.m2l(level));

                result?;

                if let Some(d) = duration {
                    self.operator_times
                        .insert(FmmOperatorType::M2L(level), OperatorTime::from_duration(d));
                }
            }
        } else {
            // Already handled M2L at level 2 during global FMM
            for level in start_level..=total_depth {
                if level > 2 {
                    let (result, duration) = optionally_time(self.timed, || self.l2l(level));

                    result?;

                    if let Some(d) = duration {
                        self.operator_times
                            .insert(FmmOperatorType::L2L(level), OperatorTime::from_duration(d));
                    }
                }

                let (result, duration) = optionally_time(self.timed, || self.m2l(level));

                result?;

                if let Some(d) = duration {
                    self.operator_times
                        .insert(FmmOperatorType::M2L(level), OperatorTime::from_duration(d));
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
                .insert(FmmOperatorType::P2P, OperatorTime::from_duration(d));
        }

        let (result, duration) = optionally_time(self.timed, || self.l2p());

        result?;

        if let Some(d) = duration {
            self.operator_times
                .insert(FmmOperatorType::L2P, OperatorTime::from_duration(d));
        }

        Ok(())
    }

    fn evaluate(&mut self) -> Result<(), crate::traits::types::FmmError> {
        // Perform partial upward pass
        self.evaluate_leaf_sources()?;
        self.evaluate_upward_pass()?;

        // Exchange ghost multipole data
        let (_, d) = optionally_time(self.timed, || {
            self.v_list_exchange_runtime();
        });

        if let Some(d) = d {
            self.communication_times.insert(
                CommunicationType::GhostExchangeVRuntime,
                OperatorTime::from_duration(d),
            );
        }

        // Gather global FMM
        let (_, d) = optionally_time(self.timed, || {
            self.gather_global_fmm_at_root();
        });

        if let Some(d) = d {
            self.communication_times.insert(
                CommunicationType::GatherGlobalFmm,
                OperatorTime::from_duration(d),
            );
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
            self.communication_times.insert(
                CommunicationType::ScatterGlobalFmm,
                OperatorTime::from_duration(d),
            );
        }

        // Perform remainder of downward pass
        self.evaluate_downward_pass()?;
        self.evaluate_leaf_targets()?;

        Ok(())
    }
}
