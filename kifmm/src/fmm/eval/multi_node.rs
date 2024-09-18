use std::time::Instant;

use crate::traits::fmm::{HomogenousKernel, MultiFmm, SourceTranslation, TargetTranslation};
use crate::traits::parallel::GhostExchange;
use crate::traits::tree::{MultiFmmTree, MultiTree};
use crate::traits::types::{FmmOperatorTime, FmmOperatorType};
use crate::traits::{
    field::SourceToTargetData as SourceToTargetDataTrait, fmm::SourceToTargetTranslation,
};
use green_kernels::traits::Kernel as KernelTrait;
use mpi::topology::SimpleCommunicator;
use mpi::traits::Equivalence;
use num::Float;
use rlst::RlstScalar;

use crate::fmm::types::{FmmEvalType, KiFmmMulti};
use crate::MultiNodeFmmTree;

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

    fn multipoles(&self, level: u64) -> Option<&[Self::Scalar]> {
        if let Some(n_sources) = self.tree().source_tree().n_keys(level) {
            let multipole_ptr = &self.level_multipoles[level as usize][0];

            unsafe {
                match self.fmm_eval_type {
                    FmmEvalType::Vector => Some(std::slice::from_raw_parts(
                        multipole_ptr.raw,
                        self.n_coeffs_equivalent_surface * n_sources,
                    )),
                    FmmEvalType::Matrix(_n) => None,
                }
            }
        } else {
            None
        }
    }

    fn multipole(
        &self,
        key: &<<<Self::Tree as MultiFmmTree>::Tree as MultiTree>::SingleTree as crate::traits::tree::SingleTree>::Node,
    ) -> Option<&[Self::Scalar]> {
        if let Some(&key_idx) = self.tree().source_tree().level_index(key) {
            let multipole_ptr = &self.level_multipoles[key.level() as usize][key_idx];

            unsafe {
                match self.fmm_eval_type {
                    FmmEvalType::Vector => Some(std::slice::from_raw_parts(
                        multipole_ptr.raw,
                        self.n_coeffs_equivalent_surface,
                    )),
                    FmmEvalType::Matrix(_n) => None,
                }
            }
        } else {
            None
        }
    }

    fn dim(&self) -> usize {
        3
    }

    fn evaluate(&mut self, timed: bool) -> Result<(), crate::traits::types::FmmError> {
        let global_depth = self.tree.source_tree().global_depth();
        let total_depth = self.tree.source_tree().total_depth();

        if timed {
            // Upward pass
            {
                let s = Instant::now();
                self.p2m()?;
                self.times
                    .push(FmmOperatorTime::from_instant(FmmOperatorType::P2M, s));

                for level in ((global_depth + 1)..=total_depth).rev() {
                    let s = Instant::now();
                    self.m2m(level)?;
                    self.times.push(FmmOperatorTime::from_instant(
                        FmmOperatorType::M2M(level),
                        s,
                    ));
                }
            }

            // Exchange ghost multipole data
            self.v_list_exchange();

            // Gather root multipoles at nominated node(s)
            self.gather_global_fmm_at_root();

            // Calculate metadata for received ghosts

            // Calculate metadata for global FMM

            // Perform upward/downward pass on nominated node(s)

            // Scatter root locals back to local tree
            self.scatter_global_fmm_from_root();

            // Downward pass
            {
                for level in global_depth..=total_depth {
                    if level > 2 {
                        let s = Instant::now();
                        self.l2l(level)?;
                        self.times.push(FmmOperatorTime::from_instant(
                            FmmOperatorType::L2L(level),
                            s,
                        ));
                    }

                    let s = Instant::now();
                    self.m2l(level)?;
                    self.times.push(FmmOperatorTime::from_instant(
                        FmmOperatorType::M2L(level),
                        s,
                    ));
                }

                // Leaf level computations
                let s = Instant::now();
                self.p2p()?;
                self.times
                    .push(FmmOperatorTime::from_instant(FmmOperatorType::P2P, s));

                let s = Instant::now();
                self.l2p()?;
                self.times
                    .push(FmmOperatorTime::from_instant(FmmOperatorType::L2P, s));
            }
        } else {
            // Upward pass
            {
                self.p2m()?;

                for level in ((global_depth + 1)..=total_depth).rev() {
                    self.m2m(level)?;
                }
            }

            // Exchange ghost multipole data
            self.v_list_exchange();

            // Gather root multipoles at nominated node(s)
            self.gather_global_fmm_at_root();

            // Calculate metadata for received ghosts

            // Calculate metadata for global FMM

            // Perform upward/downward pass on nominated node(s)

            // Scatter root locals back to local tree
            self.scatter_global_fmm_from_root();

            // Downward pass
            {
                for level in global_depth..=total_depth {
                    if level > 2 {
                        self.l2l(level)?;
                    }
                    self.m2l(level)?;
                }

                // Leaf level computations
                self.p2p()?;
                self.l2p()?;
            }
        }

        Ok(())
    }

    fn kernel<'a>(&'a self) -> &'a Self::Kernel {
        &self.kernel
    }

    fn tree<'a>(&'a self) -> &'a Self::Tree {
        &self.tree
    }
}
