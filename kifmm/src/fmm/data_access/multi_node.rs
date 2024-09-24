use mpi::{topology::SimpleCommunicator, traits::Equivalence};
use num::Float;
use rlst::RlstScalar;

use crate::{
    fmm::types::{FmmEvalType, KiFmmMulti},
    traits::{
        field::SourceToTargetData as SourceToTargetDataTrait,
        fmm::{FmmDataAccessMulti, FmmOperatorData, HomogenousKernel},
        tree::{MultiFmmTree, MultiTree},
    },
    MultiNodeFmmTree,
};

use green_kernels::traits::Kernel as KernelTrait;

impl<Scalar, Kernel, SourceToTargetData> FmmDataAccessMulti
    for KiFmmMulti<Scalar, Kernel, SourceToTargetData>
where
    Scalar: RlstScalar + Default + Float + Equivalence,
    <Scalar as RlstScalar>::Real: RlstScalar + Default + Float + Equivalence,
    Kernel: KernelTrait<T = Scalar> + HomogenousKernel + Default + Send + Sync,
    SourceToTargetData: SourceToTargetDataTrait + Send + Sync,
    Self: FmmOperatorData,
{
    type Scalar = Scalar;
    type Kernel = Kernel;
    type Tree = MultiNodeFmmTree<Scalar::Real, SimpleCommunicator>;

    fn dim(&self) -> usize {
        self.dim
    }

    fn variable_expansion_order(&self) -> bool {
        false
    }

    fn equivalent_surface_order(&self, _level: u64) -> usize {
        self.equivalent_surface_order
    }

    fn check_surface_order(&self, _level: u64) -> usize {
        self.check_surface_order
    }

    fn n_coeffs_equivalent_surface(&self, _level: u64) -> usize {
        self.n_coeffs_equivalent_surface
    }

    fn n_coeffs_check_surface(&self, _level: u64) -> usize {
        self.n_coeffs_check_surface
    }

    fn kernel(&self) -> &Self::Kernel {
        &self.kernel
    }

    fn tree(&self) -> &Self::Tree {
        &self.tree
    }

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
        key: &<<<Self::Tree as crate::traits::tree::MultiFmmTree>::Tree as crate::traits::tree::MultiTree>::SingleTree as crate::traits::tree::SingleTree>::Node,
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
}
