use mpi::{topology::SimpleCommunicator, traits::Equivalence};
use num::Float;
use rlst::RlstScalar;

use crate::{
    fmm::types::{FmmEvalType, KiFmmMulti},
    traits::{
        field::FieldTranslation as FieldTranslationTrait,
        fmm::{DataAccessMulti, HomogenousKernel, MetadataAccess},
        tree::{MultiFmmTree, MultiTree},
    },
    MultiNodeFmmTree,
};

use green_kernels::traits::Kernel as KernelTrait;

impl<Scalar, Kernel, FieldTranslation> DataAccessMulti
    for KiFmmMulti<Scalar, Kernel, FieldTranslation>
where
    Scalar: RlstScalar + Default + Float + Equivalence,
    <Scalar as RlstScalar>::Real: RlstScalar + Default + Float + Equivalence,
    Kernel: KernelTrait<T = Scalar> + HomogenousKernel + Default + Send + Sync,
    FieldTranslation: FieldTranslationTrait + Send + Sync,
    Self: MetadataAccess,
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

    fn potential(
        &self,
        leaf: &<<<Self::Tree as MultiFmmTree>::Tree as MultiTree>::SingleTree as crate::traits::tree::SingleTree>::Node,
    ) -> Option<Vec<&[Self::Scalar]>> {
        if let Some(&leaf_idx) = self.tree.target_tree().leaf_index(leaf) {
            let (l, r) = self.charge_index_pointer_targets[leaf_idx];

            match self.fmm_eval_type {
                FmmEvalType::Matrix(_) => None,
                FmmEvalType::Vector => Some(vec![
                    &self.potentials[l * self.kernel_eval_size..r * self.kernel_eval_size],
                ]),
            }
        } else {
            None
        }
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

    fn locals(&self, level: u64) -> Option<&[Self::Scalar]> {
        if let Some(n_targets) = self.tree().target_tree().n_keys(level) {
            let local_ptr = &self.level_locals[level as usize][0];

            unsafe {
                match self.fmm_eval_type {
                    FmmEvalType::Vector => Some(std::slice::from_raw_parts(
                        local_ptr.raw,
                        self.n_coeffs_equivalent_surface * n_targets,
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

    fn multipole_mut(
        &self,
        key: &<<<Self::Tree as MultiFmmTree>::Tree as MultiTree>::SingleTree as crate::traits::tree::SingleTree>::Node,
    ) -> Option<&mut [Self::Scalar]> {
        if let Some(&key_idx) = self.tree().source_tree().level_index(key) {
            let multipole_ptr = &self.level_multipoles[key.level() as usize][key_idx];

            unsafe {
                match self.fmm_eval_type {
                    FmmEvalType::Vector => Some(std::slice::from_raw_parts_mut(
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

    fn local(
        &self,
        key: &<<<Self::Tree as MultiFmmTree>::Tree as MultiTree>::SingleTree as crate::traits::tree::SingleTree>::Node,
    ) -> Option<&[Self::Scalar]> {
        if let Some(&key_idx) = self.tree().target_tree().level_index(key) {
            let local_ptr = &self.level_locals[key.level() as usize][key_idx];

            unsafe {
                match self.fmm_eval_type {
                    FmmEvalType::Vector => Some(std::slice::from_raw_parts(
                        local_ptr.raw,
                        self.n_coeffs_equivalent_surface,
                    )),
                    FmmEvalType::Matrix(_n) => None,
                }
            }
        } else {
            None
        }
    }

    fn local_mut(
        &self,
        key: &<<<Self::Tree as MultiFmmTree>::Tree as MultiTree>::SingleTree as crate::traits::tree::SingleTree>::Node,
    ) -> Option<&mut [Self::Scalar]> {
        if let Some(&key_idx) = self.tree().target_tree().level_index(key) {
            let local_ptr = &self.level_locals[key.level() as usize][key_idx];

            unsafe {
                match self.fmm_eval_type {
                    FmmEvalType::Vector => Some(std::slice::from_raw_parts_mut(
                        local_ptr.raw,
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
