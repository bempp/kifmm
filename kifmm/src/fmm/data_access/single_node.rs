use rlst::RlstScalar;

use crate::{
    fmm::{
        helpers::single_node::{
            leaf_expansion_pointers_single_node, level_expansion_pointers_single_node, map_charges,
            potential_pointers_single_node,
        },
        types::FmmEvalType,
        KiFmm,
    },
    traits::{
        field::FieldTranslation as FieldTranslationTrait,
        fmm::{DataAccess, HomogenousKernel, MetadataAccess},
        tree::{SingleFmmTree, SingleTree},
    },
    SingleNodeFmmTree,
};

use green_kernels::traits::Kernel as KernelTrait;

impl<Scalar, Kernel, FieldTranslation> DataAccess for KiFmm<Scalar, Kernel, FieldTranslation>
where
    Scalar: RlstScalar + Default,
    <Scalar as RlstScalar>::Real: Default,
    Kernel: KernelTrait<T = Scalar> + HomogenousKernel + Default + Send + Sync,
    FieldTranslation: FieldTranslationTrait + Send + Sync,
    Self: MetadataAccess,
{
    type Scalar = Scalar;
    type Kernel = Kernel;
    type Tree = SingleNodeFmmTree<Scalar::Real>;

    fn dim(&self) -> usize {
        self.dim
    }

    fn variable_expansion_order(&self) -> bool {
        self.variable_expansion_order
    }

    fn equivalent_surface_order(&self, level: u64) -> usize {
        self.equivalent_surface_order[self.expansion_index(level)]
    }

    fn check_surface_order(&self, level: u64) -> usize {
        self.check_surface_order[self.expansion_index(level)]
    }

    fn n_coeffs_equivalent_surface(&self, level: u64) -> usize {
        self.n_coeffs_equivalent_surface[self.expansion_index(level)]
    }

    fn n_coeffs_check_surface(&self, level: u64) -> usize {
        self.n_coeffs_check_surface[self.expansion_index(level)]
    }

    fn kernel(&self) -> &Self::Kernel {
        &self.kernel
    }

    fn tree(&self) -> &Self::Tree {
        &self.tree
    }

    fn multipole(
        &self,
        key: &<<Self::Tree as crate::traits::tree::SingleFmmTree>::Tree as crate::traits::tree::SingleTree>::Node,
    ) -> Option<&[Self::Scalar]> {
        if let Some(&key_idx) = self.tree().source_tree().level_index(key) {
            let multipole_ptr = &self.level_multipoles[key.level() as usize][key_idx][0];

            unsafe {
                match self.fmm_eval_type {
                    FmmEvalType::Vector => Some(std::slice::from_raw_parts(
                        multipole_ptr.raw,
                        self.n_coeffs_equivalent_surface(key.level()),
                    )),
                    FmmEvalType::Matrix(n_matvecs) => Some(std::slice::from_raw_parts(
                        multipole_ptr.raw,
                        self.n_coeffs_equivalent_surface(key.level()) * n_matvecs,
                    )),
                }
            }
        } else {
            None
        }
    }

    fn multipoles(&self, level: u64) -> Option<&[Self::Scalar]> {
        let multipole_ptr = &self.level_multipoles[level as usize][0][0];
        let n_sources = self.tree.source_tree.n_keys(level).unwrap();
        unsafe {
            match self.fmm_eval_type {
                FmmEvalType::Vector => Some(std::slice::from_raw_parts(
                    multipole_ptr.raw,
                    self.n_coeffs_equivalent_surface(level) * n_sources,
                )),
                FmmEvalType::Matrix(n_matvecs) => Some(std::slice::from_raw_parts(
                    multipole_ptr.raw,
                    self.n_coeffs_equivalent_surface(level) * n_sources * n_matvecs,
                )),
            }
        }
    }

    fn local(
        &self,
        key: &<<Self::Tree as SingleFmmTree>::Tree as SingleTree>::Node,
    ) -> Option<&[Self::Scalar]> {
        if let Some(&key_idx) = self.tree().target_tree().level_index(key) {
            let local_ptr = &self.level_locals[key.level() as usize][key_idx][0];

            unsafe {
                match self.fmm_eval_type {
                    FmmEvalType::Vector => Some(std::slice::from_raw_parts(
                        local_ptr.raw,
                        self.n_coeffs_equivalent_surface(key.level()),
                    )),
                    FmmEvalType::Matrix(n_matvecs) => Some(std::slice::from_raw_parts(
                        local_ptr.raw,
                        self.n_coeffs_equivalent_surface(key.level()) * n_matvecs,
                    )),
                }
            }
        } else {
            None
        }
    }

    fn locals(&self, level: u64) -> Option<&[Self::Scalar]> {
        let local_ptr = &self.level_locals[level as usize][0][0];
        let n_targets = self.tree.target_tree.n_keys(level).unwrap();
        unsafe {
            match self.fmm_eval_type {
                FmmEvalType::Vector => Some(std::slice::from_raw_parts(
                    local_ptr.raw,
                    self.n_coeffs_equivalent_surface(level) * n_targets,
                )),
                FmmEvalType::Matrix(n_matvecs) => Some(std::slice::from_raw_parts(
                    local_ptr.raw,
                    self.n_coeffs_equivalent_surface(level) * n_targets * n_matvecs,
                )),
            }
        }
    }

    fn potential(
        &self,
        leaf: &<<Self::Tree as SingleFmmTree>::Tree as SingleTree>::Node,
    ) -> Option<Vec<&[Self::Scalar]>> {
        if let Some(&leaf_idx) = self.tree.target_tree().leaf_index(leaf) {
            let (l, r) = self.charge_index_pointer_targets[leaf_idx];
            let n_targets = r - l;

            match self.fmm_eval_type {
                FmmEvalType::Vector => Some(vec![
                    &self.potentials[l * self.kernel_eval_size..r * self.kernel_eval_size],
                ]),
                FmmEvalType::Matrix(n_matvecs) => {
                    let n_leaves = self.tree.target_tree().n_leaves().unwrap();
                    let mut slices = Vec::new();
                    for eval_idx in 0..n_matvecs {
                        let potentials_pointer =
                            self.potentials_send_pointers[eval_idx * n_leaves + leaf_idx].raw;
                        slices.push(unsafe {
                            std::slice::from_raw_parts(
                                potentials_pointer,
                                n_targets * self.kernel_eval_size,
                            )
                        });
                    }
                    Some(slices)
                }
            }
        } else {
            None
        }
    }

    fn potentials(&self) -> Option<&Vec<Self::Scalar>> {
        Some(&self.potentials)
    }

    fn clear(&mut self, charges: &[<Self as DataAccess>::Scalar]) {
        let n_source_points = self.tree().source_tree().n_coordinates_tot().unwrap();
        let n_matvecs = charges.len() / n_source_points;

        // Clear buffers and set new buffers
        self.multipoles = vec![Scalar::default(); self.multipoles.len()];
        self.locals = vec![Scalar::default(); self.locals.len()];
        self.potentials = vec![Scalar::default(); self.potentials.len()];
        self.charges = vec![Scalar::default(); self.charges.len()];

        // Recreate mutable pointers for new buffers
        let potentials_send_pointers = potential_pointers_single_node(
            self.tree.target_tree(),
            n_matvecs,
            self.kernel_eval_size,
            &self.potentials,
        );

        let leaf_multipoles = leaf_expansion_pointers_single_node(
            self.tree().source_tree(),
            &self.n_coeffs_equivalent_surface,
            n_matvecs,
            &self.multipoles,
        );

        let level_multipoles = level_expansion_pointers_single_node(
            self.tree().source_tree(),
            &self.n_coeffs_equivalent_surface,
            n_matvecs,
            &self.multipoles,
        );

        let level_locals = level_expansion_pointers_single_node(
            self.tree().target_tree(),
            &self.n_coeffs_equivalent_surface,
            n_matvecs,
            &self.locals,
        );

        let leaf_locals = leaf_expansion_pointers_single_node(
            self.tree().target_tree(),
            &self.n_coeffs_equivalent_surface,
            n_matvecs,
            &self.locals,
        );

        // Set mutable pointers
        self.level_locals = level_locals;
        self.level_multipoles = level_multipoles;
        self.leaf_locals = leaf_locals;
        self.leaf_multipoles = leaf_multipoles;
        self.potentials_send_pointers = potentials_send_pointers;

        // Set new charges
        self.charges = map_charges(
            self.tree.source_tree().all_global_indices().unwrap(),
            charges,
            n_matvecs,
        )
        .to_vec();
    }
}
