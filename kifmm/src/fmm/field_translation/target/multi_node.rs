use itertools::Itertools;
use rayon::prelude::*;
use std::collections::HashSet;

use crate::{
    fmm::{
        constants::L2L_MAX_BLOCK_SIZE,
        helpers::single_node::chunk_size,
        types::{FmmEvalType, KiFmmMulti},
    },
    traits::{
        field::{FieldTranslation as FieldTranslationTrait, TargetTranslation},
        fmm::{DataAccessMulti, HomogenousKernel, MetadataAccess},
        tree::{MultiFmmTree, MultiTree, SingleTree},
        types::FmmError,
    },
    tree::{constants::NSIBLINGS, types::MortonKey},
    MultiNodeFmmTree,
};
use green_kernels::traits::Kernel as KernelTrait;
use mpi::{topology::SimpleCommunicator, traits::Equivalence};
use num::Float;
use rlst::{empty_array, rlst_dynamic_array2, MultIntoResize, RawAccess, RawAccessMut, RlstScalar};

impl<Scalar, Kernel, FieldTranslation> TargetTranslation
    for KiFmmMulti<Scalar, Kernel, FieldTranslation>
where
    Scalar: RlstScalar + Default + Equivalence + Float,
    <Scalar as RlstScalar>::Real: Default + Equivalence + Float,
    FieldTranslation: FieldTranslationTrait,
    Kernel: KernelTrait<T = Scalar> + HomogenousKernel + Default + Send + Sync,
    Self: MetadataAccess
        + DataAccessMulti<
            Scalar = Scalar,
            Kernel = Kernel,
            Tree = MultiNodeFmmTree<Scalar::Real, SimpleCommunicator>,
        >,
{
    fn l2l(&self, level: u64) -> Result<(), crate::traits::types::FmmError> {
        if let Some(child_targets) = self.tree.target_tree().keys(level) {
            let parent_sources: HashSet<MortonKey<_>> =
                child_targets.iter().map(|source| source.parent()).collect();

            let mut parent_sources = parent_sources.into_iter().collect_vec();
            parent_sources.sort();
            let n_parents = parent_sources.len();
            let n_coeffs_equivalent_surface = self.n_coeffs_equivalent_surface(level);
            let n_coeffs_equivalent_surface_parent = self.n_coeffs_equivalent_surface(level - 1);

            match self.fmm_eval_type {
                FmmEvalType::Vector => {
                    let mut parent_locals = Vec::new();
                    for parent in parent_sources.iter() {
                        let parent_index_pointer = *self.level_index_pointer_locals
                            [(level - 1) as usize]
                            .get(parent)
                            .unwrap();
                        let parent_local =
                            &self.level_locals[(level - 1) as usize][parent_index_pointer];
                        parent_locals.push(parent_local);
                    }

                    let mut max_chunk_size = n_parents;
                    if max_chunk_size > L2L_MAX_BLOCK_SIZE {
                        max_chunk_size = L2L_MAX_BLOCK_SIZE
                    }
                    let chunk_size = chunk_size(n_parents, max_chunk_size);

                    let child_locals = &self.level_locals[level as usize];

                    let target_vec = &self.target_vec;

                    parent_locals
                        .par_chunks_exact(chunk_size)
                        .zip(child_locals.par_chunks_exact(NSIBLINGS * chunk_size))
                        .for_each(|(parent_local_pointer_chunk, child_local_pointers_chunk)| {
                            let mut parent_locals = rlst_dynamic_array2!(
                                Scalar,
                                [n_coeffs_equivalent_surface_parent, chunk_size]
                            );
                            for (chunk_idx, parent_local_pointer) in parent_local_pointer_chunk
                                .iter()
                                .enumerate()
                                .take(chunk_size)
                            {
                                parent_locals.data_mut()[chunk_idx
                                    * n_coeffs_equivalent_surface_parent
                                    ..(chunk_idx + 1) * n_coeffs_equivalent_surface_parent]
                                    .copy_from_slice(unsafe {
                                        std::slice::from_raw_parts_mut(
                                            parent_local_pointer.raw,
                                            n_coeffs_equivalent_surface_parent,
                                        )
                                    });
                            }

                            for (i, target_vec_i) in target_vec.iter().enumerate().take(NSIBLINGS) {
                                let tmp = empty_array::<Scalar, 2>()
                                    .simple_mult_into_resize(target_vec_i.r(), parent_locals.r());

                                for j in 0..chunk_size {
                                    let chunk_displacement = j * NSIBLINGS;
                                    let child_displacement = chunk_displacement + i;
                                    let child_local = unsafe {
                                        std::slice::from_raw_parts_mut(
                                            child_local_pointers_chunk[child_displacement].raw,
                                            n_coeffs_equivalent_surface,
                                        )
                                    };
                                    child_local
                                        .iter_mut()
                                        .zip(
                                            &tmp.data()[j * n_coeffs_equivalent_surface
                                                ..(j + 1) * n_coeffs_equivalent_surface],
                                        )
                                        .for_each(|(l, t)| *l += *t);
                                }
                            }
                        });
                }

                FmmEvalType::Matrix(_) => {
                    return Err(FmmError::Unimplemented(
                        "M2L unimplemented for matrix input with BLAS field translations"
                            .to_string(),
                    ))
                }
            }
        }

        Ok(())
    }

    fn l2p(&self) -> Result<(), crate::traits::types::FmmError> {
        if let Some(_leaves) = self.tree.target_tree().all_leaves() {
            if let Some(coordinates) = self.tree.target_tree().all_coordinates() {
                let dim = 3;
                let kernel = &self.kernel;
                let kernel_eval_size = self.kernel_eval_size;
                let kernel_eval_type = self.kernel_eval_type;
                let total_depth = self.tree.target_tree().total_depth();
                let n_coeffs_equivalent_surface = self.n_coeffs_equivalent_surface(total_depth);
                let equivalent_surface_size = n_coeffs_equivalent_surface * dim;

                match self.fmm_eval_type {
                    FmmEvalType::Vector => {
                        self.leaf_downward_equivalent_surfaces_targets
                            .par_chunks_exact(equivalent_surface_size)
                            .zip(self.leaf_locals.par_iter())
                            .zip(self.charge_index_pointer_targets.par_iter())
                            .zip(self.potentials_send_pointers.par_iter())
                            .for_each(
                                |(
                                    (
                                        (leaf_downward_equivalent_surface, leaf_locals),
                                        charge_index_pointer,
                                    ),
                                    potential_send_ptr,
                                )| {
                                    let target_coordinates_row_major = &coordinates
                                        [charge_index_pointer.0 * dim
                                            ..charge_index_pointer.1 * dim];
                                    let n_targets = target_coordinates_row_major.len() / dim;

                                    // Compute direct
                                    if n_targets > 0 {
                                        let result = unsafe {
                                            std::slice::from_raw_parts_mut(
                                                potential_send_ptr.raw,
                                                n_targets * kernel_eval_size,
                                            )
                                        };

                                        let locals = unsafe {
                                            std::slice::from_raw_parts(
                                                leaf_locals.raw,
                                                n_coeffs_equivalent_surface,
                                            )
                                        };

                                        kernel.evaluate_st(
                                            kernel_eval_type,
                                            leaf_downward_equivalent_surface,
                                            target_coordinates_row_major,
                                            locals,
                                            result,
                                        );
                                    }
                                },
                            );
                    }

                    FmmEvalType::Matrix(_) => {
                        return Err(FmmError::Unimplemented(
                            "Multinode Matrix input currently unimplemented".to_string(),
                        ))
                    }
                }
            }
        }

        Ok(())
    }

    fn m2p(&self) -> Result<(), crate::traits::types::FmmError> {
        Err(FmmError::Unimplemented("M2P unimplemented".to_string()))
    }

    fn p2p(&self) -> Result<(), crate::traits::types::FmmError> {
        if let Some(leaves) = self.tree.target_tree().all_leaves() {
            let all_target_coordinates = self.tree.target_tree().all_coordinates().unwrap();
            let all_source_coordinates = [
                self.tree.source_tree().all_coordinates().unwrap(),
                self.ghost_fmm_u.tree.source_tree.all_coordinates().unwrap(),
            ];

            let dim = 3;
            let source_leaf_to_index = &self.tree.source_tree().leaf_to_index;
            let source_leaf_to_index_ghosts = &self.ghost_fmm_u.tree.source_tree.leaf_to_index;
            let charge_index_pointer_sources = [
                &self.charge_index_pointer_sources,
                &self.ghost_fmm_u.charge_index_pointer_sources,
            ];
            let all_charges = [&self.charges, &self.ghost_fmm_u.charges];
            let kernel_eval_size = self.kernel_eval_size;
            let kernel_eval_type = self.kernel_eval_type;
            let kernel = &self.kernel;

            match self.fmm_eval_type {
                FmmEvalType::Vector => {
                    leaves
                        .into_par_iter()
                        .zip(&self.charge_index_pointer_targets)
                        .zip(&self.potentials_send_pointers)
                        .for_each(
                            |((leaf, charge_index_pointer_targets), potential_send_pointer)| {
                                let target_coordinates_row_major = &all_target_coordinates
                                    [charge_index_pointer_targets.0 * dim
                                        ..charge_index_pointer_targets.1 * dim];

                                let n_targets = target_coordinates_row_major.len() / dim;

                                if n_targets > 0 {
                                    let mut u_list = leaf.neighbors().into_iter().collect_vec();
                                    u_list.push(*leaf);

                                    let mut all_u_list_indices = Vec::new();

                                    // handle locally contained source boxes
                                    all_u_list_indices.push(
                                        u_list
                                            .iter()
                                            .filter_map(|k| source_leaf_to_index.get(k))
                                            .collect_vec(),
                                    );

                                    // handle ghost source boxes
                                    all_u_list_indices.push(
                                        u_list
                                            .iter()
                                            .filter_map(|k| source_leaf_to_index_ghosts.get(k))
                                            .collect_vec(),
                                    );

                                    for (i, u_list_indices) in all_u_list_indices.iter().enumerate()
                                    {
                                        let charges = u_list_indices
                                            .iter()
                                            .map(|&idx| {
                                                let index_pointer =
                                                    &charge_index_pointer_sources[i][*idx];
                                                &all_charges[i][index_pointer.0..index_pointer.1]
                                            })
                                            .collect_vec();

                                        let sources_coordinates = u_list_indices
                                            .iter()
                                            .map(|&idx| {
                                                let index_pointer =
                                                    &charge_index_pointer_sources[i][*idx];
                                                &all_source_coordinates[i]
                                                    [index_pointer.0 * dim..index_pointer.1 * dim]
                                            })
                                            .collect_vec();

                                        for (&charges, source_coordinates_row_major) in
                                            charges.iter().zip(sources_coordinates)
                                        {
                                            let n_sources =
                                                source_coordinates_row_major.len() / dim;

                                            if n_sources > 0 {
                                                let result = unsafe {
                                                    std::slice::from_raw_parts_mut(
                                                        potential_send_pointer.raw,
                                                        n_targets * kernel_eval_size,
                                                    )
                                                };

                                                kernel.evaluate_st(
                                                    kernel_eval_type,
                                                    source_coordinates_row_major,
                                                    target_coordinates_row_major,
                                                    charges,
                                                    result,
                                                )
                                            }
                                        }
                                    }
                                }
                            },
                        );
                }

                FmmEvalType::Matrix(_) => {
                    return Err(FmmError::Unimplemented(
                        "P2P unimplemented for matrix input in multinode".to_string(),
                    ))
                }
            }
        }

        Ok(())
    }
}
