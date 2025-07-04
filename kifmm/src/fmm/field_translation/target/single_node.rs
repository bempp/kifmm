//! Local expansion translations

use std::collections::HashSet;

use itertools::Itertools;
use rayon::prelude::*;
use rlst::{empty_array, rlst_dynamic_array2, MultIntoResize, RawAccess, RawAccessMut, RlstScalar};

use green_kernels::traits::Kernel as KernelTrait;

use crate::{
    fmm::{constants::L2L_MAX_BLOCK_SIZE, helpers::single_node::chunk_size, types::FmmEvalType},
    traits::{
        field::{FieldTranslation as FieldTranslationTrait, TargetTranslation},
        fmm::{DataAccess, HomogenousKernel, MetadataAccess},
        tree::{SingleFmmTree, SingleTree},
        types::FmmError,
    },
    tree::{constants::NSIBLINGS, types::MortonKey},
    KiFmm,
};

impl<Scalar, Kernel, FieldTranslation> TargetTranslation for KiFmm<Scalar, Kernel, FieldTranslation>
where
    Scalar: RlstScalar,
    Kernel: KernelTrait<T = Scalar> + HomogenousKernel + Send + Sync,
    FieldTranslation: FieldTranslationTrait + Send + Sync,
    <Scalar as RlstScalar>::Real: Default,
    Self: MetadataAccess + DataAccess<Scalar = Scalar, Kernel = Kernel>,
{
    fn l2l(&self, level: u64) -> Result<(), FmmError> {
        let Some(child_targets) = self.tree.target_tree().keys(level) else {
            return Err(FmmError::Failed(format!(
                "L2L failed at level {level:?}, no sources found"
            )));
        };

        let parent_sources: HashSet<MortonKey<_>> =
            child_targets.iter().map(|source| source.parent()).collect();
        let mut parent_sources = parent_sources.into_iter().collect_vec();
        parent_sources.sort();
        let n_parents = parent_sources.len();
        let operator_index = self.l2l_operator_index(level);
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
                        &self.level_locals[(level - 1) as usize][parent_index_pointer][0];
                    parent_locals.push(parent_local);
                }

                let mut max_chunk_size = n_parents;
                if max_chunk_size > L2L_MAX_BLOCK_SIZE {
                    max_chunk_size = L2L_MAX_BLOCK_SIZE
                }
                let chunk_size = chunk_size(n_parents, max_chunk_size);

                let child_locals = &self.level_locals[level as usize];

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
                            parent_locals.data_mut()[chunk_idx * n_coeffs_equivalent_surface_parent
                                ..(chunk_idx + 1) * n_coeffs_equivalent_surface_parent]
                                .copy_from_slice(unsafe {
                                    std::slice::from_raw_parts_mut(
                                        parent_local_pointer.raw,
                                        n_coeffs_equivalent_surface_parent,
                                    )
                                });
                        }

                        for i in 0..NSIBLINGS {
                            let tmp = empty_array::<Scalar, 2>().simple_mult_into_resize(
                                self.target_vec[operator_index][i].r(),
                                parent_locals.r(),
                            );

                            for j in 0..chunk_size {
                                let chunk_displacement = j * NSIBLINGS;
                                let child_displacement = chunk_displacement + i;
                                let child_local = unsafe {
                                    std::slice::from_raw_parts_mut(
                                        child_local_pointers_chunk[child_displacement][0].raw,
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
                Ok(())
            }

            FmmEvalType::Matrix(n_matvecs) => {
                let mut parent_locals = vec![Vec::new(); n_parents];
                for (parent_idx, parent) in parent_sources.iter().enumerate() {
                    for charge_vec_idx in 0..n_matvecs {
                        let parent_index_pointer = *self.level_index_pointer_locals
                            [(level - 1) as usize]
                            .get(parent)
                            .unwrap();
                        let parent_local = self.level_locals[(level - 1) as usize]
                            [parent_index_pointer][charge_vec_idx];
                        parent_locals[parent_idx].push(parent_local);
                    }
                }
                let child_locals = &self.level_locals[level as usize];

                parent_locals
                    .into_par_iter()
                    .zip(child_locals.par_chunks_exact(NSIBLINGS))
                    .for_each(|(parent_local_pointers, child_locals_pointers)| {
                        let mut parent_locals = rlst_dynamic_array2!(
                            Scalar,
                            [n_coeffs_equivalent_surface_parent, n_matvecs]
                        );

                        for (charge_vec_idx, parent_local_pointer) in
                            parent_local_pointers.iter().enumerate().take(n_matvecs)
                        {
                            let tmp = unsafe {
                                std::slice::from_raw_parts(
                                    parent_local_pointer.raw,
                                    n_coeffs_equivalent_surface_parent,
                                )
                            };
                            parent_locals.data_mut()[charge_vec_idx
                                * n_coeffs_equivalent_surface_parent
                                ..(charge_vec_idx + 1) * n_coeffs_equivalent_surface_parent]
                                .copy_from_slice(tmp);
                        }

                        for (i, child_locals_i) in
                            child_locals_pointers.iter().enumerate().take(NSIBLINGS)
                        {
                            let result_i = empty_array::<Scalar, 2>().simple_mult_into_resize(
                                self.target_vec[operator_index][i].r(),
                                parent_locals.r(),
                            );

                            for (j, child_locals_ij) in
                                child_locals_i.iter().enumerate().take(n_matvecs)
                            {
                                let child_locals_ij = unsafe {
                                    std::slice::from_raw_parts_mut(
                                        child_locals_ij.raw,
                                        n_coeffs_equivalent_surface,
                                    )
                                };
                                let result_ij = &result_i.data()[j * n_coeffs_equivalent_surface
                                    ..(j + 1) * n_coeffs_equivalent_surface];
                                child_locals_ij
                                    .iter_mut()
                                    .zip(result_ij.iter())
                                    .for_each(|(c, r)| *c += *r);
                            }
                        }
                    });

                Ok(())
            }
        }
    }

    fn l2p(&self) -> Result<(), FmmError> {
        let Some(_leaves) = self.tree.target_tree().all_leaves() else {
            return Err(FmmError::Failed(
                "L2P failed, no leaves found in target tree".to_string(),
            ));
        };

        let coordinates = self.tree.target_tree().all_coordinates().unwrap();
        let depth = self.tree.target_tree().depth();
        let n_coeffs_equivalent_surface = self.n_coeffs_equivalent_surface(depth);
        let equivalent_surface_size = n_coeffs_equivalent_surface * self.dim;

        match self.fmm_eval_type {
            FmmEvalType::Vector => {
                self.leaf_downward_equivalent_surfaces_targets
                    .par_chunks_exact(equivalent_surface_size)
                    .zip(self.leaf_locals.par_iter())
                    .zip(&self.charge_index_pointer_targets)
                    .zip(&self.potentials_send_pointers)
                    .for_each(
                        |(
                            ((leaf_downward_equivalent_surface, leaf_locals), charge_index_pointer),
                            potential_send_ptr,
                        )| {
                            let target_coordinates_row_major = &coordinates[charge_index_pointer.0
                                * self.dim
                                ..charge_index_pointer.1 * self.dim];
                            let n_targets = target_coordinates_row_major.len() / self.dim;

                            // Compute direct
                            if n_targets > 0 {
                                let result = unsafe {
                                    std::slice::from_raw_parts_mut(
                                        potential_send_ptr.raw,
                                        n_targets * self.kernel_eval_size,
                                    )
                                };

                                self.kernel.evaluate_st(
                                    self.kernel_eval_type,
                                    leaf_downward_equivalent_surface,
                                    target_coordinates_row_major,
                                    unsafe {
                                        std::slice::from_raw_parts_mut(
                                            leaf_locals[0].raw,
                                            n_coeffs_equivalent_surface,
                                        )
                                    },
                                    result,
                                );
                            }
                        },
                    );

                Ok(())
            }

            FmmEvalType::Matrix(n_matvecs) => {
                let n_leaves = self.tree.target_tree().n_leaves().unwrap();
                for i in 0..n_matvecs {
                    self.leaf_downward_equivalent_surfaces_targets
                        .par_chunks_exact(equivalent_surface_size)
                        .zip(&self.leaf_locals)
                        .zip(&self.charge_index_pointer_targets)
                        .zip(&self.potentials_send_pointers[i * n_leaves..(i + 1) * n_leaves])
                        .for_each(
                            |(
                                (
                                    (leaf_downward_equivalent_surface, leaf_locals),
                                    charge_index_pointer,
                                ),
                                potential_send_ptr,
                            )| {
                                let target_coordinates_row_major =
                                    &coordinates[charge_index_pointer.0 * self.dim
                                        ..charge_index_pointer.1 * self.dim];
                                let n_targets = target_coordinates_row_major.len() / self.dim;

                                if n_targets > 0 {
                                    let local_expansion_ptr = leaf_locals[i].raw;
                                    let local_expansion = unsafe {
                                        std::slice::from_raw_parts(
                                            local_expansion_ptr,
                                            n_coeffs_equivalent_surface,
                                        )
                                    };

                                    let result = unsafe {
                                        std::slice::from_raw_parts_mut(
                                            potential_send_ptr.raw,
                                            n_targets * self.kernel_eval_size,
                                        )
                                    };

                                    self.kernel.evaluate_st(
                                        self.kernel_eval_type,
                                        leaf_downward_equivalent_surface,
                                        target_coordinates_row_major,
                                        local_expansion,
                                        result,
                                    );
                                }
                            },
                        )
                }
                Ok(())
            }
        }
    }

    fn p2p(&self) -> Result<(), FmmError> {
        let Some(leaves) = self.tree.target_tree().all_leaves() else {
            return Err(FmmError::Failed(
                "P2P failed, no leaves found in target tree".to_string(),
            ));
        };

        let all_target_coordinates = self.tree.target_tree().all_coordinates().unwrap();
        let all_source_coordinates = self.tree.source_tree().all_coordinates().unwrap();
        let n_all_source_coordinates = all_source_coordinates.len() / self.dim;

        match self.fmm_eval_type {
            FmmEvalType::Vector => {
                leaves
                    .par_iter()
                    .zip(&self.charge_index_pointer_targets)
                    .zip(&self.potentials_send_pointers)
                    .for_each(
                        |((leaf, charge_index_pointer_targets), potential_send_pointer)| {
                            let target_coordinates_row_major = &all_target_coordinates
                                [charge_index_pointer_targets.0 * self.dim
                                    ..charge_index_pointer_targets.1 * self.dim];
                            let n_targets = target_coordinates_row_major.len() / self.dim;

                            if n_targets > 0 {
                                let mut u_list = leaf.neighbors().into_iter().collect_vec();
                                u_list.push(*leaf);

                                let u_list_indices = u_list
                                    .iter()
                                    .filter_map(|k| self.tree.source_tree.leaf_to_index.get(k))
                                    .collect_vec();

                                let charges = u_list_indices
                                    .iter()
                                    .map(|&&idx| {
                                        let index_pointer = &self.charge_index_pointer_sources[idx];
                                        &self.charges[index_pointer.0..index_pointer.1]
                                    })
                                    .collect_vec();

                                let sources_coordinates = u_list_indices
                                    .into_iter()
                                    .map(|&idx| {
                                        let index_pointer = &self.charge_index_pointer_sources[idx];
                                        &all_source_coordinates
                                            [index_pointer.0 * self.dim..index_pointer.1 * self.dim]
                                    })
                                    .collect_vec();

                                for (&charges, source_coordinates_row_major) in
                                    charges.iter().zip(sources_coordinates)
                                {
                                    let n_sources = source_coordinates_row_major.len() / self.dim;

                                    if n_sources > 0 {
                                        let result = unsafe {
                                            std::slice::from_raw_parts_mut(
                                                potential_send_pointer.raw,
                                                n_targets * self.kernel_eval_size,
                                            )
                                        };

                                        self.kernel.evaluate_st(
                                            self.kernel_eval_type,
                                            source_coordinates_row_major,
                                            target_coordinates_row_major,
                                            charges,
                                            result,
                                        )
                                    }
                                }
                            }
                        },
                    );
                Ok(())
            }

            FmmEvalType::Matrix(n_matvecs) => {
                let n_leaves = self.tree.target_tree().n_leaves().unwrap();

                for i in 0..n_matvecs {
                    leaves
                        .par_iter()
                        .zip(&self.charge_index_pointer_targets)
                        .zip(&self.potentials_send_pointers[i * n_leaves..(i + 1) * n_leaves])
                        .for_each(|((leaf, charge_index_pointer), potential_send_ptr)| {
                            let target_coordinates_row_major = &all_target_coordinates
                                [charge_index_pointer.0 * self.dim
                                    ..charge_index_pointer.1 * self.dim];
                            let n_targets = target_coordinates_row_major.len() / self.dim;

                            if n_targets > 0 {
                                let mut u_list = leaf.neighbors().into_iter().collect_vec();
                                u_list.push(*leaf);

                                let u_list_indices = u_list
                                    .iter()
                                    .filter_map(|k| self.tree.source_tree.leaf_to_index.get(k))
                                    .collect_vec();

                                let charge_vec_displacement = i * n_all_source_coordinates;
                                let charges = u_list_indices
                                    .iter()
                                    .map(|&&idx| {
                                        let index_pointer = &self.charge_index_pointer_sources[idx];
                                        &self.charges[charge_vec_displacement + index_pointer.0
                                            ..charge_vec_displacement + index_pointer.1]
                                    })
                                    .collect_vec();

                                let sources_coordinates = u_list_indices
                                    .into_iter()
                                    .map(|&idx| {
                                        let index_pointer = &self.charge_index_pointer_sources[idx];
                                        &all_source_coordinates
                                            [index_pointer.0 * self.dim..index_pointer.1 * self.dim]
                                    })
                                    .collect_vec();

                                for (&charges, source_coordinates_row_major) in
                                    charges.iter().zip(sources_coordinates)
                                {
                                    let n_sources = source_coordinates_row_major.len() / self.dim;

                                    if n_sources > 0 {
                                        let result = unsafe {
                                            std::slice::from_raw_parts_mut(
                                                potential_send_ptr.raw,
                                                n_targets * self.kernel_eval_size,
                                            )
                                        };

                                        self.kernel.evaluate_st(
                                            self.kernel_eval_type,
                                            source_coordinates_row_major,
                                            target_coordinates_row_major,
                                            charges,
                                            result,
                                        );
                                    }
                                }
                            }
                        })
                }
                Ok(())
            }
        }
    }

    fn m2p(&self) -> Result<(), FmmError> {
        Err(FmmError::Unimplemented("M2P unimplemented".to_string()))
    }
}
