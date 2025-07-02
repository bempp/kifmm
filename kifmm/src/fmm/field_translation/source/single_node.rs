//! Multipole expansion translations
use std::collections::HashSet;

use green_kernels::{traits::Kernel as KernelTrait, types::GreenKernelEvalType};
use itertools::Itertools;
use rayon::prelude::*;

use rlst::{
    empty_array, rlst_array_from_slice2, rlst_dynamic_array2, MultIntoResize, RawAccess,
    RawAccessMut, RlstScalar,
};

use crate::{
    fmm::{
        constants::{M2M_MAX_BLOCK_SIZE, P2M_MAX_BLOCK_SIZE},
        helpers::single_node::{chunk_size, homogenous_kernel_scale},
        types::{FmmEvalType, KiFmm},
    },
    traits::{
        field::{FieldTranslation as FieldTranslationTrait, SourceTranslation},
        fmm::{DataAccess, HomogenousKernel, MetadataAccess},
        tree::{SingleFmmTree, SingleTree},
        types::FmmError,
    },
    tree::constants::NSIBLINGS,
};

impl<Scalar, Kernel, FieldTranslation> SourceTranslation for KiFmm<Scalar, Kernel, FieldTranslation>
where
    Scalar: RlstScalar,
    Kernel: KernelTrait<T = Scalar> + HomogenousKernel,
    FieldTranslation: FieldTranslationTrait + Send + Sync,
    <Scalar as RlstScalar>::Real: Default,
    Self: MetadataAccess + DataAccess<Scalar = Scalar, Kernel = Kernel>,
{
    fn p2m(&self) -> Result<(), FmmError> {
        let Some(_leaves) = self.tree.source_tree.all_leaves() else {
            return Err(FmmError::Failed(
                "P2M failed, no leaves found in source tree".to_string(),
            ));
        };

        let &n_coeffs_equivalent_surface = self.n_coeffs_equivalent_surface.last().unwrap();
        let &n_coeffs_check_surface = self.n_coeffs_check_surface.last().unwrap();
        let n_leaves = self.tree.source_tree().n_leaves().unwrap();
        let check_surface_size = n_coeffs_check_surface * self.dim;

        let coordinates = self.tree.source_tree.all_coordinates().unwrap();
        let n_coordinates = coordinates.len() / self.dim;
        let depth = self.tree.source_tree().depth();
        let operator_index = self.c2e_operator_index(depth);

        match self.fmm_eval_type {
            FmmEvalType::Vector => {
                let mut check_potentials =
                    rlst_dynamic_array2!(Scalar, [n_leaves * n_coeffs_check_surface, 1]);

                // Compute check potential for each box
                check_potentials
                    .data_mut()
                    .par_chunks_exact_mut(n_coeffs_check_surface)
                    .zip(
                        self.leaf_upward_check_surfaces_sources
                            .par_chunks_exact(check_surface_size),
                    )
                    .zip(&self.charge_index_pointer_sources)
                    .for_each(
                        |((check_potential, upward_check_surface), charge_index_pointer)| {
                            let charges =
                                &self.charges[charge_index_pointer.0..charge_index_pointer.1];

                            let coordinates_row_major = &coordinates[charge_index_pointer.0
                                * self.dim
                                ..charge_index_pointer.1 * self.dim];
                            let n_sources = coordinates_row_major.len() / self.dim;

                            if n_sources > 0 {
                                self.kernel.evaluate_st(
                                    GreenKernelEvalType::Value,
                                    coordinates_row_major,
                                    upward_check_surface,
                                    charges,
                                    check_potential,
                                );
                            }
                        },
                    );

                let depth = self.tree.source_tree().depth();
                let chunk_size = chunk_size(n_leaves, P2M_MAX_BLOCK_SIZE);

                let scale = if self.kernel.is_homogenous() {
                    homogenous_kernel_scale(depth)
                } else {
                    Scalar::one()
                };

                // Use check potentials to compute the multipole expansion
                check_potentials
                    .data()
                    .par_chunks_exact(n_coeffs_check_surface * chunk_size)
                    .zip(self.leaf_multipoles.par_chunks_exact(chunk_size))
                    .for_each(|(check_potential, multipole_ptrs)| {
                        let check_potential = rlst_array_from_slice2!(
                            check_potential,
                            [n_coeffs_check_surface, chunk_size]
                        );

                        let tmp = if self.kernel.is_homogenous() {
                            let mut scaled_check_potential =
                                rlst_dynamic_array2!(Scalar, [n_coeffs_check_surface, chunk_size]);
                            scaled_check_potential.fill_from(check_potential);
                            scaled_check_potential.scale_inplace(scale);

                            empty_array::<Scalar, 2>().simple_mult_into_resize(
                                self.uc2e_inv_1[operator_index].r(),
                                empty_array::<Scalar, 2>().simple_mult_into_resize(
                                    self.uc2e_inv_2[operator_index].r(),
                                    scaled_check_potential,
                                ),
                            )
                        } else {
                            empty_array::<Scalar, 2>().simple_mult_into_resize(
                                self.uc2e_inv_1[operator_index].r(),
                                empty_array::<Scalar, 2>().simple_mult_into_resize(
                                    self.uc2e_inv_2[operator_index].r(),
                                    check_potential.r(),
                                ),
                            )
                        };

                        for (i, multipole_ptr) in multipole_ptrs.iter().enumerate().take(chunk_size)
                        {
                            let multipole = unsafe {
                                std::slice::from_raw_parts_mut(
                                    multipole_ptr[0].raw,
                                    n_coeffs_equivalent_surface,
                                )
                            };
                            multipole
                                .iter_mut()
                                .zip(
                                    &tmp.data()[i * n_coeffs_equivalent_surface
                                        ..(i + 1) * n_coeffs_equivalent_surface],
                                )
                                .for_each(|(m, t)| *m += *t);
                        }
                    });
                Ok(())
            }

            FmmEvalType::Matrix(n_matvecs) => {
                let mut check_potentials = rlst_dynamic_array2!(
                    Scalar,
                    [n_leaves * n_coeffs_check_surface * n_matvecs, 1]
                );

                // Compute the check potential for each box for each charge vector
                check_potentials
                    .data_mut()
                    .par_chunks_exact_mut(n_coeffs_check_surface * n_matvecs)
                    .zip(
                        self.leaf_upward_check_surfaces_sources
                            .par_chunks_exact(check_surface_size),
                    )
                    .zip(&self.charge_index_pointer_sources)
                    .for_each(
                        |((check_potential, upward_check_surface), charge_index_pointer)| {
                            let coordinates_row_major = &coordinates[charge_index_pointer.0
                                * self.dim
                                ..charge_index_pointer.1 * self.dim];
                            let n_sources = coordinates_row_major.len() / self.dim;

                            if n_sources > 0 {
                                for i in 0..n_matvecs {
                                    let charge_vec_displacement = i * n_coordinates;
                                    let charges_i = &self.charges[charge_vec_displacement
                                        + charge_index_pointer.0
                                        ..charge_vec_displacement + charge_index_pointer.1];

                                    let check_potential_i = &mut check_potential[i
                                        * n_coeffs_check_surface
                                        ..(i + 1) * n_coeffs_check_surface];

                                    self.kernel.evaluate_st(
                                        GreenKernelEvalType::Value,
                                        coordinates_row_major,
                                        upward_check_surface,
                                        charges_i,
                                        check_potential_i,
                                    );
                                }
                            }
                        },
                    );

                let depth = self.tree.source_tree().depth();
                let scale = if self.kernel.is_homogenous() {
                    homogenous_kernel_scale(depth)
                } else {
                    Scalar::one()
                };

                // Compute multipole expansions
                check_potentials
                    .data()
                    .par_chunks_exact(n_coeffs_check_surface * n_matvecs)
                    .zip(self.leaf_multipoles.par_iter())
                    .for_each(|(check_potential, multipole_ptrs)| {
                        let check_potential = rlst_array_from_slice2!(
                            check_potential,
                            [n_coeffs_check_surface, n_matvecs]
                        );

                        let tmp = if self.kernel.is_homogenous() {
                            let mut scaled_check_potential =
                                rlst_dynamic_array2!(Scalar, [n_coeffs_check_surface, n_matvecs]);

                            scaled_check_potential.fill_from(check_potential);
                            scaled_check_potential.scale_inplace(scale);

                            empty_array::<Scalar, 2>().simple_mult_into_resize(
                                self.uc2e_inv_1[operator_index].r(),
                                empty_array::<Scalar, 2>().simple_mult_into_resize(
                                    self.uc2e_inv_2[operator_index].r(),
                                    scaled_check_potential.r(),
                                ),
                            )
                        } else {
                            empty_array::<Scalar, 2>().simple_mult_into_resize(
                                self.uc2e_inv_1[operator_index].r(),
                                empty_array::<Scalar, 2>().simple_mult_into_resize(
                                    self.uc2e_inv_2[operator_index].r(),
                                    check_potential.r(),
                                ),
                            )
                        };
                        for (i, multipole_ptr) in multipole_ptrs.iter().enumerate().take(n_matvecs)
                        {
                            let multipole = unsafe {
                                std::slice::from_raw_parts_mut(
                                    multipole_ptr.raw,
                                    n_coeffs_equivalent_surface,
                                )
                            };
                            multipole
                                .iter_mut()
                                .zip(
                                    &tmp.data()[i * n_coeffs_equivalent_surface
                                        ..(i + 1) * n_coeffs_equivalent_surface],
                                )
                                .for_each(|(m, t)| *m += *t);
                        }
                    });

                Ok(())
            }
        }
    }

    fn m2m(&self, level: u64) -> Result<(), FmmError> {
        let Some(child_sources) = self.tree.source_tree.keys(level) else {
            return Err(FmmError::Failed(format!(
                "M2M failed at level {level:?}, no sources found"
            )));
        };

        let operator_index = self.m2m_operator_index(level);
        let n_coeffs_equivalent_surface = self.n_coeffs_equivalent_surface(level);
        let n_coeffs_equivalent_surface_parent = self.n_coeffs_equivalent_surface(level - 1);

        let parent_targets: HashSet<_> =
            child_sources.iter().map(|source| source.parent()).collect();

        let mut parent_targets = parent_targets.into_iter().collect_vec();

        parent_targets.sort();
        let n_parents = parent_targets.len();
        let child_multipoles = self.multipoles(level).unwrap();

        match self.fmm_eval_type {
            FmmEvalType::Vector => {
                let mut parent_multipoles = Vec::new();
                for parent in parent_targets.iter() {
                    let &parent_index_pointer = self.level_index_pointer_multipoles
                        [(level - 1) as usize]
                        .get(parent)
                        .unwrap();
                    let parent_multipole =
                        &self.level_multipoles[(level - 1) as usize][parent_index_pointer][0];
                    parent_multipoles.push(parent_multipole);
                }

                let max_chunk_size = if n_parents > M2M_MAX_BLOCK_SIZE {
                    M2M_MAX_BLOCK_SIZE
                } else {
                    n_parents
                };

                let chunk_size = chunk_size(n_parents, max_chunk_size);

                child_multipoles
                    .par_chunks_exact(NSIBLINGS * n_coeffs_equivalent_surface * chunk_size)
                    .zip(parent_multipoles.par_chunks_exact(chunk_size))
                    .for_each(
                        |(child_multipoles_chunk, parent_multipole_pointers_chunk)| {
                            let child_multipoles_chunk_mat = rlst_array_from_slice2!(
                                child_multipoles_chunk,
                                [n_coeffs_equivalent_surface * NSIBLINGS, chunk_size]
                            );

                            let parent_multipoles_chunk = empty_array::<Scalar, 2>()
                                .simple_mult_into_resize(
                                    self.source[operator_index].r(),
                                    child_multipoles_chunk_mat,
                                );

                            for (chunk_idx, parent_multipole_pointer) in
                                parent_multipole_pointers_chunk
                                    .iter()
                                    .enumerate()
                                    .take(chunk_size)
                            {
                                let parent_multipole = unsafe {
                                    std::slice::from_raw_parts_mut(
                                        parent_multipole_pointer.raw,
                                        n_coeffs_equivalent_surface_parent,
                                    )
                                };

                                parent_multipole
                                    .iter_mut()
                                    .zip(
                                        &parent_multipoles_chunk.data()[chunk_idx
                                            * n_coeffs_equivalent_surface_parent
                                            ..(chunk_idx + 1) * n_coeffs_equivalent_surface_parent],
                                    )
                                    .for_each(|(p, t)| *p += *t);
                            }
                        },
                    );

                Ok(())
            }

            FmmEvalType::Matrix(n_matvecs) => {
                let mut parent_multipoles = vec![Vec::new(); n_parents];

                for (parent_idx, parent) in parent_targets.iter().enumerate() {
                    for charge_vec_idx in 0..n_matvecs {
                        let parent_index_pointer = *self.level_index_pointer_multipoles
                            [(level - 1) as usize]
                            .get(parent)
                            .unwrap();
                        let parent_multipole = self.level_multipoles[(level - 1) as usize]
                            [parent_index_pointer][charge_vec_idx];
                        parent_multipoles[parent_idx].push(parent_multipole);
                    }
                }

                child_multipoles
                    .par_chunks_exact(n_matvecs * n_coeffs_equivalent_surface * NSIBLINGS)
                    .zip(parent_multipoles.into_par_iter())
                    .for_each(|(child_multipoles, parent_multipole_pointers)| {
                        for i in 0..NSIBLINGS {
                            let sibling_displacement = i * n_coeffs_equivalent_surface * n_matvecs;

                            let child_multipoles_i = rlst_array_from_slice2!(
                                &child_multipoles[sibling_displacement
                                    ..sibling_displacement
                                        + n_coeffs_equivalent_surface * n_matvecs],
                                [n_coeffs_equivalent_surface, n_matvecs]
                            );

                            let result_i = empty_array::<Scalar, 2>().simple_mult_into_resize(
                                self.source_vec[operator_index][i].r(),
                                child_multipoles_i,
                            );

                            for (j, send_ptr) in
                                parent_multipole_pointers.iter().enumerate().take(n_matvecs)
                            {
                                let raw = send_ptr.raw;
                                let parent_multipole_j = unsafe {
                                    std::slice::from_raw_parts_mut(
                                        raw,
                                        n_coeffs_equivalent_surface_parent,
                                    )
                                };
                                let result_ij = &result_i.data()[j
                                    * n_coeffs_equivalent_surface_parent
                                    ..(j + 1) * n_coeffs_equivalent_surface_parent];
                                parent_multipole_j
                                    .iter_mut()
                                    .zip(result_ij.iter())
                                    .for_each(|(p, r)| *p += *r);
                            }
                        }
                    });

                Ok(())
            }
        }
    }
}
