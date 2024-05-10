use std::collections::HashSet;

use green_kernels::{traits::Kernel, types::EvalType};
use itertools::Itertools;
use rayon::prelude::*;
use rlst::{
    empty_array, rlst_array_from_slice2, rlst_dynamic_array2, MultIntoResize, RawAccess,
    RawAccessMut,
};

use crate::{
    fmm::{
        constants::{M2M_MAX_CHUNK_SIZE, P2M_MAX_CHUNK_SIZE},
        helpers::chunk_size,
        types::{FmmEvalType, KiFmmMetalLaplace},
    },
    traits::{
        fmm::{FmmOperatorData, HomogenousKernel, SourceTranslation},
        tree::{FmmTree, Tree},
        types::FmmError,
    },
    tree::constants::NSIBLINGS,
};

impl SourceTranslation for KiFmmMetalLaplace {
    fn p2m(&self) -> Result<(), crate::traits::types::FmmError> {
        let Some(_leaves) = self.tree.source_tree.all_leaves() else {
            return Err(FmmError::Failed(
                "P2M failed, no leaves found in source tree".to_string(),
            ));
        };

        let dim = self.dim.clone();
        let ncoeffs = self.ncoeffs.clone();
        let n_leaves = self.tree.source_tree().n_leaves().unwrap();
        let surface_size = ncoeffs * dim;
        let coordinates = self.tree.source_tree.all_coordinates().unwrap();
        let all_charges = &self.charges;
        let ncoordinates = coordinates.len() / dim;
        let depth = self.tree.source_tree().depth();
        let operator_index = self.c2e_operator_index(depth);
        let kernel = self.kernel.clone();
        let uc2e_inv_1 = &self.uc2e_inv_1[operator_index];
        let uc2e_inv_2 = &self.uc2e_inv_2[operator_index];

        match self.fmm_eval_type {
            FmmEvalType::Vector => {
                let mut check_potentials = rlst_dynamic_array2!(f32, [n_leaves * ncoeffs, 1]);

                // Compute check potential for each box
                check_potentials
                    .data_mut()
                    .par_chunks_exact_mut(ncoeffs)
                    .zip(
                        self.leaf_upward_surfaces_sources
                            .par_chunks_exact(surface_size),
                    )
                    .zip(&self.charge_index_pointer_sources)
                    .for_each(
                        |((check_potential, upward_check_surface), charge_index_pointer)| {
                            let charges =
                                &all_charges[charge_index_pointer.0..charge_index_pointer.1];

                            let coordinates_row_major = &coordinates
                                [charge_index_pointer.0 * dim..charge_index_pointer.1 * dim];

                            let nsources = coordinates_row_major.len() / dim;
                            if nsources > 0 {
                                let coordinates_row_major = rlst_array_from_slice2!(
                                    coordinates_row_major,
                                    [nsources, dim],
                                    [dim, 1]
                                );
                                let mut coordinates_col_major =
                                    rlst_dynamic_array2!(f32, [nsources, dim]);
                                coordinates_col_major.fill_from(coordinates_row_major.view());

                                kernel.evaluate_st(
                                    EvalType::Value,
                                    coordinates_col_major.data(),
                                    upward_check_surface,
                                    charges,
                                    check_potential,
                                );
                            }
                        },
                    );

                // Use check potentials to compute the multipole expansion
                let chunk_size = chunk_size(n_leaves, P2M_MAX_CHUNK_SIZE);
                check_potentials
                    .data()
                    .par_chunks_exact(ncoeffs * chunk_size)
                    .zip(self.leaf_multipoles.par_chunks_exact(chunk_size))
                    .zip(
                        self.leaf_scales_sources
                            .par_chunks_exact(ncoeffs * chunk_size),
                    )
                    .for_each(|((check_potential, multipole_ptrs), scale)| {
                        let check_potential =
                            rlst_array_from_slice2!(check_potential, [ncoeffs, chunk_size]);

                        let tmp = if kernel.is_homogenous() {
                            let mut scaled_check_potential =
                                rlst_dynamic_array2!(f32, [ncoeffs, chunk_size]);
                            scaled_check_potential.fill_from(check_potential);
                            scaled_check_potential.scale_inplace(scale[0]);

                            empty_array::<f32, 2>().simple_mult_into_resize(
                                uc2e_inv_1.view(),
                                empty_array::<f32, 2>().simple_mult_into_resize(
                                    uc2e_inv_2.view(),
                                    scaled_check_potential,
                                ),
                            )
                        } else {
                            empty_array::<f32, 2>().simple_mult_into_resize(
                                uc2e_inv_1.view(),
                                empty_array::<f32, 2>().simple_mult_into_resize(
                                    uc2e_inv_2.view(),
                                    check_potential.view(),
                                ),
                            )
                        };

                        for (i, multipole_ptr) in multipole_ptrs.iter().enumerate().take(chunk_size)
                        {
                            let multipole = unsafe {
                                std::slice::from_raw_parts_mut(multipole_ptr[0].raw, ncoeffs)
                            };
                            multipole
                                .iter_mut()
                                .zip(&tmp.data()[i * ncoeffs..(i + 1) * ncoeffs])
                                .for_each(|(m, t)| *m += *t);
                        }
                    });

                Ok(())
            }

            FmmEvalType::Matrix(nmatvecs) => {
                let mut check_potentials =
                    rlst_dynamic_array2!(f32, [n_leaves * ncoeffs * nmatvecs, 1]);

                // Compute the check potential for each box for each charge vector
                check_potentials
                    .data_mut()
                    .par_chunks_exact_mut(ncoeffs * nmatvecs)
                    .zip(
                        self.leaf_upward_surfaces_sources
                            .par_chunks_exact(surface_size),
                    )
                    .zip(&self.charge_index_pointer_sources)
                    .for_each(
                        |((check_potential, upward_check_surface), charge_index_pointer)| {
                            let coordinates_row_major = &coordinates
                                [charge_index_pointer.0 * dim..charge_index_pointer.1 * dim];
                            let nsources = coordinates_row_major.len() / dim;

                            if nsources > 0 {
                                for i in 0..nmatvecs {
                                    let charge_vec_displacement = i * ncoordinates;
                                    let charges_i = &all_charges[charge_vec_displacement
                                        + charge_index_pointer.0
                                        ..charge_vec_displacement + charge_index_pointer.1];

                                    let check_potential_i =
                                        &mut check_potential[i * ncoeffs..(i + 1) * ncoeffs];

                                    let coordinates_mat = rlst_array_from_slice2!(
                                        coordinates_row_major,
                                        [nsources, dim],
                                        [dim, 1]
                                    );
                                    let mut coordinates_col_major =
                                        rlst_dynamic_array2!(f32, [nsources, dim]);
                                    coordinates_col_major.fill_from(coordinates_mat.view());

                                    kernel.evaluate_st(
                                        EvalType::Value,
                                        coordinates_col_major.data(),
                                        upward_check_surface,
                                        charges_i,
                                        check_potential_i,
                                    );
                                }
                            }
                        },
                    );

                // Compute multipole expansions
                check_potentials
                    .data()
                    .par_chunks_exact(ncoeffs * nmatvecs)
                    .zip(self.leaf_multipoles.par_iter())
                    .zip(self.leaf_scales_sources.par_chunks_exact(ncoeffs))
                    .for_each(|((check_potential, multipole_ptrs), scale)| {
                        let check_potential =
                            rlst_array_from_slice2!(check_potential, [ncoeffs, nmatvecs]);

                        let tmp = if kernel.is_homogenous() {
                            let mut scaled_check_potential =
                                rlst_dynamic_array2!(f32, [ncoeffs, nmatvecs]);

                            scaled_check_potential.fill_from(check_potential);
                            scaled_check_potential.scale_inplace(scale[0]);

                            empty_array::<f32, 2>().simple_mult_into_resize(
                                uc2e_inv_1.view(),
                                empty_array::<f32, 2>().simple_mult_into_resize(
                                    uc2e_inv_2.view(),
                                    scaled_check_potential.view(),
                                ),
                            )
                        } else {
                            empty_array::<f32, 2>().simple_mult_into_resize(
                                uc2e_inv_1.view(),
                                empty_array::<f32, 2>().simple_mult_into_resize(
                                    uc2e_inv_2.view(),
                                    check_potential.view(),
                                ),
                            )
                        };

                        for (i, multipole_ptr) in multipole_ptrs.iter().enumerate().take(nmatvecs) {
                            let multipole = unsafe {
                                std::slice::from_raw_parts_mut(multipole_ptr.raw, ncoeffs)
                            };
                            multipole
                                .iter_mut()
                                .zip(&tmp.data()[i * ncoeffs..(i + 1) * ncoeffs])
                                .for_each(|(m, t)| *m += *t);
                        }
                    });
                Ok(())
            }
        }
    }

    fn m2m(&self, level: u64) -> Result<(), crate::traits::types::FmmError> {
        let Some(child_sources) = self.tree.source_tree.keys(level) else {
            return Err(FmmError::Failed(format!(
                "M2M failed at level {:?}, no sources found",
                level
            )));
        };

        let nchild_sources = self.tree.source_tree().n_keys(level).unwrap();
        let min = &child_sources[0];
        let max = &child_sources[nchild_sources - 1];
        let min_idx = self.tree.source_tree.index(min).unwrap();
        let max_idx = self.tree.source_tree.index(max).unwrap();
        let operator_index = self.m2m_operator_index(level);
        let ncoeffs = self.ncoeffs.clone();
        let dim = self.dim.clone();
        let kernel = &self.kernel;
        let source = &self.source[operator_index];
        let source_vec = &self.source_vec;

        let parent_targets: HashSet<_> =
            child_sources.iter().map(|source| source.parent()).collect();

        let mut parent_targets = parent_targets.into_iter().collect_vec();

        parent_targets.sort();
        let nparents = parent_targets.len();

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

                let child_multipoles =
                    &self.multipoles[min_idx * self.ncoeffs..(max_idx + 1) * self.ncoeffs];

                let mut max_chunk_size = nparents;
                if max_chunk_size > M2M_MAX_CHUNK_SIZE {
                    max_chunk_size = M2M_MAX_CHUNK_SIZE
                }
                let chunk_size = chunk_size(nparents, max_chunk_size);

                child_multipoles
                    .par_chunks_exact(NSIBLINGS * self.ncoeffs * chunk_size)
                    .zip(parent_multipoles.par_chunks_exact(chunk_size))
                    .for_each(
                        |(child_multipoles_chunk, parent_multipole_pointers_chunk)| {
                            let child_multipoles_chunk_mat = rlst_array_from_slice2!(
                                child_multipoles_chunk,
                                [ncoeffs * NSIBLINGS, chunk_size]
                            );

                            let parent_multipoles_chunk = empty_array::<f32, 2>()
                                .simple_mult_into_resize(source.view(), child_multipoles_chunk_mat);

                            for (chunk_idx, parent_multipole_pointer) in
                                parent_multipole_pointers_chunk
                                    .iter()
                                    .enumerate()
                                    .take(chunk_size)
                            {
                                let parent_multipole = unsafe {
                                    std::slice::from_raw_parts_mut(
                                        parent_multipole_pointer.raw,
                                        ncoeffs,
                                    )
                                };

                                parent_multipole
                                    .iter_mut()
                                    .zip(
                                        &parent_multipoles_chunk.data()
                                            [chunk_idx * ncoeffs..(chunk_idx + 1) * ncoeffs],
                                    )
                                    .for_each(|(p, t)| *p += *t);
                            }
                        },
                    );

                Ok(())
            }

            FmmEvalType::Matrix(nmatvecs) => {
                let mut parent_multipoles = vec![Vec::new(); nparents];

                for (parent_idx, parent) in parent_targets.iter().enumerate() {
                    for charge_vec_idx in 0..nmatvecs {
                        let parent_index_pointer = *self.level_index_pointer_multipoles
                            [(level - 1) as usize]
                            .get(parent)
                            .unwrap();
                        let parent_multipole = self.level_multipoles[(level - 1) as usize]
                            [parent_index_pointer][charge_vec_idx];
                        parent_multipoles[parent_idx].push(parent_multipole);
                    }
                }

                let min_key_displacement = min_idx * self.ncoeffs * nmatvecs;
                let max_key_displacement = (max_idx + 1) * self.ncoeffs * nmatvecs;

                let child_multipoles = &self.multipoles[min_key_displacement..max_key_displacement];

                child_multipoles
                    .par_chunks_exact(nmatvecs * self.ncoeffs * NSIBLINGS)
                    .zip(parent_multipoles.into_par_iter())
                    .for_each(|(child_multipoles, parent_multipole_pointers)| {
                        for i in 0..NSIBLINGS {
                            let sibling_displacement = i * ncoeffs * nmatvecs;

                            let child_multipoles_i = rlst_array_from_slice2!(
                                &child_multipoles[sibling_displacement
                                    ..sibling_displacement + ncoeffs * nmatvecs],
                                [ncoeffs, nmatvecs]
                            );

                            let result_i = empty_array::<f32, 2>().simple_mult_into_resize(
                                source_vec[operator_index][i].view(),
                                child_multipoles_i,
                            );

                            for (j, send_ptr) in
                                parent_multipole_pointers.iter().enumerate().take(nmatvecs)
                            {
                                let raw = send_ptr.raw;
                                let parent_multipole_j =
                                    unsafe { std::slice::from_raw_parts_mut(raw, ncoeffs) };
                                let result_ij = &result_i.data()[j * ncoeffs..(j + 1) * ncoeffs];
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
