//! Multipole expansion translations
use std::collections::HashSet;

use green_kernels::{traits::Kernel as KernelTrait, types::EvalType};
use itertools::Itertools;
use mpi::{
    topology::SimpleCommunicator,
    traits::{Communicator, Equivalence},
};
use num::Float;
use rayon::prelude::*;

use rlst::{
    empty_array, rlst_array_from_slice2, rlst_dynamic_array2, MultIntoResize, RawAccess,
    RawAccessMut, RlstScalar,
};

use crate::{
    fmm::{
        constants::{M2M_MAX_BLOCK_SIZE, P2M_MAX_BLOCK_SIZE},
        helpers::{chunk_size, homogenous_kernel_scale},
        types::{FmmEvalType, KiFmm, KiFmmMultiNode},
    },
    traits::{
        field::SourceToTargetData as SourceToTargetDataTrait,
        fmm::{FmmOperatorData, HomogenousKernel, MultiNodeFmm, SourceTranslation},
        tree::{SingleNodeFmmTreeTrait, SingleNodeTreeTrait},
        types::FmmError,
    },
    tree::constants::NSIBLINGS,
    Fmm,
};

impl<Scalar, Kernel, SourceToTargetData> SourceTranslation
    for KiFmm<Scalar, Kernel, SourceToTargetData>
where
    Scalar: RlstScalar,
    Kernel: KernelTrait<T = Scalar> + HomogenousKernel,
    SourceToTargetData: SourceToTargetDataTrait + Send + Sync,
    <Scalar as RlstScalar>::Real: Default,
    Self: FmmOperatorData + Fmm<Scalar = Scalar>,
{
    fn p2m(&self) -> Result<(), FmmError> {
        let Some(_leaves) = self.tree.source_tree.all_leaves() else {
            return Err(FmmError::Failed(
                "P2M failed, no leaves found in source tree".to_string(),
            ));
        };

        let &ncoeffs_equivalent_surface = self.ncoeffs_equivalent_surface.last().unwrap();
        let &ncoeffs_check_surface = self.ncoeffs_check_surface.last().unwrap();
        let n_leaves = self.tree.source_tree().n_leaves().unwrap();
        let check_surface_size = ncoeffs_check_surface * self.dim;

        let coordinates = self.tree.source_tree.all_coordinates().unwrap();
        let ncoordinates = coordinates.len() / self.dim;
        let depth = self.tree.source_tree().depth();
        let operator_index = self.c2e_operator_index(depth);

        match self.fmm_eval_type {
            FmmEvalType::Vector => {
                let mut check_potentials =
                    rlst_dynamic_array2!(Scalar, [n_leaves * ncoeffs_check_surface, 1]);

                // Compute check potential for each box
                check_potentials
                    .data_mut()
                    .par_chunks_exact_mut(ncoeffs_check_surface)
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
                            let nsources = coordinates_row_major.len() / self.dim;

                            if nsources > 0 {
                                self.kernel.evaluate_st(
                                    EvalType::Value,
                                    coordinates_row_major,
                                    upward_check_surface,
                                    charges,
                                    check_potential,
                                );
                            }
                        },
                    );

                // Use check potentials to compute the multipole expansion
                let chunk_size = chunk_size(n_leaves, P2M_MAX_BLOCK_SIZE);
                check_potentials
                    .data()
                    .par_chunks_exact(ncoeffs_check_surface * chunk_size)
                    .zip(self.leaf_multipoles.par_chunks_exact(chunk_size))
                    .zip(
                        self.leaf_scales_sources
                            .par_chunks_exact(ncoeffs_check_surface * chunk_size),
                    )
                    .for_each(|((check_potential, multipole_ptrs), scale)| {
                        let check_potential = rlst_array_from_slice2!(
                            check_potential,
                            [ncoeffs_check_surface, chunk_size]
                        );

                        let tmp = if self.kernel.is_homogenous() {
                            let mut scaled_check_potential =
                                rlst_dynamic_array2!(Scalar, [ncoeffs_check_surface, chunk_size]);
                            scaled_check_potential.fill_from(check_potential);
                            scaled_check_potential.scale_inplace(scale[0]);

                            empty_array::<Scalar, 2>().simple_mult_into_resize(
                                self.uc2e_inv_1[operator_index].view(),
                                empty_array::<Scalar, 2>().simple_mult_into_resize(
                                    self.uc2e_inv_2[operator_index].view(),
                                    scaled_check_potential,
                                ),
                            )
                        } else {
                            empty_array::<Scalar, 2>().simple_mult_into_resize(
                                self.uc2e_inv_1[operator_index].view(),
                                empty_array::<Scalar, 2>().simple_mult_into_resize(
                                    self.uc2e_inv_2[operator_index].view(),
                                    check_potential.view(),
                                ),
                            )
                        };

                        for (i, multipole_ptr) in multipole_ptrs.iter().enumerate().take(chunk_size)
                        {
                            let multipole = unsafe {
                                std::slice::from_raw_parts_mut(
                                    multipole_ptr[0].raw,
                                    ncoeffs_equivalent_surface,
                                )
                            };
                            multipole
                                .iter_mut()
                                .zip(
                                    &tmp.data()[i * ncoeffs_equivalent_surface
                                        ..(i + 1) * ncoeffs_equivalent_surface],
                                )
                                .for_each(|(m, t)| *m += *t);
                        }
                    });
                Ok(())
            }

            FmmEvalType::Matrix(nmatvecs) => {
                let mut check_potentials =
                    rlst_dynamic_array2!(Scalar, [n_leaves * ncoeffs_check_surface * nmatvecs, 1]);

                // Compute the check potential for each box for each charge vector
                check_potentials
                    .data_mut()
                    .par_chunks_exact_mut(ncoeffs_check_surface * nmatvecs)
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
                            let nsources = coordinates_row_major.len() / self.dim;

                            if nsources > 0 {
                                for i in 0..nmatvecs {
                                    let charge_vec_displacement = i * ncoordinates;
                                    let charges_i = &self.charges[charge_vec_displacement
                                        + charge_index_pointer.0
                                        ..charge_vec_displacement + charge_index_pointer.1];

                                    let check_potential_i = &mut check_potential[i
                                        * ncoeffs_check_surface
                                        ..(i + 1) * ncoeffs_check_surface];

                                    self.kernel.evaluate_st(
                                        EvalType::Value,
                                        coordinates_row_major,
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
                    .par_chunks_exact(ncoeffs_check_surface * nmatvecs)
                    .zip(self.leaf_multipoles.par_iter())
                    .zip(
                        self.leaf_scales_sources
                            .par_chunks_exact(ncoeffs_check_surface),
                    )
                    .for_each(|((check_potential, multipole_ptrs), scale)| {
                        let check_potential = rlst_array_from_slice2!(
                            check_potential,
                            [ncoeffs_check_surface, nmatvecs]
                        );

                        let tmp = if self.kernel.is_homogenous() {
                            let mut scaled_check_potential =
                                rlst_dynamic_array2!(Scalar, [ncoeffs_check_surface, nmatvecs]);

                            scaled_check_potential.fill_from(check_potential);
                            scaled_check_potential.scale_inplace(scale[0]);

                            empty_array::<Scalar, 2>().simple_mult_into_resize(
                                self.uc2e_inv_1[operator_index].view(),
                                empty_array::<Scalar, 2>().simple_mult_into_resize(
                                    self.uc2e_inv_2[operator_index].view(),
                                    scaled_check_potential.view(),
                                ),
                            )
                        } else {
                            empty_array::<Scalar, 2>().simple_mult_into_resize(
                                self.uc2e_inv_1[operator_index].view(),
                                empty_array::<Scalar, 2>().simple_mult_into_resize(
                                    self.uc2e_inv_2[operator_index].view(),
                                    check_potential.view(),
                                ),
                            )
                        };
                        for (i, multipole_ptr) in multipole_ptrs.iter().enumerate().take(nmatvecs) {
                            let multipole = unsafe {
                                std::slice::from_raw_parts_mut(
                                    multipole_ptr.raw,
                                    ncoeffs_equivalent_surface,
                                )
                            };
                            multipole
                                .iter_mut()
                                .zip(
                                    &tmp.data()[i * ncoeffs_equivalent_surface
                                        ..(i + 1) * ncoeffs_equivalent_surface],
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
                "M2M failed at level {:?}, no sources found",
                level
            )));
        };

        let operator_index = self.m2m_operator_index(level);
        let ncoeffs_equivalent_surface = self.ncoeffs_equivalent_surface(level);
        let ncoeffs_equivalent_surface_parent = self.ncoeffs_equivalent_surface(level - 1);

        let parent_targets: HashSet<_> =
            child_sources.iter().map(|source| source.parent()).collect();

        let mut parent_targets = parent_targets.into_iter().collect_vec();
        parent_targets.sort();

        let nparents = parent_targets.len();
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

                let mut max_chunk_size = nparents;
                if max_chunk_size > M2M_MAX_BLOCK_SIZE {
                    max_chunk_size = M2M_MAX_BLOCK_SIZE
                }
                let chunk_size = chunk_size(nparents, max_chunk_size);

                child_multipoles
                    .par_chunks_exact(NSIBLINGS * ncoeffs_equivalent_surface * chunk_size)
                    .zip(parent_multipoles.par_chunks_exact(chunk_size))
                    .for_each(
                        |(child_multipoles_chunk, parent_multipole_pointers_chunk)| {
                            let child_multipoles_chunk_mat = rlst_array_from_slice2!(
                                child_multipoles_chunk,
                                [ncoeffs_equivalent_surface * NSIBLINGS, chunk_size]
                            );

                            let parent_multipoles_chunk = empty_array::<Scalar, 2>()
                                .simple_mult_into_resize(
                                    self.source[operator_index].view(),
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
                                        ncoeffs_equivalent_surface_parent,
                                    )
                                };

                                parent_multipole
                                    .iter_mut()
                                    .zip(
                                        &parent_multipoles_chunk.data()[chunk_idx
                                            * ncoeffs_equivalent_surface_parent
                                            ..(chunk_idx + 1) * ncoeffs_equivalent_surface_parent],
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

                child_multipoles
                    .par_chunks_exact(nmatvecs * ncoeffs_equivalent_surface * NSIBLINGS)
                    .zip(parent_multipoles.into_par_iter())
                    .for_each(|(child_multipoles, parent_multipole_pointers)| {
                        for i in 0..NSIBLINGS {
                            let sibling_displacement = i * ncoeffs_equivalent_surface * nmatvecs;

                            let child_multipoles_i = rlst_array_from_slice2!(
                                &child_multipoles[sibling_displacement
                                    ..sibling_displacement + ncoeffs_equivalent_surface * nmatvecs],
                                [ncoeffs_equivalent_surface, nmatvecs]
                            );

                            let result_i = empty_array::<Scalar, 2>().simple_mult_into_resize(
                                self.source_vec[operator_index][i].view(),
                                child_multipoles_i,
                            );

                            for (j, send_ptr) in
                                parent_multipole_pointers.iter().enumerate().take(nmatvecs)
                            {
                                let raw = send_ptr.raw;
                                let parent_multipole_j = unsafe {
                                    std::slice::from_raw_parts_mut(
                                        raw,
                                        ncoeffs_equivalent_surface_parent,
                                    )
                                };
                                let result_ij = &result_i.data()[j
                                    * ncoeffs_equivalent_surface_parent
                                    ..(j + 1) * ncoeffs_equivalent_surface_parent];
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

impl<Scalar, Kernel, SourceToTargetData> SourceTranslation
    for KiFmmMultiNode<Scalar, Kernel, SourceToTargetData>
where
    Scalar: RlstScalar + Equivalence + Float,
    Kernel: KernelTrait<T = Scalar> + HomogenousKernel + Default + Send,
    SourceToTargetData: SourceToTargetDataTrait + Send + Sync + Default,
    <Scalar as RlstScalar>::Real: Default + Equivalence,
    Self: FmmOperatorData,
{
    fn p2m(&self) -> Result<(), FmmError> {
        let dim = 3;

        // Run P2M for each local root at this processor
        for (fmm_idx, tree) in self.tree.source_tree.trees.iter().enumerate() {
            // let &ncoeffs
            let Some(_leaves) = tree.all_leaves() else {
                return Err(FmmError::Failed(
                    "P2M failed, no leaves found in source tree".to_string(),
                ));
            };

            let ncoeffs_equivalent_surface = self.ncoeffs_equivalent_surface;
            let ncoeffs_check_surface = self.ncoeffs_check_surface;
            let n_leaves = tree.n_leaves().unwrap();
            let check_surface_size = ncoeffs_check_surface * dim;

            let coordinates = tree.all_coordinates().unwrap();
            let ncoordinates = coordinates.len() / dim;
            let operator_index = 0;
            let depth = self.tree.source_tree.local_depth + self.tree.source_tree.global_depth;

            let mut check_potentials =
                rlst_dynamic_array2!(Scalar, [n_leaves * ncoeffs_check_surface, 1]);

            let charges = &self.charges[fmm_idx];
            let kernel = &self.kernel;

            check_potentials
                .data_mut()
                .par_chunks_exact_mut(ncoeffs_check_surface)
                .zip(
                    self.leaf_upward_check_surfaces_sources[fmm_idx]
                        .par_chunks_exact(check_surface_size),
                )
                .zip(&self.charge_index_pointers_sources[fmm_idx])
                .for_each(
                    |((check_potential, upward_check_surface), charge_index_pointer)| {
                        let charges = &charges[charge_index_pointer.0..charge_index_pointer.1];

                        let coordinates_row_major = &coordinates
                            [charge_index_pointer.0 * dim..charge_index_pointer.1 * dim];
                        let nsources = coordinates_row_major.len() / dim;

                        if nsources > 0 {
                            kernel.evaluate_st(
                                EvalType::Value,
                                coordinates_row_major,
                                upward_check_surface,
                                charges,
                                check_potential,
                            );
                        }
                    },
                );

            let chunk_size = chunk_size(n_leaves, P2M_MAX_BLOCK_SIZE);
            let scale = if self.kernel.is_homogenous() {
                homogenous_kernel_scale(depth)
            } else {
                Scalar::one()
            };
            let uc2e_inv_1 = &self.uc2e_inv_1;
            let uc2e_inv_2 = &self.uc2e_inv_2;

            check_potentials
                .data()
                .par_chunks_exact(ncoeffs_check_surface * chunk_size)
                .zip(self.leaf_multipoles[fmm_idx].par_chunks_exact(chunk_size))
                .for_each(|(check_potential, multipole_ptrs)| {
                    let check_potential = rlst_array_from_slice2!(
                        check_potential,
                        [ncoeffs_check_surface, chunk_size]
                    );

                    let tmp = if kernel.is_homogenous() {
                        let mut scaled_check_potential =
                            rlst_dynamic_array2!(Scalar, [ncoeffs_check_surface, chunk_size]);
                        scaled_check_potential.fill_from(check_potential);
                        scaled_check_potential.scale_inplace(scale);

                        empty_array::<Scalar, 2>().simple_mult_into_resize(
                            uc2e_inv_1[operator_index].view(),
                            empty_array::<Scalar, 2>().simple_mult_into_resize(
                                uc2e_inv_2[operator_index].view(),
                                scaled_check_potential,
                            ),
                        )
                    } else {
                        empty_array::<Scalar, 2>().simple_mult_into_resize(
                            uc2e_inv_1[operator_index].view(),
                            empty_array::<Scalar, 2>().simple_mult_into_resize(
                                uc2e_inv_2[operator_index].view(),
                                check_potential.view(),
                            ),
                        )
                    };

                    for (i, multipole_ptr) in multipole_ptrs.iter().enumerate().take(chunk_size) {
                        let multipole = unsafe {
                            std::slice::from_raw_parts_mut(
                                multipole_ptr.raw,
                                ncoeffs_equivalent_surface,
                            )
                        };

                        multipole
                            .iter_mut()
                            .zip(
                                &tmp.data()[i * ncoeffs_equivalent_surface
                                    ..(i + 1) * ncoeffs_equivalent_surface],
                            )
                            .for_each(|(m, t)| *m += *t);
                    }
                });
        }
        Ok(())
    }

    fn m2m(&self, level: u64) -> Result<(), FmmError> {
        for (fmm_idx, tree) in self.tree.source_tree.trees.iter().enumerate() {
            let ncoeffs_equivalent_surface = self.ncoeffs_equivalent_surface;

            let Some(child_sources) = tree.keys(level) else {
                return Err(FmmError::Failed(format!(
                    "M2M failed at level {:?}, no sources found",
                    level
                )));
            };

            let parent_targets: HashSet<_> =
                child_sources.iter().map(|source| source.parent()).collect();

            let mut parent_targets = parent_targets.into_iter().collect_vec();
            parent_targets.sort();

            let nparents = parent_targets.len();

            // let multipole_ptr = &self.level_multipoles[fmm_idx][level as usize][0];
            // let nsources = tree.n_keys(level).unwrap();
            // let child_multipoles = unsafe {
            //     std::slice::from_raw_parts(multipole_ptr.raw, nsources * ncoeffs_equivalent_surface)
            // };
            let child_multipoles = self.multipoles(fmm_idx, level).unwrap();

            let mut parent_multipoles = Vec::new();
            for parent in parent_targets.iter() {
                let &parent_index_pointer = self.level_index_pointer_multipoles[fmm_idx]
                    [(level - 1) as usize]
                    .get(parent)
                    .unwrap();

                let parent_multipole =
                    &self.level_multipoles[fmm_idx][(level - 1) as usize][parent_index_pointer];
                parent_multipoles.push(parent_multipole);
            }

            let mut max_chunk_size = nparents;
            if max_chunk_size > M2M_MAX_BLOCK_SIZE {
                max_chunk_size = M2M_MAX_BLOCK_SIZE
            }

            let chunk_size = chunk_size(nparents, max_chunk_size);

            let source = &self.source[0];

            child_multipoles
                .par_chunks_exact(NSIBLINGS * ncoeffs_equivalent_surface * chunk_size)
                .zip(parent_multipoles.par_chunks_exact(chunk_size))
                .for_each(
                    |(child_multipoles_chunk, parent_multipole_pointers_chunk)| {
                        let child_multipoles_chunk_mat = rlst_array_from_slice2!(
                            child_multipoles_chunk,
                            [ncoeffs_equivalent_surface * NSIBLINGS, chunk_size]
                        );

                        let parent_multipoles_chunk = empty_array::<Scalar, 2>()
                            .simple_mult_into_resize(source.view(), child_multipoles_chunk_mat);

                        for (chunk_idx, parent_multipole_pointer) in parent_multipole_pointers_chunk
                            .iter()
                            .enumerate()
                            .take(chunk_size)
                        {
                            let parent_multipole = unsafe {
                                std::slice::from_raw_parts_mut(
                                    parent_multipole_pointer.raw,
                                    ncoeffs_equivalent_surface,
                                )
                            };

                            parent_multipole
                                .iter_mut()
                                .zip(
                                    &parent_multipoles_chunk.data()[chunk_idx
                                        * ncoeffs_equivalent_surface
                                        ..(chunk_idx + 1) * ncoeffs_equivalent_surface],
                                )
                                .for_each(|(p, t)| *p += *t);
                        }
                    },
                )
        }

        Ok(())
    }
}
