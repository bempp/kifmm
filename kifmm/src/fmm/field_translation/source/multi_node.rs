//! Multipole to Multipole field translation

use std::collections::HashSet;

use itertools::Itertools;
use mpi::traits::Equivalence;
use num::Float;
use rayon::prelude::*;
use rlst::{
    empty_array, rlst_array_from_slice2, rlst_dynamic_array2, MultIntoResize, RawAccess,
    RawAccessMut, RlstScalar,
};

use green_kernels::{traits::Kernel as KernelTrait, types::GreenKernelEvalType};

use crate::{
    fmm::{
        constants::{M2M_MAX_BLOCK_SIZE, P2M_MAX_BLOCK_SIZE},
        helpers::single_node::{chunk_size, homogenous_kernel_scale},
        types::{FmmEvalType, KiFmmMulti},
    },
    traits::{
        field::{FieldTranslation as FieldTranslationTrait, SourceTranslation},
        fmm::{DataAccessMulti, HomogenousKernel},
        tree::{MultiFmmTree, MultiTree},
        types::FmmError,
    },
    tree::constants::NSIBLINGS,
};

impl<Scalar, Kernel, FieldTranslation> SourceTranslation
    for KiFmmMulti<Scalar, Kernel, FieldTranslation>
where
    Scalar: RlstScalar + Default + Equivalence + Float,
    <Scalar as RlstScalar>::Real: Default + Equivalence + Float,
    FieldTranslation: FieldTranslationTrait,
    Kernel: KernelTrait<T = Scalar> + HomogenousKernel + Default + Send + Sync,
    Self: DataAccessMulti<Scalar = Scalar>,
{
    fn p2m(&self) -> Result<(), crate::traits::types::FmmError> {
        if let Some(_leaves) = self.tree.source_tree().all_leaves() {
            let n_coeffs_equivalent_surface = self.n_coeffs_equivalent_surface;
            let n_coeffs_check_surface = self.n_coeffs_check_surface;
            let n_leaves = self.tree.source_tree().n_leaves().unwrap();
            let check_surface_size = n_coeffs_check_surface * self.dim;
            let coordinates = self.tree.source_tree().all_coordinates().unwrap();

            let all_charges = &self.charges;
            let kernel = &self.kernel;
            let dim = self.dim;

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
                                    &all_charges[charge_index_pointer.0..charge_index_pointer.1];

                                let coordinates_row_major = &coordinates
                                    [charge_index_pointer.0 * dim..charge_index_pointer.1 * dim];
                                let n_sources = coordinates_row_major.len() / dim;

                                if n_sources > 0 {
                                    kernel.evaluate_st(
                                        GreenKernelEvalType::Value,
                                        coordinates_row_major,
                                        upward_check_surface,
                                        charges,
                                        check_potential,
                                    );
                                }
                            },
                        );

                    let total_depth = self.tree.source_tree().total_depth();
                    let chunk_size = chunk_size(n_leaves, P2M_MAX_BLOCK_SIZE);

                    let scale = if self.kernel.is_homogenous() {
                        homogenous_kernel_scale(total_depth)
                    } else {
                        Scalar::one()
                    };
                    let uc2e_inv_1 = &self.uc2e_inv_1;
                    let uc2e_inv_2 = &self.uc2e_inv_2;

                    check_potentials
                        .data()
                        .par_chunks_exact(n_coeffs_check_surface * chunk_size)
                        .zip(self.leaf_multipoles.par_chunks_exact(chunk_size))
                        .for_each(|(check_potential, multipole_ptrs)| {
                            let check_potential = rlst_array_from_slice2!(
                                check_potential,
                                [n_coeffs_check_surface, chunk_size]
                            );

                            let tmp = if kernel.is_homogenous() {
                                let mut scaled_check_potential = rlst_dynamic_array2!(
                                    Scalar,
                                    [n_coeffs_check_surface, chunk_size]
                                );
                                scaled_check_potential.fill_from(check_potential);
                                scaled_check_potential.scale_inplace(scale);

                                empty_array::<Scalar, 2>().simple_mult_into_resize(
                                    uc2e_inv_1[0].r(),
                                    empty_array::<Scalar, 2>().simple_mult_into_resize(
                                        uc2e_inv_2[0].r(),
                                        scaled_check_potential,
                                    ),
                                )
                            } else {
                                empty_array::<Scalar, 2>().simple_mult_into_resize(
                                    uc2e_inv_1[0].r(),
                                    empty_array::<Scalar, 2>().simple_mult_into_resize(
                                        uc2e_inv_2[0].r(),
                                        check_potential.r(),
                                    ),
                                )
                            };

                            for (i, multipole_ptr) in
                                multipole_ptrs.iter().enumerate().take(chunk_size)
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
                }

                FmmEvalType::Matrix(_n) => {
                    return Err(FmmError::Unimplemented(
                        "Unimplemented for matrix input".to_string(),
                    ))
                }
            }
        }

        Ok(())
    }

    fn m2m(&self, level: u64) -> Result<(), crate::traits::types::FmmError> {
        if let Some(child_sources) = self.tree.source_tree.keys(level) {
            let n_coeffs_equivalent_surface = self.n_coeffs_equivalent_surface;

            let parent_targets: HashSet<_> =
                child_sources.iter().map(|source| source.parent()).collect();

            let mut parent_targets = parent_targets.into_iter().collect_vec();

            parent_targets.sort();
            let n_parents = parent_targets.len();
            let child_multipoles = self.multipoles(level).unwrap();
            let source = &self.source;

            match self.fmm_eval_type {
                FmmEvalType::Vector => {
                    let mut parent_multipoles = Vec::new();
                    for parent in parent_targets.iter() {
                        let &parent_index_pointer = self.level_index_pointer_multipoles
                            [(level - 1) as usize]
                            .get(parent)
                            .unwrap();
                        let parent_multipole =
                            &self.level_multipoles[(level - 1) as usize][parent_index_pointer];
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
                                        source.r(),
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
                                            n_coeffs_equivalent_surface,
                                        )
                                    };

                                    parent_multipole
                                        .iter_mut()
                                        .zip(
                                            &parent_multipoles_chunk.data()[chunk_idx
                                                * n_coeffs_equivalent_surface
                                                ..(chunk_idx + 1) * n_coeffs_equivalent_surface],
                                        )
                                        .for_each(|(p, t)| *p += *t);
                                }
                            },
                        );
                }

                FmmEvalType::Matrix(_n) => {
                    return Err(FmmError::Unimplemented(
                        "Unimplemented for matrix input".to_string(),
                    ))
                }
            }
        }

        Ok(())
    }
}
