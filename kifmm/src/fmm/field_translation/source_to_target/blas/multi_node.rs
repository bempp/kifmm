//! Multipole to local field translation trait implementation using BLAS.

use std::sync::Mutex;

use itertools::Itertools;
use mpi::{topology::SimpleCommunicator, traits::Equivalence};
use num::Float;
use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use rlst::{
    empty_array, rlst_array_from_slice2, rlst_dynamic_array2, MultIntoResize, RawAccess,
    RawAccessMut, RlstScalar,
};

use green_kernels::traits::Kernel as KernelTrait;

use crate::{
    fmm::{
        helpers::single_node::{homogenous_kernel_scale, m2l_scale},
        types::{FmmEvalType, KiFmmMulti, SendPtrMut},
    },
    traits::{
        field::SourceToTargetTranslation,
        fmm::{DataAccessMulti, HomogenousKernel, MetadataAccess},
        tree::{MultiFmmTree, MultiTree, SingleTree},
        types::FmmError,
    },
    tree::constants::NTRANSFER_VECTORS_KIFMM,
    BlasFieldTranslationIa, BlasFieldTranslationSaRcmp, DataAccess, KiFmm, MultiNodeFmmTree,
};

impl<Scalar, Kernel> SourceToTargetTranslation
    for KiFmmMulti<Scalar, Kernel, BlasFieldTranslationSaRcmp<Scalar>>
where
    Scalar: RlstScalar + Default + Equivalence,
    <Scalar as RlstScalar>::Real: Default + Equivalence + Float,
    Kernel: KernelTrait<T = Scalar> + HomogenousKernel + Default + Send + Sync,
    Self: MetadataAccess
        + DataAccessMulti<
            Scalar = Scalar,
            Kernel = Kernel,
            Tree = MultiNodeFmmTree<Scalar::Real, SimpleCommunicator>,
        >,
    KiFmm<Scalar, Kernel, BlasFieldTranslationSaRcmp<Scalar>>:
        DataAccess<Scalar = Scalar, Kernel = Kernel>,
{
    fn m2l(&self, level: u64) -> Result<(), FmmError> {
        match self.fmm_eval_type {
            FmmEvalType::Vector => {
                if let Some(targets) = self.tree().target_tree().keys(level) {
                    // Metadata
                    let m2l_operator_index = self.m2l_operator_index(level);
                    let c2e_operator_index = self.c2e_operator_index(level);
                    let displacement_index = self.displacement_index(level);
                    let n_coeffs_equivalent_surface = self.n_coeffs_equivalent_surface(level);
                    let sentinel = -1i32;

                    // Parameters
                    let n_targets = targets.len();
                    let mut n_translations = 0;
                    let mut all_n_sources = Vec::new();
                    let mut all_multipoles = Vec::new();
                    let mut all_displacements = Vec::new();

                    // Handle locally contained source boxes
                    if let Some(sources) = self.tree().source_tree().keys(level) {
                        n_translations += 1;

                        all_displacements
                            .push(&self.source_to_target.displacements[displacement_index]);

                        // Number of sources at this level
                        let n_sources = sources.len();

                        // Lookup multipole data from source tree
                        let multipoles = self.multipoles(level).unwrap();

                        all_n_sources.push(n_sources);
                        all_multipoles.push(multipoles);
                    }

                    // Handle ghost sources
                    if let Some(sources) = self.ghost_fmm_v.tree.source_tree.keys(level) {
                        n_translations += 1;

                        all_displacements.push(
                            &self.ghost_fmm_v.source_to_target.displacements[displacement_index],
                        );

                        // Number of sources at this level
                        let n_sources = sources.len();

                        // Lookup multipole data from source tree
                        let multipoles = self.ghost_fmm_v.multipoles(level).unwrap();

                        all_n_sources.push(n_sources);
                        all_multipoles.push(multipoles);
                    }

                    // Allocate buffer to store compressed check potentials
                    let compressed_check_potentials = rlst_dynamic_array2!(
                        Scalar,
                        [
                            self.source_to_target.cutoff_rank[m2l_operator_index],
                            n_targets
                        ]
                    );
                    let mut compressed_check_potentials_ptrs = Vec::new();

                    for i in 0..n_targets {
                        let raw = unsafe {
                            compressed_check_potentials
                                .data()
                                .as_ptr()
                                .add(i * self.source_to_target.cutoff_rank[m2l_operator_index])
                                as *mut Scalar
                        };
                        let send_ptr = SendPtrMut { raw };
                        compressed_check_potentials_ptrs.push(send_ptr);
                    }

                    let compressed_level_check_potentials = compressed_check_potentials_ptrs
                        .iter()
                        .map(Mutex::new)
                        .collect_vec();

                    // Compute translations
                    for i in 0..n_translations {
                        let all_displacements = all_displacements[i];

                        let multipole_idxs = all_displacements
                            .iter()
                            .map(|displacement| {
                                displacement
                                    .read()
                                    .unwrap()
                                    .iter()
                                    .enumerate()
                                    .filter(|(_, &d)| d != sentinel)
                                    .map(|(i, _)| i)
                                    .collect_vec()
                            })
                            .collect_vec();

                        let local_idxs = all_displacements
                            .iter()
                            .map(|displacements| {
                                displacements
                                    .read()
                                    .unwrap()
                                    .iter()
                                    .enumerate()
                                    .filter(|(_, &d)| d != sentinel)
                                    .map(|(_, &j)| j as usize)
                                    .collect_vec()
                            })
                            .collect_vec();

                        let n_sources = all_n_sources[i];
                        let multipoles = &all_multipoles[i];

                        let multipoles = rlst_array_from_slice2!(
                            multipoles,
                            [n_coeffs_equivalent_surface, n_sources]
                        );

                        // 1. Compute the SVD compressed multipole expansions at this level
                        let mut compressed_multipoles;
                        {
                            compressed_multipoles = empty_array::<Scalar, 2>()
                                .simple_mult_into_resize(
                                    self.source_to_target.metadata[m2l_operator_index].st.r(),
                                    multipoles,
                                );

                            if self.kernel.is_homogenous() {
                                compressed_multipoles.data_mut().iter_mut().for_each(|d| {
                                    *d *= homogenous_kernel_scale::<Scalar>(level)
                                        * m2l_scale::<Scalar>(level).unwrap()
                                });
                            }
                        }

                        // 2. Apply BLAS operation
                        {
                            let all_c_u_sub =
                                &self.source_to_target.metadata[m2l_operator_index].c_u;
                            let all_c_vt_sub =
                                &self.source_to_target.metadata[m2l_operator_index].c_vt;
                            let &cutoff_rank =
                                &self.source_to_target.cutoff_rank[m2l_operator_index];

                            (0..NTRANSFER_VECTORS_KIFMM)
                                .into_par_iter()
                                .zip(multipole_idxs)
                                .zip(local_idxs)
                                .for_each(|((c_idx, multipole_idxs), local_idxs)| {
                                    let c_u_sub = &all_c_u_sub[c_idx];
                                    let c_vt_sub = &all_c_vt_sub[c_idx];

                                    let mut compressed_multipoles_subset = rlst_dynamic_array2!(
                                        Scalar,
                                        [cutoff_rank, multipole_idxs.len()]
                                    );

                                    for (i, &multipole_idx) in multipole_idxs.iter().enumerate() {
                                        compressed_multipoles_subset.data_mut()
                                            [i * cutoff_rank..(i + 1) * cutoff_rank]
                                            .copy_from_slice(
                                                &compressed_multipoles.data()[multipole_idx
                                                    * cutoff_rank
                                                    ..(multipole_idx + 1) * cutoff_rank],
                                            );
                                    }

                                    let compressed_check_potential = empty_array::<Scalar, 2>()
                                        .simple_mult_into_resize(
                                            c_u_sub.r(),
                                            empty_array::<Scalar, 2>().simple_mult_into_resize(
                                                c_vt_sub.r(),
                                                compressed_multipoles_subset.r(),
                                            ),
                                        );

                                    for (multipole_idx, &local_idx) in local_idxs.iter().enumerate()
                                    {
                                        let check_potential_lock =
                                            compressed_level_check_potentials[local_idx]
                                                .lock()
                                                .unwrap();
                                        let check_potential_ptr = check_potential_lock.raw;
                                        let check_potential = unsafe {
                                            std::slice::from_raw_parts_mut(
                                                check_potential_ptr,
                                                cutoff_rank,
                                            )
                                        };
                                        let tmp = &compressed_check_potential.data()[multipole_idx
                                            * cutoff_rank
                                            ..(multipole_idx + 1) * cutoff_rank];
                                        check_potential
                                            .iter_mut()
                                            .zip(tmp)
                                            .for_each(|(l, r)| *l += *r);
                                    }
                                });
                        }
                    }

                    // 3. Compute local expansions from compressed check potentials
                    {
                        let locals = empty_array::<Scalar, 2>().simple_mult_into_resize(
                            self.dc2e_inv_1[c2e_operator_index].r(),
                            empty_array::<Scalar, 2>().simple_mult_into_resize(
                                self.dc2e_inv_2[c2e_operator_index].r(),
                                empty_array::<Scalar, 2>().simple_mult_into_resize(
                                    self.source_to_target.metadata[m2l_operator_index].u.r(),
                                    compressed_check_potentials.r(),
                                ),
                            ),
                        );

                        let ptr = self.level_locals[level as usize][0].raw;
                        let all_locals = unsafe {
                            std::slice::from_raw_parts_mut(
                                ptr,
                                n_targets * n_coeffs_equivalent_surface,
                            )
                        };
                        all_locals
                            .iter_mut()
                            .zip(locals.data().iter())
                            .for_each(|(l, r)| *l += *r);
                    }
                }

                Ok(())
            }
            FmmEvalType::Matrix(_) => Err(FmmError::Unimplemented(
                "M2L unimplemented for matrix input with BLAS field translations".to_string(),
            )),
        }
    }

    fn p2l(&self, _level: u64) -> Result<(), FmmError> {
        Ok(())
    }
}

impl<Scalar, Kernel> SourceToTargetTranslation
    for KiFmmMulti<Scalar, Kernel, BlasFieldTranslationIa<Scalar>>
where
    Scalar: RlstScalar + Default + Equivalence,
    <Scalar as RlstScalar>::Real: Default + Equivalence + Float,
    Kernel: KernelTrait<T = Scalar> + HomogenousKernel + Default + Send + Sync,
    Self: MetadataAccess
        + DataAccessMulti<
            Scalar = Scalar,
            Kernel = Kernel,
            Tree = MultiNodeFmmTree<Scalar::Real, SimpleCommunicator>,
        >,
    KiFmm<Scalar, Kernel, BlasFieldTranslationIa<Scalar>>:
        DataAccess<Scalar = Scalar, Kernel = Kernel>,
{
    fn m2l(&self, level: u64) -> Result<(), FmmError> {
        match self.fmm_eval_type {
            FmmEvalType::Vector => {
                if let Some(targets) = self.tree().target_tree().keys(level) {
                    // Metadata
                    let m2l_operator_index = self.m2l_operator_index(level);
                    let c2e_operator_index = self.c2e_operator_index(level);
                    let displacement_index = self.displacement_index(level);
                    let n_coeffs_equivalent_surface = self.n_coeffs_equivalent_surface(level);
                    let n_coeffs_check_surface = self.n_coeffs_check_surface(level);
                    let sentinel = -1i32;

                    // Parameters
                    let n_targets = targets.len();
                    let mut n_translations = 0;
                    let mut all_n_sources = Vec::new();
                    let mut all_multipoles = Vec::new();
                    let mut all_displacements = Vec::new();

                    // Handle locally contained source boxes
                    if let Some(sources) = self.tree().source_tree().keys(level) {
                        n_translations += 1;

                        all_displacements
                            .push(&self.source_to_target.displacements[displacement_index]);

                        // Number of sources at this level
                        let n_sources = sources.len();

                        // Lookup multipole data from source tree
                        let multipoles = self.multipoles(level).unwrap();

                        all_n_sources.push(n_sources);
                        all_multipoles.push(multipoles);
                    }

                    // Handle ghost sources
                    if let Some(sources) = self.ghost_fmm_v.tree.source_tree.keys(level) {
                        n_translations += 1;

                        all_displacements.push(
                            &self.ghost_fmm_v.source_to_target.displacements[displacement_index],
                        );

                        // Number of sources at this level
                        let n_sources = sources.len();

                        // Lookup multipole data from source tree
                        let multipoles = self.ghost_fmm_v.multipoles(level).unwrap();

                        all_n_sources.push(n_sources);
                        all_multipoles.push(multipoles);
                    }

                    // Allocate buffer to store check potentials
                    let all_check_potentials =
                        rlst_dynamic_array2!(Scalar, [n_coeffs_check_surface, n_targets]);

                    let mut all_check_potentials_ptrs = Vec::new();

                    for i in 0..n_targets {
                        let raw = unsafe {
                            all_check_potentials
                                .data()
                                .as_ptr()
                                .add(i * n_coeffs_check_surface)
                                as *mut Scalar
                        };
                        let send_ptr = SendPtrMut { raw };
                        all_check_potentials_ptrs.push(send_ptr);
                    }

                    let level_check_potentials = all_check_potentials_ptrs
                        .iter()
                        .map(Mutex::new)
                        .collect_vec();

                    // Compute translations
                    for i in 0..n_translations {
                        let all_displacements_i = all_displacements[i];

                        let multipole_idxs = all_displacements_i
                            .iter()
                            .map(|displacement| {
                                displacement
                                    .read()
                                    .unwrap()
                                    .iter()
                                    .enumerate()
                                    .filter(|(_, &d)| d != sentinel)
                                    .map(|(i, _)| i)
                                    .collect_vec()
                            })
                            .collect_vec();

                        let local_idxs = all_displacements_i
                            .iter()
                            .map(|displacements| {
                                displacements
                                    .read()
                                    .unwrap()
                                    .iter()
                                    .enumerate()
                                    .filter(|(_, &d)| d != sentinel)
                                    .map(|(_, &j)| j as usize)
                                    .collect_vec()
                            })
                            .collect_vec();

                        let n_sources = all_n_sources[i];
                        let multipoles = &all_multipoles[i];

                        let multipoles = rlst_array_from_slice2!(
                            multipoles,
                            [n_coeffs_equivalent_surface, n_sources]
                        );

                        // 1. Apply BLAS operation
                        {
                            let all_u_sub = &self.source_to_target.metadata[m2l_operator_index].u;
                            let all_vt_sub = &self.source_to_target.metadata[m2l_operator_index].vt;

                            (0..NTRANSFER_VECTORS_KIFMM)
                                .into_par_iter()
                                .zip(multipole_idxs)
                                .zip(local_idxs)
                                .for_each(|((c_idx, multipole_idxs), local_idxs)| {
                                    let u = &all_u_sub[c_idx];
                                    let vt = &all_vt_sub[c_idx];

                                    let mut multipoles_subset = rlst_dynamic_array2!(
                                        Scalar,
                                        [n_coeffs_equivalent_surface, multipole_idxs.len()]
                                    );

                                    for (local_multipole_idx, &global_multipole_idx) in
                                        multipole_idxs.iter().enumerate()
                                    {
                                        multipoles_subset.data_mut()[local_multipole_idx
                                            * n_coeffs_equivalent_surface
                                            ..(local_multipole_idx + 1)
                                                * n_coeffs_equivalent_surface]
                                            .copy_from_slice(
                                                &multipoles.data()[global_multipole_idx
                                                    * n_coeffs_equivalent_surface
                                                    ..(global_multipole_idx + 1)
                                                        * n_coeffs_equivalent_surface],
                                            );
                                    }

                                    let check_potential = empty_array::<Scalar, 2>()
                                        .simple_mult_into_resize(
                                            u.r(),
                                            empty_array::<Scalar, 2>().simple_mult_into_resize(
                                                vt.r(),
                                                multipoles_subset.r(),
                                            ),
                                        );

                                    for (multipole_idx, &local_idx) in local_idxs.iter().enumerate()
                                    {
                                        let tmp = &check_potential.data()[multipole_idx
                                            * n_coeffs_check_surface
                                            ..(multipole_idx + 1) * n_coeffs_check_surface];

                                        let check_potential_lock =
                                            level_check_potentials[local_idx].lock().unwrap();
                                        let check_potential_ptr = check_potential_lock.raw;
                                        let global_check_potential = unsafe {
                                            std::slice::from_raw_parts_mut(
                                                check_potential_ptr,
                                                n_coeffs_check_surface,
                                            )
                                        };

                                        global_check_potential
                                            .iter_mut()
                                            .zip(tmp)
                                            .for_each(|(l, r)| *l += *r);
                                    }
                                });
                        }
                    }

                    // 2. Compute local expansions from compressed check potentials
                    {
                        let locals = empty_array::<Scalar, 2>().simple_mult_into_resize(
                            self.dc2e_inv_1[c2e_operator_index].r(),
                            empty_array::<Scalar, 2>().simple_mult_into_resize(
                                self.dc2e_inv_2[c2e_operator_index].r(),
                                all_check_potentials.r(),
                            ),
                        );

                        let ptr = self.level_locals[level as usize][0].raw;
                        let all_locals = unsafe {
                            std::slice::from_raw_parts_mut(
                                ptr,
                                n_targets * n_coeffs_equivalent_surface,
                            )
                        };

                        all_locals
                            .iter_mut()
                            .zip(locals.data().iter())
                            .for_each(|(l, r)| *l += *r);
                    }
                }

                Ok(())
            }
            FmmEvalType::Matrix(_) => Err(FmmError::Unimplemented(
                "M2L unimplemented for matrix input with BLAS field translations".to_string(),
            )),
        }
    }

    fn p2l(&self, _level: u64) -> Result<(), FmmError> {
        Ok(())
    }
}
