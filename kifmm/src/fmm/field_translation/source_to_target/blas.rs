//! Multipole to local field translation trait implementation using BLAS.

use std::{
    collections::{HashMap, HashSet},
    sync::Mutex, time::{Duration, Instant},
};

use itertools::Itertools;
use rayon::prelude::*;
use rlst::{
    empty_array, rlst_array_from_slice2, rlst_dynamic_array2, rlst_metal_array2, DefaultIterator, MetalDevice, MultIntoResize, RawAccess, RawAccessMut, RlstScalar, Shape
};

use green_kernels::traits::Kernel as KernelTrait;

use crate::{
    fmm::{
        helpers::{homogenous_kernel_scale, m2l_scale},
        types::{BlasFieldTranslationIa, FmmEvalType, KiFmmLaplaceMetal, SendPtrMut},
        KiFmm,
    },
    traits::{
        fmm::{FmmOperatorData, HomogenousKernel, SourceToTargetTranslation},
        tree::{FmmTree, Tree},
        types::{FmmError, M2LResult},
    },
    tree::constants::NTRANSFER_VECTORS_KIFMM,
    BlasFieldTranslationSaRcmp, Fmm,
};

impl<Scalar, Kernel> KiFmm<Scalar, Kernel, BlasFieldTranslationSaRcmp<Scalar>>
where
    Scalar: RlstScalar + Default,
    Kernel: KernelTrait<T = Scalar> + HomogenousKernel + Default,
    <Scalar as RlstScalar>::Real: Default,
    Self: FmmOperatorData,
{
    /// Map between each transfer vector for homogenous kernels, and the source boxes involved in that translation
    /// at this octree level.
    ///
    /// Returns a vector of length 316, the maximum number of unique transfer vectors at a tree level for homogenous
    /// kernels, where each item is a Mutex locked vector of indices, of a length equal to the number of source boxes
    /// at this tree level, where an index value of -1 indicates that the box isn't involved in the translation, and
    /// an index values of `usize` gives the target box index that the source box density is being translated to.
    fn displacements(&self, level: u64) -> Vec<Mutex<Vec<i64>>> {
        let sources = self.tree.source_tree().keys(level).unwrap();
        let nsources = sources.len();

        let all_displacements = vec![vec![-1i64; nsources]; 316];
        let all_displacements = all_displacements.into_iter().map(Mutex::new).collect_vec();

        sources
            .into_par_iter()
            .enumerate()
            .for_each(|(source_idx, source)| {
                // Find interaction list of each source, as this defines scatter locations
                let interaction_list = source
                    .parent()
                    .neighbors()
                    .iter()
                    .flat_map(|pn| pn.children())
                    .filter(|pnc| {
                        !source.is_adjacent(pnc)
                            && self
                                .tree
                                .target_tree()
                                .all_keys_set()
                                .unwrap()
                                .contains(pnc)
                    })
                    .collect_vec();

                let transfer_vectors = interaction_list
                    .iter()
                    .map(|target| source.find_transfer_vector(target).unwrap())
                    .collect_vec();

                let mut transfer_vectors_map = HashMap::new();
                for (i, v) in transfer_vectors.iter().enumerate() {
                    transfer_vectors_map.insert(v, i);
                }

                let transfer_vectors_set: HashSet<_> = transfer_vectors.iter().cloned().collect();

                // Mark items in interaction list for scattering
                for (tv_idx, tv) in self.source_to_target.transfer_vectors.iter().enumerate() {
                    let mut all_displacements_lock = all_displacements[tv_idx].lock().unwrap();
                    if transfer_vectors_set.contains(&tv.hash) {
                        // Look up scatter location in target tree
                        let target =
                            &interaction_list[*transfer_vectors_map.get(&tv.hash).unwrap()];
                        let &target_idx = self.level_index_pointer_locals[level as usize]
                            .get(target)
                            .unwrap();
                        all_displacements_lock[source_idx] = target_idx as i64;
                    }
                }
            });
        all_displacements
    }
}

impl<Scalar, Kernel> SourceToTargetTranslation
    for KiFmm<Scalar, Kernel, BlasFieldTranslationSaRcmp<Scalar>>
where
    Scalar: RlstScalar + Default,
    Kernel: KernelTrait<T = Scalar> + HomogenousKernel + Default + Send + Sync,
    <Scalar as RlstScalar>::Real: Default,
    Self: FmmOperatorData,
{
    fn m2l(&self, level: u64) -> Result<M2LResult, FmmError> {
        let Some(targets) = self.tree().target_tree().keys(level) else {
            return Err(FmmError::Failed(
                "No target boxes at this level".to_string(),
            ));
        };
        let Some(sources) = self.tree().source_tree().keys(level) else {
            return Err(FmmError::Failed(
                "No source boxes at this level".to_string(),
            ));
        };

        let m2l_operator_index = self.m2l_operator_index(level);
        let c2e_operator_index = self.c2e_operator_index(level);

        // Compute the displacements
        let all_displacements = self.displacements(level);

        let multipole_idxs = all_displacements
            .iter()
            .map(|displacement| {
                displacement
                    .lock()
                    .unwrap()
                    .iter()
                    .enumerate()
                    .filter(|(_, &d)| d != -1)
                    .map(|(i, _)| i)
                    .collect_vec()
            })
            .collect_vec();

        let local_idxs = all_displacements
            .iter()
            .map(|displacements| {
                displacements
                    .lock()
                    .unwrap()
                    .iter()
                    .enumerate()
                    .filter(|(_, &d)| d != -1)
                    .map(|(_, &j)| j as usize)
                    .collect_vec()
            })
            .collect_vec();

        // Number of sources at this level
        let nsources = sources.len();
        let ntargets = targets.len();

        match self.fmm_eval_type {
            FmmEvalType::Vector => {
                // Lookup multipole data from source tree
                let multipoles = rlst_array_from_slice2!(
                    unsafe {
                        std::slice::from_raw_parts(
                            self.level_multipoles[level as usize][0][0].raw,
                            self.ncoeffs * nsources,
                        )
                    },
                    [self.ncoeffs, nsources]
                );

                // Allocate buffer to store compressed check potentials
                let compressed_check_potentials =
                    rlst_dynamic_array2!(Scalar, [self.source_to_target.cutoff_rank, ntargets]);
                let mut compressed_check_potentials_ptrs = Vec::new();

                for i in 0..ntargets {
                    let raw = unsafe {
                        compressed_check_potentials
                            .data()
                            .as_ptr()
                            .add(i * self.source_to_target.cutoff_rank)
                            as *mut Scalar
                    };
                    let send_ptr = SendPtrMut { raw };
                    compressed_check_potentials_ptrs.push(send_ptr);
                }

                let compressed_level_check_potentials = compressed_check_potentials_ptrs
                    .iter()
                    .map(Mutex::new)
                    .collect_vec();

                // 1. Compute the SVD compressed multipole expansions at this level
                let mut compressed_multipoles;
                {
                    #[cfg(target_os = "linux")]
                    rlst::threading::enable_threading();
                    compressed_multipoles = empty_array::<Scalar, 2>().simple_mult_into_resize(
                        self.source_to_target.metadata[m2l_operator_index].st.view(),
                        multipoles,
                    );
                    #[cfg(target_os = "linux")]
                    rlst::threading::disable_threading();

                    if self.kernel.is_homogenous() {
                        compressed_multipoles.data_mut().iter_mut().for_each(|d| {
                            *d *= homogenous_kernel_scale::<Scalar>(level)
                                * m2l_scale::<Scalar>(level).unwrap()
                        });
                    }
                }

                // 2. Apply BLAS operation
                {
                    (0..NTRANSFER_VECTORS_KIFMM)
                        .into_par_iter()
                        .zip(multipole_idxs)
                        .zip(local_idxs)
                        .for_each(|((c_idx, multipole_idxs), local_idxs)| {
                            let c_u_sub =
                                &self.source_to_target.metadata[m2l_operator_index].c_u[c_idx];
                            let c_vt_sub =
                                &self.source_to_target.metadata[m2l_operator_index].c_vt[c_idx];

                            let mut compressed_multipoles_subset = rlst_dynamic_array2!(
                                Scalar,
                                [self.source_to_target.cutoff_rank, multipole_idxs.len()]
                            );

                            for (i, &multipole_idx) in multipole_idxs.iter().enumerate() {
                                compressed_multipoles_subset.data_mut()[i * self
                                    .source_to_target
                                    .cutoff_rank
                                    ..(i + 1) * self.source_to_target.cutoff_rank]
                                    .copy_from_slice(
                                        &compressed_multipoles.data()[multipole_idx
                                            * self.source_to_target.cutoff_rank
                                            ..(multipole_idx + 1)
                                                * self.source_to_target.cutoff_rank],
                                    );
                            }

                            let compressed_check_potential = empty_array::<Scalar, 2>()
                                .simple_mult_into_resize(
                                    c_u_sub.view(),
                                    empty_array::<Scalar, 2>().simple_mult_into_resize(
                                        c_vt_sub.view(),
                                        compressed_multipoles_subset.view(),
                                    ),
                                );

                            for (multipole_idx, &local_idx) in local_idxs.iter().enumerate() {
                                let check_potential_lock =
                                    compressed_level_check_potentials[local_idx].lock().unwrap();
                                let check_potential_ptr = check_potential_lock.raw;
                                let check_potential = unsafe {
                                    std::slice::from_raw_parts_mut(
                                        check_potential_ptr,
                                        self.source_to_target.cutoff_rank,
                                    )
                                };
                                let tmp = &compressed_check_potential.data()[multipole_idx
                                    * self.source_to_target.cutoff_rank
                                    ..(multipole_idx + 1) * self.source_to_target.cutoff_rank];
                                check_potential
                                    .iter_mut()
                                    .zip(tmp)
                                    .for_each(|(l, r)| *l += *r);
                            }
                        });
                }

                // 3. Compute local expansions from compressed check potentials
                {
                    #[cfg(target_os = "linux")]
                    rlst::threading::enable_threading();
                    let locals = empty_array::<Scalar, 2>().simple_mult_into_resize(
                        self.dc2e_inv_1[c2e_operator_index].view(),
                        empty_array::<Scalar, 2>().simple_mult_into_resize(
                            self.dc2e_inv_2[c2e_operator_index].view(),
                            empty_array::<Scalar, 2>().simple_mult_into_resize(
                                self.source_to_target.metadata[m2l_operator_index].u.view(),
                                compressed_check_potentials,
                            ),
                        ),
                    );
                    #[cfg(target_os = "linux")]
                    rlst::threading::disable_threading();

                    let ptr = self.level_locals[level as usize][0][0].raw;
                    let all_locals =
                        unsafe { std::slice::from_raw_parts_mut(ptr, ntargets * self.ncoeffs) };
                    all_locals
                        .iter_mut()
                        .zip(locals.data().iter())
                        .for_each(|(l, r)| *l += *r);
                }

                return Ok(M2LResult(Duration::from_secs(0), Duration::from_secs(0), Duration::from_secs(0), Duration::from_secs(0), 0.));
            }
            FmmEvalType::Matrix(nmatvecs) => {
                // Lookup multipole data from source tree
                let multipoles = rlst_array_from_slice2!(
                    unsafe {
                        std::slice::from_raw_parts(
                            self.level_multipoles[level as usize][0][0].raw,
                            self.ncoeffs * nsources * nmatvecs,
                        )
                    },
                    [self.ncoeffs, nsources * nmatvecs]
                );

                let compressed_check_potentials = rlst_dynamic_array2!(
                    Scalar,
                    [self.source_to_target.cutoff_rank, nsources * nmatvecs]
                );
                let mut compressed_check_potentials_ptrs = Vec::new();

                for i in 0..ntargets {
                    let key_displacement = i * self.source_to_target.cutoff_rank * nmatvecs;
                    let mut tmp = Vec::new();
                    for charge_vec_idx in 0..nmatvecs {
                        let charge_vec_displacement =
                            charge_vec_idx * self.source_to_target.cutoff_rank;

                        let raw = unsafe {
                            compressed_check_potentials
                                .data()
                                .as_ptr()
                                .add(key_displacement + charge_vec_displacement)
                                as *mut Scalar
                        };
                        let send_ptr = SendPtrMut { raw };
                        tmp.push(send_ptr)
                    }
                    compressed_check_potentials_ptrs.push(tmp);
                }

                let compressed_level_check_potentials = compressed_check_potentials_ptrs
                    .iter()
                    .map(Mutex::new)
                    .collect_vec();

                // 1. Compute the SVD compressed multipole expansions at this level
                let mut compressed_multipoles;
                {
                    #[cfg(target_os = "linux")]
                    rlst::threading::enable_threading();
                    compressed_multipoles = empty_array::<Scalar, 2>().simple_mult_into_resize(
                        self.source_to_target.metadata[m2l_operator_index].st.view(),
                        multipoles,
                    );
                    #[cfg(target_os = "linux")]
                    rlst::threading::disable_threading();

                    if self.kernel.is_homogenous() {
                        compressed_multipoles.data_mut().iter_mut().for_each(|d| {
                            *d *= homogenous_kernel_scale::<Scalar>(level)
                                * m2l_scale::<Scalar>(level).unwrap()
                        });
                    }
                }

                // 2. Apply the BLAS operation
                {
                    (0..NTRANSFER_VECTORS_KIFMM)
                        .into_par_iter()
                        .zip(multipole_idxs)
                        .zip(local_idxs)
                        .for_each(|((c_idx, multipole_idxs), local_idxs)| {
                            let c_u_sub =
                                &self.source_to_target.metadata[m2l_operator_index].c_u[c_idx];
                            let c_vt_sub =
                                &self.source_to_target.metadata[m2l_operator_index].c_vt[c_idx];

                            let mut compressed_multipoles_subset = rlst_dynamic_array2!(
                                Scalar,
                                [
                                    self.source_to_target.cutoff_rank,
                                    multipole_idxs.len() * nmatvecs
                                ]
                            );

                            for (local_multipole_idx, &global_multipole_idx) in
                                multipole_idxs.iter().enumerate()
                            {
                                let key_displacement_global = global_multipole_idx
                                    * self.source_to_target.cutoff_rank
                                    * nmatvecs;

                                let key_displacement_local = local_multipole_idx
                                    * self.source_to_target.cutoff_rank
                                    * nmatvecs;

                                for charge_vec_idx in 0..nmatvecs {
                                    let charge_vec_displacement =
                                        charge_vec_idx * self.source_to_target.cutoff_rank;

                                    compressed_multipoles_subset.data_mut()[key_displacement_local
                                        + charge_vec_displacement
                                        ..key_displacement_local
                                            + charge_vec_displacement
                                            + self.source_to_target.cutoff_rank]
                                        .copy_from_slice(
                                            &compressed_multipoles.data()[key_displacement_global
                                                + charge_vec_displacement
                                                ..key_displacement_global
                                                    + charge_vec_displacement
                                                    + self.source_to_target.cutoff_rank],
                                        );
                                }
                            }

                            let compressed_check_potential = empty_array::<Scalar, 2>()
                                .simple_mult_into_resize(
                                    c_u_sub.view(),
                                    empty_array::<Scalar, 2>().simple_mult_into_resize(
                                        c_vt_sub.view(),
                                        compressed_multipoles_subset.view(),
                                    ),
                                );

                            for (local_multipole_idx, &global_local_idx) in
                                local_idxs.iter().enumerate()
                            {
                                let check_potential_lock = compressed_level_check_potentials
                                    [global_local_idx]
                                    .lock()
                                    .unwrap();

                                for charge_vec_idx in 0..nmatvecs {
                                    let check_potential_ptr =
                                        check_potential_lock[charge_vec_idx].raw;
                                    let check_potential = unsafe {
                                        std::slice::from_raw_parts_mut(
                                            check_potential_ptr,
                                            self.source_to_target.cutoff_rank,
                                        )
                                    };

                                    let key_displacement = local_multipole_idx
                                        * self.source_to_target.cutoff_rank
                                        * nmatvecs;
                                    let charge_vec_displacement =
                                        charge_vec_idx * self.source_to_target.cutoff_rank;

                                    let tmp = &compressed_check_potential.data()[key_displacement
                                        + charge_vec_displacement
                                        ..key_displacement
                                            + charge_vec_displacement
                                            + self.source_to_target.cutoff_rank];
                                    check_potential
                                        .iter_mut()
                                        .zip(tmp)
                                        .for_each(|(l, r)| *l += *r);
                                }
                            }
                        });
                }

                // 3. Compute local expansions from compressed check potentials
                {
                    #[cfg(target_os = "linux")]
                    rlst::threading::enable_threading();
                    let locals = empty_array::<Scalar, 2>().simple_mult_into_resize(
                        self.dc2e_inv_1[c2e_operator_index].view(),
                        empty_array::<Scalar, 2>().simple_mult_into_resize(
                            self.dc2e_inv_2[c2e_operator_index].view(),
                            empty_array::<Scalar, 2>().simple_mult_into_resize(
                                self.source_to_target.metadata[m2l_operator_index].u.view(),
                                compressed_check_potentials,
                            ),
                        ),
                    );
                    #[cfg(target_os = "linux")]
                    rlst::threading::disable_threading();
                    let ptr = self.level_locals[level as usize][0][0].raw;
                    let all_locals = unsafe {
                        std::slice::from_raw_parts_mut(ptr, ntargets * self.ncoeffs * nmatvecs)
                    };
                    all_locals
                        .iter_mut()
                        .zip(locals.data().iter())
                        .for_each(|(l, r)| *l += *r);
                }
            }
        }

        Ok(M2LResult(Duration::from_secs(0), Duration::from_secs(0), Duration::from_secs(0), Duration::from_secs(0), 0.))
    }

    fn p2l(&self, _level: u64) -> Result<(), FmmError> {
        Ok(())
    }
}

impl<Scalar, Kernel> KiFmm<Scalar, Kernel, BlasFieldTranslationIa<Scalar>>
where
    Scalar: RlstScalar + Default,
    Kernel: KernelTrait<T = Scalar> + HomogenousKernel + Default,
    <Scalar as RlstScalar>::Real: Default,
    Self: FmmOperatorData,
{
    /// Map between each transfer vector for homogenous kernels, and the source boxes involved in that translation
    /// at this octree level.
    ///
    /// Returns a vector of length 316, the maximum number of unique transfer vectors at a tree level for homogenous
    /// kernels, where each item is a Mutex locked vector of indices, of a length equal to the number of source boxes
    /// at this tree level, where an index value of -1 indicates that the box isn't involved in the translation, and
    /// an index values of `usize` gives the target box index that the source box density is being translated to.
    fn displacements(&self, level: u64) -> Vec<Mutex<Vec<i64>>> {
        let sources = self.tree.source_tree().keys(level).unwrap();
        let nsources = sources.len();
        let m2l_operator_index = self.m2l_operator_index(level);

        let all_displacements = vec![vec![-1i64; nsources]; 316];
        let all_displacements = all_displacements.into_iter().map(Mutex::new).collect_vec();

        sources
            .into_par_iter()
            .enumerate()
            .for_each(|(source_idx, source)| {
                // Find interaction list of each source, as this defines scatter locations
                let interaction_list = source
                    .parent()
                    .neighbors()
                    .iter()
                    .flat_map(|pn| pn.children())
                    .filter(|pnc| {
                        !source.is_adjacent(pnc)
                            && self
                                .tree
                                .target_tree()
                                .all_keys_set()
                                .unwrap()
                                .contains(pnc)
                    })
                    .collect_vec();

                let transfer_vectors = interaction_list
                    .iter()
                    .map(|target| source.find_transfer_vector(target).unwrap())
                    .collect_vec();

                let mut transfer_vectors_map = HashMap::new();
                for (i, v) in transfer_vectors.iter().enumerate() {
                    transfer_vectors_map.insert(v, i);
                }

                let transfer_vectors_set: HashSet<_> = transfer_vectors.iter().cloned().collect();

                // Mark items in interaction list for scattering
                for (tv_idx, tv) in self.source_to_target.transfer_vectors[m2l_operator_index]
                    .iter()
                    .enumerate()
                {
                    let mut all_displacements_lock = all_displacements[tv_idx].lock().unwrap();
                    if transfer_vectors_set.contains(&tv.hash) {
                        // Look up scatter location in target tree
                        let target =
                            &interaction_list[*transfer_vectors_map.get(&tv.hash).unwrap()];
                        let &target_idx = self.level_index_pointer_locals[level as usize]
                            .get(target)
                            .unwrap();
                        all_displacements_lock[source_idx] = target_idx as i64;
                    }
                }
            });
        all_displacements
    }
}

impl<Scalar, Kernel> SourceToTargetTranslation
    for KiFmm<Scalar, Kernel, BlasFieldTranslationIa<Scalar>>
where
    Scalar: RlstScalar + Default,
    Kernel: KernelTrait<T = Scalar> + HomogenousKernel + Default + Send + Sync,
    <Scalar as RlstScalar>::Real: Default,
    Self: FmmOperatorData,
{
    fn m2l(&self, level: u64) -> Result<M2LResult, FmmError> {
        let Some(targets) = self.tree().target_tree().keys(level) else {
            return Err(FmmError::Failed(format!(
                "M2L failed at level {:?}, no targets found",
                level
            )));
        };
        let Some(sources) = self.tree().source_tree().keys(level) else {
            return Err(FmmError::Failed(format!(
                "M2L failed at level {:?}, no sources found",
                level
            )));
        };

        if self.kernel.is_homogenous() {
            return Err(FmmError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "IA based M2L only implemented for Inhomogenous kernels",
            )));
        }

        let m2l_operator_index = self.m2l_operator_index(level);
        let c2e_operator_index = self.c2e_operator_index(level);

        // Compute the displacements
        let all_displacements = self.displacements(level);

        let multipole_idxs = all_displacements
            .iter()
            .map(|displacement| {
                displacement
                    .lock()
                    .unwrap()
                    .iter()
                    .enumerate()
                    .filter(|(_, &d)| d != -1)
                    .map(|(i, _)| i)
                    .collect_vec()
            })
            .collect_vec();

        let local_idxs = all_displacements
            .iter()
            .map(|displacements| {
                displacements
                    .lock()
                    .unwrap()
                    .iter()
                    .enumerate()
                    .filter(|(_, &d)| d != -1)
                    .map(|(_, &j)| j as usize)
                    .collect_vec()
            })
            .collect_vec();

        // Number of sources at this level
        let nsources = sources.len();
        let ntargets = targets.len();

        match self.fmm_eval_type {
            FmmEvalType::Vector => {
                // Lookup multipole data from source tree
                let multipoles = rlst_array_from_slice2!(
                    unsafe {
                        std::slice::from_raw_parts(
                            self.level_multipoles[level as usize][0][0].raw,
                            self.ncoeffs * nsources,
                        )
                    },
                    [self.ncoeffs, nsources]
                );

                // Allocate buffer to store check potentials
                let check_potentials = rlst_dynamic_array2!(Scalar, [self.ncoeffs, ntargets]);
                let mut check_potentials_ptrs = Vec::new();

                for i in 0..ntargets {
                    let raw = unsafe {
                        check_potentials.data().as_ptr().add(i * self.ncoeffs) as *mut Scalar
                    };
                    let send_ptr = SendPtrMut { raw };
                    check_potentials_ptrs.push(send_ptr);
                }

                let level_check_potentials =
                    check_potentials_ptrs.iter().map(Mutex::new).collect_vec();

                // 1. Apply BLAS operation
                {
                    (0..NTRANSFER_VECTORS_KIFMM)
                        .into_par_iter()
                        .zip(multipole_idxs)
                        .zip(local_idxs)
                        .for_each(|((c_idx, multipole_idxs), local_idxs)| {
                            let u = &self.source_to_target.metadata[m2l_operator_index].u[c_idx];
                            let vt = &self.source_to_target.metadata[m2l_operator_index].vt[c_idx];

                            let mut multipoles_subset =
                                rlst_dynamic_array2!(Scalar, [self.ncoeffs, multipole_idxs.len()]);

                            for (local_multipole_idx, &global_multipole_idx) in
                                multipole_idxs.iter().enumerate()
                            {
                                multipoles_subset.data_mut()[local_multipole_idx * self.ncoeffs
                                    ..(local_multipole_idx + 1) * self.ncoeffs]
                                    .copy_from_slice(
                                        &multipoles.data()[global_multipole_idx * self.ncoeffs
                                            ..(global_multipole_idx + 1) * self.ncoeffs],
                                    );
                            }

                            let check_potential = empty_array::<Scalar, 2>()
                                .simple_mult_into_resize(
                                    u.view(),
                                    empty_array::<Scalar, 2>().simple_mult_into_resize(
                                        vt.view(),
                                        multipoles_subset.view(),
                                    ),
                                );

                            for (multipole_idx, &local_idx) in local_idxs.iter().enumerate() {
                                let tmp = &check_potential.data()[multipole_idx * self.ncoeffs
                                    ..(multipole_idx + 1) * self.ncoeffs];

                                let check_potential_lock =
                                    level_check_potentials[local_idx].lock().unwrap();
                                let check_potential_ptr = check_potential_lock.raw;
                                let global_check_potential = unsafe {
                                    std::slice::from_raw_parts_mut(
                                        check_potential_ptr,
                                        self.ncoeffs,
                                    )
                                };

                                global_check_potential
                                    .iter_mut()
                                    .zip(tmp)
                                    .for_each(|(l, r)| *l += *r);
                            }
                        });
                }

                // 3. Compute local expansions from compressed check potentials
                {
                    #[cfg(target_os = "linux")]
                    rlst::threading::enable_threading();
                    let locals = empty_array::<Scalar, 2>().simple_mult_into_resize(
                        self.dc2e_inv_1[c2e_operator_index].view(),
                        empty_array::<Scalar, 2>().simple_mult_into_resize(
                            self.dc2e_inv_2[c2e_operator_index].view(),
                            check_potentials,
                        ),
                    );
                    #[cfg(target_os = "linux")]
                    rlst::threading::disable_threading();

                    let ptr = self.level_locals[level as usize][0][0].raw;
                    let all_locals =
                        unsafe { std::slice::from_raw_parts_mut(ptr, ntargets * self.ncoeffs) };
                    all_locals
                        .iter_mut()
                        .zip(locals.data().iter())
                        .for_each(|(l, r)| *l += *r);
                }

                return Ok(M2LResult(Duration::from_secs(0), Duration::from_secs(0), Duration::from_secs(0), Duration::from_secs(0), 0.))

            }
            FmmEvalType::Matrix(nmatvecs) => {
                // Lookup multipole data from source tree
                let multipoles = rlst_array_from_slice2!(
                    unsafe {
                        std::slice::from_raw_parts(
                            self.level_multipoles[level as usize][0][0].raw,
                            self.ncoeffs * nsources * nmatvecs,
                        )
                    },
                    [self.ncoeffs, nsources * nmatvecs]
                );

                let check_potentials =
                    rlst_dynamic_array2!(Scalar, [self.ncoeffs, nsources * nmatvecs]);

                let mut check_potentials_ptrs = Vec::new();

                for i in 0..ntargets {
                    let key_displacement = i * self.ncoeffs * nmatvecs;
                    let mut tmp = Vec::new();
                    for charge_vec_idx in 0..nmatvecs {
                        let charge_vec_displacement = charge_vec_idx * self.ncoeffs;

                        let raw = unsafe {
                            check_potentials
                                .data()
                                .as_ptr()
                                .add(key_displacement + charge_vec_displacement)
                                as *mut Scalar
                        };
                        let send_ptr = SendPtrMut { raw };
                        tmp.push(send_ptr)
                    }
                    check_potentials_ptrs.push(tmp);
                }

                let level_check_potentials =
                    check_potentials_ptrs.iter().map(Mutex::new).collect_vec();

                // 2. Apply the BLAS operation
                {
                    (0..NTRANSFER_VECTORS_KIFMM)
                        .into_par_iter()
                        .zip(multipole_idxs)
                        .zip(local_idxs)
                        .for_each(|((c_idx, multipole_idxs), local_idxs)| {
                            let u = &self.source_to_target.metadata[m2l_operator_index].u[c_idx];
                            let vt = &self.source_to_target.metadata[m2l_operator_index].vt[c_idx];

                            let mut multipoles_subset = rlst_dynamic_array2!(
                                Scalar,
                                [self.ncoeffs, multipole_idxs.len() * nmatvecs]
                            );

                            for (local_multipole_idx, &global_multipole_idx) in
                                multipole_idxs.iter().enumerate()
                            {
                                let key_displacement_global =
                                    global_multipole_idx * self.ncoeffs * nmatvecs;

                                let key_displacement_local =
                                    local_multipole_idx * self.ncoeffs * nmatvecs;

                                for charge_vec_idx in 0..nmatvecs {
                                    let charge_vec_displacement = charge_vec_idx * self.ncoeffs;

                                    multipoles_subset.data_mut()[key_displacement_local
                                        + charge_vec_displacement
                                        ..key_displacement_local
                                            + charge_vec_displacement
                                            + self.ncoeffs]
                                        .copy_from_slice(
                                            &multipoles.data()[key_displacement_global
                                                + charge_vec_displacement
                                                ..key_displacement_global
                                                    + charge_vec_displacement
                                                    + self.ncoeffs],
                                        );
                                }
                            }

                            let check_potential = empty_array::<Scalar, 2>()
                                .simple_mult_into_resize(
                                    u.view(),
                                    empty_array::<Scalar, 2>().simple_mult_into_resize(
                                        vt.view(),
                                        multipoles_subset.view(),
                                    ),
                                );

                            for (local_multipole_idx, &global_local_idx) in
                                local_idxs.iter().enumerate()
                            {
                                let check_potential_lock =
                                    level_check_potentials[global_local_idx].lock().unwrap();

                                for charge_vec_idx in 0..nmatvecs {
                                    let key_displacement =
                                        local_multipole_idx * self.ncoeffs * nmatvecs;
                                    let charge_vec_displacement = charge_vec_idx * self.ncoeffs;

                                    let tmp = &check_potential.data()[key_displacement
                                        + charge_vec_displacement
                                        ..key_displacement
                                            + charge_vec_displacement
                                            + self.ncoeffs];

                                    let check_potential_ptr =
                                        check_potential_lock[charge_vec_idx].raw;
                                    let check_potential = unsafe {
                                        std::slice::from_raw_parts_mut(
                                            check_potential_ptr,
                                            self.ncoeffs,
                                        )
                                    };

                                    check_potential
                                        .iter_mut()
                                        .zip(tmp)
                                        .for_each(|(l, r)| *l += *r);
                                }
                            }
                        });
                }

                // 3. Compute local expansions from compressed check potentials
                {
                    #[cfg(target_os = "linux")]
                    rlst::threading::enable_threading();
                    let locals = empty_array::<Scalar, 2>().simple_mult_into_resize(
                        self.dc2e_inv_1[c2e_operator_index].view(),
                        empty_array::<Scalar, 2>().simple_mult_into_resize(
                            self.dc2e_inv_2[c2e_operator_index].view(),
                            check_potentials,
                        ),
                    );
                    #[cfg(target_os = "linux")]
                    rlst::threading::disable_threading();

                    let ptr = self.level_locals[level as usize][0][0].raw;
                    let all_locals = unsafe {
                        std::slice::from_raw_parts_mut(ptr, ntargets * self.ncoeffs * nmatvecs)
                    };
                    all_locals
                        .iter_mut()
                        .zip(locals.data().iter())
                        .for_each(|(l, r)| *l += *r);
                }
            }
        }
        return Ok(M2LResult(Duration::from_secs(0), Duration::from_secs(0), Duration::from_secs(0), Duration::from_secs(0), 0.))
    }

    fn p2l(&self, _level: u64) -> Result<(), FmmError> {
        Err(FmmError::Unimplemented("P2L unimplemented".to_string()))
    }
}

impl KiFmmLaplaceMetal
where
    Self: FmmOperatorData,
{
    /// Map between each transfer vector for homogenous kernels, and the source boxes involved in that translation
    /// at this octree level.
    ///
    /// Returns a vector of length 316, the maximum number of unique transfer vectors at a tree level for homogenous
    /// kernels, where each item is a Mutex locked vector of indices, of a length equal to the number of source boxes
    /// at this tree level, where an index value of -1 indicates that the box isn't involved in the translation, and
    /// an index values of `usize` gives the target box index that the source box density is being translated to.
    fn displacements(&self, level: u64) -> Vec<Mutex<Vec<i64>>> {
        let sources = self.tree.source_tree().keys(level).unwrap();
        let nsources = sources.len();

        let all_displacements = vec![vec![-1i64; nsources]; 316];
        let all_displacements = all_displacements.into_iter().map(Mutex::new).collect_vec();

        sources
            .into_par_iter()
            .enumerate()
            .for_each(|(source_idx, source)| {
                // Find interaction list of each source, as this defines scatter locations
                let interaction_list = source
                    .parent()
                    .neighbors()
                    .iter()
                    .flat_map(|pn| pn.children())
                    .filter(|pnc| {
                        !source.is_adjacent(pnc)
                            && self
                                .tree
                                .target_tree()
                                .all_keys_set()
                                .unwrap()
                                .contains(pnc)
                    })
                    .collect_vec();

                let transfer_vectors = interaction_list
                    .iter()
                    .map(|target| source.find_transfer_vector(target).unwrap())
                    .collect_vec();

                let mut transfer_vectors_map = HashMap::new();
                for (i, v) in transfer_vectors.iter().enumerate() {
                    transfer_vectors_map.insert(v, i);
                }

                let transfer_vectors_set: HashSet<_> = transfer_vectors.iter().cloned().collect();

                // Mark items in interaction list for scattering
                for (tv_idx, tv) in self.source_to_target.transfer_vectors.iter().enumerate() {
                    let mut all_displacements_lock = all_displacements[tv_idx].lock().unwrap();
                    if transfer_vectors_set.contains(&tv.hash) {
                        // Look up scatter location in target tree
                        let target =
                            &interaction_list[*transfer_vectors_map.get(&tv.hash).unwrap()];
                        let &target_idx = self.level_index_pointer_locals[level as usize]
                            .get(target)
                            .unwrap();
                        all_displacements_lock[source_idx] = target_idx as i64;
                    }
                }
            });
        all_displacements
    }
}

impl SourceToTargetTranslation for KiFmmLaplaceMetal
where
    Self: FmmOperatorData,
{
    fn m2l(&self, level: u64) -> Result<M2LResult, FmmError> {
        let Some(targets) = self.tree().target_tree().keys(level) else {
            return Err(FmmError::Failed(
                "No target boxes at this level".to_string(),
            ));
        };
        let Some(sources) = self.tree().source_tree().keys(level) else {
            return Err(FmmError::Failed(
                "No source boxes at this level".to_string(),
            ));
        };

        let m2l_operator_index = self.m2l_operator_index(level);
        let c2e_operator_index = self.c2e_operator_index(level);

        // Compute the displacements
        let all_displacements = self.displacements(level);

        let multipole_idxs = all_displacements
            .iter()
            .map(|displacement| {
                displacement
                    .lock()
                    .unwrap()
                    .iter()
                    .enumerate()
                    .filter(|(_, &d)| d != -1)
                    .map(|(i, _)| i)
                    .collect_vec()
            })
            .collect_vec();

        let local_idxs = all_displacements
            .iter()
            .map(|displacements| {
                displacements
                    .lock()
                    .unwrap()
                    .iter()
                    .enumerate()
                    .filter(|(_, &d)| d != -1)
                    .map(|(_, &j)| j as usize)
                    .collect_vec()
            })
            .collect_vec();

        // Number of sources at this level
        let nsources = sources.len();
        let ntargets = targets.len();

        let matmul_time = Mutex::new(Duration::from_secs(0));
        let organisation_time = Mutex::new(Duration::from_secs(0));
        let allocation_time = Mutex::new(Duration::from_secs(0));
        let saving_time = Mutex::new(Duration::from_secs(0));
        let flops = Mutex::new(0);

        match self.fmm_eval_type {
            FmmEvalType::Vector => {
                // Lookup multipole data from source tree
                let multipoles = rlst_array_from_slice2!(
                    unsafe {
                        std::slice::from_raw_parts(
                            self.level_multipoles[level as usize][0][0].raw,
                            self.ncoeffs * nsources,
                        )
                    },
                    [self.ncoeffs, nsources]
                );

                // Allocate buffer to store compressed check potentials
                let compressed_check_potentials =
                    rlst_dynamic_array2!(f32, [self.source_to_target.cutoff_rank, ntargets]);
                let mut compressed_check_potentials_ptrs = Vec::new();

                for i in 0..ntargets {
                    let raw = unsafe {
                        compressed_check_potentials
                            .data()
                            .as_ptr()
                            .add(i * self.source_to_target.cutoff_rank)
                            as *mut f32
                    };
                    let send_ptr = SendPtrMut { raw };
                    compressed_check_potentials_ptrs.push(send_ptr);
                }

                let compressed_level_check_potentials = compressed_check_potentials_ptrs
                    .iter()
                    .map(Mutex::new)
                    .collect_vec();

                // 1. Compute the SVD compressed multipole expansions at this level
                let mut compressed_multipoles;
                let scale =
                    homogenous_kernel_scale::<f32>(level) * m2l_scale::<f32>(level).unwrap();
                // {
                compressed_multipoles = empty_array::<f32, 2>().simple_mult_into_resize(
                    self.source_to_target.metadata[m2l_operator_index].st.view(),
                    multipoles,
                );

                // 2. Apply BLAS operation
                if level < self.metal_level {
                    // Apply scale outside of matmul
                    compressed_multipoles.data_mut().iter_mut().for_each(|d| {
                        *d *=
                            homogenous_kernel_scale::<f32>(level) * m2l_scale::<f32>(level).unwrap()
                    });

                    (0..NTRANSFER_VECTORS_KIFMM)
                        .into_par_iter()
                        .zip(multipole_idxs)
                        .zip(local_idxs)
                        .for_each(|((c_idx, multipole_idxs), local_idxs)| {

                            let mut org_time = Duration::new(0, 0);
                            let mut mat_time = Duration::new(0, 0);
                            let mut alloc_time = Duration::new(0, 0);
                            let mut save_time = Duration::new(0, 0);

                            let s = Instant::now();
                            let c_u_sub =
                                &self.source_to_target.metadata[m2l_operator_index].c_u[c_idx];
                            let c_vt_sub =
                                &self.source_to_target.metadata[m2l_operator_index].c_vt[c_idx];

                            let mut compressed_multipoles_subset = rlst_dynamic_array2!(
                                f32,
                                [self.source_to_target.cutoff_rank, multipole_idxs.len()]
                            );
                            alloc_time += s.elapsed();

                            let s = Instant::now();
                            for (i, &multipole_idx) in multipole_idxs.iter().enumerate() {
                                compressed_multipoles_subset.data_mut()[i * self
                                    .source_to_target
                                    .cutoff_rank
                                    ..(i + 1) * self.source_to_target.cutoff_rank]
                                    .copy_from_slice(
                                        &compressed_multipoles.data()[multipole_idx
                                            * self.source_to_target.cutoff_rank
                                            ..(multipole_idx + 1)
                                                * self.source_to_target.cutoff_rank],
                                    );
                            }
                            org_time += s.elapsed();


                            let total_flops = c_vt_sub.shape().iter().product::<usize>() * compressed_check_potentials.shape()[1] + c_u_sub.shape().iter().product::<usize>() * compressed_check_potentials.shape()[1];
                            *flops .lock().unwrap() += total_flops;

                            let s = Instant::now();
                            let compressed_check_potential = empty_array::<f32, 2>()
                                .simple_mult_into_resize(
                                    c_u_sub.view(),
                                    empty_array::<f32, 2>().simple_mult_into_resize(
                                        c_vt_sub.view(),
                                        compressed_multipoles_subset.view(),
                                    ),
                                );
                            mat_time += s.elapsed();

                            let s = Instant::now();
                            for (multipole_idx, &local_idx) in local_idxs.iter().enumerate() {
                                let check_potential_lock =
                                    compressed_level_check_potentials[local_idx].lock().unwrap();
                                let check_potential_ptr = check_potential_lock.raw;
                                let check_potential = unsafe {
                                    std::slice::from_raw_parts_mut(
                                        check_potential_ptr,
                                        self.source_to_target.cutoff_rank,
                                    )
                                };
                                let tmp = &compressed_check_potential.data()[multipole_idx
                                    * self.source_to_target.cutoff_rank
                                    ..(multipole_idx + 1) * self.source_to_target.cutoff_rank];
                                check_potential
                                    .iter_mut()
                                    .zip(tmp)
                                    .for_each(|(l, r)| *l += *r);
                            }
                            save_time += s.elapsed();

                            *matmul_time.lock().unwrap() += mat_time;
                            *organisation_time.lock().unwrap() += org_time;
                            *allocation_time.lock().unwrap() += alloc_time;
                            *saving_time.lock().unwrap() += save_time;

                        });
                } else {
                    let device = MetalDevice::from_default();
                    (0..NTRANSFER_VECTORS_KIFMM)
                        .into_par_iter()
                        .zip(multipole_idxs)
                        .zip(local_idxs)
                        .for_each(|((c_idx, multipole_idxs), local_idxs)| {

                            let mut org_time = Duration::new(0, 0);
                            let mut mat_time = Duration::new(0, 0);
                            let mut alloc_time = Duration::new(0, 0);
                            let mut save_time = Duration::new(0, 0);

                            let s = Instant::now();
                            let c_u_sub_metal = &self.source_to_target.metadata[m2l_operator_index]
                                .c_u_metal[c_idx];
                            let c_vt_sub_metal = &self.source_to_target.metadata
                                [m2l_operator_index]
                                .c_vt_metal[c_idx];
                            // let c_sub_metal = &self.source_to_target.metadata[m2l_operator_index].c_metal[c_idx];

                            let mut compressed_multipoles_subset_metal = rlst_metal_array2!(
                                &device,
                                f32,
                                [self.source_to_target.cutoff_rank, multipole_idxs.len()]
                            );
                            alloc_time += s.elapsed();

                            let s = Instant::now();
                            for (i, &multipole_idx) in multipole_idxs.iter().enumerate() {
                                compressed_multipoles_subset_metal.data_mut()[i * self
                                    .source_to_target
                                    .cutoff_rank
                                    ..(i + 1) * self.source_to_target.cutoff_rank]
                                    .copy_from_slice(
                                        &compressed_multipoles.data()[multipole_idx
                                            * self.source_to_target.cutoff_rank
                                            ..(multipole_idx + 1)
                                                * self.source_to_target.cutoff_rank],
                                    );
                            }
                            org_time += s.elapsed();

                            let s = Instant::now();
                            let mut tmp = rlst_metal_array2!(
                                &device,
                                f32,
                                [
                                    c_vt_sub_metal.shape()[0],
                                    compressed_multipoles_subset_metal.shape()[1]
                                ]
                            );
                            let mut compressed_check_potential = rlst_metal_array2!(
                                &device,
                                f32,
                                [c_u_sub_metal.shape()[0], tmp.shape()[1]]
                            );
                            alloc_time += s.elapsed();

                            let total_flops = c_vt_sub_metal.shape().iter().product::<usize>() * compressed_check_potentials.shape()[1] + c_u_sub_metal.shape().iter().product::<usize>() * compressed_check_potentials.shape()[1];
                            *flops.lock().unwrap() += total_flops;

                            let s = Instant::now();
                            tmp.view_mut().metal_mult_into(
                                rlst::TransMode::NoTrans,
                                rlst::TransMode::NoTrans,
                                scale,
                                c_vt_sub_metal.view(),
                                compressed_multipoles_subset_metal.view(),
                                0.0,
                            );

                            compressed_check_potential.view_mut().metal_mult_into(
                                rlst::TransMode::NoTrans,
                                rlst::TransMode::NoTrans,
                                1.0,
                                c_u_sub_metal.view(),
                                tmp.view(),
                                0.0,
                            );
                            mat_time += s.elapsed();

                            let s = Instant::now();
                            for (multipole_idx, &local_idx) in local_idxs.iter().enumerate() {
                                let check_potential_lock =
                                    compressed_level_check_potentials[local_idx].lock().unwrap();
                                let check_potential_ptr = check_potential_lock.raw;
                                let check_potential = unsafe {
                                    std::slice::from_raw_parts_mut(
                                        check_potential_ptr,
                                        self.source_to_target.cutoff_rank,
                                    )
                                };
                                let tmp = &compressed_check_potential.data()[multipole_idx
                                    * self.source_to_target.cutoff_rank
                                    ..(multipole_idx + 1) * self.source_to_target.cutoff_rank];
                                check_potential
                                    .iter_mut()
                                    .zip(tmp)
                                    .for_each(|(l, r)| *l += *r);
                            }
                            save_time += s.elapsed();

                            *matmul_time.lock().unwrap() += mat_time;
                            *organisation_time.lock().unwrap() += org_time;
                            *allocation_time.lock().unwrap() += alloc_time;
                            *saving_time.lock().unwrap() += save_time;
                        });
                }

                // 3. Compute local expansions from compressed check potentials
                {
                    let locals = empty_array::<f32, 2>().simple_mult_into_resize(
                        self.dc2e_inv_1[c2e_operator_index].view(),
                        empty_array::<f32, 2>().simple_mult_into_resize(
                            self.dc2e_inv_2[c2e_operator_index].view(),
                            empty_array::<f32, 2>().simple_mult_into_resize(
                                self.source_to_target.metadata[m2l_operator_index].u.view(),
                                compressed_check_potentials,
                            ),
                        ),
                    );

                    let ptr = self.level_locals[level as usize][0][0].raw;
                    let all_locals =
                        unsafe { std::slice::from_raw_parts_mut(ptr, ntargets * self.ncoeffs) };
                    all_locals
                        .iter_mut()
                        .zip(locals.data().iter())
                        .for_each(|(l, r)| *l += *r);
                }

                let mut mean_flops = *flops.lock().unwrap() as f64 / (NTRANSFER_VECTORS_KIFMM as f64);

                return Ok(M2LResult(matmul_time.lock().unwrap().clone(), organisation_time.lock().unwrap().clone(), allocation_time.lock().unwrap().clone(), saving_time.lock().unwrap().clone(), mean_flops))
            }
            FmmEvalType::Matrix(nmatvecs) => {
                // Lookup multipole data from source tree
                let multipoles = rlst_array_from_slice2!(
                    unsafe {
                        std::slice::from_raw_parts(
                            self.level_multipoles[level as usize][0][0].raw,
                            self.ncoeffs * nsources * nmatvecs,
                        )
                    },
                    [self.ncoeffs, nsources * nmatvecs]
                );

                let compressed_check_potentials = rlst_dynamic_array2!(
                    f32,
                    [self.source_to_target.cutoff_rank, nsources * nmatvecs]
                );
                let mut compressed_check_potentials_ptrs = Vec::new();

                for i in 0..ntargets {
                    let key_displacement = i * self.source_to_target.cutoff_rank * nmatvecs;
                    let mut tmp = Vec::new();
                    for charge_vec_idx in 0..nmatvecs {
                        let charge_vec_displacement =
                            charge_vec_idx * self.source_to_target.cutoff_rank;

                        let raw = unsafe {
                            compressed_check_potentials
                                .data()
                                .as_ptr()
                                .add(key_displacement + charge_vec_displacement)
                                as *mut f32
                        };
                        let send_ptr = SendPtrMut { raw };
                        tmp.push(send_ptr)
                    }
                    compressed_check_potentials_ptrs.push(tmp);
                }

                let compressed_level_check_potentials = compressed_check_potentials_ptrs
                    .iter()
                    .map(Mutex::new)
                    .collect_vec();

                // 1. Compute the SVD compressed multipole expansions at this level
                let mut compressed_multipoles;
                let scale =
                    homogenous_kernel_scale::<f32>(level) * m2l_scale::<f32>(level).unwrap();
                {
                    compressed_multipoles = empty_array::<f32, 2>().simple_mult_into_resize(
                        self.source_to_target.metadata[m2l_operator_index].st.view(),
                        multipoles,
                    );
                }

                // 2. Apply the BLAS operation
                if level < self.metal_level {
                    compressed_multipoles.data_mut().iter_mut().for_each(|d| {
                        *d *=
                            homogenous_kernel_scale::<f32>(level) * m2l_scale::<f32>(level).unwrap()
                    });

                    (0..NTRANSFER_VECTORS_KIFMM)
                        .into_par_iter()
                        .zip(multipole_idxs)
                        .zip(local_idxs)
                        .for_each(|((c_idx, multipole_idxs), local_idxs)| {

                            let mut org_time = Duration::new(0, 0);
                            let mut mat_time = Duration::new(0, 0);
                            let mut alloc_time = Duration::new(0, 0);
                            let mut save_time = Duration::new(0, 0);

                            let s = Instant::now();
                            let c_u_sub =
                                &self.source_to_target.metadata[m2l_operator_index].c_u[c_idx];
                            let c_vt_sub =
                                &self.source_to_target.metadata[m2l_operator_index].c_vt[c_idx];

                            let mut compressed_multipoles_subset = rlst_dynamic_array2!(
                                f32,
                                [
                                    self.source_to_target.cutoff_rank,
                                    multipole_idxs.len() * nmatvecs
                                ]
                            );
                            alloc_time += s.elapsed();

                            let s = Instant::now();
                            for (local_multipole_idx, &global_multipole_idx) in
                                multipole_idxs.iter().enumerate()
                            {
                                let key_displacement_global = global_multipole_idx
                                    * self.source_to_target.cutoff_rank
                                    * nmatvecs;

                                let key_displacement_local = local_multipole_idx
                                    * self.source_to_target.cutoff_rank
                                    * nmatvecs;

                                for charge_vec_idx in 0..nmatvecs {
                                    let charge_vec_displacement =
                                        charge_vec_idx * self.source_to_target.cutoff_rank;

                                    compressed_multipoles_subset.data_mut()[key_displacement_local
                                        + charge_vec_displacement
                                        ..key_displacement_local
                                            + charge_vec_displacement
                                            + self.source_to_target.cutoff_rank]
                                        .copy_from_slice(
                                            &compressed_multipoles.data()[key_displacement_global
                                                + charge_vec_displacement
                                                ..key_displacement_global
                                                    + charge_vec_displacement
                                                    + self.source_to_target.cutoff_rank],
                                        );
                                }
                            }
                            org_time += s.elapsed();

                            let total_flops = c_vt_sub.shape().iter().product::<usize>() * compressed_check_potentials.shape()[1] + c_u_sub.shape().iter().product::<usize>() * compressed_check_potentials.shape()[1];
                            *flops .lock().unwrap() += total_flops;

                            let s = Instant::now();
                            let compressed_check_potential = empty_array::<f32, 2>()
                                .simple_mult_into_resize(
                                    c_u_sub.view(),
                                    empty_array::<f32, 2>().simple_mult_into_resize(
                                        c_vt_sub.view(),
                                        compressed_multipoles_subset.view(),
                                    ),
                                );
                            mat_time += s.elapsed();

                            let s = Instant::now();
                            for (local_multipole_idx, &global_local_idx) in
                                local_idxs.iter().enumerate()
                            {
                                let check_potential_lock = compressed_level_check_potentials
                                    [global_local_idx]
                                    .lock()
                                    .unwrap();

                                for charge_vec_idx in 0..nmatvecs {
                                    let check_potential_ptr =
                                        check_potential_lock[charge_vec_idx].raw;
                                    let check_potential = unsafe {
                                        std::slice::from_raw_parts_mut(
                                            check_potential_ptr,
                                            self.source_to_target.cutoff_rank,
                                        )
                                    };

                                    let key_displacement = local_multipole_idx
                                        * self.source_to_target.cutoff_rank
                                        * nmatvecs;
                                    let charge_vec_displacement =
                                        charge_vec_idx * self.source_to_target.cutoff_rank;

                                    let tmp = &compressed_check_potential.data()[key_displacement
                                        + charge_vec_displacement
                                        ..key_displacement
                                            + charge_vec_displacement
                                            + self.source_to_target.cutoff_rank];
                                    check_potential
                                        .iter_mut()
                                        .zip(tmp)
                                        .for_each(|(l, r)| *l += *r);
                                }
                            }
                            save_time += s.elapsed();

                            *matmul_time.lock().unwrap() += mat_time;
                            *organisation_time.lock().unwrap() += org_time;
                            *allocation_time.lock().unwrap() += alloc_time;
                            *saving_time.lock().unwrap() += save_time;
                        });
                } else {

                    let device = MetalDevice::from_default();

                    (0..NTRANSFER_VECTORS_KIFMM)
                        .into_par_iter()
                        .zip(multipole_idxs)
                        .zip(local_idxs)
                        .for_each(|((c_idx, multipole_idxs), local_idxs)| {

                            let mut org_time = Duration::new(0, 0);
                            let mut mat_time = Duration::new(0, 0);
                            let mut alloc_time = Duration::new(0, 0);
                            let mut save_time = Duration::new(0, 0);

                            let s = Instant::now();
                            let c_u_sub_metal = &self.source_to_target.metadata[m2l_operator_index]
                                .c_u_metal[c_idx];
                            let c_vt_sub_metal = &self.source_to_target.metadata
                                [m2l_operator_index]
                                .c_vt_metal[c_idx];

                            let mut compressed_multipoles_subset_metal = rlst_metal_array2!(
                                &device,
                                f32,
                                [
                                    self.source_to_target.cutoff_rank,
                                    multipole_idxs.len() * nmatvecs
                                ]
                            );
                            alloc_time += s.elapsed();

                            let s = Instant::now();
                            for (local_multipole_idx, &global_multipole_idx) in
                                multipole_idxs.iter().enumerate()
                            {
                                let key_displacement_global = global_multipole_idx
                                    * self.source_to_target.cutoff_rank
                                    * nmatvecs;

                                let key_displacement_local = local_multipole_idx
                                    * self.source_to_target.cutoff_rank
                                    * nmatvecs;

                                for charge_vec_idx in 0..nmatvecs {
                                    let charge_vec_displacement =
                                        charge_vec_idx * self.source_to_target.cutoff_rank;

                                    compressed_multipoles_subset_metal.data_mut()
                                        [key_displacement_local + charge_vec_displacement
                                            ..key_displacement_local
                                                + charge_vec_displacement
                                                + self.source_to_target.cutoff_rank]
                                        .copy_from_slice(
                                            &compressed_multipoles.data()[key_displacement_global
                                                + charge_vec_displacement
                                                ..key_displacement_global
                                                    + charge_vec_displacement
                                                    + self.source_to_target.cutoff_rank],
                                        );
                                }
                            }
                            org_time += s.elapsed();

                            let s = Instant::now();
                            let mut tmp = rlst_metal_array2!(
                                &device,
                                f32,
                                [
                                    c_vt_sub_metal.shape()[0],
                                    compressed_multipoles_subset_metal.shape()[1]
                                ]
                            );

                            let mut compressed_check_potential = rlst_metal_array2!(
                                &device,
                                f32,
                                [c_u_sub_metal.shape()[0], tmp.shape()[1]]
                            );
                            alloc_time += s.elapsed();

                            let total_flops = c_vt_sub_metal.shape().iter().product::<usize>() * compressed_check_potentials.shape()[1] + c_u_sub_metal.shape().iter().product::<usize>() * compressed_check_potentials.shape()[1];
                            *flops .lock().unwrap() += total_flops;

                            let s = Instant::now();
                            tmp.view_mut().metal_mult_into(
                                rlst::TransMode::NoTrans,
                                rlst::TransMode::NoTrans,
                                scale,
                                c_vt_sub_metal.view(),
                                compressed_multipoles_subset_metal.view(),
                                0.0,
                            );

                            compressed_check_potential.view_mut().metal_mult_into(
                                rlst::TransMode::NoTrans,
                                rlst::TransMode::NoTrans,
                                1.0,
                                c_u_sub_metal.view(),
                                tmp.view(),
                                0.0,
                            );
                            mat_time += s.elapsed();

                            let s = Instant::now();
                            for (local_multipole_idx, &global_local_idx) in
                                local_idxs.iter().enumerate()
                            {
                                let check_potential_lock = compressed_level_check_potentials
                                    [global_local_idx]
                                    .lock()
                                    .unwrap();

                                for charge_vec_idx in 0..nmatvecs {
                                    let check_potential_ptr =
                                        check_potential_lock[charge_vec_idx].raw;
                                    let check_potential = unsafe {
                                        std::slice::from_raw_parts_mut(
                                            check_potential_ptr,
                                            self.source_to_target.cutoff_rank,
                                        )
                                    };

                                    let key_displacement = local_multipole_idx
                                        * self.source_to_target.cutoff_rank
                                        * nmatvecs;
                                    let charge_vec_displacement =
                                        charge_vec_idx * self.source_to_target.cutoff_rank;

                                    let tmp = &compressed_check_potential.data()[key_displacement
                                        + charge_vec_displacement
                                        ..key_displacement
                                            + charge_vec_displacement
                                            + self.source_to_target.cutoff_rank];
                                    check_potential
                                        .iter_mut()
                                        .zip(tmp)
                                        .for_each(|(l, r)| *l += *r);
                                }
                            }
                            save_time += s.elapsed();

                            *matmul_time.lock().unwrap() += mat_time;
                            *organisation_time.lock().unwrap() += org_time;
                            *allocation_time.lock().unwrap() += alloc_time;
                            *saving_time.lock().unwrap() += save_time;
                        });
                }

                // 3. Compute local expansions from compressed check potentials
                {
                    let locals = empty_array::<f32, 2>().simple_mult_into_resize(
                        self.dc2e_inv_1[c2e_operator_index].view(),
                        empty_array::<f32, 2>().simple_mult_into_resize(
                            self.dc2e_inv_2[c2e_operator_index].view(),
                            empty_array::<f32, 2>().simple_mult_into_resize(
                                self.source_to_target.metadata[m2l_operator_index].u.view(),
                                compressed_check_potentials,
                            ),
                        ),
                    );

                    let ptr = self.level_locals[level as usize][0][0].raw;
                    let all_locals = unsafe {
                        std::slice::from_raw_parts_mut(ptr, ntargets * self.ncoeffs * nmatvecs)
                    };
                    all_locals
                        .iter_mut()
                        .zip(locals.data().iter())
                        .for_each(|(l, r)| *l += *r);
                }

            let mut mean_flops = *flops.lock().unwrap() as f64 / (NTRANSFER_VECTORS_KIFMM as f64);
            return Ok(M2LResult(matmul_time.lock().unwrap().clone(), organisation_time.lock().unwrap().clone(), allocation_time.lock().unwrap().clone(), saving_time.lock().unwrap().clone(), mean_flops))
            }
        }

    }

    fn p2l(&self, _level: u64) -> Result<(), FmmError> {
        Ok(())
    }
}
