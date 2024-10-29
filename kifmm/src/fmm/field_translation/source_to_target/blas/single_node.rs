//! Multipole to local field translation trait implementation using BLAS.

use std::sync::Mutex;

use itertools::Itertools;
use rayon::prelude::*;
use rlst::{
    empty_array, rlst_array_from_slice2, rlst_dynamic_array2, MultIntoResize, RawAccess,
    RawAccessMut, RlstScalar,
};

use green_kernels::traits::Kernel as KernelTrait;

use crate::{
    fmm::{
        helpers::single_node::{homogenous_kernel_scale, m2l_scale},
        types::{BlasFieldTranslationIa, FmmEvalType, SendPtrMut},
    },
    traits::{
        field::SourceToTargetTranslation,
        fmm::{DataAccess, HomogenousKernel, MetadataAccess},
        tree::{SingleFmmTree, SingleTree},
        types::FmmError,
    },
    tree::constants::NTRANSFER_VECTORS_KIFMM,
    BlasFieldTranslationSaRcmp, KiFmm,
};

impl<Scalar, Kernel> SourceToTargetTranslation
    for KiFmm<Scalar, Kernel, BlasFieldTranslationSaRcmp<Scalar>>
where
    Scalar: RlstScalar + Default,
    Kernel: KernelTrait<T = Scalar> + HomogenousKernel + Default + Send + Sync,
    <Scalar as RlstScalar>::Real: Default,
    Self: MetadataAccess + DataAccess<Scalar = Scalar, Kernel = Kernel>,
{
    fn m2l(&self, level: u64) -> Result<(), FmmError> {
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
        let displacement_index = self.displacement_index(level);
        let n_coeffs_equivalent_surface = self.n_coeffs_equivalent_surface(level);

        // let sentinel = sources.len();
        let sentinel = -1i32;

        // Compute the displacements
        let all_displacements = &self.source_to_target.displacements[displacement_index];

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

        // Number of sources at this level
        let n_sources = sources.len();
        let n_targets = targets.len();

        // Lookup multipole data from source tree
        let multipoles = self.multipoles(level).unwrap();

        match self.fmm_eval_type {
            FmmEvalType::Vector => {
                let multipoles =
                    rlst_array_from_slice2!(multipoles, [n_coeffs_equivalent_surface, n_sources]);

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

                // 1. Compute the SVD compressed multipole expansions at this level
                let mut compressed_multipoles;
                {
                    compressed_multipoles = empty_array::<Scalar, 2>().simple_mult_into_resize(
                        self.source_to_target.metadata[m2l_operator_index].st.view(),
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
                                    self.source_to_target.cutoff_rank[m2l_operator_index],
                                    multipole_idxs.len()
                                ]
                            );

                            for (i, &multipole_idx) in multipole_idxs.iter().enumerate() {
                                compressed_multipoles_subset.data_mut()[i * self
                                    .source_to_target
                                    .cutoff_rank[m2l_operator_index]
                                    ..(i + 1)
                                        * self.source_to_target.cutoff_rank[m2l_operator_index]]
                                    .copy_from_slice(
                                        &compressed_multipoles.data()[multipole_idx
                                            * self.source_to_target.cutoff_rank[m2l_operator_index]
                                            ..(multipole_idx + 1)
                                                * self.source_to_target.cutoff_rank
                                                    [m2l_operator_index]],
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
                                        self.source_to_target.cutoff_rank[m2l_operator_index],
                                    )
                                };
                                let tmp = &compressed_check_potential.data()[multipole_idx
                                    * self.source_to_target.cutoff_rank[m2l_operator_index]
                                    ..(multipole_idx + 1)
                                        * self.source_to_target.cutoff_rank[m2l_operator_index]];
                                check_potential
                                    .iter_mut()
                                    .zip(tmp)
                                    .for_each(|(l, r)| *l += *r);
                            }
                        });
                }

                // 3. Compute local expansions from compressed check potentials
                {
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

                    let ptr = self.level_locals[level as usize][0][0].raw;
                    let all_locals = unsafe {
                        std::slice::from_raw_parts_mut(ptr, n_targets * n_coeffs_equivalent_surface)
                    };
                    all_locals
                        .iter_mut()
                        .zip(locals.data().iter())
                        .for_each(|(l, r)| *l += *r);
                }

                return Ok(());
            }
            FmmEvalType::Matrix(n_matvecs) => {
                let multipoles = rlst_array_from_slice2!(
                    multipoles,
                    [n_coeffs_equivalent_surface, n_sources * n_matvecs]
                );

                let compressed_check_potentials = rlst_dynamic_array2!(
                    Scalar,
                    [
                        self.source_to_target.cutoff_rank[m2l_operator_index],
                        n_targets * n_matvecs
                    ]
                );

                let mut compressed_check_potentials_ptrs = Vec::new();

                for i in 0..n_targets {
                    let key_displacement =
                        i * self.source_to_target.cutoff_rank[m2l_operator_index] * n_matvecs;
                    let mut tmp = Vec::new();
                    for charge_vec_idx in 0..n_matvecs {
                        let charge_vec_displacement =
                            charge_vec_idx * self.source_to_target.cutoff_rank[m2l_operator_index];

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
                    compressed_multipoles = empty_array::<Scalar, 2>().simple_mult_into_resize(
                        self.source_to_target.metadata[m2l_operator_index].st.view(),
                        multipoles,
                    );

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
                                    self.source_to_target.cutoff_rank[m2l_operator_index],
                                    multipole_idxs.len() * n_matvecs
                                ]
                            );

                            for (local_multipole_idx, &global_multipole_idx) in
                                multipole_idxs.iter().enumerate()
                            {
                                let key_displacement_global = global_multipole_idx
                                    * self.source_to_target.cutoff_rank[m2l_operator_index]
                                    * n_matvecs;

                                let key_displacement_local = local_multipole_idx
                                    * self.source_to_target.cutoff_rank[m2l_operator_index]
                                    * n_matvecs;

                                for charge_vec_idx in 0..n_matvecs {
                                    let charge_vec_displacement = charge_vec_idx
                                        * self.source_to_target.cutoff_rank[m2l_operator_index];

                                    compressed_multipoles_subset.data_mut()[key_displacement_local
                                        + charge_vec_displacement
                                        ..key_displacement_local
                                            + charge_vec_displacement
                                            + self.source_to_target.cutoff_rank
                                                [m2l_operator_index]]
                                        .copy_from_slice(
                                            &compressed_multipoles.data()[key_displacement_global
                                                + charge_vec_displacement
                                                ..key_displacement_global
                                                    + charge_vec_displacement
                                                    + self.source_to_target.cutoff_rank
                                                        [m2l_operator_index]],
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

                                for charge_vec_idx in 0..n_matvecs {
                                    let check_potential_ptr =
                                        check_potential_lock[charge_vec_idx].raw;
                                    let check_potential = unsafe {
                                        std::slice::from_raw_parts_mut(
                                            check_potential_ptr,
                                            self.source_to_target.cutoff_rank[m2l_operator_index],
                                        )
                                    };

                                    let key_displacement = local_multipole_idx
                                        * self.source_to_target.cutoff_rank[m2l_operator_index]
                                        * n_matvecs;
                                    let charge_vec_displacement = charge_vec_idx
                                        * self.source_to_target.cutoff_rank[m2l_operator_index];

                                    let tmp = &compressed_check_potential.data()[key_displacement
                                        + charge_vec_displacement
                                        ..key_displacement
                                            + charge_vec_displacement
                                            + self.source_to_target.cutoff_rank
                                                [m2l_operator_index]];
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
                    let ptr = self.level_locals[level as usize][0][0].raw;
                    let all_locals = unsafe {
                        std::slice::from_raw_parts_mut(
                            ptr,
                            n_targets * n_coeffs_equivalent_surface * n_matvecs,
                        )
                    };
                    all_locals
                        .iter_mut()
                        .zip(locals.data().iter())
                        .for_each(|(l, r)| *l += *r);
                }
            }
        }

        Ok(())
    }

    fn p2l(&self, _level: u64) -> Result<(), FmmError> {
        Ok(())
    }
}

impl<Scalar, Kernel> SourceToTargetTranslation
    for KiFmm<Scalar, Kernel, BlasFieldTranslationIa<Scalar>>
where
    Scalar: RlstScalar + Default,
    Kernel: KernelTrait<T = Scalar> + HomogenousKernel + Default + Send + Sync,
    <Scalar as RlstScalar>::Real: Default,
    Self: MetadataAccess,
{
    fn m2l(&self, level: u64) -> Result<(), FmmError> {
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
        let displacement_index = self.displacement_index(level);
        let n_coeffs_equivalent_surface = self.n_coeffs_equivalent_surface(level);
        let n_coeffs_check_surface = self.n_coeffs_check_surface(level);
        let sentinel = sources.len();

        // Compute the displacements
        let all_displacements = &self.source_to_target.displacements[displacement_index];

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
                    .map(|(_, &j)| j)
                    .collect_vec()
            })
            .collect_vec();

        // Number of sources at this level
        let n_sources = sources.len();
        let n_targets = targets.len();

        match self.fmm_eval_type {
            FmmEvalType::Vector => {
                // Lookup multipole data from source tree
                let multipoles = rlst_array_from_slice2!(
                    unsafe {
                        std::slice::from_raw_parts(
                            self.level_multipoles[level as usize][0][0].raw,
                            n_coeffs_equivalent_surface * n_sources,
                        )
                    },
                    [n_coeffs_equivalent_surface, n_sources]
                );

                // Allocate buffer to store check potentials
                let check_potentials =
                    rlst_dynamic_array2!(Scalar, [n_coeffs_check_surface, n_targets]);
                let mut check_potentials_ptrs = Vec::new();

                for i in 0..n_targets {
                    let raw = unsafe {
                        check_potentials
                            .data()
                            .as_ptr()
                            .add(i * n_coeffs_check_surface) as *mut Scalar
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

                            let mut multipoles_subset = rlst_dynamic_array2!(
                                Scalar,
                                [n_coeffs_equivalent_surface, multipole_idxs.len()]
                            );

                            for (local_multipole_idx, &global_multipole_idx) in
                                multipole_idxs.iter().enumerate()
                            {
                                multipoles_subset.data_mut()[local_multipole_idx
                                    * n_coeffs_equivalent_surface
                                    ..(local_multipole_idx + 1) * n_coeffs_equivalent_surface]
                                    .copy_from_slice(
                                        &multipoles.data()[global_multipole_idx
                                            * n_coeffs_equivalent_surface
                                            ..(global_multipole_idx + 1)
                                                * n_coeffs_equivalent_surface],
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

                // 2. Compute local expansions from compressed check potentials
                {
                    let locals = empty_array::<Scalar, 2>().simple_mult_into_resize(
                        self.dc2e_inv_1[c2e_operator_index].view(),
                        empty_array::<Scalar, 2>().simple_mult_into_resize(
                            self.dc2e_inv_2[c2e_operator_index].view(),
                            check_potentials,
                        ),
                    );

                    let ptr = self.level_locals[level as usize][0][0].raw;
                    let all_locals = unsafe {
                        std::slice::from_raw_parts_mut(ptr, n_targets * n_coeffs_equivalent_surface)
                    };
                    all_locals
                        .iter_mut()
                        .zip(locals.data().iter())
                        .for_each(|(l, r)| *l += *r);
                }

                return Ok(());
            }
            FmmEvalType::Matrix(n_matvecs) => {
                // Lookup multipole data from source tree
                let multipoles = rlst_array_from_slice2!(
                    unsafe {
                        std::slice::from_raw_parts(
                            self.level_multipoles[level as usize][0][0].raw,
                            n_coeffs_equivalent_surface * n_sources * n_matvecs,
                        )
                    },
                    [n_coeffs_equivalent_surface, n_sources * n_matvecs]
                );

                let check_potentials =
                    rlst_dynamic_array2!(Scalar, [n_coeffs_check_surface, n_sources * n_matvecs]);

                let mut check_potentials_ptrs = Vec::new();

                for i in 0..n_targets {
                    let key_displacement = i * n_coeffs_check_surface * n_matvecs;
                    let mut tmp = Vec::new();
                    for charge_vec_idx in 0..n_matvecs {
                        let charge_vec_displacement = charge_vec_idx * n_coeffs_check_surface;

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

                // 1. Apply the BLAS operation
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
                                [
                                    n_coeffs_equivalent_surface,
                                    multipole_idxs.len() * n_matvecs
                                ]
                            );

                            for (local_multipole_idx, &global_multipole_idx) in
                                multipole_idxs.iter().enumerate()
                            {
                                let key_displacement_global =
                                    global_multipole_idx * n_coeffs_equivalent_surface * n_matvecs;

                                let key_displacement_local =
                                    local_multipole_idx * n_coeffs_equivalent_surface * n_matvecs;

                                for charge_vec_idx in 0..n_matvecs {
                                    let charge_vec_displacement =
                                        charge_vec_idx * n_coeffs_equivalent_surface;

                                    multipoles_subset.data_mut()[key_displacement_local
                                        + charge_vec_displacement
                                        ..key_displacement_local
                                            + charge_vec_displacement
                                            + n_coeffs_equivalent_surface]
                                        .copy_from_slice(
                                            &multipoles.data()[key_displacement_global
                                                + charge_vec_displacement
                                                ..key_displacement_global
                                                    + charge_vec_displacement
                                                    + n_coeffs_equivalent_surface],
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

                                for charge_vec_idx in 0..n_matvecs {
                                    let key_displacement =
                                        local_multipole_idx * n_coeffs_check_surface * n_matvecs;
                                    let charge_vec_displacement =
                                        charge_vec_idx * n_coeffs_check_surface;

                                    let tmp = &check_potential.data()[key_displacement
                                        + charge_vec_displacement
                                        ..key_displacement
                                            + charge_vec_displacement
                                            + n_coeffs_check_surface];

                                    let check_potential_ptr =
                                        check_potential_lock[charge_vec_idx].raw;
                                    let check_potential = unsafe {
                                        std::slice::from_raw_parts_mut(
                                            check_potential_ptr,
                                            n_coeffs_check_surface,
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

                // 2. Compute local expansions from compressed check potentials
                {
                    let locals = empty_array::<Scalar, 2>().simple_mult_into_resize(
                        self.dc2e_inv_1[c2e_operator_index].view(),
                        empty_array::<Scalar, 2>().simple_mult_into_resize(
                            self.dc2e_inv_2[c2e_operator_index].view(),
                            check_potentials,
                        ),
                    );

                    let ptr = self.level_locals[level as usize][0][0].raw;
                    let all_locals = unsafe {
                        std::slice::from_raw_parts_mut(
                            ptr,
                            n_targets * n_coeffs_equivalent_surface * n_matvecs,
                        )
                    };
                    all_locals
                        .iter_mut()
                        .zip(locals.data().iter())
                        .for_each(|(l, r)| *l += *r);
                }
            }
        }
        Ok(())
    }

    fn p2l(&self, _level: u64) -> Result<(), FmmError> {
        Err(FmmError::Unimplemented("P2L unimplemented".to_string()))
    }
}
