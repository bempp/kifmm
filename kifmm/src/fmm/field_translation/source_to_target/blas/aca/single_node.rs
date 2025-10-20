//! Multipole to local field translation trait implementation using BLAS and ACA+ compression

use std::sync::Mutex;

use crate::{
    fmm::{
        helpers::single_node::{homogenous_kernel_scale, m2l_scale},
        types::{BlasFieldTranslationAca, FmmEvalType, SendPtrMut},
    },
    traits::{
        field::SourceToTargetTranslation,
        fmm::{HomogenousKernel, MetadataAccess},
        general::single_node::{ArgmaxValue, Cast, Epsilon, Upcast},
        tree::{SingleFmmTree, SingleTree},
        types::FmmError,
    },
    tree::constants::NTRANSFER_VECTORS_KIFMM,
    DataAccess, KiFmm,
};
use green_kernels::traits::Kernel as KernelTrait;
use itertools::Itertools;
use rayon::prelude::*;
use rlst::{
    rlst_array_from_slice2, rlst_dynamic_array2, MatrixQr, MatrixSvd, MultInto, RawAccess,
    RawAccessMut, RlstScalar, Shape,
};

impl<Scalar, Kernel> SourceToTargetTranslation
    for KiFmm<Scalar, Kernel, BlasFieldTranslationAca<Scalar>>
where
    Scalar: RlstScalar
        + Default
        + Epsilon
        + MatrixSvd
        + Epsilon
        + MatrixQr
        + Upcast
        + ArgmaxValue<Scalar>
        + Cast<<Scalar as Upcast>::Higher>,
    <Scalar as RlstScalar>::Real: Default
        + Epsilon
        + Upcast
        + Cast<<<Scalar as Upcast>::Higher as RlstScalar>::Real>
        + ArgmaxValue<<Scalar as RlstScalar>::Real>,
    <Scalar as Upcast>::Higher: RlstScalar + MatrixSvd + Epsilon + Cast<Scalar>,
    <<Scalar as Upcast>::Higher as RlstScalar>::Real: Epsilon + Cast<Scalar::Real>,
    Kernel: KernelTrait<T = Scalar> + HomogenousKernel + Default + Send + Sync,
    Self: MetadataAccess + DataAccess<Scalar = Scalar, Kernel = Kernel>,
{
    fn m2l(&self, level: u64) -> Result<(), crate::traits::types::FmmError> {
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
        let n_coeffs_check_surface = self.n_coeffs_check_surface(level);

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

                // Allocate buffer to store check potentials
                let check_potentials =
                    rlst_dynamic_array2!(Scalar, [n_coeffs_check_surface, n_targets]);

                let mut check_potential_ptrs = Vec::new();

                for i in 0..n_targets {
                    let raw = unsafe {
                        check_potentials
                            .data()
                            .as_ptr()
                            .add(i * n_coeffs_check_surface) as *mut Scalar
                    };
                    let send_ptr = SendPtrMut { raw };
                    check_potential_ptrs.push(send_ptr);
                }

                let level_check_potentials =
                    check_potential_ptrs.iter().map(Mutex::new).collect_vec();

                let mut scale = Scalar::one();
                if self.kernel.is_homogenous() {
                    scale = homogenous_kernel_scale::<Scalar>(level) * m2l_scale(level).unwrap();
                }

                // 1. Apply BLAS operation
                {
                    (0..NTRANSFER_VECTORS_KIFMM)
                        .into_par_iter()
                        .zip(multipole_idxs)
                        .zip(local_idxs)
                        .for_each(|((t_idx, multipole_idxs), local_idxs)| {
                            let u_i = &self.source_to_target.metadata[m2l_operator_index].u[t_idx];
                            let vt_i =
                                &self.source_to_target.metadata[m2l_operator_index].vt[t_idx];

                            let mut multipoles_subset = rlst_dynamic_array2!(
                                Scalar,
                                [n_coeffs_equivalent_surface, multipole_idxs.len()]
                            );

                            for (i, &multipole_idx) in multipole_idxs.iter().enumerate() {
                                multipoles_subset.data_mut()[i * n_coeffs_equivalent_surface
                                    ..(i + 1) * n_coeffs_equivalent_surface]
                                    .copy_from_slice(
                                        &multipoles.data()[multipole_idx
                                            * n_coeffs_equivalent_surface
                                            ..(multipole_idx + 1) * n_coeffs_equivalent_surface],
                                    );
                            }

                            // Apply scale
                            multipoles_subset
                                .data_mut()
                                .iter_mut()
                                .for_each(|x| *x *= scale);

                            // Apply right decomposition
                            let [m, _k] = vt_i.shape();
                            let [_k, n] = multipoles_subset.shape();

                            let mut tmp1 = rlst_dynamic_array2!(Scalar, [m, n]);
                            tmp1.r_mut()
                                .simple_mult_into(vt_i.r(), multipoles_subset.r());

                            // Apply left decomposition
                            let [m, _k] = u_i.shape();
                            let [_k, n] = tmp1.shape();
                            let mut check_potential = rlst_dynamic_array2!(Scalar, [m, n]);
                            check_potential.r_mut().simple_mult_into(u_i.r(), tmp1.r());

                            // Save results to global vector
                            for (multipole_idx, &local_idx) in local_idxs.iter().enumerate() {
                                let check_potential_lock =
                                    level_check_potentials[local_idx].lock().unwrap();
                                let check_potential_ptr = check_potential_lock.raw;
                                let check_potential_global = unsafe {
                                    std::slice::from_raw_parts_mut(
                                        check_potential_ptr,
                                        n_coeffs_check_surface,
                                    )
                                };

                                let tmp = &check_potential.data()[multipole_idx
                                    * n_coeffs_check_surface
                                    ..(multipole_idx + 1) * n_coeffs_check_surface];

                                check_potential_global
                                    .iter_mut()
                                    .zip(tmp)
                                    .for_each(|(l, &r)| *l += r);
                            }
                        });
                }

                // 2. Compute local expansions from check potentials
                {
                    let [m, _k] = self.dc2e_inv_2[c2e_operator_index].shape();
                    let [_k, n] = check_potentials.shape();
                    let mut tmp = rlst_dynamic_array2!(Scalar, [m, n]);
                    tmp.r_mut().simple_mult_into(
                        self.dc2e_inv_2[c2e_operator_index].r(),
                        check_potentials.r(),
                    );

                    let mut locals =
                        rlst_dynamic_array2!(Scalar, [n_coeffs_equivalent_surface, n_targets]);
                    locals
                        .r_mut()
                        .simple_mult_into(self.dc2e_inv_1[c2e_operator_index].r(), tmp.r());

                    let ptr = self.level_locals[level as usize][0][0].raw;
                    let all_locals = unsafe {
                        std::slice::from_raw_parts_mut(ptr, n_targets * n_coeffs_equivalent_surface)
                    };
                    all_locals
                        .iter_mut()
                        .zip(locals.data().iter())
                        .for_each(|(l, r)| *l += *r);
                }

                Ok(())
            }

            FmmEvalType::Matrix(n_matvecs) => {
                let multipoles = rlst_array_from_slice2!(
                    multipoles,
                    [n_coeffs_equivalent_surface, n_sources * n_matvecs]
                );

                let check_potentials =
                    rlst_dynamic_array2!(Scalar, [n_coeffs_check_surface, n_targets * n_matvecs]);

                let mut check_potentials_ptrs = Vec::new();

                for i in 0..n_targets {
                    let key_displacement = i * n_coeffs_equivalent_surface * n_matvecs;

                    let mut tmp = Vec::new();
                    for charge_vec_idx in 0..n_matvecs {
                        let charge_vec_displacement = n_coeffs_check_surface * charge_vec_idx;
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
                    check_potentials_ptrs.push(tmp)
                }

                let level_check_potentials =
                    check_potentials_ptrs.iter().map(Mutex::new).collect_vec();

                let mut scale = Scalar::one();
                if self.kernel.is_homogenous() {
                    scale = homogenous_kernel_scale::<Scalar>(level) * m2l_scale(level).unwrap();
                }

                // 1. Apply BLAS operation
                {
                    (0..NTRANSFER_VECTORS_KIFMM)
                        .into_par_iter()
                        .zip(multipole_idxs)
                        .zip(local_idxs)
                        .for_each(|((t_idx, multipole_idxs), local_idxs)| {
                            let u_i = &self.source_to_target.metadata[m2l_operator_index].u[t_idx];
                            let vt_i =
                                &self.source_to_target.metadata[m2l_operator_index].vt[t_idx];

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
                                        )
                                }
                            }

                            // Apply scale
                            multipoles_subset
                                .data_mut()
                                .iter_mut()
                                .for_each(|x| *x *= scale);

                            let [m, _k] = vt_i.shape();
                            let [_k, n] = multipoles_subset.shape();

                            // Apply right decomposition
                            let mut tmp1 = rlst_dynamic_array2!(Scalar, [m, n]);
                            tmp1.r_mut()
                                .simple_mult_into(vt_i.r(), multipoles_subset.r());

                            // Apply left decomposition
                            let [m, _k] = u_i.shape();
                            let [_k, n] = tmp1.shape();
                            let mut check_potential = rlst_dynamic_array2!(Scalar, [m, n]);
                            check_potential.r_mut().simple_mult_into(u_i.r(), tmp1.r());

                            // Save results to global vector
                            for (local_multipole_idx, &global_local_idx) in
                                local_idxs.iter().enumerate()
                            {
                                let check_potential_lock =
                                    level_check_potentials[global_local_idx].lock().unwrap();

                                for charge_vec_idx in 0..n_matvecs {
                                    let check_potential_ptr =
                                        check_potential_lock[charge_vec_idx].raw;
                                    let global_check_potential = unsafe {
                                        std::slice::from_raw_parts_mut(
                                            check_potential_ptr,
                                            n_coeffs_check_surface,
                                        )
                                    };

                                    let key_displacement =
                                        local_multipole_idx * n_coeffs_check_surface * n_matvecs;
                                    let charge_vec_displacement = charge_vec_idx
                                        * n_coeffs_check_surface
                                        * n_coeffs_check_surface;

                                    let tmp = &check_potentials.data()[key_displacement
                                        + charge_vec_displacement
                                        ..key_displacement
                                            + charge_vec_displacement
                                            + n_coeffs_check_surface];

                                    global_check_potential
                                        .iter_mut()
                                        .zip(tmp)
                                        .for_each(|(l, r)| *l += *r);
                                }
                            }
                        });
                }

                Ok(())
            }
        }
    }

    fn p2l(&self, _level: u64) -> Result<(), crate::traits::types::FmmError> {
        Ok(())
    }
}
