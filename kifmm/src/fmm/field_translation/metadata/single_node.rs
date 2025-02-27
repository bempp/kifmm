//! Implementation of traits to compute metadata for field translation operations.
use std::{
    collections::{HashMap, HashSet},
    sync::{Mutex, RwLock},
};

use green_kernels::{
    helmholtz_3d::Helmholtz3dKernel, laplace_3d::Laplace3dKernel, traits::Kernel as KernelTrait,
    types::GreenKernelEvalType,
};
use itertools::Itertools;
use num::{Float, Zero};
use rayon::prelude::*;
use rlst::{
    empty_array, rlst_array_from_slice2, rlst_dynamic_array2, rlst_dynamic_array3, Array,
    BaseArray, Gemm, MatrixSvd, MultIntoResize, RawAccess, RawAccessMut, RlstScalar, Shape,
    SvdMode, UnsafeRandomAccessByRef, UnsafeRandomAccessMut, VectorContainer,
};

use crate::{
    fmm::{
        constants::DEFAULT_M2L_FFT_BLOCK_SIZE,
        field_translation::source_to_target::transfer_vector::compute_transfer_vectors_at_level,
        helpers::single_node::{
            coordinate_index_pointer_single_node, flip3, homogenous_kernel_scale,
            leaf_expansion_pointers_single_node, leaf_scales_single_node,
            leaf_surfaces_single_node, level_expansion_pointers_single_node,
            level_index_pointer_single_node, ncoeffs_kifmm, potential_pointers_single_node,
        },
        types::{
            BlasFieldTranslationIa, BlasFieldTranslationSaRcmp, BlasMetadataIa, BlasMetadataSaRcmp,
            FftFieldTranslation, FftMetadata, FmmSvdMode,
        },
    },
    linalg::{pinv::pinv, rsvd::MatrixRsvd},
    traits::{
        fftw::{Dft, DftType},
        field::{
            FieldTranslation as FieldTranslationTrait, SourceToTargetTranslationMetadata,
            SourceTranslationMetadata, TargetTranslationMetadata,
        },
        fmm::{DataAccess, HomogenousKernel, Metadata, MetadataAccess},
        general::single_node::{AsComplex, Epsilon},
        tree::{Domain as DomainTrait, FmmTreeNode, SingleFmmTree, SingleTree},
    },
    tree::{
        constants::{
            ALPHA_INNER, ALPHA_OUTER, NHALO, NSIBLINGS, NSIBLINGS_SQUARED, NTRANSFER_VECTORS_KIFMM,
        },
        helpers::find_corners,
        types::MortonKey,
    },
    Evaluate, KiFmm,
};

/// Compute the cutoff rank for an SVD decomposition of a matrix from its singular values
/// using a specified `threshold` as a tolerance parameter
pub fn find_cutoff_rank<T: Float + RlstScalar + Gemm>(
    singular_values: &[T],
    threshold: T,
    max_rank: usize,
) -> usize {
    let len = singular_values.len().min(max_rank);

    for (i, &s) in singular_values.iter().take(len).enumerate() {
        if s <= threshold {
            return i;
        }
    }

    len - 1
}
impl<Scalar, FieldTranslation> SourceTranslationMetadata
    for KiFmm<Scalar, Laplace3dKernel<Scalar>, FieldTranslation>
where
    Scalar: RlstScalar + Default + Epsilon + MatrixSvd,
    FieldTranslation: FieldTranslationTrait + Send + Sync,
    <Scalar as RlstScalar>::Real: Default,
    Self: DataAccess,
{
    fn source(&mut self) {
        let root = MortonKey::<Scalar::Real>::root();

        // Cast surface parameters
        let alpha_outer = Scalar::from(ALPHA_OUTER).unwrap().re();
        let alpha_inner = Scalar::from(ALPHA_INNER).unwrap().re();
        let domain = self.tree.domain();

        let mut m2m = Vec::new();
        let mut m2m_vec = Vec::new();
        let mut uc2e_inv_1 = Vec::new();
        let mut uc2e_inv_2 = Vec::new();

        for (&equivalent_surface_order, &check_surface_order) in self
            .equivalent_surface_order
            .iter()
            .zip(self.check_surface_order.iter())
        {
            // Compute required surfaces
            let upward_equivalent_surface =
                root.surface_grid(equivalent_surface_order, domain, alpha_inner);
            let upward_check_surface = root.surface_grid(check_surface_order, domain, alpha_outer);

            let nequiv_surface = ncoeffs_kifmm(equivalent_surface_order);
            let ncheck_surface = ncoeffs_kifmm(check_surface_order);

            // Assemble matrix of kernel evaluations between upward check to equivalent, and downward check to equivalent matrices
            // As well as estimating their inverses using SVD
            let mut uc2e = rlst_dynamic_array2!(Scalar, [ncheck_surface, nequiv_surface]);
            self.kernel.assemble_st(
                GreenKernelEvalType::Value,
                &upward_check_surface[..],
                &upward_equivalent_surface[..],
                uc2e.data_mut(),
            );

            let (s, ut, v) = pinv(&uc2e, None, None).unwrap();

            let mut mat_s = rlst_dynamic_array2!(Scalar, [s.len(), s.len()]);
            for i in 0..s.len() {
                mat_s[[i, i]] = Scalar::from_real(s[i]);
            }

            uc2e_inv_1.push(empty_array::<Scalar, 2>().simple_mult_into_resize(v.r(), mat_s.r()));
            uc2e_inv_2.push(ut);
        }

        let iterator = if self.equivalent_surface_order.len() > 1 {
            0..self.equivalent_surface_order.len() - 1
        } else {
            0..1
        };

        // Calculate M2M operator matrices on each level, if required
        for parent_level in iterator {
            let check_surface_order_parent = self.check_surface_order(parent_level as u64);
            let equivalent_surface_order_parent =
                self.equivalent_surface_order(parent_level as u64);
            let equivalent_surface_order_child =
                self.equivalent_surface_order((parent_level + 1) as u64);

            let parent_upward_check_surface =
                root.surface_grid(check_surface_order_parent, domain, alpha_outer);

            let children = root.children();
            let ncheck_surface_parent = ncoeffs_kifmm(check_surface_order_parent);
            let nequiv_surface_child = ncoeffs_kifmm(equivalent_surface_order_child);
            let nequiv_surface_parent = ncoeffs_kifmm(equivalent_surface_order_parent);

            let mut m2m_level =
                rlst_dynamic_array2!(Scalar, [nequiv_surface_parent, 8 * nequiv_surface_child]);
            let mut m2m_vec_level = Vec::new();

            for (i, child) in children.iter().enumerate() {
                let child_upward_equivalent_surface =
                    child.surface_grid(equivalent_surface_order_child, domain, alpha_inner);

                let mut ce2pc =
                    rlst_dynamic_array2!(Scalar, [ncheck_surface_parent, nequiv_surface_child]);

                // Note, this way around due to calling convention of kernel, source/targets are 'swapped'
                self.kernel.assemble_st(
                    GreenKernelEvalType::Value,
                    &parent_upward_check_surface,
                    &child_upward_equivalent_surface,
                    ce2pc.data_mut(),
                );

                let tmp = empty_array::<Scalar, 2>().simple_mult_into_resize(
                    uc2e_inv_1[self.expansion_index(parent_level as u64)].r(),
                    empty_array::<Scalar, 2>().simple_mult_into_resize(
                        uc2e_inv_2[self.expansion_index(parent_level as u64)].r(),
                        ce2pc.r(),
                    ),
                );

                let l = i * nequiv_surface_child * nequiv_surface_parent;
                let r = l + nequiv_surface_child * nequiv_surface_parent;

                m2m_level.data_mut()[l..r].copy_from_slice(tmp.data());
                m2m_vec_level.push(tmp);
            }

            m2m_vec.push(m2m_vec_level);
            m2m.push(m2m_level);
        }

        self.source = m2m;
        self.source_vec = m2m_vec;
        self.uc2e_inv_1 = uc2e_inv_1;
        self.uc2e_inv_2 = uc2e_inv_2;
    }
}

impl<Scalar, FieldTranslation> TargetTranslationMetadata
    for KiFmm<Scalar, Laplace3dKernel<Scalar>, FieldTranslation>
where
    Scalar: RlstScalar + Default + Epsilon + MatrixSvd,
    FieldTranslation: FieldTranslationTrait + Send + Sync,
    <Scalar as RlstScalar>::Real: Default,
    Self: DataAccess,
{
    fn target(&mut self) {
        let root = MortonKey::<Scalar::Real>::root();

        // Cast surface parameters
        let alpha_outer = Scalar::from(ALPHA_OUTER).unwrap().re();
        let alpha_inner = Scalar::from(ALPHA_INNER).unwrap().re();
        let domain = self.tree.domain();

        let mut l2l = Vec::new();
        let mut dc2e_inv_1 = Vec::new();
        let mut dc2e_inv_2 = Vec::new();

        for (&equivalent_surface_order, &check_surface_order) in self
            .equivalent_surface_order
            .iter()
            .zip(self.check_surface_order.iter())
        {
            // Compute required surfaces
            let downward_equivalent_surface =
                root.surface_grid(equivalent_surface_order, domain, alpha_outer);
            let downward_check_surface =
                root.surface_grid(check_surface_order, domain, alpha_inner);

            let nequiv_surface = ncoeffs_kifmm(equivalent_surface_order);
            let ncheck_surface = ncoeffs_kifmm(check_surface_order);

            // Assemble matrix of kernel evaluations between upward check to equivalent, and downward check to equivalent matrices
            // As well as estimating their inverses using SVD
            let mut dc2e = rlst_dynamic_array2!(Scalar, [ncheck_surface, nequiv_surface]);
            self.kernel.assemble_st(
                GreenKernelEvalType::Value,
                &downward_check_surface[..],
                &downward_equivalent_surface[..],
                dc2e.data_mut(),
            );

            let (s, ut, v) = pinv::<Scalar>(&dc2e, None, None).unwrap();

            let mut mat_s = rlst_dynamic_array2!(Scalar, [s.len(), s.len()]);
            for i in 0..s.len() {
                mat_s[[i, i]] = Scalar::from_real(s[i]);
            }

            dc2e_inv_1.push(empty_array::<Scalar, 2>().simple_mult_into_resize(v.r(), mat_s.r()));
            dc2e_inv_2.push(ut);
        }

        let depth = self.tree.target_tree().depth();

        let iterator = if self.equivalent_surface_order.len() > 1 {
            0..depth
        } else {
            0..1
        };

        for parent_level in iterator {
            let equivalent_surface_order_parent = self.equivalent_surface_order(parent_level);
            let check_surface_order_child = self.check_surface_order(parent_level + 1);

            let parent_downward_equivalent_surface =
                root.surface_grid(equivalent_surface_order_parent, domain, alpha_outer);

            // Calculate L2L operator matrices
            let children = root.children();
            let ncheck_surface_child = ncoeffs_kifmm(check_surface_order_child);
            let nequiv_surface_parent = ncoeffs_kifmm(equivalent_surface_order_parent);

            let mut l2l_level = Vec::new();

            for child in children.iter() {
                let child_downward_check_surface =
                    child.surface_grid(check_surface_order_child, domain, alpha_inner);

                // Note, this way around due to calling convention of kernel, source/targets are 'swapped'
                let mut pe2cc =
                    rlst_dynamic_array2!(Scalar, [ncheck_surface_child, nequiv_surface_parent]);
                self.kernel.assemble_st(
                    GreenKernelEvalType::Value,
                    &child_downward_check_surface,
                    &parent_downward_equivalent_surface,
                    pe2cc.data_mut(),
                );

                let mut tmp = empty_array::<Scalar, 2>().simple_mult_into_resize(
                    dc2e_inv_1[self.expansion_index(parent_level + 1)].r(),
                    empty_array::<Scalar, 2>().simple_mult_into_resize(
                        dc2e_inv_2[self.expansion_index(parent_level + 1)].r(),
                        pe2cc.r(),
                    ),
                );

                tmp.data_mut()
                    .iter_mut()
                    .for_each(|d| *d *= homogenous_kernel_scale(child.level()));

                l2l_level.push(tmp);
            }

            l2l.push(l2l_level);
        }

        self.target_vec = l2l;
        self.dc2e_inv_1 = dc2e_inv_1;
        self.dc2e_inv_2 = dc2e_inv_2;
    }
}

impl<Scalar, FieldTranslation> SourceTranslationMetadata
    for KiFmm<Scalar, Helmholtz3dKernel<Scalar>, FieldTranslation>
where
    Scalar: RlstScalar<Complex = Scalar> + Default + Epsilon + MatrixSvd,
    FieldTranslation: FieldTranslationTrait + Send + Sync,
    <Scalar as RlstScalar>::Real: Default,
    Self: Evaluate,
{
    fn source(&mut self) {
        let root = MortonKey::<Scalar::Real>::root();

        // Cast surface parameters
        let alpha_outer = Scalar::from(ALPHA_OUTER).unwrap().re();
        let alpha_inner = Scalar::from(ALPHA_INNER).unwrap().re();
        let domain = self.tree.domain();

        let depth = self.tree.source_tree().depth();

        let mut curr = root;
        let mut uc2e_inv_1 = Vec::new();
        let mut uc2e_inv_2 = Vec::new();

        // Calculate inverse upward check to equivalent matrices on each level
        let iterator = if self.equivalent_surface_order.len() > 1 {
            self.equivalent_surface_order
                .iter()
                .cloned()
                .zip(self.check_surface_order.iter().cloned())
                .collect_vec()
        } else {
            vec![
                (
                    *self.equivalent_surface_order.last().unwrap(),
                    *self.check_surface_order.last().unwrap()
                );
                (depth + 1) as usize
            ]
        };

        for (equivalent_surface_order, check_surface_order) in iterator {
            // Compute required surfaces
            let upward_equivalent_surface =
                curr.surface_grid(equivalent_surface_order, domain, alpha_inner);
            let upward_check_surface = curr.surface_grid(check_surface_order, domain, alpha_outer);

            let nequiv_surface = ncoeffs_kifmm(equivalent_surface_order);
            let ncheck_surface = ncoeffs_kifmm(check_surface_order);

            // Assemble matrix of kernel evaluations between upward check to equivalent, and downward check to equivalent matrices
            // As well as estimating their inverses using SVD
            let mut uc2e = rlst_dynamic_array2!(Scalar, [ncheck_surface, nequiv_surface]);
            self.kernel.assemble_st(
                GreenKernelEvalType::Value,
                &upward_check_surface[..],
                &upward_equivalent_surface[..],
                uc2e.data_mut(),
            );

            let (s, ut, v) = pinv(&uc2e, None, None).unwrap();

            let mut mat_s = rlst_dynamic_array2!(Scalar, [s.len(), s.len()]);
            for i in 0..s.len() {
                mat_s[[i, i]] = Scalar::from_real(s[i]);
            }

            uc2e_inv_1.push(empty_array::<Scalar, 2>().simple_mult_into_resize(v.r(), mat_s.r()));
            uc2e_inv_2.push(ut);

            curr = curr.first_child();
        }

        let mut curr = root;
        let mut source = Vec::new();
        let mut source_vec = Vec::new();

        let iterator = if self.equivalent_surface_order.len() > 1 {
            (0..depth)
                .zip(
                    self.check_surface_order
                        .iter()
                        .cloned()
                        .take(depth as usize)
                        .zip(
                            self.equivalent_surface_order
                                .iter()
                                .skip(1)
                                .cloned()
                                .take(depth as usize),
                        ),
                )
                .collect_vec()
        } else {
            (0..depth)
                .zip(
                    vec![*self.check_surface_order.last().unwrap(); depth as usize]
                        .into_iter()
                        .zip(vec![
                            *self.equivalent_surface_order.last().unwrap();
                            depth as usize
                        ]),
                )
                .collect_vec()
        };

        // Calculate M2M operator matrices on each level
        for (level, (check_surface_order_parent, equivalent_surface_order_child)) in iterator {
            // Compute required surfaces
            let parent_upward_check_surface =
                curr.surface_grid(check_surface_order_parent, domain, alpha_outer);

            let equivalent_surface_order_parent = self.equivalent_surface_order(level);

            let ncheck_surface_parent = ncoeffs_kifmm(check_surface_order_parent);
            let nequiv_surface_child = ncoeffs_kifmm(equivalent_surface_order_child);
            let nequiv_surface_parent = ncoeffs_kifmm(equivalent_surface_order_parent);

            let children = curr.children();
            let mut m2m =
                rlst_dynamic_array2!(Scalar, [nequiv_surface_parent, 8 * nequiv_surface_child]);
            let mut m2m_vec = Vec::new();

            for (i, child) in children.iter().enumerate() {
                let child_upward_equivalent_surface =
                    child.surface_grid(equivalent_surface_order_child, domain, alpha_inner);

                let mut ce2pc =
                    rlst_dynamic_array2!(Scalar, [ncheck_surface_parent, nequiv_surface_child]);

                self.kernel.assemble_st(
                    GreenKernelEvalType::Value,
                    &parent_upward_check_surface,
                    &child_upward_equivalent_surface,
                    ce2pc.data_mut(),
                );

                let tmp = empty_array::<Scalar, 2>().simple_mult_into_resize(
                    uc2e_inv_1[level as usize].r(),
                    empty_array::<Scalar, 2>()
                        .simple_mult_into_resize(uc2e_inv_2[level as usize].r(), ce2pc.r()),
                );
                let l = i * nequiv_surface_child * nequiv_surface_parent;
                let r = l + nequiv_surface_child * nequiv_surface_parent;

                m2m.data_mut()[l..r].copy_from_slice(tmp.data());
                m2m_vec.push(tmp);
            }

            source.push(m2m);
            source_vec.push(m2m_vec);
            curr = curr.first_child();
        }

        self.uc2e_inv_1 = uc2e_inv_1;
        self.uc2e_inv_2 = uc2e_inv_2;
        self.source = source;
        self.source_vec = source_vec;
    }
}

impl<Scalar, FieldTranslation> TargetTranslationMetadata
    for KiFmm<Scalar, Helmholtz3dKernel<Scalar>, FieldTranslation>
where
    Scalar: RlstScalar<Complex = Scalar> + Default + Epsilon + MatrixSvd,
    FieldTranslation: FieldTranslationTrait + Send + Sync,
    <Scalar as RlstScalar>::Real: Default,
    Self: Evaluate,
{
    fn target(&mut self) {
        let root = MortonKey::<Scalar::Real>::root();

        // Cast surface parameters
        let alpha_outer = Scalar::from(ALPHA_OUTER).unwrap().re();
        let alpha_inner = Scalar::from(ALPHA_INNER).unwrap().re();
        let domain = self.tree.domain();

        let depth = self.tree.source_tree().depth();

        let mut curr = root;
        let mut dc2e_inv_1 = Vec::new();
        let mut dc2e_inv_2 = Vec::new();

        // Calculate inverse upward check to equivalent matrices on each level
        let iterator = if self.equivalent_surface_order.len() > 1 {
            self.equivalent_surface_order
                .iter()
                .cloned()
                .zip(self.check_surface_order.iter().cloned())
                .collect_vec()
        } else {
            vec![
                (
                    *self.equivalent_surface_order.last().unwrap(),
                    *self.check_surface_order.last().unwrap()
                );
                (depth + 1) as usize
            ]
        };

        for (equivalent_surface_order, check_surface_order) in iterator {
            // Compute required surfaces
            let downward_equivalent_surface =
                curr.surface_grid(equivalent_surface_order, domain, alpha_outer);
            let downward_check_surface =
                curr.surface_grid(check_surface_order, domain, alpha_inner);

            let nequiv_surface = ncoeffs_kifmm(equivalent_surface_order);
            let ncheck_surface = ncoeffs_kifmm(check_surface_order);

            // Assemble matrix of kernel evaluations between upward check to equivalent, and downward check to equivalent matrices
            // As well as estimating their inverses using SVD
            let mut dc2e = rlst_dynamic_array2!(Scalar, [ncheck_surface, nequiv_surface]);
            self.kernel.assemble_st(
                GreenKernelEvalType::Value,
                &downward_check_surface[..],
                &downward_equivalent_surface[..],
                dc2e.data_mut(),
            );

            let (s, ut, v) = pinv::<Scalar>(&dc2e, None, None).unwrap();

            let mut mat_s = rlst_dynamic_array2!(Scalar, [s.len(), s.len()]);
            for i in 0..s.len() {
                mat_s[[i, i]] = Scalar::from_real(s[i]);
            }

            dc2e_inv_1.push(empty_array::<Scalar, 2>().simple_mult_into_resize(v.r(), mat_s.r()));
            dc2e_inv_2.push(ut);
            curr = curr.first_child();
        }

        let mut curr = root;
        let mut target_vec = Vec::new();

        let iterator = if self.equivalent_surface_order.len() > 1 {
            (0..depth)
                .zip(
                    self.equivalent_surface_order
                        .iter()
                        .cloned()
                        .take(depth as usize)
                        .zip(
                            self.check_surface_order
                                .iter()
                                .skip(1)
                                .cloned()
                                .take(depth as usize),
                        ),
                )
                .collect_vec()
        } else {
            (0..depth)
                .zip(
                    vec![*self.equivalent_surface_order.last().unwrap(); depth as usize]
                        .into_iter()
                        .zip(vec![
                            *self.check_surface_order.last().unwrap();
                            depth as usize
                        ]),
                )
                .collect_vec()
        };

        for (level, (equivalent_surface_order_parent, check_surface_order_child)) in iterator {
            // Compute required surfaces
            let parent_downward_equivalent_surface =
                curr.surface_grid(equivalent_surface_order_parent, domain, alpha_outer);

            let ncheck_surface_child = ncoeffs_kifmm(check_surface_order_child);
            let nequiv_surface_parent = ncoeffs_kifmm(equivalent_surface_order_parent);

            // Calculate l2l operator matrices on each level
            let children = curr.children();
            let mut l2l = Vec::new();

            for child in children.iter() {
                let child_downward_check_surface =
                    child.surface_grid(check_surface_order_child, domain, alpha_inner);

                let mut pe2cc =
                    rlst_dynamic_array2!(Scalar, [ncheck_surface_child, nequiv_surface_parent]);
                self.kernel.assemble_st(
                    GreenKernelEvalType::Value,
                    &child_downward_check_surface,
                    &parent_downward_equivalent_surface,
                    pe2cc.data_mut(),
                );

                let tmp = empty_array::<Scalar, 2>().simple_mult_into_resize(
                    dc2e_inv_1[(level + 1) as usize].r(),
                    empty_array::<Scalar, 2>()
                        .simple_mult_into_resize(dc2e_inv_2[(level + 1) as usize].r(), pe2cc.r()),
                );

                l2l.push(tmp);
            }

            target_vec.push(l2l);
            curr = curr.first_child();
        }

        self.dc2e_inv_1 = dc2e_inv_1;
        self.dc2e_inv_2 = dc2e_inv_2;
        self.target_vec = target_vec;
    }
}

impl<Scalar> SourceToTargetTranslationMetadata
    for KiFmm<Scalar, Helmholtz3dKernel<Scalar>, BlasFieldTranslationIa<Scalar>>
where
    Scalar: RlstScalar<Complex = Scalar> + Default + MatrixRsvd,
    <Scalar as RlstScalar>::Real: Default,
{
    fn displacements(&mut self, start_level: Option<u64>) {
        let mut displacements = Vec::new();
        let start_level = if let Some(start_level) = start_level {
            if start_level >= 2 {
                start_level
            } else {
                2
            }
        } else {
            2
        };

        for level in start_level..=self.tree.source_tree().depth() {
            let sources = self.tree.source_tree().keys(level).unwrap();
            let n_sources = sources.len();
            let m2l_operator_index = self.m2l_operator_index(level);
            let sentinel = -1i32;

            let result = vec![vec![sentinel; n_sources]; 316];
            let result = result.into_iter().map(RwLock::new).collect_vec();

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

                    let transfer_vectors_set: HashSet<_> =
                        transfer_vectors.iter().cloned().collect();

                    // Mark items in interaction list for scattering
                    for (tv_idx, tv) in self.source_to_target.transfer_vectors[m2l_operator_index]
                        .iter()
                        .enumerate()
                    {
                        let mut all_displacements_lock = result[tv_idx].write().unwrap();
                        if transfer_vectors_set.contains(&tv.hash) {
                            // Look up scatter location in target tree
                            let target =
                                &interaction_list[*transfer_vectors_map.get(&tv.hash).unwrap()];
                            let &target_idx = self.level_index_pointer_locals[level as usize]
                                .get(target)
                                .unwrap();
                            all_displacements_lock[source_idx] = target_idx as i32;
                        }
                    }
                });

            displacements.push(result);
        }

        self.source_to_target.displacements = displacements;
    }

    fn source_to_target(&mut self) {
        // Compute unique M2L interactions at Level 3 (smallest choice with all vectors)
        // Compute interaction matrices between source and unique targets, defined by unique transfer vectors
        let alpha = Scalar::real(ALPHA_INNER);
        let depth = self.tree.source_tree().depth();

        let mut result = BlasFieldTranslationIa::<Scalar>::default();

        let iterator = if self.equivalent_surface_order.len() > 1 {
            (2..=depth)
                .zip(self.equivalent_surface_order.iter().skip(2).cloned())
                .zip(self.check_surface_order.iter().skip(2).cloned())
                .collect_vec()
        } else {
            (2..=depth)
                .zip(vec![
                    *self.equivalent_surface_order.last().unwrap();
                    (depth - 1) as usize
                ])
                .zip(vec![
                    *self.check_surface_order.last().unwrap();
                    (depth - 1) as usize
                ])
                .collect_vec()
        };

        for ((level, equivalent_surface_order), check_surface_order) in iterator {
            let transfer_vectors =
                compute_transfer_vectors_at_level::<Scalar::Real>(level).unwrap();

            let mut level_result = BlasMetadataIa::default();
            let level_u = Mutex::new(Vec::new());
            let level_vt = Mutex::new(Vec::new());
            let level_cutoff_rank = Mutex::new(vec![0usize; NTRANSFER_VECTORS_KIFMM]);

            for _ in 0..NTRANSFER_VECTORS_KIFMM {
                level_u
                    .lock()
                    .unwrap()
                    .push(rlst_dynamic_array2!(Scalar, [1, 1]));
                level_vt
                    .lock()
                    .unwrap()
                    .push(rlst_dynamic_array2!(Scalar, [1, 1]));
            }

            transfer_vectors
                .into_par_iter()
                .enumerate()
                .for_each(|(i, t)| {
                    let source_equivalent_surface = t.source.surface_grid(
                        equivalent_surface_order,
                        self.tree.source_tree().domain(),
                        alpha,
                    );
                    let n_sources = ncoeffs_kifmm(equivalent_surface_order);

                    let target_check_surface = t.target.surface_grid(
                        check_surface_order,
                        self.tree.source_tree().domain(),
                        alpha,
                    );
                    let n_targets = ncoeffs_kifmm(check_surface_order);

                    let mut tmp_gram = rlst_dynamic_array2!(Scalar, [n_targets, n_sources]);

                    self.kernel.assemble_st(
                        GreenKernelEvalType::Value,
                        &target_check_surface[..],
                        &source_equivalent_surface[..],
                        tmp_gram.data_mut(),
                    );

                    let mu = tmp_gram.shape()[0];
                    let nvt = tmp_gram.shape()[1];
                    let k = std::cmp::min(mu, nvt);

                    let mut u = rlst_dynamic_array2!(Scalar, [mu, k]);
                    let mut sigma = vec![Scalar::zero().re(); k];
                    let mut vt = rlst_dynamic_array2!(Scalar, [k, nvt]);

                    let target_rank;

                    match &self.source_to_target.svd_mode {
                        &FmmSvdMode::Random {
                            n_components,
                            normaliser,
                            n_oversamples,
                            random_state,
                        } => {
                            // Estimate targget rank if unspecified by user
                            if let Some(n_components) = n_components {
                                target_rank = n_components
                            } else {
                                let max_equivalent_surface_ncoeffs =
                                    self.n_coeffs_equivalent_surface.iter().max().unwrap();
                                let max_check_surface_ncoeffs =
                                    self.n_coeffs_check_surface.iter().max().unwrap();
                                target_rank =
                                    *max_equivalent_surface_ncoeffs.max(max_check_surface_ncoeffs);
                            }

                            (sigma, u, vt) = Scalar::rsvd_fixed_rank(
                                &tmp_gram,
                                target_rank,
                                n_oversamples,
                                normaliser,
                                random_state,
                            )
                            .unwrap();
                        }

                        FmmSvdMode::Deterministic => {
                            tmp_gram
                                .into_svd_alloc(
                                    u.r_mut(),
                                    vt.r_mut(),
                                    &mut sigma[..],
                                    SvdMode::Reduced,
                                )
                                .unwrap();
                        }
                    }

                    let mut sigma_mat = rlst_dynamic_array2!(Scalar, [k, k]);

                    for (j, s) in sigma.iter().enumerate().take(k) {
                        unsafe {
                            *sigma_mat.get_unchecked_mut([j, j]) = Scalar::from(*s).unwrap();
                        }
                    }

                    let vt =
                        empty_array::<Scalar, 2>().simple_mult_into_resize(sigma_mat.r(), vt.r());

                    let cutoff_rank =
                        find_cutoff_rank(&sigma, self.source_to_target.threshold, n_sources);

                    let mut u_compressed = rlst_dynamic_array2!(Scalar, [mu, cutoff_rank]);
                    let mut vt_compressed = rlst_dynamic_array2!(Scalar, [cutoff_rank, nvt]);

                    u_compressed.fill_from(u.into_subview([0, 0], [mu, cutoff_rank]));
                    vt_compressed.fill_from(vt.into_subview([0, 0], [cutoff_rank, nvt]));
                    level_u.lock().unwrap()[i] = u_compressed;
                    level_vt.lock().unwrap()[i] = vt_compressed;
                    level_cutoff_rank.lock().unwrap()[i] = cutoff_rank;
                });

            let level_u = std::mem::take(&mut *level_u.lock().unwrap());
            let level_vt = std::mem::take(&mut *level_vt.lock().unwrap());
            let level_cutoff_rank = std::mem::take(&mut *level_cutoff_rank.lock().unwrap());
            level_result.u = level_u;
            level_result.vt = level_vt;

            let transfer_vectors =
                compute_transfer_vectors_at_level::<Scalar::Real>(level).unwrap();
            result.cutoff_ranks.push(level_cutoff_rank);
            result.metadata.push(level_result);
            result.transfer_vectors.push(transfer_vectors);
        }

        self.source_to_target = result;
    }
}

impl<Scalar> SourceToTargetTranslationMetadata
    for KiFmm<Scalar, Helmholtz3dKernel<Scalar>, BlasFieldTranslationSaRcmp<Scalar>>
where
    Scalar: RlstScalar<Complex = Scalar> + Default + MatrixRsvd,
    <Scalar as RlstScalar>::Real: Default,
{
    fn displacements(&mut self, start_level: Option<u64>) {
        let mut displacements = Vec::new();
        let start_level = if let Some(start_level) = start_level {
            if start_level >= 2 {
                start_level
            } else {
                2
            }
        } else {
            2
        };

        for level in start_level..=self.tree.source_tree().depth() {
            let sources = self.tree.source_tree().keys(level).unwrap();
            let n_sources = sources.len();
            let sentinel = -1i32;

            let result = vec![vec![sentinel; n_sources]; 316];
            let result = result.into_iter().map(RwLock::new).collect_vec();

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

                    let transfer_vectors_set: HashSet<_> =
                        transfer_vectors.iter().cloned().collect();

                    // Mark items in interaction list for scattering
                    for (tv_idx, tv) in self.source_to_target.transfer_vectors.iter().enumerate() {
                        let mut all_displacements_lock = result[tv_idx].write().unwrap();
                        if transfer_vectors_set.contains(&tv.hash) {
                            // Look up scatter location in target tree
                            let target =
                                &interaction_list[*transfer_vectors_map.get(&tv.hash).unwrap()];
                            let &target_idx = self.level_index_pointer_locals[level as usize]
                                .get(target)
                                .unwrap();
                            all_displacements_lock[source_idx] = target_idx as i32;
                        }
                    }
                });

            displacements.push(result);
        }

        self.source_to_target.displacements = displacements;
    }

    fn source_to_target(&mut self) {
        // Compute unique M2L interactions at Level 3 (smallest choice with all vectors)
        // Compute interaction matrices between source and unique targets, defined by unique transfer vectors
        let depth = self.tree.source_tree().depth();

        let iterator = if self.equivalent_surface_order.len() > 1 {
            (2..=depth)
                .zip(self.equivalent_surface_order.iter().skip(2).cloned())
                .zip(self.check_surface_order.iter().skip(2).cloned())
                .collect_vec()
        } else {
            (2..=depth)
                .zip(vec![
                    *self.equivalent_surface_order.last().unwrap();
                    (depth - 1) as usize
                ])
                .zip(vec![
                    *self.check_surface_order.last().unwrap();
                    (depth - 1) as usize
                ])
                .collect_vec()
        };

        for ((level, equivalent_surface_order), check_surface_order) in iterator {
            let transfer_vectors =
                compute_transfer_vectors_at_level::<Scalar::Real>(level).unwrap();

            let nrows = ncoeffs_kifmm(check_surface_order);
            let ncols = ncoeffs_kifmm(equivalent_surface_order);

            let mut se2tc_fat =
                rlst_dynamic_array2!(Scalar, [nrows, ncols * NTRANSFER_VECTORS_KIFMM]);
            let mut se2tc_thin =
                rlst_dynamic_array2!(Scalar, [nrows * NTRANSFER_VECTORS_KIFMM, ncols]);
            let alpha = Scalar::real(ALPHA_INNER);

            transfer_vectors.iter().enumerate().for_each(|(i, t)| {
                let source_equivalent_surface = t.source.surface_grid(
                    equivalent_surface_order,
                    self.tree.source_tree().domain(),
                    alpha,
                );
                let target_check_surface = t.target.surface_grid(
                    check_surface_order,
                    self.tree.source_tree().domain(),
                    alpha,
                );

                let mut tmp_gram = rlst_dynamic_array2!(Scalar, [nrows, ncols]);

                self.kernel.assemble_st(
                    GreenKernelEvalType::Value,
                    &target_check_surface[..],
                    &source_equivalent_surface[..],
                    tmp_gram.data_mut(),
                );

                let mut block = se2tc_fat
                    .r_mut()
                    .into_subview([0, i * ncols], [nrows, ncols]);
                block.fill_from(tmp_gram.r());

                let mut block_column = se2tc_thin
                    .r_mut()
                    .into_subview([i * nrows, 0], [nrows, ncols]);
                block_column.fill_from(tmp_gram.r());
            });

            let mu = se2tc_fat.shape()[0];
            let nvt = se2tc_fat.shape()[1];
            let k = std::cmp::min(mu, nvt);

            let mut u_big = rlst_dynamic_array2!(Scalar, [mu, k]);
            let mut sigma = vec![Scalar::zero().re(); k];
            let mut vt_big = rlst_dynamic_array2!(Scalar, [k, nvt]);

            // Target rank defined by max dimension before cutoff
            let mut target_rank = k;

            match &self.source_to_target.svd_mode {
                &FmmSvdMode::Random {
                    n_components,
                    normaliser,
                    n_oversamples,
                    random_state,
                } => {
                    // Estimate target rank if unspecified by user
                    if let Some(n_components) = n_components {
                        target_rank = n_components
                    } else {
                        let max_equivalent_surface_ncoeffs =
                            self.n_coeffs_equivalent_surface.iter().max().unwrap();
                        let max_check_surface_ncoeffs =
                            self.n_coeffs_check_surface.iter().max().unwrap();
                        target_rank =
                            max_equivalent_surface_ncoeffs.max(max_check_surface_ncoeffs) / 2;
                    }

                    let mut se2tc_fat_transpose =
                        rlst_dynamic_array2!(Scalar, se2tc_fat.r().transpose().shape());
                    se2tc_fat_transpose
                        .r_mut()
                        .fill_from(se2tc_fat.r().transpose());

                    let (sigma_t, u_big_t, vt_big_t) = Scalar::rsvd_fixed_rank(
                        &se2tc_fat_transpose,
                        target_rank,
                        n_oversamples,
                        normaliser,
                        random_state,
                    )
                    .unwrap();
                    u_big = rlst_dynamic_array2!(Scalar, [mu, sigma_t.len()]);
                    vt_big = rlst_dynamic_array2!(Scalar, [sigma_t.len(), nvt]);

                    vt_big.fill_from(u_big_t.transpose());
                    u_big.fill_from(vt_big_t.transpose());
                    sigma = sigma_t;
                }
                FmmSvdMode::Deterministic => {
                    se2tc_fat
                        .into_svd_alloc(
                            u_big.r_mut(),
                            vt_big.r_mut(),
                            &mut sigma[..],
                            SvdMode::Reduced,
                        )
                        .unwrap();
                }
            }

            // Cutoff rank is the minimum of the target rank and the value found by user threshold
            let cutoff_rank =
                find_cutoff_rank(&sigma, self.source_to_target.threshold, ncols).min(target_rank);

            let mut u = rlst_dynamic_array2!(Scalar, [mu, cutoff_rank]);
            let mut sigma_mat = rlst_dynamic_array2!(Scalar, [cutoff_rank, cutoff_rank]);
            let mut vt = rlst_dynamic_array2!(Scalar, [cutoff_rank, nvt]);

            // Store compressed M2L operators
            let thin_nrows = se2tc_thin.shape()[0];
            let nst = se2tc_thin.shape()[1];
            let k = std::cmp::min(thin_nrows, nst);
            let mut st;
            let mut _gamma;
            let mut _r;

            if self.source_to_target.surface_diff() == 0 {
                st = rlst_dynamic_array2!(Scalar, u_big.r().transpose().shape());
                st.fill_from(u_big.r().transpose())
            } else {
                match &self.source_to_target.svd_mode {
                    &FmmSvdMode::Random {
                        n_components,
                        normaliser,
                        n_oversamples,
                        random_state,
                    } => {
                        let target_rank;
                        if let Some(n_components) = n_components {
                            target_rank = n_components
                        } else {
                            // Estimate target rank
                            let max_equivalent_surface_ncoeffs =
                                self.n_coeffs_equivalent_surface.iter().max().unwrap();
                            let max_check_surface_ncoeffs =
                                self.n_coeffs_check_surface.iter().max().unwrap();
                            target_rank =
                                max_equivalent_surface_ncoeffs.max(max_check_surface_ncoeffs) / 2;
                        }

                        (_gamma, _r, st) = Scalar::rsvd_fixed_rank(
                            &se2tc_thin,
                            target_rank,
                            n_oversamples,
                            normaliser,
                            random_state,
                        )
                        .unwrap();
                    }
                    FmmSvdMode::Deterministic => {
                        _r = rlst_dynamic_array2!(Scalar, [thin_nrows, k]);
                        _gamma = vec![Scalar::zero().re(); k];
                        st = rlst_dynamic_array2!(Scalar, [k, nst]);
                        se2tc_thin
                            .into_svd_alloc(
                                _r.r_mut(),
                                st.r_mut(),
                                &mut _gamma[..],
                                SvdMode::Reduced,
                            )
                            .unwrap();
                    }
                }
            }

            u.fill_from(u_big.into_subview([0, 0], [mu, cutoff_rank]));
            vt.fill_from(vt_big.into_subview([0, 0], [cutoff_rank, nvt]));
            for (j, s) in sigma.iter().enumerate().take(cutoff_rank) {
                unsafe {
                    *sigma_mat.get_unchecked_mut([j, j]) = Scalar::from(*s).unwrap();
                }
            }

            let mut s_trunc = rlst_dynamic_array2!(Scalar, [nst, cutoff_rank]);
            for j in 0..cutoff_rank {
                for i in 0..nst {
                    unsafe { *s_trunc.get_unchecked_mut([i, j]) = *st.get_unchecked([j, i]) }
                }
            }

            let c_u = Mutex::new(Vec::new());
            let c_vt = Mutex::new(Vec::new());
            let directional_cutoff_ranks =
                Mutex::new(vec![0usize; self.source_to_target.transfer_vectors.len()]);

            for _ in 0..NTRANSFER_VECTORS_KIFMM {
                c_u.lock()
                    .unwrap()
                    .push(rlst_dynamic_array2!(Scalar, [1, 1]));
                c_vt.lock()
                    .unwrap()
                    .push(rlst_dynamic_array2!(Scalar, [1, 1]));
            }

            (0..NTRANSFER_VECTORS_KIFMM).into_par_iter().for_each(|i| {
                let vt_block = vt.r().into_subview([0, i * ncols], [cutoff_rank, ncols]);

                let tmp = empty_array::<Scalar, 2>().simple_mult_into_resize(
                    sigma_mat.r(),
                    empty_array::<Scalar, 2>().simple_mult_into_resize(vt_block.r(), s_trunc.r()),
                );

                let mut u_i = rlst_dynamic_array2!(Scalar, [cutoff_rank, cutoff_rank]);
                let mut sigma_i = vec![Scalar::zero().re(); cutoff_rank];
                let mut vt_i = rlst_dynamic_array2!(Scalar, [cutoff_rank, cutoff_rank]);

                tmp.into_svd_alloc(u_i.r_mut(), vt_i.r_mut(), &mut sigma_i, SvdMode::Full)
                    .unwrap();

                let directional_cutoff_rank =
                    find_cutoff_rank(&sigma_i, self.source_to_target.threshold, cutoff_rank);

                let mut u_i_compressed =
                    rlst_dynamic_array2!(Scalar, [cutoff_rank, directional_cutoff_rank]);
                let mut vt_i_compressed_ =
                    rlst_dynamic_array2!(Scalar, [directional_cutoff_rank, cutoff_rank]);

                let mut sigma_mat_i_compressed = rlst_dynamic_array2!(
                    Scalar,
                    [directional_cutoff_rank, directional_cutoff_rank]
                );

                u_i_compressed
                    .fill_from(u_i.into_subview([0, 0], [cutoff_rank, directional_cutoff_rank]));
                vt_i_compressed_
                    .fill_from(vt_i.into_subview([0, 0], [directional_cutoff_rank, cutoff_rank]));

                for (j, s) in sigma_i.iter().enumerate().take(directional_cutoff_rank) {
                    unsafe {
                        *sigma_mat_i_compressed.get_unchecked_mut([j, j]) =
                            Scalar::from(*s).unwrap();
                    }
                }

                let vt_i_compressed = empty_array::<Scalar, 2>()
                    .simple_mult_into_resize(sigma_mat_i_compressed.r(), vt_i_compressed_.r());

                directional_cutoff_ranks.lock().unwrap()[i] = directional_cutoff_rank;
                c_u.lock().unwrap()[i] = u_i_compressed;
                c_vt.lock().unwrap()[i] = vt_i_compressed;
            });

            let mut st_trunc = rlst_dynamic_array2!(Scalar, [cutoff_rank, nst]);
            st_trunc.fill_from(s_trunc.transpose());

            let c_vt = std::mem::take(&mut *c_vt.lock().unwrap());
            let c_u = std::mem::take(&mut *c_u.lock().unwrap());
            let directional_cutoff_ranks =
                std::mem::take(&mut *directional_cutoff_ranks.lock().unwrap());

            let result = BlasMetadataSaRcmp {
                u,
                st: st_trunc,
                c_u,
                c_vt,
            };

            self.source_to_target.metadata.push(result);
            self.source_to_target.cutoff_rank.push(cutoff_rank);
            self.source_to_target
                .directional_cutoff_ranks
                .push(directional_cutoff_ranks);
        }

        // self.source_to_target = result;
    }
}

impl<Scalar> SourceToTargetTranslationMetadata
    for KiFmm<Scalar, Helmholtz3dKernel<Scalar>, FftFieldTranslation<Scalar>>
where
    Scalar: RlstScalar<Complex = Scalar>
        + Default
        + AsComplex
        + Dft<InputType = Scalar, OutputType = <Scalar as AsComplex>::ComplexType>,
    <Scalar as RlstScalar>::Real: RlstScalar + Default,
{
    fn displacements(&mut self, start_level: Option<u64>) {
        let mut displacements = Vec::new();
        let start_level = if let Some(start_level) = start_level {
            if start_level >= 2 {
                start_level
            } else {
                2
            }
        } else {
            2
        };

        for level in start_level..=self.tree.source_tree().depth() {
            let targets = self.tree.target_tree().keys(level).unwrap();
            let targets_parents: HashSet<MortonKey<_>> =
                targets.iter().map(|target| target.parent()).collect();
            let mut targets_parents = targets_parents.into_iter().collect_vec();
            targets_parents.sort();
            let ntargets_parents = targets_parents.len();

            let sources = self.tree.source_tree().keys(level).unwrap();

            let sources_parents: HashSet<MortonKey<_>> =
                sources.iter().map(|source| source.parent()).collect();
            let mut sources_parents = sources_parents.into_iter().collect_vec();
            sources_parents.sort();
            let nsources_parents = sources_parents.len();

            let result = vec![Vec::new(); NHALO];
            let result = result.into_iter().map(RwLock::new).collect_vec();

            let targets_parents_neighbors = targets_parents
                .iter()
                .map(|parent| parent.all_neighbors())
                .collect_vec();

            let zero_displacement = nsources_parents * NSIBLINGS;

            (0..NHALO).into_par_iter().for_each(|i| {
                let mut result_i = result[i].write().unwrap();
                for all_neighbors in targets_parents_neighbors.iter().take(ntargets_parents) {
                    // Check if neighbor exists in a valid tree
                    if let Some(neighbor) = all_neighbors[i] {
                        // If it does, check if first child exists in the source tree
                        let first_child = neighbor.first_child();
                        if let Some(neighbor_displacement) =
                            self.level_index_pointer_multipoles[level as usize].get(&first_child)
                        {
                            result_i.push(*neighbor_displacement)
                        } else {
                            result_i.push(zero_displacement)
                        }
                    } else {
                        result_i.push(zero_displacement)
                    }
                }
            });

            displacements.push(result);
        }

        self.source_to_target.displacements = displacements;
    }

    fn source_to_target(&mut self) {
        // Compute the field translation operators

        let depth = self.tree.source_tree().depth();

        // Pick a point in the middle of the domain
        let two = Scalar::real(2.0);
        let midway = self
            .tree
            .source_tree()
            .domain()
            .side_length()
            .iter()
            .map(|d| *d / two)
            .collect_vec();

        let point = midway
            .iter()
            .zip(self.tree.source_tree().domain().origin())
            .map(|(m, o)| *m + *o)
            .collect_vec();
        let point = [point[0], point[1], point[2]];
        let domain = self.tree.source_tree().domain();

        let mut metadata = Vec::new();

        if depth >= 2 {
            // Find unique transfer vectors in correct order at level 3
            let key = MortonKey::from_point(&point, domain, 3);
            let siblings = key.siblings();
            let parent = key.parent();

            let halo = parent.neighbors();
            let halo_children = halo.iter().map(|h| h.children()).collect_vec();

            let mut transfer_vector_index = vec![vec![0usize; NSIBLINGS_SQUARED]; NHALO];

            for (i, halo_child_set) in halo_children.iter().enumerate() {
                let outer_displacement = i;

                for (j, sibling) in siblings.iter().enumerate() {
                    for (k, halo_child) in halo_child_set.iter().enumerate() {
                        let tv = halo_child.find_transfer_vector(sibling).unwrap();

                        let inner_displacement = NSIBLINGS * j + k;
                        transfer_vector_index[outer_displacement][inner_displacement] = tv;
                    }
                }
            }

            // Compute data for level 2 separately
            let equivalent_surface_order = if self.equivalent_surface_order.len() > 2 {
                self.equivalent_surface_order[2]
            } else {
                *self.equivalent_surface_order.last().unwrap()
            };

            let shape = <Scalar as Dft>::shape_in(equivalent_surface_order);
            let transform_shape = <Scalar as Dft>::shape_out(equivalent_surface_order);
            let transform_size = <Scalar as Dft>::size_out(equivalent_surface_order);

            // Need to find valid source/target pairs at this level with matching transfer vectors;
            let all_keys = MortonKey::<Scalar::Real>::root().descendants(2).unwrap();

            // The child boxes in the halo of the sibling set
            let mut sources = vec![];
            // The sibling set
            let mut targets = vec![];

            // Green's function evaluations for each source, target pair interaction
            let mut kernel_data_vec = vec![];

            for _ in 0..NHALO {
                sources.push(vec![
                    MortonKey::<Scalar::Real>::default();
                    NSIBLINGS_SQUARED
                ]);
                targets.push(vec![
                    MortonKey::<Scalar::Real>::default();
                    NSIBLINGS_SQUARED
                ]);
                kernel_data_vec.push(vec![]);
            }

            let mut tv_source_target_pair_map = HashMap::new();
            for source in all_keys.iter() {
                for target in all_keys.iter() {
                    let transfer_vector = source.find_transfer_vector(target).unwrap();

                    if !tv_source_target_pair_map.keys().contains(&transfer_vector) {
                        tv_source_target_pair_map.insert(transfer_vector, (source, target));
                    }
                }
            }

            let alpha = Scalar::real(ALPHA_INNER);

            // Iterate over each set of convolutions in the halo (26)
            for i in 0..NHALO {
                // Iterate over each unique convolution between sibling set, and halo siblings (64)
                for j in 0..NSIBLINGS_SQUARED {
                    let tv = transfer_vector_index[i][j];
                    let (source, target) = tv_source_target_pair_map.get(&tv).unwrap();

                    let source_equivalent_surface = source.surface_grid(
                        equivalent_surface_order,
                        self.tree.source_tree().domain(),
                        alpha,
                    );

                    let target_check_surface = target.surface_grid(
                        equivalent_surface_order,
                        self.tree.source_tree().domain(),
                        alpha,
                    );

                    let v_list: HashSet<MortonKey<_>> = target
                        .parent()
                        .neighbors()
                        .iter()
                        .flat_map(|pn| pn.children())
                        .filter(|pnc| !target.is_adjacent(pnc))
                        .collect();

                    if v_list.contains(source) {
                        // Compute convolution grid around the source box
                        let conv_point_corner_index = 7;
                        let corners = find_corners(&source_equivalent_surface[..]);
                        let conv_point_corner = [
                            corners[self.dim * conv_point_corner_index],
                            corners[self.dim * conv_point_corner_index + 1],
                            corners[self.dim * conv_point_corner_index + 2],
                        ];

                        let (conv_grid, _) = source.convolution_grid(
                            equivalent_surface_order,
                            self.tree.source_tree().domain(),
                            alpha,
                            &conv_point_corner,
                            conv_point_corner_index,
                        );

                        // Calculate Green's fct evaluations with respect to a 'kernel point' on the target box
                        let kernel_point_index = 0;
                        let kernel_point = [
                            target_check_surface[self.dim * kernel_point_index],
                            target_check_surface[self.dim * kernel_point_index + 1],
                            target_check_surface[self.dim * kernel_point_index + 2],
                        ];

                        // Compute Green's fct evaluations
                        let mut kernel = flip3(&self.evaluate_greens_fct_convolution_grid(
                            equivalent_surface_order,
                            &conv_grid,
                            kernel_point,
                        ));

                        // Compute FFT of padded kernel
                        let mut kernel_hat =
                            rlst_dynamic_array3!(<Scalar as DftType>::OutputType, transform_shape);

                        let plan = Scalar::plan_forward(
                            kernel.data_mut(),
                            kernel_hat.data_mut(),
                            &shape,
                            None,
                        )
                        .unwrap();
                        let _ = Scalar::forward_dft(
                            kernel.data_mut(),
                            kernel_hat.data_mut(),
                            &shape,
                            &plan,
                        );

                        kernel_data_vec[i].push(kernel_hat);
                    } else {
                        // Fill with zeros when interaction doesn't exist
                        let kernel_hat_zeros =
                            rlst_dynamic_array3!(<Scalar as DftType>::OutputType, transform_shape);
                        kernel_data_vec[i].push(kernel_hat_zeros);
                    }
                }
            }
            // Each element corresponds to all evaluations for each sibling (in order) at that halo position
            let mut kernel_data = vec![
                vec![
                    <Scalar as DftType>::OutputType::zero();
                    NSIBLINGS_SQUARED * transform_size
                ];
                halo_children.len()
            ];

            // For each halo position
            for i in 0..halo_children.len() {
                // For each unique interaction
                for j in 0..NSIBLINGS_SQUARED {
                    let offset = j * transform_size;
                    kernel_data[i][offset..offset + transform_size]
                        .copy_from_slice(kernel_data_vec[i][j].data())
                }
            }

            // We want to use this data by frequency in the implementation of FFT M2L
            // Rearrangement: Grouping by frequency, then halo child, then sibling
            let mut kernel_data_f = vec![];
            for _ in &halo_children {
                kernel_data_f.push(vec![]);
            }
            for i in 0..halo_children.len() {
                let current_vector = &kernel_data[i];
                for l in 0..transform_size {
                    // halo child
                    for k in 0..NSIBLINGS {
                        // sibling
                        for j in 0..NSIBLINGS {
                            let index = j * transform_size * 8 + k * transform_size + l;
                            kernel_data_f[i].push(current_vector[index]);
                        }
                    }
                }
            }

            // TODO: Get rid of this transpose
            // Transpose results for better cache locality in application
            let mut kernel_data_ft = Vec::new();
            for freq in 0..transform_size {
                let frequency_offset = NSIBLINGS_SQUARED * freq;
                for kernel_f in kernel_data_f.iter().take(NHALO) {
                    let k_f = &kernel_f[frequency_offset..(frequency_offset + NSIBLINGS_SQUARED)]
                        .to_vec();
                    let k_f_ = rlst_array_from_slice2!(k_f.as_slice(), [NSIBLINGS, NSIBLINGS]);
                    let mut k_ft = rlst_dynamic_array2!(
                        <Scalar as DftType>::OutputType,
                        [NSIBLINGS, NSIBLINGS]
                    );
                    k_ft.fill_from(k_f_.r());
                    kernel_data_ft.push(k_ft.data().to_vec());
                }
            }

            metadata.push(FftMetadata {
                kernel_data,
                kernel_data_f: kernel_data_ft,
            });

            let iterator = if self.equivalent_surface_order.len() > 1 {
                (3..=depth)
                    .zip(self.equivalent_surface_order.iter().cloned().skip(3))
                    .collect_vec()
            } else {
                (3..=depth)
                    .zip(vec![
                        *self.equivalent_surface_order.last().unwrap();
                        (depth - 2) as usize
                    ])
                    .collect_vec()
            };

            for &(level, equivalent_surface_order) in &iterator {
                let shape = <Scalar as Dft>::shape_in(equivalent_surface_order);
                let transform_shape = <Scalar as Dft>::shape_out(equivalent_surface_order);
                let transform_size = <Scalar as Dft>::size_out(equivalent_surface_order);

                // Encode point in centre of domain and compute halo of parent, and their resp. children
                let key = MortonKey::from_point(&point, domain, level);
                let siblings = key.siblings();
                let parent = key.parent();

                let halo = parent.neighbors();
                let halo_children = halo.iter().map(|h| h.children()).collect_vec();

                // The child boxes in the halo of the sibling set
                let mut sources = vec![];
                // The sibling set
                let mut targets = vec![];
                // The transfer vectors corresponding to source->target translations
                let mut transfer_vectors = vec![];
                // Green's function evaluations for each source, target pair interaction
                let mut kernel_data_vec = vec![];

                for _ in &halo_children {
                    sources.push(vec![]);
                    targets.push(vec![]);
                    transfer_vectors.push(vec![]);
                    kernel_data_vec.push(vec![]);
                }

                // Each set of 64 M2L operators will correspond to a point in the halo
                // Computing transfer of potential from sibling set to halo
                for (i, halo_child_set) in halo_children.iter().enumerate() {
                    let mut tmp_transfer_vectors = vec![];
                    let mut tmp_targets = vec![];
                    let mut tmp_sources = vec![];

                    // Consider all halo children for a given sibling at a time
                    for sibling in siblings.iter() {
                        for halo_child in halo_child_set.iter() {
                            tmp_transfer_vectors.push(halo_child.find_transfer_vector(sibling));
                            tmp_targets.push(sibling);
                            tmp_sources.push(halo_child);
                        }
                    }

                    // From source to target
                    transfer_vectors[i] = tmp_transfer_vectors;
                    targets[i] = tmp_targets;
                    sources[i] = tmp_sources;
                }

                let alpha = Scalar::real(ALPHA_INNER);

                // Iterate over each set of convolutions in the halo (26)
                for i in 0..transfer_vectors.len() {
                    // Iterate over each unique convolution between sibling set, and halo siblings (64)
                    for j in 0..transfer_vectors[i].len() {
                        // Translating from sibling set to boxes in its M2L halo
                        let target = targets[i][j];
                        let source = sources[i][j];

                        let source_equivalent_surface = source.surface_grid(
                            equivalent_surface_order,
                            self.tree.source_tree().domain(),
                            alpha,
                        );
                        let target_check_surface = target.surface_grid(
                            equivalent_surface_order,
                            self.tree.source_tree().domain(),
                            alpha,
                        );

                        let v_list: HashSet<MortonKey<_>> = target
                            .parent()
                            .neighbors()
                            .iter()
                            .flat_map(|pn| pn.children())
                            .filter(|pnc| !target.is_adjacent(pnc))
                            .collect();

                        if v_list.contains(source) {
                            // Compute convolution grid around the source box
                            let conv_point_corner_index = 7;
                            let corners = find_corners(&source_equivalent_surface[..]);
                            let conv_point_corner = [
                                corners[self.dim * conv_point_corner_index],
                                corners[self.dim * conv_point_corner_index + 1],
                                corners[self.dim * conv_point_corner_index + 2],
                            ];

                            let (conv_grid, _) = source.convolution_grid(
                                equivalent_surface_order,
                                self.tree.source_tree().domain(),
                                alpha,
                                &conv_point_corner,
                                conv_point_corner_index,
                            );

                            // Calculate Green's fct evaluations with respect to a 'kernel point' on the target box
                            let kernel_point_index = 0;
                            let kernel_point = [
                                target_check_surface[self.dim * kernel_point_index],
                                target_check_surface[self.dim * kernel_point_index + 1],
                                target_check_surface[self.dim * kernel_point_index + 2],
                            ];
                            // Compute Green's fct evaluations
                            let mut kernel = flip3(&self.evaluate_greens_fct_convolution_grid(
                                equivalent_surface_order,
                                &conv_grid,
                                kernel_point,
                            ));

                            // Compute FFT of padded kernel
                            let mut kernel_hat = rlst_dynamic_array3!(
                                <Scalar as DftType>::OutputType,
                                transform_shape
                            );

                            let plan = Scalar::plan_forward(
                                kernel.data_mut(),
                                kernel_hat.data_mut(),
                                &shape,
                                None,
                            )
                            .unwrap();
                            let _ = Scalar::forward_dft(
                                kernel.data_mut(),
                                kernel_hat.data_mut(),
                                &shape,
                                &plan,
                            );

                            kernel_data_vec[i].push(kernel_hat);
                        } else {
                            // Fill with zeros when interaction doesn't exist
                            let kernel_hat_zeros = rlst_dynamic_array3!(
                                <Scalar as DftType>::OutputType,
                                transform_shape
                            );
                            kernel_data_vec[i].push(kernel_hat_zeros);
                        }
                    }
                }

                // Each element corresponds to all evaluations for each sibling (in order) at that halo position
                let mut kernel_data = vec![
                    vec![
                        <Scalar as DftType>::OutputType::zero();
                        NSIBLINGS_SQUARED * transform_size
                    ];
                    halo_children.len()
                ];

                // For each halo position
                for i in 0..halo_children.len() {
                    // For each unique interaction
                    for j in 0..NSIBLINGS_SQUARED {
                        let offset = j * transform_size;
                        kernel_data[i][offset..offset + transform_size]
                            .copy_from_slice(kernel_data_vec[i][j].data())
                    }
                }

                // We want to use this data by frequency in the implementation of FFT M2L
                // Rearrangement: Grouping by frequency, then halo child, then sibling
                let mut kernel_data_f = vec![];
                for _ in &halo_children {
                    kernel_data_f.push(vec![]);
                }
                for i in 0..halo_children.len() {
                    let current_vector = &kernel_data[i];
                    for l in 0..transform_size {
                        // halo child
                        for k in 0..NSIBLINGS {
                            // sibling
                            for j in 0..NSIBLINGS {
                                let index = j * transform_size * 8 + k * transform_size + l;
                                kernel_data_f[i].push(current_vector[index]);
                            }
                        }
                    }
                }

                // Re-order
                let mut kernel_data_ft = Vec::new();
                for freq in 0..transform_size {
                    let frequency_offset = NSIBLINGS_SQUARED * freq;
                    for kernel_f in kernel_data_f.iter().take(NHALO) {
                        let k_f = &kernel_f
                            [frequency_offset..(frequency_offset + NSIBLINGS_SQUARED)]
                            .to_vec();
                        let k_f_ = rlst_array_from_slice2!(k_f.as_slice(), [NSIBLINGS, NSIBLINGS]);
                        let mut k_ft = rlst_dynamic_array2!(
                            <Scalar as DftType>::OutputType,
                            [NSIBLINGS, NSIBLINGS]
                        );
                        k_ft.fill_from(k_f_.r());
                        kernel_data_ft.push(k_ft.data().to_vec());
                    }
                }

                metadata.push(FftMetadata {
                    kernel_data,
                    kernel_data_f: kernel_data_ft,
                })
            }
        }

        // Set operator data
        self.source_to_target.metadata = metadata;

        let iterator = if self.equivalent_surface_order.len() > 1 {
            self.equivalent_surface_order
                .iter()
                .skip(2)
                .cloned()
                .collect_vec()
        } else {
            self.equivalent_surface_order.clone()
        };

        // Set required maps
        let mut tmp1 = Vec::new();
        let mut tmp2 = Vec::new();
        for equivalent_surface_order in iterator {
            let (surf_to_conv_map, conv_to_surf_map) =
                Self::compute_surf_to_conv_map(equivalent_surface_order);
            tmp1.push(surf_to_conv_map);
            tmp2.push(conv_to_surf_map)
        }
        self.source_to_target.surf_to_conv_map = tmp1;
        self.source_to_target.conv_to_surf_map = tmp2;
    }
}

impl<Scalar> SourceToTargetTranslationMetadata
    for KiFmm<Scalar, Laplace3dKernel<Scalar>, BlasFieldTranslationSaRcmp<Scalar>>
where
    Scalar: RlstScalar + Default + MatrixRsvd + Clone,
    <Scalar as RlstScalar>::Real: Default + Clone,
{
    fn displacements(&mut self, start_level: Option<u64>) {
        let mut displacements = Vec::new();

        let start_level = if let Some(start_level) = start_level {
            if start_level >= 2 {
                start_level
            } else {
                2
            }
        } else {
            2
        };

        for level in start_level..=self.tree.source_tree().depth() {
            let mut result = Vec::default();

            if let Some(sources) = self.tree.source_tree().keys(level) {
                let n_sources = sources.len();
                let sentinel = -1_i32;

                let tmp = vec![vec![sentinel; n_sources]; 316];
                result = tmp.into_iter().map(RwLock::new).collect_vec();

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
                        for (i, &v) in transfer_vectors.iter().enumerate() {
                            transfer_vectors_map.insert(v, i);
                        }

                        let transfer_vectors_set: HashSet<_> =
                            transfer_vectors.into_iter().collect();

                        // Mark items in interaction list for scattering
                        for (tv_idx, tv) in
                            self.source_to_target.transfer_vectors.iter().enumerate()
                        {
                            let mut result_lock = result[tv_idx].write().unwrap();
                            if transfer_vectors_set.contains(&tv.hash) {
                                // Look up scatter location in target tree
                                let target =
                                    &interaction_list[*transfer_vectors_map.get(&tv.hash).unwrap()];
                                let &target_idx = self.level_index_pointer_locals[level as usize]
                                    .get(target)
                                    .unwrap();
                                result_lock[source_idx] = target_idx as i32;
                            }
                        }
                    });
            }

            displacements.push(result);
        }

        self.source_to_target.displacements = displacements;
    }

    fn source_to_target(&mut self) {
        // Compute unique M2L interactions at Level 3 (smallest choice with all vectors)
        // Compute interaction matrices between source and unique targets, defined by unique transfer vectors

        let iterator = if self.variable_expansion_order() {
            self.equivalent_surface_order
                .iter()
                .skip(2)
                .cloned()
                .zip(self.check_surface_order.iter().skip(2).cloned())
                .collect_vec()
        } else {
            vec![(
                self.equivalent_surface_order[0],
                self.check_surface_order[0],
            )]
        };

        for (equivalent_surface_order, check_surface_order) in iterator {
            let nrows = ncoeffs_kifmm(check_surface_order);
            let ncols = ncoeffs_kifmm(equivalent_surface_order);

            let mut se2tc_fat =
                rlst_dynamic_array2!(Scalar, [nrows, ncols * NTRANSFER_VECTORS_KIFMM]);
            let mut se2tc_thin =
                rlst_dynamic_array2!(Scalar, [nrows * NTRANSFER_VECTORS_KIFMM, ncols]);
            let alpha = Scalar::real(ALPHA_INNER);

            for (i, t) in self.source_to_target.transfer_vectors.iter().enumerate() {
                let source_equivalent_surface = t.source.surface_grid(
                    equivalent_surface_order,
                    self.tree.source_tree().domain(),
                    alpha,
                );

                let target_check_surface = t.target.surface_grid(
                    check_surface_order,
                    self.tree.source_tree().domain(),
                    alpha,
                );

                let mut tmp_gram = rlst_dynamic_array2!(Scalar, [nrows, ncols]);

                self.kernel.assemble_st(
                    GreenKernelEvalType::Value,
                    &target_check_surface[..],
                    &source_equivalent_surface[..],
                    tmp_gram.data_mut(),
                );

                let mut block = se2tc_fat
                    .r_mut()
                    .into_subview([0, i * ncols], [nrows, ncols]);
                block.fill_from(tmp_gram.r());

                let mut block_column = se2tc_thin
                    .r_mut()
                    .into_subview([i * nrows, 0], [nrows, ncols]);
                block_column.fill_from(tmp_gram.r());
            }

            let mu = se2tc_fat.shape()[0];
            let nvt = se2tc_fat.shape()[1];
            let k = std::cmp::min(mu, nvt);

            let mut u_big = rlst_dynamic_array2!(Scalar, [mu, k]);
            let mut sigma = vec![Scalar::zero().re(); k];
            let mut vt_big = rlst_dynamic_array2!(Scalar, [k, nvt]);

            // Target rank defined by max dimension before cutoff
            let mut target_rank = k;

            match &self.source_to_target.svd_mode {
                &FmmSvdMode::Random {
                    n_components,
                    normaliser,
                    n_oversamples,
                    random_state,
                } => {
                    // Estimate targget rank if unspecified by user
                    if let Some(n_components) = n_components {
                        target_rank = n_components
                    } else {
                        let max_equivalent_surface_ncoeffs =
                            self.n_coeffs_equivalent_surface.iter().max().unwrap();
                        let max_check_surface_ncoeffs =
                            self.n_coeffs_check_surface.iter().max().unwrap();
                        target_rank =
                            max_equivalent_surface_ncoeffs.max(max_check_surface_ncoeffs) / 2;
                    }

                    let mut se2tc_fat_transpose =
                        rlst_dynamic_array2!(Scalar, se2tc_fat.r().transpose().shape());
                    se2tc_fat_transpose
                        .r_mut()
                        .fill_from(se2tc_fat.r().transpose());

                    let (sigma_t, u_big_t, vt_big_t) = Scalar::rsvd_fixed_rank(
                        &se2tc_fat_transpose,
                        target_rank,
                        n_oversamples,
                        normaliser,
                        random_state,
                    )
                    .unwrap();
                    u_big = rlst_dynamic_array2!(Scalar, [mu, sigma_t.len()]);
                    vt_big = rlst_dynamic_array2!(Scalar, [sigma_t.len(), nvt]);

                    vt_big.fill_from(u_big_t.transpose());
                    u_big.fill_from(vt_big_t.transpose());
                    sigma = sigma_t;
                }
                FmmSvdMode::Deterministic => {
                    se2tc_fat
                        .into_svd_alloc(
                            u_big.r_mut(),
                            vt_big.r_mut(),
                            &mut sigma[..],
                            SvdMode::Reduced,
                        )
                        .unwrap();
                }
            }

            // Cutoff rank is the minimum of the target rank and the value found by user threshold
            let cutoff_rank =
                find_cutoff_rank(&sigma, self.source_to_target.threshold, ncols).min(target_rank);

            let mut u = rlst_dynamic_array2!(Scalar, [mu, cutoff_rank]);
            let mut sigma_mat = rlst_dynamic_array2!(Scalar, [cutoff_rank, cutoff_rank]);
            let mut vt = rlst_dynamic_array2!(Scalar, [cutoff_rank, nvt]);

            // Store compressed M2L operators
            let thin_nrows = se2tc_thin.shape()[0];
            let nst = se2tc_thin.shape()[1];
            let k = std::cmp::min(thin_nrows, nst);
            let mut st;
            let mut _gamma;
            let mut _r;

            if self.source_to_target.surface_diff() == 0 {
                st = rlst_dynamic_array2!(Scalar, u_big.r().transpose().shape());
                st.fill_from(u_big.r().transpose())
            } else {
                match &self.source_to_target.svd_mode {
                    &FmmSvdMode::Random {
                        n_components,
                        normaliser,
                        n_oversamples,
                        random_state,
                    } => {
                        let target_rank;
                        if let Some(n_components) = n_components {
                            target_rank = n_components
                        } else {
                            // Estimate target rank
                            let max_equivalent_surface_ncoeffs =
                                self.n_coeffs_equivalent_surface.iter().max().unwrap();
                            let max_check_surface_ncoeffs =
                                self.n_coeffs_check_surface.iter().max().unwrap();
                            target_rank =
                                max_equivalent_surface_ncoeffs.max(max_check_surface_ncoeffs) / 2;
                        }

                        (_gamma, _r, st) = Scalar::rsvd_fixed_rank(
                            &se2tc_thin,
                            target_rank,
                            n_oversamples,
                            normaliser,
                            random_state,
                        )
                        .unwrap();
                    }
                    FmmSvdMode::Deterministic => {
                        _r = rlst_dynamic_array2!(Scalar, [thin_nrows, k]);
                        _gamma = vec![Scalar::zero().re(); k];
                        st = rlst_dynamic_array2!(Scalar, [k, nst]);
                        se2tc_thin
                            .into_svd_alloc(
                                _r.r_mut(),
                                st.r_mut(),
                                &mut _gamma[..],
                                SvdMode::Reduced,
                            )
                            .unwrap();
                    }
                }
            }

            u.fill_from(u_big.into_subview([0, 0], [mu, cutoff_rank]));
            vt.fill_from(vt_big.into_subview([0, 0], [cutoff_rank, nvt]));
            for (j, s) in sigma.iter().enumerate().take(cutoff_rank) {
                unsafe {
                    *sigma_mat.get_unchecked_mut([j, j]) = Scalar::from(*s).unwrap();
                }
            }

            let mut s_trunc = rlst_dynamic_array2!(Scalar, [nst, cutoff_rank]);
            for j in 0..cutoff_rank {
                for i in 0..nst {
                    unsafe { *s_trunc.get_unchecked_mut([i, j]) = *st.get_unchecked([j, i]) }
                }
            }

            let c_u = Mutex::new(Vec::new());
            let c_vt = Mutex::new(Vec::new());
            let directional_cutoff_ranks =
                Mutex::new(vec![0usize; self.source_to_target.transfer_vectors.len()]);

            for _ in 0..NTRANSFER_VECTORS_KIFMM {
                c_u.lock()
                    .unwrap()
                    .push(rlst_dynamic_array2!(Scalar, [1, 1]));
                c_vt.lock()
                    .unwrap()
                    .push(rlst_dynamic_array2!(Scalar, [1, 1]));
            }

            (0..NTRANSFER_VECTORS_KIFMM).into_par_iter().for_each(|i| {
                let vt_block = vt.r().into_subview([0, i * ncols], [cutoff_rank, ncols]);

                let tmp = empty_array::<Scalar, 2>().simple_mult_into_resize(
                    sigma_mat.r(),
                    empty_array::<Scalar, 2>().simple_mult_into_resize(vt_block.r(), s_trunc.r()),
                );

                let mut u_i = rlst_dynamic_array2!(Scalar, [cutoff_rank, cutoff_rank]);
                let mut sigma_i = vec![Scalar::zero().re(); cutoff_rank];
                let mut vt_i = rlst_dynamic_array2!(Scalar, [cutoff_rank, cutoff_rank]);

                tmp.into_svd_alloc(u_i.r_mut(), vt_i.r_mut(), &mut sigma_i, SvdMode::Full)
                    .unwrap();

                let directional_cutoff_rank =
                    find_cutoff_rank(&sigma_i, self.source_to_target.threshold, cutoff_rank);

                let mut u_i_compressed =
                    rlst_dynamic_array2!(Scalar, [cutoff_rank, directional_cutoff_rank]);
                let mut vt_i_compressed_ =
                    rlst_dynamic_array2!(Scalar, [directional_cutoff_rank, cutoff_rank]);

                let mut sigma_mat_i_compressed = rlst_dynamic_array2!(
                    Scalar,
                    [directional_cutoff_rank, directional_cutoff_rank]
                );

                u_i_compressed
                    .fill_from(u_i.into_subview([0, 0], [cutoff_rank, directional_cutoff_rank]));
                vt_i_compressed_
                    .fill_from(vt_i.into_subview([0, 0], [directional_cutoff_rank, cutoff_rank]));

                for (j, s) in sigma_i.iter().enumerate().take(directional_cutoff_rank) {
                    unsafe {
                        *sigma_mat_i_compressed.get_unchecked_mut([j, j]) =
                            Scalar::from(*s).unwrap();
                    }
                }

                let vt_i_compressed = empty_array::<Scalar, 2>()
                    .simple_mult_into_resize(sigma_mat_i_compressed.r(), vt_i_compressed_.r());

                directional_cutoff_ranks.lock().unwrap()[i] = directional_cutoff_rank;
                c_u.lock().unwrap()[i] = u_i_compressed;
                c_vt.lock().unwrap()[i] = vt_i_compressed;
            });

            let mut st_trunc = rlst_dynamic_array2!(Scalar, [cutoff_rank, nst]);
            st_trunc.fill_from(s_trunc.transpose());

            let c_vt = std::mem::take(&mut *c_vt.lock().unwrap());
            let c_u = std::mem::take(&mut *c_u.lock().unwrap());
            let directional_cutoff_ranks =
                std::mem::take(&mut *directional_cutoff_ranks.lock().unwrap());

            let result = BlasMetadataSaRcmp {
                u,
                st: st_trunc,
                c_u,
                c_vt,
            };

            self.source_to_target.metadata.push(result);
            self.source_to_target.cutoff_rank.push(cutoff_rank);
            self.source_to_target
                .directional_cutoff_ranks
                .push(directional_cutoff_ranks);
        }
    }
}

impl<Scalar, Kernel> KiFmm<Scalar, Kernel, FftFieldTranslation<Scalar>>
where
    Scalar: RlstScalar + AsComplex + Default + Dft,
    Kernel: KernelTrait<T = Scalar> + HomogenousKernel + Default + Send + Sync,
    <Scalar as RlstScalar>::Real: Default,
{
    /// Computes the unique Green's function evaluations and places them on a convolution grid on the source box wrt to a given
    /// target point on the target box surface grid.
    ///
    /// # Arguments
    /// * `expansion_order` - The expansion order of the FMM
    /// * `convolution_grid` - Cartesian coordinates of points on the convolution grid at a source box, expected in row major order.
    /// * `target_pt` - The point on the target box's surface grid, with which kernels are being evaluated with respect to.
    pub fn evaluate_greens_fct_convolution_grid(
        &self,
        expansion_order: usize,
        convolution_grid: &[Scalar::Real],
        target_pt: [Scalar::Real; 3],
    ) -> Array<Scalar, BaseArray<Scalar, VectorContainer<Scalar>, 3>, 3> {
        let n = 2 * expansion_order - 1; // size of convolution grid
        let npad = n + 1; // padded size
        let nconv = n.pow(3); // length of buffer storing values on convolution grid

        let mut result = rlst_dynamic_array3!(Scalar, [npad, npad, npad]);

        let mut kernel_evals = vec![Scalar::zero(); nconv];
        self.kernel.assemble_st(
            GreenKernelEvalType::Value,
            convolution_grid,
            &target_pt,
            &mut kernel_evals[..],
        );

        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let conv_idx = i + j * n + k * n * n;
                    let save_idx = i + j * npad + k * npad * npad;
                    result.data_mut()[save_idx..(save_idx + 1)]
                        .copy_from_slice(&kernel_evals[(conv_idx)..(conv_idx + 1)]);
                }
            }
        }

        result
    }

    /// Place charge data on the convolution grid.
    ///
    /// # Arguments
    /// * `expansion_order` - The expansion order of the FMM
    /// * `charges` - A vector of charges.
    pub fn evaluate_charges_convolution_grid(
        &self,
        expansion_order: usize,
        expansion_order_index: usize,
        charges: &[Scalar],
    ) -> Array<Scalar, BaseArray<Scalar, VectorContainer<Scalar>, 3>, 3> {
        let n = 2 * expansion_order - 1;
        let npad = n + 1;
        let mut result = rlst_dynamic_array3!(Scalar, [npad, npad, npad]);
        for (i, &j) in self.source_to_target.surf_to_conv_map[expansion_order_index]
            .iter()
            .enumerate()
        {
            result.data_mut()[j] = charges[i];
        }

        result
    }

    /// Compute map between convolution grid indices and surface indices, return mapping and inverse mapping.
    ///
    /// # Arguments
    /// * `expansion_order` - The expansion order of the FMM
    pub fn compute_surf_to_conv_map(expansion_order: usize) -> (Vec<usize>, Vec<usize>) {
        // Number of points along each axis of convolution grid
        let n = 2 * expansion_order - 1;
        let npad = n + 1;

        let nsurf_grid = 6 * (expansion_order - 1).pow(2) + 2;

        // Index maps between surface and convolution grids
        let mut surf_to_conv = vec![0usize; nsurf_grid];
        let mut conv_to_surf = vec![0usize; nsurf_grid];

        // Initialise surface grid index
        let mut surf_index = 0;

        // The boundaries of the surface grid when embedded within the convolution grid
        let lower = expansion_order;
        let upper = 2 * expansion_order - 1;

        for k in 0..npad {
            for j in 0..npad {
                for i in 0..npad {
                    let conv_index = i + npad * j + npad * npad * k;
                    if (i >= lower && j >= lower && (k == lower || k == upper))
                        || (j >= lower && k >= lower && (i == lower || i == upper))
                        || (k >= lower && i >= lower && (j == lower || j == upper))
                    {
                        surf_to_conv[surf_index] = conv_index;
                        surf_index += 1;
                    }
                }
            }
        }

        let lower = 0;
        let upper = expansion_order - 1;
        let mut surf_index = 0;

        for k in 0..npad {
            for j in 0..npad {
                for i in 0..npad {
                    let conv_index = i + npad * j + npad * npad * k;
                    if (i <= upper && j <= upper && (k == lower || k == upper))
                        || (j <= upper && k <= upper && (i == lower || i == upper))
                        || (k <= upper && i <= upper && (j == lower || j == upper))
                    {
                        conv_to_surf[surf_index] = conv_index;
                        surf_index += 1;
                    }
                }
            }
        }

        (surf_to_conv, conv_to_surf)
    }
}

impl<Scalar> SourceToTargetTranslationMetadata
    for KiFmm<Scalar, Laplace3dKernel<Scalar>, FftFieldTranslation<Scalar>>
where
    Scalar: RlstScalar
        + AsComplex
        + Default
        + Dft<InputType = Scalar, OutputType = <Scalar as AsComplex>::ComplexType>,
    <Scalar as RlstScalar>::Real: RlstScalar + Default,
{
    fn displacements(&mut self, start_level: Option<u64>) {
        let mut displacements = Vec::new();

        let start_level = if let Some(start_level) = start_level {
            if start_level >= 2 {
                start_level
            } else {
                2
            }
        } else {
            2
        };

        for level in start_level..=self.tree.source_tree().depth() {
            let mut result = Vec::default();
            if let Some(targets) = self.tree.target_tree().keys(level) {
                if let Some(sources) = self.tree.source_tree().keys(level) {
                    let targets_parents: HashSet<MortonKey<_>> =
                        targets.iter().map(|target| target.parent()).collect();
                    let mut targets_parents = targets_parents.into_iter().collect_vec();
                    targets_parents.sort();
                    let ntargets_parents = targets_parents.len();

                    let sources_parents: HashSet<MortonKey<_>> =
                        sources.iter().map(|source| source.parent()).collect();
                    let mut sources_parents = sources_parents.into_iter().collect_vec();
                    sources_parents.sort();
                    let nsources_parents = sources_parents.len();

                    let tmp = vec![Vec::new(); NHALO];
                    result = tmp.into_iter().map(RwLock::new).collect_vec();

                    let targets_parents_neighbors = targets_parents
                        .iter()
                        .map(|parent| parent.all_neighbors())
                        .collect_vec();

                    let zero_displacement = nsources_parents * NSIBLINGS;

                    (0..NHALO).into_par_iter().for_each(|i| {
                        let mut result_i = result[i].write().unwrap();
                        for all_neighbors in targets_parents_neighbors.iter().take(ntargets_parents)
                        {
                            // Check if neighbor exists in a valid tree
                            if let Some(neighbor) = all_neighbors[i] {
                                // If it does, check if first child exists in the source tree
                                let first_child = neighbor.first_child();
                                if let Some(neighbor_displacement) =
                                    self.level_index_pointer_multipoles[level as usize]
                                        .get(&first_child)
                                {
                                    result_i.push(*neighbor_displacement)
                                } else {
                                    result_i.push(zero_displacement)
                                }
                            } else {
                                result_i.push(zero_displacement)
                            }
                        }
                    });
                }
            }

            displacements.push(result);
        }

        self.source_to_target.displacements = displacements;
    }

    fn source_to_target(&mut self) {
        // Compute the field translation operators

        // Pick a point in the middle of the domain
        let two = Scalar::real(2.0);
        let midway = self
            .tree
            .source_tree()
            .domain()
            .side_length()
            .iter()
            .map(|d| *d / two)
            .collect_vec();

        let point = midway
            .iter()
            .zip(self.tree.source_tree().domain().origin())
            .map(|(m, o)| *m + *o)
            .collect_vec();

        let point = [point[0], point[1], point[2]];

        // Encode point in centre of domain and compute halo of parent, and their resp. children
        let key = MortonKey::from_point(&point, self.tree.source_tree().domain(), 3);
        let siblings = key.siblings();
        let parent = key.parent();
        let halo = parent.neighbors();
        let halo_children = halo.iter().map(|h| h.children()).collect_vec();

        let iterator = if self.equivalent_surface_order.len() > 1 {
            self.equivalent_surface_order
                .iter()
                .skip(2)
                .cloned()
                .collect_vec()
        } else {
            self.equivalent_surface_order.clone()
        };

        for &equivalent_surface_order in &iterator {
            // The child boxes in the halo of the sibling set
            let mut sources = vec![];
            // The sibling set
            let mut targets = vec![];
            // The transfer vectors corresponding to source->target translations
            let mut transfer_vectors = vec![];
            // Green's function evaluations for each source, target pair interaction
            let mut kernel_data_vec = vec![];

            for _ in &halo_children {
                sources.push(vec![]);
                targets.push(vec![]);
                transfer_vectors.push(vec![]);
                kernel_data_vec.push(vec![]);
            }

            // Each set of 64 M2L operators will correspond to a point in the halo
            // Computing transfer of potential from sibling set to halo
            for (i, halo_child_set) in halo_children.iter().enumerate() {
                let mut tmp_transfer_vectors = vec![];
                let mut tmp_targets = vec![];
                let mut tmp_sources = vec![];

                // Consider all halo children for a given sibling at a time
                for sibling in siblings.iter() {
                    for halo_child in halo_child_set.iter() {
                        tmp_transfer_vectors.push(halo_child.find_transfer_vector(sibling));
                        tmp_targets.push(sibling);
                        tmp_sources.push(halo_child);
                    }
                }

                // From source to target
                transfer_vectors[i] = tmp_transfer_vectors;
                targets[i] = tmp_targets;
                sources[i] = tmp_sources;
            }

            let shape = <Scalar as Dft>::shape_in(equivalent_surface_order);
            let transform_shape = <Scalar as Dft>::shape_out(equivalent_surface_order);
            let transform_size = <Scalar as Dft>::size_out(equivalent_surface_order);
            let alpha = Scalar::real(ALPHA_INNER);

            // Iterate over each set of convolutions in the halo (26)
            for i in 0..transfer_vectors.len() {
                // Iterate over each unique convolution between sibling set, and halo siblings (64)
                for j in 0..transfer_vectors[i].len() {
                    // Translating from sibling set to boxes in its M2L halo
                    let target = targets[i][j];
                    let source = sources[i][j];

                    let source_equivalent_surface = source.surface_grid(
                        equivalent_surface_order,
                        self.tree.source_tree().domain(),
                        alpha,
                    );
                    let target_check_surface = target.surface_grid(
                        equivalent_surface_order,
                        self.tree.source_tree().domain(),
                        alpha,
                    );

                    let v_list: HashSet<MortonKey<_>> = target
                        .parent()
                        .neighbors()
                        .iter()
                        .flat_map(|pn| pn.children())
                        .filter(|pnc| !target.is_adjacent(pnc))
                        .collect();

                    if v_list.contains(source) {
                        // Compute convolution grid around the source box
                        let conv_point_corner_index = 7;
                        let corners = find_corners(&source_equivalent_surface[..]);
                        let conv_point_corner = [
                            corners[self.dim * conv_point_corner_index],
                            corners[self.dim * conv_point_corner_index + 1],
                            corners[self.dim * conv_point_corner_index + 2],
                        ];

                        let (conv_grid, _) = source.convolution_grid(
                            equivalent_surface_order,
                            self.tree.source_tree().domain(),
                            alpha,
                            &conv_point_corner,
                            conv_point_corner_index,
                        );

                        // Calculate Green's fct evaluations with respect to a 'kernel point' on the target box
                        let kernel_point_index = 0;
                        let kernel_point = [
                            target_check_surface[self.dim * kernel_point_index],
                            target_check_surface[self.dim * kernel_point_index + 1],
                            target_check_surface[self.dim * kernel_point_index + 2],
                        ];

                        // Compute Green's fct evaluations
                        let mut kernel = flip3(&self.evaluate_greens_fct_convolution_grid(
                            equivalent_surface_order,
                            &conv_grid,
                            kernel_point,
                        ));

                        // Compute FFT of padded kernel
                        let mut kernel_hat =
                            rlst_dynamic_array3!(<Scalar as DftType>::OutputType, transform_shape);

                        let plan = Scalar::plan_forward(
                            kernel.data_mut(),
                            kernel_hat.data_mut(),
                            &shape,
                            None,
                        )
                        .unwrap();
                        let _ = Scalar::forward_dft(
                            kernel.data_mut(),
                            kernel_hat.data_mut(),
                            &shape,
                            &plan,
                        );

                        kernel_data_vec[i].push(kernel_hat);
                    } else {
                        // Fill with zeros when interaction doesn't exist
                        let kernel_hat_zeros =
                            rlst_dynamic_array3!(<Scalar as DftType>::OutputType, transform_shape);
                        kernel_data_vec[i].push(kernel_hat_zeros);
                    }
                }
            }

            // Each element corresponds to all evaluations for each sibling (in order) at that halo position
            let mut kernel_data = vec![
                vec![
                    <Scalar as DftType>::OutputType::zero();
                    NSIBLINGS_SQUARED * transform_size
                ];
                halo_children.len()
            ];

            // For each halo position
            for i in 0..halo_children.len() {
                // For each unique interaction
                for j in 0..NSIBLINGS_SQUARED {
                    let offset = j * transform_size;
                    kernel_data[i][offset..offset + transform_size]
                        .copy_from_slice(kernel_data_vec[i][j].data())
                }
            }

            // We want to use this data by frequency in the implementation of FFT M2L
            // Rearrangement: Grouping by frequency, then halo child, then sibling
            let mut kernel_data_f = vec![];
            for _ in &halo_children {
                kernel_data_f.push(vec![]);
            }
            for i in 0..halo_children.len() {
                let current_vector = &kernel_data[i];
                for l in 0..transform_size {
                    // halo child
                    for k in 0..NSIBLINGS {
                        // sibling
                        for j in 0..NSIBLINGS {
                            let index = j * transform_size * 8 + k * transform_size + l;
                            kernel_data_f[i].push(current_vector[index]);
                        }
                    }
                }
            }

            // Re-order
            let mut kernel_data_ft = Vec::new();
            for freq in 0..transform_size {
                let frequency_offset = NSIBLINGS_SQUARED * freq;
                for kernel_f in kernel_data_f.iter().take(NHALO) {
                    let k_f = &kernel_f[frequency_offset..(frequency_offset + NSIBLINGS_SQUARED)]
                        .to_vec();
                    let k_f_ = rlst_array_from_slice2!(k_f.as_slice(), [NSIBLINGS, NSIBLINGS]);
                    let mut k_ft = rlst_dynamic_array2!(
                        <Scalar as DftType>::OutputType,
                        [NSIBLINGS, NSIBLINGS]
                    );
                    k_ft.fill_from(k_f_.r());
                    kernel_data_ft.push(k_ft.data().to_vec());
                }
            }

            let metadata = FftMetadata {
                kernel_data,
                kernel_data_f: kernel_data_ft,
            };

            // Set operator data
            self.source_to_target.metadata.push(metadata);
        }

        // Set required maps
        let mut tmp1 = Vec::new();
        let mut tmp2 = Vec::new();
        for &expansion_order in &iterator {
            let (surf_to_conv_map, conv_to_surf_map) =
                Self::compute_surf_to_conv_map(expansion_order);
            tmp1.push(surf_to_conv_map);
            tmp2.push(conv_to_surf_map)
        }
        self.source_to_target.surf_to_conv_map = tmp1;
        self.source_to_target.conv_to_surf_map = tmp2;
    }
}

impl<Scalar, FieldTranslation> MetadataAccess
    for KiFmm<Scalar, Laplace3dKernel<Scalar>, FieldTranslation>
where
    Scalar: RlstScalar + Default,
    FieldTranslation: FieldTranslationTrait + Send + Sync,
    <Scalar as RlstScalar>::Real: Default,
{
    fn fft_map_index(&self, level: u64) -> usize {
        if self.variable_expansion_order {
            (level - 2) as usize
        } else {
            0
        }
    }

    fn expansion_index(&self, level: u64) -> usize {
        if self.variable_expansion_order {
            level as usize
        } else {
            0
        }
    }

    fn c2e_operator_index(&self, level: u64) -> usize {
        if self.variable_expansion_order {
            level as usize
        } else {
            0
        }
    }

    fn m2m_operator_index(&self, level: u64) -> usize {
        if self.variable_expansion_order {
            (level - 1) as usize
        } else {
            0
        }
    }

    fn l2l_operator_index(&self, level: u64) -> usize {
        if self.variable_expansion_order {
            (level - 1) as usize
        } else {
            0
        }
    }

    fn m2l_operator_index(&self, level: u64) -> usize {
        if self.variable_expansion_order {
            (level - 2) as usize
        } else {
            0
        }
    }

    fn displacement_index(&self, level: u64) -> usize {
        (level - 2) as usize
    }
}

impl<Scalar, FieldTranslation> MetadataAccess
    for KiFmm<Scalar, Helmholtz3dKernel<Scalar>, FieldTranslation>
where
    Scalar: RlstScalar<Complex = Scalar> + Default,
    FieldTranslation: FieldTranslationTrait + Send + Sync,
    <Scalar as RlstScalar>::Real: Default,
{
    fn fft_map_index(&self, level: u64) -> usize {
        if self.equivalent_surface_order.len() > 1 {
            (level - 2) as usize
        } else {
            0
        }
    }

    fn expansion_index(&self, level: u64) -> usize {
        if self.equivalent_surface_order.len() > 1 {
            level as usize
        } else {
            0
        }
    }

    fn c2e_operator_index(&self, level: u64) -> usize {
        level as usize
    }

    fn m2m_operator_index(&self, level: u64) -> usize {
        (level - 1) as usize
    }

    fn l2l_operator_index(&self, level: u64) -> usize {
        (level - 1) as usize
    }

    fn m2l_operator_index(&self, level: u64) -> usize {
        (level - 2) as usize
    }

    fn displacement_index(&self, level: u64) -> usize {
        (level - 2) as usize
    }
}

impl<Scalar, Kernel, FieldTranslation> Metadata for KiFmm<Scalar, Kernel, FieldTranslation>
where
    Scalar: RlstScalar + Default,
    Kernel: KernelTrait<T = Scalar> + HomogenousKernel + Default + Send + Sync,
    FieldTranslation: FieldTranslationTrait + Send + Sync,
    <Scalar as RlstScalar>::Real: Default,
{
    type Scalar = Scalar;

    fn metadata(&mut self, eval_type: GreenKernelEvalType, charges: &[Self::Scalar]) {
        let alpha_outer = Scalar::real(ALPHA_OUTER);
        let alpha_inner = Scalar::real(ALPHA_INNER);

        // Check if computing potentials, or potentials and derivatives
        let kernel_eval_size = match eval_type {
            GreenKernelEvalType::Value => 1,
            GreenKernelEvalType::ValueDeriv => self.dim + 1,
        };

        let n_target_points = self.tree.target_tree.n_coordinates_tot().unwrap();
        let n_source_points = self.tree.source_tree.n_coordinates_tot().unwrap();
        let n_matvecs = charges.len() / n_source_points;
        let n_source_keys = self.tree.source_tree.n_keys_tot().unwrap();
        let n_target_keys = self.tree.target_tree.n_keys_tot().unwrap();

        // Buffers to store all multipole and local data
        let n_multipole_coeffs;
        let n_local_coeffs;
        if self.equivalent_surface_order.len() > 1 {
            n_multipole_coeffs = (0..=self.tree.source_tree().depth())
                .zip(self.n_coeffs_equivalent_surface.iter())
                .fold(0usize, |acc, (level, &ncoeffs)| {
                    acc + self.tree.source_tree().n_keys(level).unwrap() * ncoeffs
                });

            n_local_coeffs = (0..=self.tree.target_tree().depth())
                .zip(self.n_coeffs_equivalent_surface.iter())
                .fold(0usize, |acc, (level, &ncoeffs)| {
                    acc + self.tree.target_tree().n_keys(level).unwrap() * ncoeffs
                })
        } else {
            n_multipole_coeffs = n_source_keys * self.n_coeffs_equivalent_surface.last().unwrap();
            n_local_coeffs = n_target_keys * self.n_coeffs_equivalent_surface.last().unwrap();
        }

        let multipoles = vec![Scalar::default(); n_multipole_coeffs * n_matvecs];
        let locals = vec![Scalar::default(); n_local_coeffs * n_matvecs];

        // Index pointers of multipole and local data, indexed by level
        let level_index_pointer_multipoles =
            level_index_pointer_single_node(&self.tree.source_tree);
        let level_index_pointer_locals = level_index_pointer_single_node(&self.tree.target_tree);

        // Buffer to store evaluated potentials and/or gradients at target points
        let potentials = vec![Scalar::default(); n_target_points * kernel_eval_size * n_matvecs];

        // Kernel scale at each target and source leaf
        let leaf_scales_sources = leaf_scales_single_node::<Scalar>(
            &self.tree.source_tree,
            *self.n_coeffs_check_surface.last().unwrap(),
        );

        // Pre compute check surfaces
        let leaf_upward_equivalent_surfaces_sources = leaf_surfaces_single_node(
            &self.tree.source_tree,
            *self.n_coeffs_equivalent_surface.last().unwrap(),
            alpha_inner,
            *self.equivalent_surface_order.last().unwrap(),
        );

        let leaf_upward_check_surfaces_sources = leaf_surfaces_single_node(
            &self.tree.source_tree,
            *self.n_coeffs_check_surface.last().unwrap(),
            alpha_outer,
            *self.check_surface_order.last().unwrap(),
        );

        let leaf_downward_equivalent_surfaces_targets = leaf_surfaces_single_node(
            &self.tree.target_tree,
            *self.n_coeffs_equivalent_surface.last().unwrap(),
            alpha_outer,
            *self.equivalent_surface_order.last().unwrap(),
        );

        // Mutable pointers to multipole and local data, indexed by level
        let level_multipoles = level_expansion_pointers_single_node(
            &self.tree.source_tree,
            &self.n_coeffs_equivalent_surface,
            n_matvecs,
            &multipoles,
        );

        let level_locals = level_expansion_pointers_single_node(
            &self.tree.source_tree,
            &self.n_coeffs_equivalent_surface,
            n_matvecs,
            &locals,
        );

        // Mutable pointers to multipole and local data only at leaf level
        let leaf_multipoles = leaf_expansion_pointers_single_node(
            &self.tree.source_tree,
            &self.n_coeffs_equivalent_surface,
            n_matvecs,
            &multipoles,
        );

        let leaf_locals = leaf_expansion_pointers_single_node(
            &self.tree.target_tree,
            &self.n_coeffs_equivalent_surface,
            n_matvecs,
            &locals,
        );

        // Mutable pointers to potential data at each target leaf
        let potentials_send_pointers = potential_pointers_single_node(
            &self.tree.target_tree,
            n_matvecs,
            kernel_eval_size,
            &potentials,
        );

        // Index pointer of charge data at each target leaf
        let charge_index_pointer_targets =
            coordinate_index_pointer_single_node(&self.tree.target_tree);
        let charge_index_pointer_sources =
            coordinate_index_pointer_single_node(&self.tree.source_tree);

        // Set data
        self.multipoles = multipoles;
        self.leaf_multipoles = leaf_multipoles;
        self.level_multipoles = level_multipoles;
        self.locals = locals;
        self.leaf_locals = leaf_locals;
        self.level_locals = level_locals;
        self.level_index_pointer_locals = level_index_pointer_locals;
        self.level_index_pointer_multipoles = level_index_pointer_multipoles;
        self.potentials = potentials;
        self.potentials_send_pointers = potentials_send_pointers;
        self.leaf_upward_equivalent_surfaces_sources = leaf_upward_equivalent_surfaces_sources;
        self.leaf_upward_check_surfaces_sources = leaf_upward_check_surfaces_sources;
        self.leaf_downward_equivalent_surfaces_targets = leaf_downward_equivalent_surfaces_targets;
        self.charges = charges.to_vec();
        self.charge_index_pointer_targets = charge_index_pointer_targets;
        self.charge_index_pointer_sources = charge_index_pointer_sources;
        self.leaf_scales_sources = leaf_scales_sources;
        self.kernel_eval_size = kernel_eval_size;
    }
}

impl<Scalar> FftFieldTranslation<Scalar>
where
    Scalar: RlstScalar + AsComplex + Dft + Default,
    <Scalar as RlstScalar>::Real: RlstScalar + Default,
{
    /// Constructor for FFT based field translations
    pub fn new(block_size: Option<usize>) -> Self {
        Self {
            transfer_vectors: compute_transfer_vectors_at_level::<Scalar::Real>(3).unwrap(),
            block_size: block_size.unwrap_or(DEFAULT_M2L_FFT_BLOCK_SIZE),
            ..Default::default()
        }
    }
}

impl<Scalar> FieldTranslationTrait for FftFieldTranslation<Scalar>
where
    Scalar: RlstScalar + AsComplex + Default + Dft,
    <Scalar as RlstScalar>::Real: RlstScalar + Default,
{
    type Metadata = FftMetadata<<Scalar as AsComplex>::ComplexType>;

    fn overdetermined(&self) -> bool {
        false
    }

    fn surface_diff(&self) -> usize {
        0
    }
}

impl<Scalar> BlasFieldTranslationSaRcmp<Scalar>
where
    Scalar: RlstScalar + Epsilon + Default,
    <Scalar as RlstScalar>::Real: Default,
{
    /// Constructor for BLAS based field translations, specify a compression threshold for the SVD compressed operators
    /// TODO: More docs
    pub fn new(
        threshold: Option<Scalar::Real>,
        surface_diff: Option<usize>,
        svd_mode: FmmSvdMode,
    ) -> Self {
        let tmp = Scalar::epsilon().re();

        Self {
            threshold: threshold.unwrap_or(tmp),
            transfer_vectors: compute_transfer_vectors_at_level::<Scalar::Real>(3).unwrap(),
            surface_diff: surface_diff.unwrap_or_default(),
            svd_mode,
            ..Default::default()
        }
    }
}

impl<Scalar> FieldTranslationTrait for BlasFieldTranslationSaRcmp<Scalar>
where
    Scalar: RlstScalar,
{
    type Metadata = BlasMetadataSaRcmp<Scalar>;

    fn overdetermined(&self) -> bool {
        true
    }

    fn surface_diff(&self) -> usize {
        self.surface_diff
    }
}

impl<Scalar> BlasFieldTranslationIa<Scalar>
where
    Scalar: RlstScalar + Epsilon + Default,
    <Scalar as RlstScalar>::Real: Default,
{
    /// Constructor for BLAS based field translations, specify a compression threshold for the SVD compressed operators
    /// TODO: More docs
    pub fn new(
        threshold: Option<Scalar::Real>,
        surface_diff: Option<usize>,
        svd_mode: FmmSvdMode,
    ) -> Self {
        let tmp = Scalar::epsilon().re();

        Self {
            threshold: threshold.unwrap_or(tmp),
            surface_diff: surface_diff.unwrap_or_default(),
            svd_mode,
            ..Default::default()
        }
    }
}

impl<Scalar> FieldTranslationTrait for BlasFieldTranslationIa<Scalar>
where
    Scalar: RlstScalar,
{
    type Metadata = BlasFieldTranslationIa<Scalar>;

    fn overdetermined(&self) -> bool {
        true
    }

    fn surface_diff(&self) -> usize {
        self.surface_diff
    }
}

#[cfg(test)]
mod test {
    use green_kernels::laplace_3d::Laplace3dKernel;
    use num::Complex;
    use rand::rngs::StdRng;
    use rand::Rng;
    use rand::SeedableRng;
    use rlst::c64;
    use rlst::RandomAccessByRef;
    use rlst::RandomAccessMut;

    use crate::fmm::helpers::single_node::flip3;
    use crate::traits::fmm::DataAccess;
    use crate::tree::helpers::points_fixture;
    use crate::SingleNodeBuilder;

    use super::*;

    #[test]
    fn test_blas_field_translation_laplace() {
        // Setup random sources and targets
        let n_sources = 10000;
        let n_targets = 10000;
        let sources = points_fixture::<f64>(n_sources, None, None, Some(1));
        let targets = points_fixture::<f64>(n_targets, None, None, Some(1));

        // FMM parameters
        let n_crit = Some(100);
        let depth = None;
        let expansion_order = [6];
        let prune_empty = true;

        // Charge data
        let nvecs = 1;
        let mut rng = StdRng::seed_from_u64(0);
        let mut charges = rlst_dynamic_array2!(f64, [n_sources, nvecs]);
        charges.data_mut().iter_mut().for_each(|c| *c = rng.gen());

        let fmm = SingleNodeBuilder::new(false)
            .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
            .unwrap()
            .parameters(
                charges.data(),
                &expansion_order,
                Laplace3dKernel::new(),
                GreenKernelEvalType::Value,
                BlasFieldTranslationSaRcmp::new(Some(1e-5), None, FmmSvdMode::Deterministic),
            )
            .unwrap()
            .build()
            .unwrap();

        let idx = 123;
        let level = 3;
        let transfer_vectors = compute_transfer_vectors_at_level::<f64>(level).unwrap();
        let transfer_vector = &transfer_vectors[idx];

        // Lookup correct components of SVD compressed M2L operator matrix
        let c_idx = fmm
            .source_to_target
            .transfer_vectors
            .iter()
            .position(|x| x.hash == transfer_vector.hash)
            .unwrap();

        let c_u = &fmm.source_to_target.metadata[0].c_u[c_idx];
        let c_vt = &fmm.source_to_target.metadata[0].c_vt[c_idx];

        let mut multipole = rlst_dynamic_array2!(f64, [fmm.n_coeffs_equivalent_surface(level), 1]);
        for i in 0..fmm.n_coeffs_equivalent_surface(level) {
            *multipole.get_mut([i, 0]).unwrap() = i as f64;
        }

        let compressed_multipole = empty_array::<f64, 2>()
            .simple_mult_into_resize(fmm.source_to_target.metadata[0].st.r(), multipole.r());

        let compressed_check_potential = empty_array::<f64, 2>().simple_mult_into_resize(
            c_u.r(),
            empty_array::<f64, 2>().simple_mult_into_resize(c_vt.r(), compressed_multipole.r()),
        );

        // Post process to find check potential
        let check_potential = empty_array::<f64, 2>().simple_mult_into_resize(
            fmm.source_to_target.metadata[0].u.r(),
            compressed_check_potential.r(),
        );

        let alpha = ALPHA_INNER;

        let sources = transfer_vector.source.surface_grid(
            fmm.equivalent_surface_order(level),
            &fmm.tree.domain,
            alpha,
        );

        let targets = transfer_vector.target.surface_grid(
            fmm.equivalent_surface_order(level),
            &fmm.tree.domain,
            alpha,
        );

        let mut direct = vec![0f64; fmm.n_coeffs_check_surface(level)];

        fmm.kernel.evaluate_st(
            GreenKernelEvalType::Value,
            &sources[..],
            &targets[..],
            multipole.data(),
            &mut direct[..],
        );

        let abs_error: f64 = check_potential
            .data()
            .iter()
            .zip(direct.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        let rel_error: f64 = abs_error / (direct.iter().sum::<f64>());

        assert!(rel_error < 1e-5);
    }

    #[test]
    fn test_blas_field_translation_helmholtz() {
        // Setup random sources and targets
        let n_sources = 10000;
        let n_targets = 10000;
        let sources = points_fixture::<f64>(n_sources, None, None, Some(1));
        let targets = points_fixture::<f64>(n_targets, None, None, Some(1));

        // FMM parameters
        let n_crit = Some(100);
        let depth = None;
        let expansion_order = [6];
        let prune_empty = true;
        let wavenumber = 2.5;

        // Charge data
        let nvecs = 1;
        let mut rng = StdRng::seed_from_u64(0);
        let mut charges = rlst_dynamic_array2!(c64, [n_sources, nvecs]);
        charges.data_mut().iter_mut().for_each(|c| *c = rng.gen());

        let fmm = SingleNodeBuilder::new(false)
            .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
            .unwrap()
            .parameters(
                charges.data(),
                &expansion_order,
                Helmholtz3dKernel::new(wavenumber),
                GreenKernelEvalType::Value,
                BlasFieldTranslationIa::new(None, None, FmmSvdMode::Deterministic),
            )
            .unwrap()
            .build()
            .unwrap();

        let level = 2;

        let target = fmm.tree.target_tree().keys(level).unwrap()[0];

        let interaction_list = target
            .parent()
            .neighbors()
            .iter()
            .flat_map(|pn| pn.children())
            .filter(|pnc| {
                !target.is_adjacent(pnc)
                    && fmm.tree.source_tree().all_keys_set().unwrap().contains(pnc)
            })
            .collect_vec();

        let source = interaction_list[0];
        let transfer_vector = source.find_transfer_vector(&target).unwrap();

        let transfer_vectors = compute_transfer_vectors_at_level::<f64>(level).unwrap();

        let m2l_operator_index = fmm.m2l_operator_index(level);

        // Lookup correct components of SVD compressed M2L operator matrix
        let c_idx = transfer_vectors
            .iter()
            .position(|x| x.hash == transfer_vector)
            .unwrap();

        let u = &fmm.source_to_target.metadata[m2l_operator_index].u[c_idx];
        let vt = &fmm.source_to_target.metadata[m2l_operator_index].vt[c_idx];

        let mut multipole = rlst_dynamic_array2!(c64, [fmm.n_coeffs_equivalent_surface(level), 1]);
        for i in 0..fmm.n_coeffs_equivalent_surface(level) {
            *multipole.get_mut([i, 0]).unwrap() = c64::from(i as f64);
        }

        let check_potential = empty_array::<c64, 2>().simple_mult_into_resize(
            u.r(),
            empty_array::<c64, 2>().simple_mult_into_resize(vt.r(), multipole.r()),
        );

        let alpha = ALPHA_INNER;

        let sources =
            source.surface_grid(fmm.equivalent_surface_order(level), &fmm.tree.domain, alpha);
        let targets = target.surface_grid(fmm.check_surface_order(level), &fmm.tree.domain, alpha);

        let mut direct = vec![c64::zero(); fmm.n_coeffs_check_surface(level)];

        fmm.kernel.evaluate_st(
            GreenKernelEvalType::Value,
            &sources[..],
            &targets[..],
            multipole.data(),
            &mut direct[..],
        );

        let abs_error: f64 = check_potential
            .data()
            .iter()
            .zip(direct.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        let rel_error: f64 = abs_error / (direct.iter().sum::<c64>().abs());

        println!("abs {:?} rel {:?}", abs_error, rel_error);
        assert!(rel_error < 1e-5);
    }

    #[test]
    fn test_kernel_rearrangement() {
        // Dummy data mirroring unrearranged kernels
        // here each '1000' corresponds to a sibling index
        // each '100' to a child in a given halo element
        // and each '1' to a frequency
        let mut kernel_data_mat = vec![];
        for _ in 0..26 {
            kernel_data_mat.push(vec![]);
        }
        let size_real = 10;

        for elem in kernel_data_mat.iter_mut().take(26) {
            // sibling index
            for j in 0..8 {
                // halo child index
                for k in 0..8 {
                    // frequency
                    for l in 0..size_real {
                        elem.push(Complex::new((1000 * j + 100 * k + l) as f64, 0.))
                    }
                }
            }
        }

        // We want to use this data by frequency in the implementation of FFT M2L
        // Rearrangement: Grouping by frequency, then halo child, then sibling
        let mut rearranged = vec![];
        for _ in 0..26 {
            rearranged.push(vec![]);
        }
        for i in 0..26 {
            let current_vector = &kernel_data_mat[i];
            for l in 0..size_real {
                // halo child
                for k in 0..8 {
                    // sibling
                    for j in 0..8 {
                        let index = j * size_real * 8 + k * size_real + l;
                        rearranged[i].push(current_vector[index]);
                    }
                }
            }
        }

        // We expect the first 64 elements to correspond to the first frequency components of all
        // siblings with all elements in a given halo position
        let freq = 4;
        let offset = freq * 64;
        let result = &rearranged[0][offset..offset + 64];

        // For each halo child
        for i in 0..8 {
            // for each sibling
            for j in 0..8 {
                let expected = (i * 100 + j * 1000 + freq) as f64;
                assert!(expected == result[i * 8 + j].re)
            }
        }
    }

    #[test]
    fn test_fft_field_translation_laplace() {
        // Setup random sources and targets
        let n_sources = 10000;
        let n_targets = 10000;
        let sources = points_fixture::<f64>(n_sources, None, None, Some(1));
        let targets = points_fixture::<f64>(n_targets, None, None, Some(1));

        // FMM parameters
        let n_crit = Some(100);
        let depth = None;
        let expansion_order = [6];
        let prune_empty = true;

        // Charge data
        let nvecs = 1;
        let mut rng = StdRng::seed_from_u64(0);
        let mut charges = rlst_dynamic_array2!(f64, [n_sources, nvecs]);
        charges.data_mut().iter_mut().for_each(|c| *c = rng.gen());

        let fmm = SingleNodeBuilder::new(false)
            .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
            .unwrap()
            .parameters(
                charges.data(),
                &expansion_order,
                Laplace3dKernel::new(),
                GreenKernelEvalType::Value,
                FftFieldTranslation::new(None),
            )
            .unwrap()
            .build()
            .unwrap();

        let level = 3;
        let coeff_idx = fmm.c2e_operator_index(level);

        let mut multipole = rlst_dynamic_array2!(f64, [fmm.n_coeffs_equivalent_surface(level), 1]);

        for i in 0..fmm.n_coeffs_equivalent_surface(level) {
            *multipole.get_mut([i, 0]).unwrap() = i as f64;
        }

        // Compute all M2L operators
        // Pick a random source/target pair
        let idx = 123;
        let all_transfer_vectors = compute_transfer_vectors_at_level::<f64>(level).unwrap();

        let transfer_vector = &all_transfer_vectors[idx];

        // Compute FFT of the representative signal
        let mut signal = fmm.evaluate_charges_convolution_grid(
            expansion_order[coeff_idx],
            coeff_idx,
            multipole.data(),
        );
        let [m, n, o] = signal.shape();
        let mut signal_hat = rlst_dynamic_array3!(Complex<f64>, [m, n, o / 2 + 1]);

        let plan =
            f64::plan_forward(signal.data_mut(), signal_hat.data_mut(), &[m, n, o], None).unwrap();
        let _ = f64::forward_dft(signal.data_mut(), signal_hat.data_mut(), &[m, n, o], &plan);

        let source_equivalent_surface = transfer_vector.source.surface_grid(
            expansion_order[coeff_idx],
            &fmm.tree.source_tree.domain,
            ALPHA_INNER,
        );
        let target_check_surface = transfer_vector.target.surface_grid(
            expansion_order[coeff_idx],
            &fmm.tree.source_tree.domain,
            ALPHA_INNER,
        );
        let n_targets = target_check_surface.len() / 3;

        // Compute conv grid
        let conv_point_corner_index = 7;
        let corners = find_corners(&source_equivalent_surface[..]);
        let conv_point_corner = [
            corners[fmm.dim * conv_point_corner_index],
            corners[fmm.dim * conv_point_corner_index + 1],
            corners[fmm.dim * conv_point_corner_index + 2],
        ];

        let (conv_grid, _) = transfer_vector.source.convolution_grid(
            expansion_order[coeff_idx],
            &fmm.tree.source_tree.domain,
            ALPHA_INNER,
            &conv_point_corner,
            conv_point_corner_index,
        );

        let kernel_point_index = 0;
        let kernel_point = [
            target_check_surface[fmm.dim() * kernel_point_index],
            target_check_surface[fmm.dim() * kernel_point_index + 1],
            target_check_surface[fmm.dim() * kernel_point_index + 2],
        ];

        // Compute kernel
        let kernel = fmm.evaluate_greens_fct_convolution_grid(
            expansion_order[coeff_idx],
            &conv_grid,
            kernel_point,
        );
        let [m, n, o] = kernel.shape();

        let mut kernel = flip3(&kernel);

        // Compute FFT of padded kernel
        let mut kernel_hat = rlst_dynamic_array3!(Complex<f64>, [m, n, o / 2 + 1]);
        let plan =
            f64::plan_forward(kernel.data_mut(), kernel_hat.data_mut(), &[m, n, o], None).unwrap();
        let _ = f64::forward_dft(kernel.data_mut(), kernel_hat.data_mut(), &[m, n, o], &plan);

        let mut hadamard_product = rlst_dynamic_array3!(Complex<f64>, [m, n, o / 2 + 1]);
        for k in 0..o / 2 + 1 {
            for j in 0..n {
                for i in 0..m {
                    *hadamard_product.get_mut([i, j, k]).unwrap() =
                        kernel_hat.get([i, j, k]).unwrap() * signal_hat.get([i, j, k]).unwrap();
                }
            }
        }
        let mut potentials = rlst_dynamic_array3!(f64, [m, n, o]);

        let plan = f64::plan_backward(
            hadamard_product.data_mut(),
            potentials.data_mut(),
            &[m, n, o],
            None,
        )
        .unwrap();
        let _ = f64::backward_dft(
            hadamard_product.data_mut(),
            potentials.data_mut(),
            &[m, n, o],
            &plan,
        );

        let mut result = vec![0f64; n_targets];
        for (i, &idx) in fmm.source_to_target.conv_to_surf_map[coeff_idx]
            .iter()
            .enumerate()
        {
            result[i] = potentials.data()[idx];
        }

        // Get direct evaluations for testing
        let mut direct = vec![0f64; fmm.n_coeffs_check_surface(level)];
        fmm.kernel.evaluate_st(
            GreenKernelEvalType::Value,
            &source_equivalent_surface[..],
            &target_check_surface[..],
            multipole.data(),
            &mut direct[..],
        );

        let abs_error: f64 = result
            .iter()
            .zip(direct.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        let rel_error: f64 = abs_error / (direct.iter().sum::<f64>());

        assert!(rel_error < 1e-15);
    }

    #[test]
    fn test_fft_field_translation_helmholtz() {
        // Setup random sources and targets
        let n_sources = 10000;
        let n_targets = 10000;
        let sources = points_fixture::<f64>(n_sources, None, None, Some(1));
        let targets = points_fixture::<f64>(n_targets, None, None, Some(1));

        // FMM parameters
        let n_crit = Some(100);
        let depth = None;
        let expansion_order = [6];
        let prune_empty = true;
        let wavenumber = 1.0;

        // Charge data
        let nvecs = 1;
        let mut rng = StdRng::seed_from_u64(0);
        let mut charges = rlst_dynamic_array2!(c64, [n_sources, nvecs]);
        charges.data_mut().iter_mut().for_each(|c| *c = rng.gen());

        let fmm = SingleNodeBuilder::new(false)
            .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
            .unwrap()
            .parameters(
                charges.data(),
                &expansion_order,
                Helmholtz3dKernel::new(wavenumber),
                GreenKernelEvalType::Value,
                FftFieldTranslation::new(None),
            )
            .unwrap()
            .build()
            .unwrap();

        let level = 2;
        let coeff_index = fmm.expansion_index(level);
        let mut multipole = rlst_dynamic_array2!(c64, [fmm.n_coeffs_equivalent_surface(level), 1]);

        for i in 0..fmm.n_coeffs_equivalent_surface(level) {
            *multipole.get_mut([i, 0]).unwrap() = c64::from(i as f64);
        }

        let source = fmm.tree().source_tree().keys(level).unwrap()[0];

        let v_list: HashSet<MortonKey<_>> = source
            .parent()
            .neighbors()
            .iter()
            .flat_map(|pn| pn.children())
            .filter(|pnc| {
                !source.is_adjacent(pnc) && fmm.tree().source_tree().keys_set.contains(pnc)
            })
            .collect();

        let v_list = v_list.into_iter().collect_vec();
        let target = v_list[0];

        // Compute FFT of the representative signal
        let mut signal = fmm.evaluate_charges_convolution_grid(
            expansion_order[coeff_index],
            coeff_index,
            multipole.data(),
        );
        let [m, n, o] = signal.shape();
        let mut signal_hat = rlst_dynamic_array3!(Complex<f64>, [m, n, o]);

        let plan =
            c64::plan_forward(signal.data_mut(), signal_hat.data_mut(), &[m, n, o], None).unwrap();
        let _ = c64::forward_dft(signal.data_mut(), signal_hat.data_mut(), &[m, n, o], &plan);

        let source_equivalent_surface = source.surface_grid(
            expansion_order[coeff_index],
            &fmm.tree.source_tree.domain,
            ALPHA_INNER,
        );
        let target_check_surface = target.surface_grid(
            expansion_order[coeff_index],
            &fmm.tree.source_tree.domain,
            ALPHA_INNER,
        );
        let n_targets = target_check_surface.len() / 3;

        // Compute conv grid
        let conv_point_corner_index = 7;
        let corners = find_corners(&source_equivalent_surface[..]);
        let conv_point_corner = [
            corners[fmm.dim * conv_point_corner_index],
            corners[fmm.dim * conv_point_corner_index + 1],
            corners[fmm.dim * conv_point_corner_index + 2],
        ];

        let (conv_grid, _) = source.convolution_grid(
            expansion_order[coeff_index],
            &fmm.tree.source_tree.domain,
            ALPHA_INNER,
            &conv_point_corner,
            conv_point_corner_index,
        );

        let kernel_point_index = 0;
        let kernel_point = [
            target_check_surface[fmm.dim() * kernel_point_index],
            target_check_surface[fmm.dim() * kernel_point_index + 1],
            target_check_surface[fmm.dim() * kernel_point_index + 2],
        ];

        // Compute kernel
        let kernel = fmm.evaluate_greens_fct_convolution_grid(
            expansion_order[coeff_index],
            &conv_grid,
            kernel_point,
        );
        let [m, n, o] = kernel.shape();

        let mut kernel = flip3(&kernel);

        // Compute FFT of padded kernel
        let mut kernel_hat = rlst_dynamic_array3!(Complex<f64>, [m, n, o]);
        let plan =
            c64::plan_forward(kernel.data_mut(), kernel_hat.data_mut(), &[m, n, o], None).unwrap();
        let _ = c64::forward_dft(kernel.data_mut(), kernel_hat.data_mut(), &[m, n, o], &plan);

        let mut hadamard_product = rlst_dynamic_array3!(Complex<f64>, [m, n, o]);
        for k in 0..o {
            for j in 0..n {
                for i in 0..m {
                    *hadamard_product.get_mut([i, j, k]).unwrap() =
                        kernel_hat.get([i, j, k]).unwrap() * signal_hat.get([i, j, k]).unwrap();
                }
            }
        }
        let mut potentials = rlst_dynamic_array3!(c64, [m, n, o]);

        let plan = c64::plan_backward(
            hadamard_product.data_mut(),
            potentials.data_mut(),
            &[m, n, o],
            None,
        )
        .unwrap();
        let _ = c64::backward_dft(
            hadamard_product.data_mut(),
            potentials.data_mut(),
            &[m, n, o],
            &plan,
        );

        let mut result = vec![c64::zero(); n_targets];
        for (i, &idx) in fmm.source_to_target.conv_to_surf_map[coeff_index]
            .iter()
            .enumerate()
        {
            result[i] = potentials.data()[idx];
        }

        // Get direct evaluations for testing
        let mut direct = vec![c64::zero(); fmm.n_coeffs_check_surface(level)];
        fmm.kernel.evaluate_st(
            GreenKernelEvalType::Value,
            &source_equivalent_surface[..],
            &target_check_surface[..],
            multipole.data(),
            &mut direct[..],
        );

        let abs_error: f64 = result
            .iter()
            .zip(direct.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        let rel_error: f64 = abs_error / (direct.iter().sum::<c64>().abs());

        assert!(rel_error < 1e-15);
    }
}
