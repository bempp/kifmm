use green_kernels::laplace_3d::Laplace3dKernel;
use green_kernels::traits::Kernel as KernelTrait;
use green_kernels::types::GreenKernelEvalType;
use itertools::Itertools;
use mpi::traits::Equivalence;
use num::{Float, Zero};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use rlst::{
    empty_array, rlst_array_from_slice2, rlst_dynamic_array2, rlst_dynamic_array3, Array,
    BaseArray, MatrixSvd, MultIntoResize, RawAccess, RawAccessMut, RlstScalar, Shape, SvdMode,
    UnsafeRandomAccessByRef, UnsafeRandomAccessMut, VectorContainer,
};
use std::collections::{HashMap, HashSet};
use std::sync::{Mutex, RwLock};

use crate::fmm::helpers::multi_node::{
    coordinate_index_pointer_multi_node, leaf_expansion_pointers_multi_node,
    leaf_scales_multi_node, leaf_surfaces_multi_node, level_expansion_pointers_multi_node,
    level_index_pointer_multi_node, potential_pointers_multi_node,
};
use crate::fmm::helpers::single_node::{flip3, homogenous_kernel_scale};
use crate::fmm::types::{BlasMetadataSaRcmp, FftMetadata};
use crate::fmm::KiFmm;
use crate::linalg::pinv::pinv;
use crate::traits::fftw::{Dft, DftType};
use crate::traits::fmm::{DataAccess, DataAccessMulti, Metadata, MetadataAccess};
use crate::traits::general::{
    multi_node::GhostExchange,
    single_node::{AsComplex, Epsilon},
};
use crate::traits::tree::{Domain, FmmTreeNode, MultiFmmTree, MultiTree};
use crate::tree::constants::{NHALO, NSIBLINGS, NSIBLINGS_SQUARED, NTRANSFER_VECTORS_KIFMM};
use crate::tree::helpers::find_corners;
use crate::tree::types::MortonKey;
use crate::{
    fmm::types::{KiFmmMulti, NeighbourhoodCommunicator},
    linalg::rsvd::MatrixRsvd,
    traits::{
        field::{
            FieldTranslation as FieldTranslationTrait, SourceToTargetTranslationMetadata,
            SourceTranslationMetadata, TargetTranslationMetadata,
        },
        fmm::HomogenousKernel,
    },
    tree::constants::{ALPHA_INNER, ALPHA_OUTER},
    BlasFieldTranslationSaRcmp,
};
use crate::{FftFieldTranslation, FmmSvdMode, SingleNodeFmmTree};

use super::single_node::find_cutoff_rank;

impl<Scalar, FieldTranslation> SourceTranslationMetadata
    for KiFmmMulti<Scalar, Laplace3dKernel<Scalar>, FieldTranslation>
where
    Scalar: RlstScalar + Default + Epsilon + MatrixSvd + Equivalence + Float,
    <Scalar as RlstScalar>::Real: Default + Equivalence + Float,
    FieldTranslation: FieldTranslationTrait + Send + Sync,
{
    fn source(&mut self) {
        let root = MortonKey::<Scalar::Real>::root();
        let equivalent_surface_order = self.equivalent_surface_order;
        let check_surface_order = self.check_surface_order;
        let n_coeffs_equivalent_surface = self.n_coeffs_equivalent_surface;
        let n_coeffs_check_surface = self.n_coeffs_check_surface;

        // Cast surface parameters
        let alpha_outer = Scalar::from(ALPHA_OUTER).unwrap().re();
        let alpha_inner = Scalar::from(ALPHA_INNER).unwrap().re();
        let domain = self.tree.domain();

        let mut m2m = rlst_dynamic_array2!(
            Scalar,
            [n_coeffs_equivalent_surface, 8 * n_coeffs_equivalent_surface]
        );
        let mut m2m_vec = Vec::new();

        let mut m2m_global = vec![rlst_dynamic_array2!(
            Scalar,
            [n_coeffs_equivalent_surface, 8 * n_coeffs_equivalent_surface]
        )];
        let mut m2m_vec_global = vec![Vec::new()];

        // Compute required surfaces
        let upward_equivalent_surface =
            root.surface_grid(equivalent_surface_order, domain, alpha_inner);

        let upward_check_surface = root.surface_grid(check_surface_order, domain, alpha_outer);

        // Assemble matrix of kernel evaluations between upward check to equivalent, and downward check to equivalent matrices
        // As well as estimating their inverses using SVD
        let mut uc2e = rlst_dynamic_array2!(
            Scalar,
            [n_coeffs_check_surface, n_coeffs_equivalent_surface]
        );

        self.kernel.assemble_st(
            GreenKernelEvalType::Value,
            &upward_check_surface,
            &upward_equivalent_surface,
            uc2e.data_mut(),
        );

        let (s, ut, v) = pinv(&uc2e, None, None).unwrap();

        let mut mat_s = rlst_dynamic_array2!(Scalar, [s.len(), s.len()]);
        for i in 0..s.len() {
            mat_s[[i, i]] = Scalar::from_real(s[i]);
        }

        let uc2e_inv_1 =
            vec![empty_array::<Scalar, 2>().simple_mult_into_resize(v.view(), mat_s.view())];

        let uc2e_inv_2 = vec![ut];

        let parent_upward_check_surface =
            root.surface_grid(check_surface_order, domain, alpha_outer);
        let children = root.children();

        for (i, child) in children.iter().enumerate() {
            let child_upward_equivalent_surface =
                child.surface_grid(equivalent_surface_order, domain, alpha_inner);

            let mut ce2pc = rlst_dynamic_array2!(
                Scalar,
                [n_coeffs_check_surface, n_coeffs_equivalent_surface]
            );

            self.kernel.assemble_st(
                GreenKernelEvalType::Value,
                &parent_upward_check_surface,
                &child_upward_equivalent_surface,
                ce2pc.data_mut(),
            );

            let tmp = empty_array::<Scalar, 2>().simple_mult_into_resize(
                uc2e_inv_1[0].view(),
                empty_array::<Scalar, 2>()
                    .simple_mult_into_resize(uc2e_inv_2[0].view(), ce2pc.view()),
            );

            let tmp1 = empty_array::<Scalar, 2>().simple_mult_into_resize(
                uc2e_inv_1[0].view(),
                empty_array::<Scalar, 2>()
                    .simple_mult_into_resize(uc2e_inv_2[0].view(), ce2pc.view()),
            );

            let l = i * n_coeffs_equivalent_surface * n_coeffs_equivalent_surface;
            let r = l + n_coeffs_equivalent_surface * n_coeffs_equivalent_surface;

            m2m.data_mut()[l..r].copy_from_slice(tmp.data());
            m2m_vec.push(tmp);

            m2m_global[0].data_mut()[l..r].copy_from_slice(tmp1.data());
            m2m_vec_global[0].push(tmp1);
        }

        self.global_fmm.source = m2m_global;
        self.global_fmm.source_vec = m2m_vec_global;
        self.source = m2m;
        self.source_vec = m2m_vec;
        self.uc2e_inv_1 = uc2e_inv_1;
        self.uc2e_inv_2 = uc2e_inv_2;
    }
}

impl<Scalar, FieldTranslation> TargetTranslationMetadata
    for KiFmmMulti<Scalar, Laplace3dKernel<Scalar>, FieldTranslation>
where
    Scalar: RlstScalar + Default + Epsilon + MatrixSvd + Equivalence + Float,
    <Scalar as RlstScalar>::Real: Default + Equivalence + Float,
    FieldTranslation: FieldTranslationTrait + Send + Sync,
{
    fn displacements(&mut self) {}

    fn target(&mut self) {
        let root = MortonKey::<Scalar::Real>::root();
        let equivalent_surface_order = self.equivalent_surface_order;
        let check_surface_order = self.check_surface_order;
        let n_coeffs_equivalent_surface = self.n_coeffs_equivalent_surface;
        let n_coeffs_check_surface = self.n_coeffs_check_surface;

        // Cast surface parameters
        let alpha_outer = Scalar::from(ALPHA_OUTER).unwrap().re();
        let alpha_inner = Scalar::from(ALPHA_INNER).unwrap().re();
        let domain = self.tree.domain();

        let mut l2l = Vec::new();
        let mut l2l_global = Vec::new();

        let downward_equivalent_surface =
            root.surface_grid(equivalent_surface_order, domain, alpha_outer);
        let downward_check_surface = root.surface_grid(check_surface_order, domain, alpha_inner);

        let mut dc2e = rlst_dynamic_array2!(
            Scalar,
            [n_coeffs_check_surface, n_coeffs_equivalent_surface]
        );
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

        let dc2e_inv_1 =
            vec![empty_array::<Scalar, 2>().simple_mult_into_resize(v.view(), mat_s.view())];
        let dc2e_inv_2 = vec![ut];

        let mut dc2e_inv_1_global = dc2e_inv_1
            .iter()
            .map(|x| rlst_dynamic_array2!(Scalar, x.shape()))
            .collect_vec();
        let mut dc2e_inv_2_global = dc2e_inv_2
            .iter()
            .map(|x| rlst_dynamic_array2!(Scalar, x.shape()))
            .collect_vec();

        dc2e_inv_1_global
            .iter_mut()
            .enumerate()
            .for_each(|(i, x)| x.data_mut().copy_from_slice(dc2e_inv_1[i].data()));
        dc2e_inv_2_global
            .iter_mut()
            .enumerate()
            .for_each(|(i, x)| x.data_mut().copy_from_slice(dc2e_inv_2[i].data()));

        let parent_downward_equivalent_surface =
            root.surface_grid(equivalent_surface_order, domain, alpha_outer);

        let children = root.children();

        for child in children.iter() {
            let child_downward_check_surface =
                child.surface_grid(check_surface_order, domain, alpha_inner);

            // Note, this way around due to calling convention of kernel, source/targets are 'swapped'
            let mut pe2cc = rlst_dynamic_array2!(
                Scalar,
                [n_coeffs_check_surface, n_coeffs_equivalent_surface]
            );
            self.kernel.assemble_st(
                GreenKernelEvalType::Value,
                &child_downward_check_surface,
                &parent_downward_equivalent_surface,
                pe2cc.data_mut(),
            );

            let mut tmp = empty_array::<Scalar, 2>().simple_mult_into_resize(
                dc2e_inv_1[0].view(),
                empty_array::<Scalar, 2>()
                    .simple_mult_into_resize(dc2e_inv_2[0].view(), pe2cc.view()),
            );

            tmp.data_mut()
                .iter_mut()
                .for_each(|d| *d *= homogenous_kernel_scale(child.level()));

            let mut tmp2 = rlst_dynamic_array2!(Scalar, tmp.shape());
            tmp2.data_mut().copy_from_slice(tmp.data());

            l2l.push(tmp);
            l2l_global.push(tmp2);
        }

        self.global_fmm.target_vec = vec![l2l_global];
        self.global_fmm.dc2e_inv_1 = dc2e_inv_1_global;
        self.global_fmm.dc2e_inv_2 = dc2e_inv_2_global;

        self.target_vec = l2l;
        self.dc2e_inv_1 = dc2e_inv_1;
        self.dc2e_inv_2 = dc2e_inv_2;
    }
}

impl<Scalar> SourceToTargetTranslationMetadata
    for KiFmmMulti<Scalar, Laplace3dKernel<Scalar>, BlasFieldTranslationSaRcmp<Scalar>>
where
    Scalar: RlstScalar + Default + MatrixRsvd + Equivalence + Float,
    <Scalar as RlstScalar>::Real: Default + Equivalence + Float,
{
    fn displacements(&mut self) {
        let mut displacements = Vec::new();

        for level in 2..=self.tree.source_tree().total_depth() {
            let sources = self.tree.source_tree().keys(level).unwrap_or_default();
            let n_sources = sources.len();

            let sentinel = n_sources;
            let result = vec![vec![sentinel; n_sources]; 316];
            let result = result.into_iter().map(RwLock::new).collect_vec();

            let tmp = HashSet::new();
            let target_tree_keys_set = self.tree.target_tree().all_keys_set().unwrap_or(&tmp);

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
                            !source.is_adjacent(pnc) && target_tree_keys_set.contains(pnc)
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

                    let transfer_vectors_set: HashSet<_> = transfer_vectors.into_iter().collect();

                    // Mark items in interaction list for scattering
                    for (tv_idx, tv) in self.source_to_target.transfer_vectors.iter().enumerate() {
                        let mut result_lock = result[tv_idx].write().unwrap();
                        if transfer_vectors_set.contains(&tv.hash) {
                            // Look up scatter location in target tree
                            let target =
                                &interaction_list[*transfer_vectors_map.get(&tv.hash).unwrap()];
                            let &target_idx = self.level_index_pointer_locals[level as usize]
                                .get(target)
                                .unwrap();
                            result_lock[source_idx] = target_idx;
                        }
                    }
                });

            displacements.push(result);
        }

        self.source_to_target.displacements = displacements;
    }

    fn source_to_target(&mut self) {
        // Compute unique M2L interactions at level 3, shallowest level which contains them all
        // Compute interaction matrices between source and unique targets, defined by unique transfer vectors

        let equivalent_surface_order = self.equivalent_surface_order;
        let check_surface_order = self.check_surface_order;
        let n_coeffs_equivalent_surface = self.n_coeffs_equivalent_surface; //cols
        let n_coeffs_check_surface = self.n_coeffs_check_surface; // rows

        let mut se2tc_fat = rlst_dynamic_array2!(
            Scalar,
            [
                n_coeffs_check_surface,
                n_coeffs_equivalent_surface * NTRANSFER_VECTORS_KIFMM
            ]
        );
        let mut se2tc_thin = rlst_dynamic_array2!(
            Scalar,
            [
                n_coeffs_check_surface * NTRANSFER_VECTORS_KIFMM,
                n_coeffs_equivalent_surface
            ]
        );
        let alpha = Scalar::real(ALPHA_INNER);

        for (i, t) in self.source_to_target.transfer_vectors.iter().enumerate() {
            let source_equivalent_surface =
                t.source
                    .surface_grid(equivalent_surface_order, self.tree.domain(), alpha);

            let target_check_surface =
                t.target
                    .surface_grid(check_surface_order, self.tree.domain(), alpha);

            let mut tmp_gram = rlst_dynamic_array2!(
                Scalar,
                [n_coeffs_check_surface, n_coeffs_equivalent_surface]
            );

            self.kernel.assemble_st(
                GreenKernelEvalType::Value,
                &target_check_surface,
                &source_equivalent_surface,
                tmp_gram.data_mut(),
            );

            let mut block = se2tc_fat.view_mut().into_subview(
                [0, i * n_coeffs_equivalent_surface],
                [n_coeffs_check_surface, n_coeffs_equivalent_surface],
            );
            block.fill_from(tmp_gram.view_mut());

            let mut block_column = se2tc_thin.view_mut().into_subview(
                [i * n_coeffs_check_surface, 0],
                [n_coeffs_check_surface, n_coeffs_equivalent_surface],
            );
            block_column.fill_from(tmp_gram.view());
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
                    let max_equivalent_surface_ncoeffs = n_coeffs_equivalent_surface;
                    let max_check_surface_ncoeffs = n_coeffs_check_surface;
                    target_rank = max_equivalent_surface_ncoeffs.max(max_check_surface_ncoeffs) / 2;
                }

                let mut se2tc_fat_transpose =
                    rlst_dynamic_array2!(Scalar, se2tc_fat.view().transpose().shape());
                se2tc_fat_transpose
                    .view_mut()
                    .fill_from(se2tc_fat.view().transpose());

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
                        u_big.view_mut(),
                        vt_big.view_mut(),
                        &mut sigma[..],
                        SvdMode::Reduced,
                    )
                    .unwrap();
            }
        }

        // Cutoff rank is the minimum of the target rank and the value found by user threshold
        let cutoff_rank = find_cutoff_rank(
            &sigma,
            self.source_to_target.threshold,
            n_coeffs_equivalent_surface,
        )
        .min(target_rank);

        let mut u = rlst_dynamic_array2!(Scalar, [mu, cutoff_rank]);
        let mut sigma_mat = rlst_dynamic_array2!(Scalar, [cutoff_rank, cutoff_rank]);
        let mut vt = rlst_dynamic_array2!(Scalar, [cutoff_rank, nvt]);

        // Store compressed M2L operators
        let nst = se2tc_thin.shape()[1];
        let mut st = rlst_dynamic_array2!(Scalar, u_big.view().transpose().shape());
        st.fill_from(u_big.view().transpose());
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
            let vt_block = vt.view().into_subview(
                [0, i * n_coeffs_check_surface],
                [cutoff_rank, n_coeffs_check_surface],
            );

            let tmp = empty_array::<Scalar, 2>().simple_mult_into_resize(
                sigma_mat.view(),
                empty_array::<Scalar, 2>().simple_mult_into_resize(vt_block.view(), s_trunc.view()),
            );

            let mut u_i = rlst_dynamic_array2!(Scalar, [cutoff_rank, cutoff_rank]);
            let mut sigma_i = vec![Scalar::zero().re(); cutoff_rank];
            let mut vt_i = rlst_dynamic_array2!(Scalar, [cutoff_rank, cutoff_rank]);

            tmp.into_svd_alloc(u_i.view_mut(), vt_i.view_mut(), &mut sigma_i, SvdMode::Full)
                .unwrap();

            let directional_cutoff_rank =
                find_cutoff_rank(&sigma_i, self.source_to_target.threshold, cutoff_rank);

            let mut u_i_compressed =
                rlst_dynamic_array2!(Scalar, [cutoff_rank, directional_cutoff_rank]);
            let mut vt_i_compressed_ =
                rlst_dynamic_array2!(Scalar, [directional_cutoff_rank, cutoff_rank]);

            let mut sigma_mat_i_compressed =
                rlst_dynamic_array2!(Scalar, [directional_cutoff_rank, directional_cutoff_rank]);

            u_i_compressed
                .fill_from(u_i.into_subview([0, 0], [cutoff_rank, directional_cutoff_rank]));
            vt_i_compressed_
                .fill_from(vt_i.into_subview([0, 0], [directional_cutoff_rank, cutoff_rank]));

            for (j, s) in sigma_i.iter().enumerate().take(directional_cutoff_rank) {
                unsafe {
                    *sigma_mat_i_compressed.get_unchecked_mut([j, j]) = Scalar::from(*s).unwrap();
                }
            }

            let vt_i_compressed = empty_array::<Scalar, 2>()
                .simple_mult_into_resize(sigma_mat_i_compressed.view(), vt_i_compressed_.view());

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

        let mut u_ = rlst_dynamic_array2!(Scalar, u.shape());
        let mut st_ = rlst_dynamic_array2!(Scalar, st_trunc.shape());
        let mut c_u_ = c_u
            .iter()
            .map(|x| rlst_dynamic_array2!(Scalar, x.shape()))
            .collect_vec();
        let mut c_vt_ = c_vt
            .iter()
            .map(|x| rlst_dynamic_array2!(Scalar, x.shape()))
            .collect_vec();
        u_.data_mut().copy_from_slice(u.data());
        st_.data_mut().copy_from_slice(st_trunc.data());

        c_u_.iter_mut()
            .enumerate()
            .for_each(|(i, x)| x.data_mut().copy_from_slice(c_u[i].data()));
        c_vt_
            .iter_mut()
            .enumerate()
            .for_each(|(i, x)| x.data_mut().copy_from_slice(c_vt[i].data()));

        let result = BlasMetadataSaRcmp {
            u,
            st: st_trunc,
            c_u,
            c_vt,
        };

        let result_ = BlasMetadataSaRcmp {
            u: u_,
            st: st_,
            c_u: c_u_,
            c_vt: c_vt_,
        };

        self.global_fmm.source_to_target.metadata.push(result_);
        self.global_fmm
            .source_to_target
            .cutoff_rank
            .push(cutoff_rank);
        self.global_fmm
            .source_to_target
            .directional_cutoff_ranks
            .push(directional_cutoff_ranks.clone());

        self.source_to_target.metadata.push(result);
        self.source_to_target.cutoff_rank.push(cutoff_rank);
        self.source_to_target
            .directional_cutoff_ranks
            .push(directional_cutoff_ranks);
    }
}

impl<Scalar> SourceToTargetTranslationMetadata
    for KiFmmMulti<Scalar, Laplace3dKernel<Scalar>, FftFieldTranslation<Scalar>>
where
    Scalar: RlstScalar
        + AsComplex
        + Default
        + Dft<InputType = Scalar, OutputType = <Scalar as AsComplex>::ComplexType>
        + Equivalence
        + Float,
    <Scalar as RlstScalar>::Real: RlstScalar + Default + Equivalence + Float,
{
    fn displacements(&mut self) {
        let mut displacements = Vec::new();

        for level in 2..=self.tree.source_tree.total_depth() {
            let targets = self.tree.target_tree().keys(level).unwrap_or_default();
            let targets_parents: HashSet<MortonKey<_>> =
                targets.iter().map(|target| target.parent()).collect();
            let mut targets_parents = targets_parents.into_iter().collect_vec();
            targets_parents.sort();
            let ntargets_parents = targets_parents.len();

            let sources = self.tree.source_tree().keys(level).unwrap_or_default();

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

        // Pick a point in the middle of the domain
        let two = Scalar::real(2.0);
        let midway = self
            .tree
            .domain()
            .side_length()
            .iter()
            .map(|d| *d / two)
            .collect_vec();

        let point = midway
            .iter()
            .zip(self.tree.domain().origin())
            .map(|(m, o)| *m + *o)
            .collect_vec();

        let point = [point[0], point[1], point[2]];

        // Encode point in centre of domain and compute halo of parent, and their resp. children
        let key = MortonKey::from_point(&point, self.tree.source_tree().domain(), 3);
        let siblings = key.siblings();
        let parent = key.parent();
        let halo = parent.neighbors();
        let halo_children = halo.iter().map(|h| h.children()).collect_vec();

        let equivalent_surface_order = self.equivalent_surface_order;

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

                let source_equivalent_surface =
                    source.surface_grid(equivalent_surface_order, self.tree.domain(), alpha);
                let target_check_surface =
                    target.surface_grid(equivalent_surface_order, self.tree.domain(), alpha);

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
                        self.tree.domain(),
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
        let mut kernel_data =
            vec![
                vec![<Scalar as DftType>::OutputType::zero(); NSIBLINGS_SQUARED * transform_size];
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
                let k_f =
                    &kernel_f[frequency_offset..(frequency_offset + NSIBLINGS_SQUARED)].to_vec();
                let k_f_ = rlst_array_from_slice2!(k_f.as_slice(), [NSIBLINGS, NSIBLINGS]);
                let mut k_ft =
                    rlst_dynamic_array2!(<Scalar as DftType>::OutputType, [NSIBLINGS, NSIBLINGS]);
                k_ft.fill_from(k_f_.view());
                kernel_data_ft.push(k_ft.data().to_vec());
            }
        }

        let metadata = FftMetadata {
            kernel_data,
            kernel_data_f: kernel_data_ft,
        };

        // Set operator data
        self.source_to_target.metadata.push(metadata.clone());

        // Set required maps
        let (surf_to_conv_map, conv_to_surf_map) =
            Self::compute_surf_to_conv_map(equivalent_surface_order);
        self.source_to_target.surf_to_conv_map = vec![surf_to_conv_map];
        self.source_to_target.conv_to_surf_map = vec![conv_to_surf_map];

        // Copy for global FMM
        self.global_fmm
            .source_to_target
            .metadata
            .push(metadata.clone());
        self.global_fmm.source_to_target.surf_to_conv_map =
            self.source_to_target.surf_to_conv_map.clone();
        self.global_fmm.source_to_target.conv_to_surf_map =
            self.source_to_target.conv_to_surf_map.clone();
    }
}

impl<Scalar, Kernel, FieldTranslation> Metadata for KiFmmMulti<Scalar, Kernel, FieldTranslation>
where
    Scalar: RlstScalar + Default + Float + Equivalence,
    <Scalar as RlstScalar>::Real: Default + Float + Equivalence,
    Kernel: KernelTrait<T = Scalar> + HomogenousKernel + Default + Send + Sync,
    FieldTranslation: FieldTranslationTrait + Send + Sync,
    Self: DataAccessMulti + GhostExchange,
{
    type Scalar = Scalar;

    fn metadata(&mut self, eval_type: GreenKernelEvalType, _charges: &[Self::Scalar]) {
        // Check if computing potentials, or potentials and derivatives
        match eval_type {
            GreenKernelEvalType::Value => {}
            GreenKernelEvalType::ValueDeriv => {
                panic!("Only potential computation supported for now")
            }
        }
        let kernel_eval_size = match eval_type {
            GreenKernelEvalType::Value => 1,
            GreenKernelEvalType::ValueDeriv => 4,
        };

        let alpha_outer = Scalar::from(ALPHA_OUTER).unwrap().re();
        let alpha_inner = Scalar::from(ALPHA_INNER).unwrap().re();
        let n_target_points = self.tree.target_tree.n_coordinates_tot().unwrap();
        let n_source_keys = self.tree.source_tree().n_keys_tot().unwrap();
        let n_target_keys = self.tree.target_tree().n_keys_tot().unwrap();

        let equivalent_surface_order = self.equivalent_surface_order;
        let check_surface_order = self.check_surface_order;
        let n_coeffs_equivalent_surface = self.n_coeffs_equivalent_surface;

        // Allocate multipole and local buffers for all locally owned source/target octants
        let multipoles = vec![Scalar::default(); n_source_keys * n_coeffs_equivalent_surface];
        let locals = vec![Scalar::default(); n_target_keys * n_coeffs_equivalent_surface];

        // Index pointers of multipole and local data, indexed by level
        let level_index_pointer_multipoles = level_index_pointer_multi_node(&self.tree.source_tree);
        let level_index_pointer_locals = level_index_pointer_multi_node(&self.tree.target_tree);

        // Allocate buffers for local potential data
        let potentials = vec![Scalar::default(); n_target_points * kernel_eval_size];

        // Kernel scale at each target and source leaf
        let leaf_scales_sources =
            leaf_scales_multi_node::<Scalar>(&self.tree.source_tree, n_coeffs_equivalent_surface);

        // Pre compute check surfaces
        let leaf_upward_equivalent_surfaces_sources = leaf_surfaces_multi_node(
            &self.tree.source_tree,
            n_coeffs_equivalent_surface,
            alpha_inner,
            equivalent_surface_order,
        );

        let leaf_upward_check_surfaces_sources = leaf_surfaces_multi_node(
            &self.tree.source_tree,
            n_coeffs_equivalent_surface,
            alpha_outer,
            check_surface_order,
        );

        let leaf_downward_equivalent_surfaces_targets = leaf_surfaces_multi_node(
            &self.tree.target_tree,
            n_coeffs_equivalent_surface,
            alpha_outer,
            equivalent_surface_order,
        );

        // Mutable pointers to multipole and local data, indexed by level
        let level_multipoles = level_expansion_pointers_multi_node(
            &self.tree.source_tree,
            n_coeffs_equivalent_surface,
            &multipoles,
        );

        let level_locals = level_expansion_pointers_multi_node(
            &self.tree.target_tree,
            n_coeffs_equivalent_surface,
            &locals,
        );

        // Mutable pointers to multipole and local data only at leaf level, for utility
        let leaf_multipoles = leaf_expansion_pointers_multi_node(
            &self.tree.source_tree,
            n_coeffs_equivalent_surface,
            &multipoles,
        );

        let leaf_locals = leaf_expansion_pointers_multi_node(
            &self.tree.target_tree,
            n_coeffs_equivalent_surface,
            &locals,
        );

        // Mutable pointers to potential data at each target leaf
        let potential_send_pointers =
            potential_pointers_multi_node(&self.tree.target_tree, kernel_eval_size, &potentials);

        // TODO: Add functionality for charges at some point
        let charges = vec![Scalar::one(); self.tree.source_tree().n_coordinates_tot().unwrap()];
        let charge_index_pointer_targets =
            coordinate_index_pointer_multi_node(&self.tree.target_tree);
        let charge_index_pointer_sources =
            coordinate_index_pointer_multi_node(&self.tree.source_tree);

        // Set neighbourhood communicators
        self.neighbourhood_communicator_v = NeighbourhoodCommunicator::new(
            &self.communicator,
            &self.tree.v_list_query.send_marker,
            &self.tree.v_list_query.receive_marker,
        );

        self.neighbourhood_communicator_u = NeighbourhoodCommunicator::new(
            &self.communicator,
            &self.tree.u_list_query.send_marker,
            &self.tree.u_list_query.receive_marker,
        );

        // Set metadata
        self.multipoles = multipoles;
        self.leaf_multipoles = leaf_multipoles;
        self.level_multipoles = level_multipoles;
        self.level_index_pointer_multipoles = level_index_pointer_multipoles;
        self.locals = locals;
        self.leaf_locals = leaf_locals;
        self.level_locals = level_locals;
        self.level_index_pointer_locals = level_index_pointer_locals;
        self.potentials = potentials;
        self.potentials_send_pointers = potential_send_pointers;
        self.leaf_upward_equivalent_surfaces_sources = leaf_upward_equivalent_surfaces_sources;
        self.leaf_upward_check_surfaces_sources = leaf_upward_check_surfaces_sources;
        self.leaf_downward_equivalent_surfaces_targets = leaf_downward_equivalent_surfaces_targets;
        self.charges = charges;
        self.charge_index_pointer_sources = charge_index_pointer_sources;
        self.charge_index_pointer_targets = charge_index_pointer_targets;
        self.leaf_scales_sources = leaf_scales_sources;
        self.kernel_eval_size = kernel_eval_size;

        // Can perform U list exchange now
        self.u_list_exchange();
    }
}

impl<Scalar, Kernel> KiFmmMulti<Scalar, Kernel, FftFieldTranslation<Scalar>>
where
    Scalar: RlstScalar + AsComplex + Default + Dft + Float + Equivalence,
    Kernel: KernelTrait<T = Scalar> + HomogenousKernel + Default + Send + Sync,
    <Scalar as RlstScalar>::Real: Default + Float + Equivalence,
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

impl<Scalar, FieldTranslation> MetadataAccess
    for KiFmmMulti<Scalar, Laplace3dKernel<Scalar>, FieldTranslation>
where
    Scalar: RlstScalar + Default + Equivalence + Float,
    <Scalar as RlstScalar>::Real: RlstScalar + Default + Equivalence + Float,
    FieldTranslation: FieldTranslationTrait + Send + Sync,
    KiFmm<Scalar, Laplace3dKernel<Scalar>, FieldTranslation>: DataAccess<Scalar = Scalar, Tree = SingleNodeFmmTree<Scalar::Real>>
        + SourceToTargetTranslationMetadata,
{
    fn fft_map_index(&self, _level: u64) -> usize {
        0
    }

    fn expansion_index(&self, _level: u64) -> usize {
        0
    }

    fn c2e_operator_index(&self, _level: u64) -> usize {
        0
    }

    fn m2l_operator_index(&self, _level: u64) -> usize {
        0
    }

    fn l2l_operator_index(&self, _level: u64) -> usize {
        0
    }

    fn m2m_operator_index(&self, _level: u64) -> usize {
        0
    }

    fn displacement_index(&self, level: u64) -> usize {
        (level - 2) as usize
    }
}
