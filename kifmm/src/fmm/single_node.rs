//! Single Node FMM
use std::collections::HashSet;

use green_kernels::{
    helmholtz_3d::Helmholtz3dKernel, laplace_3d::Laplace3dKernel, traits::Kernel as KernelTrait,
    types::EvalType,
};
use itertools::Itertools;
use num::{Float, Zero};
use rlst::{
    empty_array, rlst_array_from_slice2, rlst_dynamic_array2, rlst_dynamic_array3, Array,
    BaseArray, Gemm, MatrixSvd, MultIntoResize, RawAccess, RawAccessMut, RlstScalar, Shape,
    SvdMode, UnsafeRandomAccessByRef, UnsafeRandomAccessMut, VectorContainer,
};

use crate::{
    fmm::{
        field_translation::source_to_target::transfer_vector::compute_transfer_vectors,
        types::{FmmEvalType, KiFmm},
    },
    traits::{
        fftw::{Dft, DftType},
        field::{
            KernelMetadataFieldTranslation, KernelMetadataSourceTarget,
            SourceToTargetData as SourceToTargetDataTrait,
        },
        fmm::{FmmKernel, FmmMetadata, SourceToTargetTranslation, SourceTranslation, TargetTranslation},
        general::{AsComplex, Epsilon},
        tree::{Domain, FmmTree, FmmTreeNode, Tree},
    },
    tree::{
        constants::{
            ALPHA_INNER, ALPHA_OUTER, NCORNERS, NHALO, NSIBLINGS, NSIBLINGS_SQUARED,
            NTRANSFER_VECTORS_KIFMM,
        },
        helpers::find_corners,
        types::MortonKey,
    },
    BlasFieldTranslation, FftFieldTranslation, Fmm, SingleNodeFmmTree,
};

use super::{
    helpers::{
        coordinate_index_pointer, flip3, homogenous_kernel_scale, leaf_expansion_pointers,
        leaf_scales, leaf_surfaces, level_expansion_pointers, level_index_pointer, map_charges,
        ncoeffs_kifmm, potential_pointers,
    },
    pinv::pinv,
    types::{BlasMetadata, Charges, FftMetadata},
};

impl<Scalar, SourceToTargetData> KernelMetadataSourceTarget
    for KiFmm<Scalar, Laplace3dKernel<Scalar>, SourceToTargetData>
where
    Scalar: RlstScalar + Default + Epsilon,
    Array<Scalar, BaseArray<Scalar, VectorContainer<Scalar>, 2>, 2>: MatrixSvd<Item = Scalar>,
    SourceToTargetData: SourceToTargetDataTrait + Send + Sync,
    <Scalar as RlstScalar>::Real: Default,
    Self: SourceToTargetTranslation,
{
    fn source(&mut self) {
        let root = MortonKey::<Scalar::Real>::root();

        // Cast surface parameters
        let alpha_outer = Scalar::from(ALPHA_OUTER).unwrap().re();
        let alpha_inner = Scalar::from(ALPHA_INNER).unwrap().re();
        let domain = self.tree.domain;

        // Compute required surfaces
        let upward_equivalent_surface =
            root.surface_grid(self.expansion_order, &domain, alpha_inner);
        let upward_check_surface = root.surface_grid(self.expansion_order, &domain, alpha_outer);

        let nequiv_surface = upward_equivalent_surface.len() / self.dim;
        let ncheck_surface = upward_check_surface.len() / self.dim;

        // Assemble matrix of kernel evaluations between upward check to equivalent, and downward check to equivalent matrices
        // As well as estimating their inverses using SVD
        let mut uc2e_t = rlst_dynamic_array2!(Scalar, [ncheck_surface, nequiv_surface]);
        self.kernel.assemble_st(
            EvalType::Value,
            &upward_equivalent_surface[..],
            &upward_check_surface[..],
            uc2e_t.data_mut(),
        );

        // Need to transpose so that rows correspond to targets and columns to sources
        let mut uc2e = rlst_dynamic_array2!(Scalar, [nequiv_surface, ncheck_surface]);
        uc2e.fill_from(uc2e_t.transpose());

        let (s, ut, v) = pinv(&uc2e, None, None).unwrap();

        let mut mat_s = rlst_dynamic_array2!(Scalar, [s.len(), s.len()]);
        for i in 0..s.len() {
            mat_s[[i, i]] = Scalar::from_real(s[i]);
        }

        let uc2e_inv_1 = vec![empty_array::<Scalar, 2>().simple_mult_into_resize(v.view(), mat_s.view())];
        let uc2e_inv_2 = vec![ut];

        // Calculate M2M operator matrices
        let children = root.children();
        let mut m2m = vec![rlst_dynamic_array2!(Scalar, [nequiv_surface, 8 * nequiv_surface])];
        let mut m2m_vec = vec![Vec::new()];

        for (i, child) in children.iter().enumerate() {
            let child_upward_equivalent_surface =
                child.surface_grid(self.expansion_order, &domain, alpha_inner);

            let mut pc2ce_t = rlst_dynamic_array2!(Scalar, [ncheck_surface, nequiv_surface]);

            self.kernel.assemble_st(
                EvalType::Value,
                &child_upward_equivalent_surface,
                &upward_check_surface,
                pc2ce_t.data_mut(),
            );

            // Need to transpose so that rows correspond to targets, and columns to sources
            let mut pc2ce = rlst_dynamic_array2!(Scalar, [nequiv_surface, ncheck_surface]);
            pc2ce.fill_from(pc2ce_t.transpose());

            let tmp = empty_array::<Scalar, 2>().simple_mult_into_resize(
                uc2e_inv_1[0].view(),
                empty_array::<Scalar, 2>().simple_mult_into_resize(uc2e_inv_2[0].view(), pc2ce.view()),
            );
            let l = i * nequiv_surface * nequiv_surface;
            let r = l + nequiv_surface * nequiv_surface;

            m2m[0].data_mut()[l..r].copy_from_slice(tmp.data());
            m2m_vec[0].push(tmp);
        }

        self.source = m2m;
        self.source_vec = m2m_vec;
        self.uc2e_inv_1 = uc2e_inv_1;
        self.uc2e_inv_2 = uc2e_inv_2;
    }

    fn target(&mut self) {
        let root = MortonKey::<Scalar::Real>::root();

        // Cast surface parameters
        let alpha_outer = Scalar::from(ALPHA_OUTER).unwrap().re();
        let alpha_inner = Scalar::from(ALPHA_INNER).unwrap().re();
        let domain = self.tree.domain;

        // Compute required surfaces
        let downward_equivalent_surface =
            root.surface_grid(self.expansion_order, &domain, alpha_outer);
        let downward_check_surface = root.surface_grid(self.expansion_order, &domain, alpha_inner);

        let nequiv_surface = downward_equivalent_surface.len() / self.dim;
        let ncheck_surface = downward_check_surface.len() / self.dim;

        // Assemble matrix of kernel evaluations between upward check to equivalent, and downward check to equivalent matrices
        // As well as estimating their inverses using SVD
        let mut dc2e_t = rlst_dynamic_array2!(Scalar, [ncheck_surface, nequiv_surface]);
        self.kernel.assemble_st(
            EvalType::Value,
            &downward_equivalent_surface[..],
            &downward_check_surface[..],
            dc2e_t.data_mut(),
        );

        // Need to transpose so that rows correspond to targets and columns to sources
        let mut dc2e = rlst_dynamic_array2!(Scalar, [nequiv_surface, ncheck_surface]);
        dc2e.fill_from(dc2e_t.transpose());

        let (s, ut, v) = pinv::<Scalar>(&dc2e, None, None).unwrap();

        let mut mat_s = rlst_dynamic_array2!(Scalar, [s.len(), s.len()]);
        for i in 0..s.len() {
            mat_s[[i, i]] = Scalar::from_real(s[i]);
        }

        let dc2e_inv_1 = vec![empty_array::<Scalar, 2>().simple_mult_into_resize(v.view(), mat_s.view())];
        let dc2e_inv_2 = vec![ut];

        // Calculate M2M and L2L operator matrices
        let children = root.children();
        let mut l2l = vec![Vec::new()];

        for (_i, child) in children.iter().enumerate() {
            let child_downward_check_surface =
                child.surface_grid(self.expansion_order, &domain, alpha_inner);
            // Need to transpose so that rows correspond to targets, and columns to sources

            let mut cc2pe_t = rlst_dynamic_array2!(Scalar, [ncheck_surface, nequiv_surface]);
            self.kernel.assemble_st(
                EvalType::Value,
                &downward_equivalent_surface,
                &child_downward_check_surface,
                cc2pe_t.data_mut(),
            );

            // Need to transpose so that rows correspond to targets, and columns to sources
            let mut cc2pe = rlst_dynamic_array2!(Scalar, [nequiv_surface, ncheck_surface]);
            cc2pe.fill_from(cc2pe_t.transpose());
            let mut tmp = empty_array::<Scalar, 2>().simple_mult_into_resize(
                dc2e_inv_1[0].view(),
                empty_array::<Scalar, 2>().simple_mult_into_resize(dc2e_inv_2[0].view(), cc2pe.view()),
            );
            tmp.data_mut()
                .iter_mut()
                .for_each(|d| *d *= homogenous_kernel_scale(child.level()));

            l2l[0].push(tmp);
        }

        self.target_vec = l2l;
        self.dc2e_inv_1 = dc2e_inv_1;
        self.dc2e_inv_2 = dc2e_inv_2;
    }
}

impl<Scalar, SourceToTargetData> KernelMetadataSourceTarget
    for KiFmm<Scalar, Helmholtz3dKernel<Scalar>, SourceToTargetData>
where
    Scalar: RlstScalar<Complex = Scalar> + Default + Epsilon,
    Array<Scalar, BaseArray<Scalar, VectorContainer<Scalar>, 2>, 2>: MatrixSvd<Item = Scalar>,
    SourceToTargetData: SourceToTargetDataTrait + Send + Sync,
    <Scalar as RlstScalar>::Real: Default,
    Self: SourceToTargetTranslation,
{
    fn source(&mut self) {
        let root = MortonKey::<Scalar::Real>::root();

        // Cast surface parameters
        let alpha_outer = Scalar::from(ALPHA_OUTER).unwrap().re();
        let alpha_inner = Scalar::from(ALPHA_INNER).unwrap().re();
        let domain = self.tree.domain;

        let depth = self.tree.source_tree().depth();

        let mut curr = root;
        let mut uc2e_inv_1 = Vec::new();
        let mut uc2e_inv_2= Vec::new();

        // Calculate inverse upward check to equivalent matrices on each level
        for _level in 0..=depth {

            // Compute required surfaces
            let upward_equivalent_surface =
                curr.surface_grid(self.expansion_order, &domain, alpha_inner);
            let upward_check_surface = curr.surface_grid(self.expansion_order, &domain, alpha_outer);

            let nequiv_surface = upward_equivalent_surface.len() / self.dim;
            let ncheck_surface = upward_check_surface.len() / self.dim;

            // Assemble matrix of kernel evaluations between upward check to equivalent, and downward check to equivalent matrices
            // As well as estimating their inverses using SVD
            let mut uc2e_t = rlst_dynamic_array2!(Scalar, [ncheck_surface, nequiv_surface]);
            self.kernel.assemble_st(
                EvalType::Value,
                &upward_equivalent_surface[..],
                &upward_check_surface[..],
                uc2e_t.data_mut(),
            );

            // Need to transpose so that rows correspond to targets and columns to sources
            let mut uc2e = rlst_dynamic_array2!(Scalar, [nequiv_surface, ncheck_surface]);
            uc2e.fill_from(uc2e_t.transpose());

            let (s, ut, v) = pinv(&uc2e, None, None).unwrap();

            let mut mat_s = rlst_dynamic_array2!(Scalar, [s.len(), s.len()]);
            for i in 0..s.len() {
                mat_s[[i, i]] = Scalar::from_real(s[i]);
            }

            uc2e_inv_1.push(empty_array::<Scalar, 2>().simple_mult_into_resize(v.view(), mat_s.view()));
            uc2e_inv_2.push(ut);

            curr = curr.first_child();
        }


        let mut curr = root;
        let mut source = Vec::new();
        let mut source_vec = Vec::new();

        for level in 0..depth {

            // Compute required surfaces
            let upward_equivalent_surface =
                curr.surface_grid(self.expansion_order, &domain, alpha_inner);
            let upward_check_surface = curr.surface_grid(self.expansion_order, &domain, alpha_outer);

            let nequiv_surface = upward_equivalent_surface.len() / self.dim;
            let ncheck_surface = upward_check_surface.len() / self.dim;

            // Calculate M2M operator matrices on each level
            let children = curr.children();
            let mut m2m = rlst_dynamic_array2!(Scalar, [nequiv_surface, 8 * nequiv_surface]);
            let mut m2m_vec = Vec::new();

            for (i, child) in children.iter().enumerate() {
                let child_upward_equivalent_surface =
                    child.surface_grid(self.expansion_order, &domain, alpha_inner);

                let mut pc2ce_t = rlst_dynamic_array2!(Scalar, [ncheck_surface, nequiv_surface]);

                self.kernel.assemble_st(
                    EvalType::Value,
                    &child_upward_equivalent_surface,
                    &upward_check_surface,
                    pc2ce_t.data_mut(),
                );

                // Need to transpose so that rows correspond to targets, and columns to sources
                let mut pc2ce = rlst_dynamic_array2!(Scalar, [nequiv_surface, ncheck_surface]);
                pc2ce.fill_from(pc2ce_t.transpose());

                let tmp = empty_array::<Scalar, 2>().simple_mult_into_resize(
                    uc2e_inv_1[level as usize].view(),
                    empty_array::<Scalar, 2>().simple_mult_into_resize(uc2e_inv_2[level as usize].view(), pc2ce.view()),
                );
                let l = i * nequiv_surface * nequiv_surface;
                let r = l + nequiv_surface * nequiv_surface;

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

    fn target(&mut self) {
        let root = MortonKey::<Scalar::Real>::root();

        // Cast surface parameters
        let alpha_outer = Scalar::from(ALPHA_OUTER).unwrap().re();
        let alpha_inner = Scalar::from(ALPHA_INNER).unwrap().re();
        let domain = self.tree.domain;

        let depth = self.tree.source_tree().depth();

        let mut curr = root;
        let mut dc2e_inv_1 = Vec::new();
        let mut dc2e_inv_2= Vec::new();

        // Calculate inverse upward check to equivalent matrices on each level
        for _level in 0..=depth {
            // Compute required surfaces
            let downward_equivalent_surface =
                curr.surface_grid(self.expansion_order, &domain, alpha_outer);
            let downward_check_surface = curr.surface_grid(self.expansion_order, &domain, alpha_inner);

            let nequiv_surface = downward_equivalent_surface.len() / self.dim;
            let ncheck_surface = downward_check_surface.len() / self.dim;

            // Assemble matrix of kernel evaluations between upward check to equivalent, and downward check to equivalent matrices
            // As well as estimating their inverses using SVD
            let mut dc2e_t = rlst_dynamic_array2!(Scalar, [ncheck_surface, nequiv_surface]);
            self.kernel.assemble_st(
                EvalType::Value,
                &downward_equivalent_surface[..],
                &downward_check_surface[..],
                dc2e_t.data_mut(),
            );

            // Need to transpose so that rows correspond to targets and columns to sources
            let mut dc2e = rlst_dynamic_array2!(Scalar, [nequiv_surface, ncheck_surface]);
            dc2e.fill_from(dc2e_t.transpose());

            let (s, ut, v) = pinv::<Scalar>(&dc2e, None, None).unwrap();

            let mut mat_s = rlst_dynamic_array2!(Scalar, [s.len(), s.len()]);
            for i in 0..s.len() {
                mat_s[[i, i]] = Scalar::from_real(s[i]);
            }

            dc2e_inv_1.push(empty_array::<Scalar, 2>().simple_mult_into_resize(v.view(), mat_s.view()));
            dc2e_inv_2.push(ut);
            curr = curr.first_child();
        }

        let mut curr = root;
        let mut target = Vec::new();

        for level in 0..depth {

            // Compute required surfaces
            let downward_equivalent_surface =
                curr.surface_grid(self.expansion_order, &domain, alpha_outer);
            let downard_check_surface = curr.surface_grid(self.expansion_order, &domain, alpha_inner);

            let nequiv_surface = downward_equivalent_surface.len() / self.dim;
            let ncheck_surface = downard_check_surface.len() / self.dim;

            // Calculate M2M operator matrices on each level
            let children = curr.children();
            let mut l2l = Vec::new();

            for (_i, child) in children.iter().enumerate() {
                let child_downward_check_surface =
                    child.surface_grid(self.expansion_order, &domain, alpha_inner);
                // Need to transpose so that rows correspond to targets, and columns to sources

                let mut cc2pe_t = rlst_dynamic_array2!(Scalar, [ncheck_surface, nequiv_surface]);
                self.kernel.assemble_st(
                    EvalType::Value,
                    &downward_equivalent_surface,
                    &child_downward_check_surface,
                    cc2pe_t.data_mut(),
                );

                // Need to transpose so that rows correspond to targets, and columns to sources
                let mut cc2pe = rlst_dynamic_array2!(Scalar, [nequiv_surface, ncheck_surface]);
                cc2pe.fill_from(cc2pe_t.transpose());
                let tmp = empty_array::<Scalar, 2>().simple_mult_into_resize(
                    dc2e_inv_1[(level + 1) as usize].view(),
                    empty_array::<Scalar, 2>().simple_mult_into_resize(dc2e_inv_2[(level + 1) as usize].view(), cc2pe.view()),
                );

                l2l.push(tmp);
            }

            target.push(l2l);
            curr = curr.first_child();
        }

        self.dc2e_inv_1 = dc2e_inv_1;
        self.dc2e_inv_2 = dc2e_inv_2;

        self.target_vec = target;
    }
}

impl<Scalar> KernelMetadataFieldTranslation
    for KiFmm<
        Scalar,
        Helmholtz3dKernel<Scalar>,
        BlasFieldTranslation<Scalar, Helmholtz3dKernel<Scalar>>,
    >
where
    Scalar: RlstScalar<Complex = Scalar> + Default,
    <Scalar as RlstScalar>::Real: Default,
{
    fn field_translation(&mut self) {}
}

impl<Scalar> KernelMetadataFieldTranslation
    for KiFmm<
        Scalar,
        Helmholtz3dKernel<Scalar>,
        FftFieldTranslation<Scalar, Helmholtz3dKernel<Scalar>>,
    >
where
    Scalar: RlstScalar<Complex = Scalar> + Default + AsComplex + Dft,
    <Scalar as RlstScalar>::Real: RlstScalar + Default,
{
    fn field_translation(&mut self) {}
}

/// Compute the cutoff rank for an SVD decomposition of a matrix from its singular values
/// using a specified `threshold` as a tolerance parameter
fn find_cutoff_rank<T: Float + RlstScalar + Gemm>(singular_values: &[T], threshold: T) -> usize {
    for (i, &s) in singular_values.iter().enumerate() {
        if s <= threshold {
            return i;
        }
    }

    singular_values.len() - 1
}

impl<Scalar> KernelMetadataFieldTranslation
    for KiFmm<
        Scalar,
        Laplace3dKernel<Scalar>,
        BlasFieldTranslation<Scalar, Laplace3dKernel<Scalar>>,
    >
where
    Scalar: RlstScalar + Default,
    <Scalar as RlstScalar>::Real: Default,
    Array<Scalar, BaseArray<Scalar, VectorContainer<Scalar>, 2>, 2>: MatrixSvd<Item = Scalar>,
{
    fn field_translation(&mut self) {
        // Compute unique M2L interactions at Level 3 (smallest choice with all vectors)
        // Compute interaction matrices between source and unique targets, defined by unique transfer vectors
        let nrows = ncoeffs_kifmm(self.expansion_order);
        let ncols = ncoeffs_kifmm(self.expansion_order);

        let mut se2tc_fat = rlst_dynamic_array2!(Scalar, [nrows, ncols * NTRANSFER_VECTORS_KIFMM]);
        let mut se2tc_thin = rlst_dynamic_array2!(Scalar, [nrows * NTRANSFER_VECTORS_KIFMM, ncols]);

        let alpha = Scalar::from(ALPHA_INNER).unwrap().re();

        for (i, t) in self.source_to_target.transfer_vectors.iter().enumerate() {
            let source_equivalent_surface = t.source.surface_grid(
                self.expansion_order,
                &self.tree.source_tree().domain(),
                alpha,
            );
            let nsources = source_equivalent_surface.len() / self.kernel.space_dimension();

            let target_check_surface = t.target.surface_grid(
                self.expansion_order,
                &self.tree.source_tree().domain(),
                alpha,
            );
            let ntargets = target_check_surface.len() / self.kernel.space_dimension();

            let mut tmp_gram_t = rlst_dynamic_array2!(Scalar, [ntargets, nsources]);

            self.kernel.assemble_st(
                EvalType::Value,
                &source_equivalent_surface[..],
                &target_check_surface[..],
                tmp_gram_t.data_mut(),
            );

            // Need to transpose so that rows correspond to targets, and columns to sources
            let mut tmp_gram = rlst_dynamic_array2!(Scalar, [nsources, ntargets]);
            tmp_gram.fill_from(tmp_gram_t.transpose());

            let mut block = se2tc_fat
                .view_mut()
                .into_subview([0, i * ncols], [nrows, ncols]);
            block.fill_from(tmp_gram.view());

            let mut block_column = se2tc_thin
                .view_mut()
                .into_subview([i * nrows, 0], [nrows, ncols]);
            block_column.fill_from(tmp_gram.view());
        }

        let mu = se2tc_fat.shape()[0];
        let nvt = se2tc_fat.shape()[1];
        let k = std::cmp::min(mu, nvt);

        let mut u_big = rlst_dynamic_array2!(Scalar, [mu, k]);
        let mut sigma = vec![Scalar::zero().re(); k];
        let mut vt_big = rlst_dynamic_array2!(Scalar, [k, nvt]);

        se2tc_fat
            .into_svd_alloc(
                u_big.view_mut(),
                vt_big.view_mut(),
                &mut sigma[..],
                SvdMode::Reduced,
            )
            .unwrap();
        let cutoff_rank = find_cutoff_rank(&sigma, self.source_to_target.threshold);
        let mut u = rlst_dynamic_array2!(Scalar, [mu, cutoff_rank]);
        let mut sigma_mat = rlst_dynamic_array2!(Scalar, [cutoff_rank, cutoff_rank]);
        let mut vt = rlst_dynamic_array2!(Scalar, [cutoff_rank, nvt]);

        u.fill_from(u_big.into_subview([0, 0], [mu, cutoff_rank]));
        vt.fill_from(vt_big.into_subview([0, 0], [cutoff_rank, nvt]));
        for (j, s) in sigma.iter().enumerate().take(cutoff_rank) {
            unsafe {
                *sigma_mat.get_unchecked_mut([j, j]) = Scalar::from(*s).unwrap();
            }
        }

        // Store compressed M2L operators
        let thin_nrows = se2tc_thin.shape()[0];
        let nst = se2tc_thin.shape()[1];
        let k = std::cmp::min(thin_nrows, nst);
        let mut _gamma = rlst_dynamic_array2!(Scalar, [thin_nrows, k]);
        let mut _r = vec![Scalar::zero().re(); k];
        let mut st = rlst_dynamic_array2!(Scalar, [k, nst]);

        se2tc_thin
            .into_svd_alloc(
                _gamma.view_mut(),
                st.view_mut(),
                &mut _r[..],
                SvdMode::Reduced,
            )
            .unwrap();

        let mut s_trunc = rlst_dynamic_array2!(Scalar, [nst, cutoff_rank]);
        for j in 0..cutoff_rank {
            for i in 0..nst {
                unsafe { *s_trunc.get_unchecked_mut([i, j]) = *st.get_unchecked([j, i]) }
            }
        }

        let mut c_u = Vec::new();
        let mut c_vt = Vec::new();

        for i in 0..self.source_to_target.transfer_vectors.len() {
            let vt_block = vt.view().into_subview([0, i * ncols], [cutoff_rank, ncols]);

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
                find_cutoff_rank(&sigma_i, self.source_to_target.threshold);

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

            c_u.push(u_i_compressed);
            c_vt.push(vt_i_compressed);
        }

        let mut st_trunc = rlst_dynamic_array2!(Scalar, [cutoff_rank, nst]);
        st_trunc.fill_from(s_trunc.transpose());

        let result = BlasMetadata {
            u,
            st: st_trunc,
            c_u,
            c_vt,
        };
        self.source_to_target.metadata = result;
        self.source_to_target.cutoff_rank = cutoff_rank;
    }
}

impl<Scalar, Kernel> KiFmm<Scalar, Kernel, FftFieldTranslation<Scalar, Kernel>>
where
    Scalar: RlstScalar + AsComplex + Default + Dft,
    Kernel: KernelTrait<T = Scalar> + FmmKernel + Default + Send + Sync,
    <Scalar as RlstScalar>::Real: Default,
{
    /// Computes the unique Green's function evaluations and places them on a convolution grid on the source box wrt to a given
    /// target point on the target box surface grid.
    ///
    /// # Arguments
    /// * `expansion_order` - The expansion order of the FMM
    /// * `convolution_grid` - Cartesian coordinates of points on the convolution grid at a source box, expected in column major order.
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
            EvalType::Value,
            convolution_grid,
            &target_pt[..],
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
        charges: &[Scalar],
    ) -> Array<Scalar, BaseArray<Scalar, VectorContainer<Scalar>, 3>, 3> {
        let n = 2 * expansion_order - 1;
        let npad = n + 1;

        let mut result = rlst_dynamic_array3!(Scalar, [npad, npad, npad]);

        for (i, &j) in self.source_to_target.surf_to_conv_map.iter().enumerate() {
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

impl<Scalar> KernelMetadataFieldTranslation
    for KiFmm<Scalar, Laplace3dKernel<Scalar>, FftFieldTranslation<Scalar, Laplace3dKernel<Scalar>>>
where
    Scalar: RlstScalar
        + AsComplex
        + Default
        + Dft<InputType = Scalar, OutputType = <Scalar as AsComplex>::ComplexType>,
    <Scalar as RlstScalar>::Real: RlstScalar + Default,
{
    fn field_translation(&mut self) {
        // Calculate source to target translation metadata
        // Set the expansion order
        self.source_to_target.expansion_order = self.expansion_order;

        // Set the associated kernel
        self.source_to_target.kernel = self.kernel.clone();

        // Compute the field translation operators
        // Parameters related to the FFT and Tree
        let shape = <Scalar as Dft>::shape_in(self.expansion_order);
        let transform_shape = <Scalar as Dft>::shape_out(self.expansion_order);
        let transform_size = <Scalar as Dft>::size_out(self.expansion_order);

        // Pick a point in the middle of the domain
        let two = Scalar::real(2.0);
        let midway = self
            .tree
            .source_tree()
            .domain
            .side_length
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
        let key = MortonKey::from_point(&point, &self.tree.source_tree().domain(), 3);
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

        let n_source_equivalent_surface = 6 * (self.expansion_order - 1).pow(2) + 2;
        let n_target_check_surface = n_source_equivalent_surface;
        let alpha = Scalar::real(ALPHA_INNER);

        // Iterate over each set of convolutions in the halo (26)
        for i in 0..transfer_vectors.len() {
            // Iterate over each unique convolution between sibling set, and halo siblings (64)
            for j in 0..transfer_vectors[i].len() {
                // Translating from sibling set to boxes in its M2L halo
                let target = targets[i][j];
                let source = sources[i][j];

                let source_equivalent_surface = source.surface_grid(
                    self.expansion_order,
                    &self.tree.source_tree().domain(),
                    alpha,
                );
                let target_check_surface = target.surface_grid(
                    self.expansion_order,
                    &&self.tree.source_tree().domain(),
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
                        corners[conv_point_corner_index],
                        corners[NCORNERS + conv_point_corner_index],
                        corners[2 * NCORNERS + conv_point_corner_index],
                    ];

                    let (conv_grid, _) = source.convolution_grid(
                        self.expansion_order,
                        &&self.tree.source_tree().domain(),
                        alpha,
                        &conv_point_corner,
                        conv_point_corner_index,
                    );

                    // Calculate Green's fct evaluations with respect to a 'kernel point' on the target box
                    let kernel_point_index = 0;
                    let kernel_point = [
                        target_check_surface[kernel_point_index],
                        target_check_surface[n_target_check_surface + kernel_point_index],
                        target_check_surface[2 * n_target_check_surface + kernel_point_index],
                    ];

                    // Compute Green's fct evaluations
                    let mut kernel = flip3(&self.evaluate_greens_fct_convolution_grid(
                        self.expansion_order,
                        &conv_grid,
                        kernel_point,
                    ));

                    // Compute FFT of padded kernel
                    let mut kernel_hat =
                        rlst_dynamic_array3!(<Scalar as DftType>::OutputType, transform_shape);

                    let _ = Scalar::forward_dft(kernel.data_mut(), kernel_hat.data_mut(), &shape);

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

        // Transpose results for better cache locality in application
        let mut kernel_data_ft = Vec::new();
        for freq in 0..transform_size {
            let frequency_offset = NSIBLINGS_SQUARED * freq;
            for kernel_f in kernel_data_f.iter().take(NHALO) {
                let k_f =
                    &kernel_f[frequency_offset..(frequency_offset + NSIBLINGS_SQUARED)].to_vec();
                let k_f_ = rlst_array_from_slice2!(k_f.as_slice(), [NSIBLINGS, NSIBLINGS]);
                let mut k_ft =
                    rlst_dynamic_array2!(<Scalar as DftType>::OutputType, [NSIBLINGS, NSIBLINGS]);
                k_ft.fill_from(k_f_.view().transpose());
                kernel_data_ft.push(k_ft.data().to_vec());
            }
        }

        let result = FftMetadata {
            kernel_data,
            kernel_data_f: kernel_data_ft,
        };

        // Set operator data
        self.source_to_target.metadata = result;

        // Set required maps, TODO: Should be a part of operator data
        (
            self.source_to_target.surf_to_conv_map,
            self.source_to_target.conv_to_surf_map,
        ) = Self::compute_surf_to_conv_map(self.expansion_order);

        // Set transfer vectors
        self.source_to_target.transfer_vectors = compute_transfer_vectors();
    }
}

impl<Scalar, Kernel, SourceToTargetData> FmmMetadata for KiFmm<Scalar, Kernel, SourceToTargetData>
where
    Scalar: RlstScalar + Default,
    Kernel: KernelTrait<T = Scalar> + FmmKernel + Default + Send + Sync,
    SourceToTargetData: SourceToTargetDataTrait + Send + Sync,
    <Scalar as RlstScalar>::Real: Default,
{
    // type Scalar = <Self as
    type Scalar = Scalar;

    fn metadata(&mut self, eval_type: EvalType, charges: &Charges<Self::Scalar>) {
        let alpha_outer = Scalar::real(ALPHA_OUTER);

        // Check if computing potentials, or potentials and derivatives
        let eval_size = match eval_type {
            EvalType::Value => 1,
            EvalType::ValueDeriv => self.dim + 1,
        };

        // Check if we are computing matvec or matmul
        let [_ncharges, nmatvecs] = charges.shape();
        let ntarget_points = self.tree.target_tree.all_coordinates().unwrap().len() / self.dim;
        let nsource_keys = self.tree.source_tree.n_keys_tot().unwrap();
        let ntarget_keys = self.tree.target_tree.n_keys_tot().unwrap();
        let ntarget_leaves = self.tree.target_tree.n_leaves().unwrap();
        let nsource_leaves = self.tree.source_tree.n_leaves().unwrap();

        // Buffers to store all multipole and local data
        let multipoles = vec![Scalar::default(); self.ncoeffs * nsource_keys * nmatvecs];
        let locals = vec![Scalar::default(); self.ncoeffs * ntarget_keys * nmatvecs];

        // Index pointers of multipole and local data, indexed by level
        let level_index_pointer_multipoles = level_index_pointer(&self.tree.source_tree);
        let level_index_pointer_locals = level_index_pointer(&self.tree.target_tree);

        // Buffer to store evaluated potentials and/or gradients at target points
        let potentials = vec![Scalar::default(); ntarget_points * eval_size * nmatvecs];

        // Kernel scale at each target and source leaf
        // let source_leaf_scales = leaf_scales(&self.tree.source_tree, self.ncoeffs);
        let mut source_leaf_scales = vec![Scalar::default(); self.tree.source_tree.n_leaves().unwrap() * self.ncoeffs];

        for (i, leaf) in self.tree.source_tree.all_leaves().unwrap().iter().enumerate() {
            // Assign scales
            let l = i * self.ncoeffs;
            let r = l + self.ncoeffs;
            source_leaf_scales[l..r]
                .copy_from_slice(vec![self.kernel.scale(leaf.level()); self.ncoeffs].as_slice());
        }

        // Pre compute check surfaces
        let leaf_upward_surfaces_sources = leaf_surfaces(
            &self.tree.source_tree,
            self.ncoeffs,
            alpha_outer,
            self.expansion_order,
        );
        let leaf_upward_surfaces_targets = leaf_surfaces(
            &self.tree.target_tree,
            self.ncoeffs,
            alpha_outer,
            self.expansion_order,
        );

        // Mutable pointers to multipole and local data, indexed by level
        let level_multipoles =
            level_expansion_pointers(&self.tree.source_tree, self.ncoeffs, nmatvecs, &multipoles);

        let level_locals =
            level_expansion_pointers(&self.tree.source_tree, self.ncoeffs, nmatvecs, &locals);

        // Mutable pointers to multipole and local data only at leaf level
        let leaf_multipoles = leaf_expansion_pointers(
            &self.tree.source_tree,
            self.ncoeffs,
            nmatvecs,
            nsource_leaves,
            &multipoles,
        );

        let leaf_locals = leaf_expansion_pointers(
            &self.tree.target_tree,
            self.ncoeffs,
            nmatvecs,
            ntarget_leaves,
            &locals,
        );

        // Mutable pointers to potential data at each target leaf
        let potentials_send_pointers = potential_pointers(
            &self.tree.target_tree,
            nmatvecs,
            ntarget_leaves,
            ntarget_points,
            eval_size,
            &potentials,
        );

        // Index pointer of charge data at each target leaf
        let charge_index_pointer_targets = coordinate_index_pointer(&self.tree.target_tree);
        let charge_index_pointer_sources = coordinate_index_pointer(&self.tree.source_tree);

        // Set data
        self.multipoles = multipoles;
        self.locals = locals;
        self.leaf_multipoles = leaf_multipoles;
        self.level_multipoles = level_multipoles;
        self.leaf_locals = leaf_locals;
        self.level_locals = level_locals;
        self.level_index_pointer_locals = level_index_pointer_locals;
        self.level_index_pointer_multipoles = level_index_pointer_multipoles;
        self.potentials = potentials;
        self.potentials_send_pointers = potentials_send_pointers;
        self.leaf_upward_surfaces_sources = leaf_upward_surfaces_sources;
        self.leaf_upward_surfaces_targets = leaf_upward_surfaces_targets;
        self.charges = charges.data().to_vec();
        self.charge_index_pointer_targets = charge_index_pointer_targets;
        self.charge_index_pointer_sources = charge_index_pointer_sources;
        self.leaf_scales_sources = source_leaf_scales;
        self.kernel_eval_size = eval_size;
    }
}

impl<Scalar, Kernel, SourceToTargetData> Fmm for KiFmm<Scalar, Kernel, SourceToTargetData>
where
    Scalar: RlstScalar + Default,
    Kernel: KernelTrait<T = Scalar> + FmmKernel +Default + Send + Sync,
    SourceToTargetData: SourceToTargetDataTrait + Send + Sync,
    <Scalar as RlstScalar>::Real: Default,
    Self: SourceToTargetTranslation,
{
    type Scalar = Scalar;
    type Kernel = Kernel;
    type Tree = SingleNodeFmmTree<Scalar::Real>;

    fn dim(&self) -> usize {
        self.dim
    }

    fn expansion_order(&self) -> usize {
        self.expansion_order
    }

    fn kernel(&self) -> &Self::Kernel {
        &self.kernel
    }

    fn tree(&self) -> &Self::Tree {
        &self.tree
    }

    fn multipole(
        &self,
        key: &<<Self::Tree as crate::traits::tree::FmmTree>::Tree as crate::traits::tree::Tree>::Node,
    ) -> Option<&[Self::Scalar]> {
        if let Some(index) = self.tree().source_tree().index(key) {
            match self.fmm_eval_type {
                FmmEvalType::Vector => {
                    Some(&self.multipoles[index * self.ncoeffs..(index + 1) * self.ncoeffs])
                }
                FmmEvalType::Matrix(nmatvecs) => Some(
                    &self.multipoles
                        [index * self.ncoeffs * nmatvecs..(index + 1) * self.ncoeffs * nmatvecs],
                ),
            }
        } else {
            None
        }
    }

    fn local(
        &self,
        key: &<<Self::Tree as FmmTree>::Tree as Tree>::Node,
    ) -> Option<&[Self::Scalar]> {
        if let Some(index) = self.tree.target_tree().index(key) {
            match self.fmm_eval_type {
                FmmEvalType::Vector => {
                    Some(&self.locals[index * self.ncoeffs..(index + 1) * self.ncoeffs])
                }
                FmmEvalType::Matrix(nmatvecs) => Some(
                    &self.locals
                        [index * self.ncoeffs * nmatvecs..(index + 1) * self.ncoeffs * nmatvecs],
                ),
            }
        } else {
            None
        }
    }

    fn potential(
        &self,
        leaf: &<<Self::Tree as FmmTree>::Tree as Tree>::Node,
    ) -> Option<Vec<&[Self::Scalar]>> {
        if let Some(&leaf_idx) = self.tree.target_tree().leaf_index(leaf) {
            let (l, r) = self.charge_index_pointer_targets[leaf_idx];
            let ntargets = r - l;

            match self.fmm_eval_type {
                FmmEvalType::Vector => Some(vec![
                    &self.potentials[l * self.kernel_eval_size..r * self.kernel_eval_size],
                ]),
                FmmEvalType::Matrix(nmatvecs) => {
                    let n_leaves = self.tree.target_tree().n_leaves().unwrap();
                    let mut slices = Vec::new();
                    for eval_idx in 0..nmatvecs {
                        let potentials_pointer =
                            self.potentials_send_pointers[eval_idx * n_leaves + leaf_idx].raw;
                        slices.push(unsafe {
                            std::slice::from_raw_parts(
                                potentials_pointer,
                                ntargets * self.kernel_eval_size,
                            )
                        });
                    }
                    Some(slices)
                }
            }
        } else {
            None
        }
    }

    fn evaluate(&self) {
        // Upward pass
        {
            self.p2m();
            for level in (1..=self.tree().source_tree().depth()).rev() {
                self.m2m(level)
            }
        }

        // Downward pass
        {
            for level in 2..=self.tree().target_tree().depth() {
                if level > 2 {
                    self.l2l(level);
                }
                self.m2l(level);
                self.p2l(level);
            }

            // Leaf level computation
            self.m2p();
            self.p2p();
            self.l2p();
        }
    }

    fn clear(&mut self, charges: &Charges<Self::Scalar>) {
        let [_ncharges, nmatvecs] = charges.shape();
        let ntarget_points = self.tree().target_tree().n_coordinates_tot().unwrap();
        let nsource_leaves = self.tree().source_tree().n_leaves().unwrap();
        let ntarget_leaves = self.tree().target_tree().n_leaves().unwrap();

        // Clear buffers and set new buffers
        self.multipoles = vec![Scalar::default(); self.multipoles.len()];
        self.locals = vec![Scalar::default(); self.locals.len()];
        self.potentials = vec![Scalar::default(); self.potentials.len()];
        self.charges = vec![Scalar::default(); self.charges.len()];

        // Recreate mutable pointers for new buffers
        let potentials_send_pointers = potential_pointers(
            self.tree.target_tree(),
            nmatvecs,
            ntarget_leaves,
            ntarget_points,
            self.kernel_eval_size,
            &self.potentials,
        );

        let leaf_multipoles = leaf_expansion_pointers(
            self.tree().source_tree(),
            self.ncoeffs,
            nmatvecs,
            nsource_leaves,
            &self.multipoles,
        );

        let level_multipoles = level_expansion_pointers(
            self.tree().source_tree(),
            self.ncoeffs,
            nmatvecs,
            &self.multipoles,
        );

        let level_locals = level_expansion_pointers(
            self.tree().target_tree(),
            self.ncoeffs,
            nmatvecs,
            &self.locals,
        );

        let leaf_locals = leaf_expansion_pointers(
            self.tree().target_tree(),
            self.ncoeffs,
            nmatvecs,
            ntarget_leaves,
            &self.locals,
        );

        // Set mutable pointers
        self.level_locals = level_locals;
        self.level_multipoles = level_multipoles;
        self.leaf_locals = leaf_locals;
        self.leaf_multipoles = leaf_multipoles;
        self.potentials_send_pointers = potentials_send_pointers;

        // Set new charges
        self.charges = map_charges(
            self.tree.source_tree().all_global_indices().unwrap(),
            charges,
        )
        .data()
        .to_vec();
    }
}

#[allow(clippy::type_complexity)]
#[cfg(test)]
mod test {
    use green_kernels::{
        helmholtz_3d::Helmholtz3dKernel, laplace_3d::Laplace3dKernel, traits::Kernel,
        types::EvalType,
    };
    use num::{Float, One, Zero};
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use rlst::{
        c64, empty_array, rlst_array_from_slice2, rlst_dynamic_array2, Array, BaseArray, MultIntoResize, RawAccess, RawAccessMut, RlstScalar, Shape, VectorContainer
    };

    use crate::{
        fmm, traits::{fmm::SourceTranslation, tree::{FmmTree, FmmTreeNode, Tree}}, tree::{constants::{ALPHA_INNER, ALPHA_OUTER}, helpers::points_fixture, types::MortonKey}, BlasFieldTranslation, FftFieldTranslation, Fmm, SingleNodeBuilder, SingleNodeFmmTree
    };

    fn test_single_node_laplace_fmm_matrix_helper<T: RlstScalar + Float + Default>(
        fmm: Box<
            dyn Fmm<
                Scalar = T::Real,
                Kernel = Laplace3dKernel<T::Real>,
                Tree = SingleNodeFmmTree<T::Real>,
            >,
        >,
        eval_type: EvalType,
        sources: &Array<T::Real, BaseArray<T::Real, VectorContainer<T::Real>, 2>, 2>,
        charges: &Array<T::Real, BaseArray<T::Real, VectorContainer<T::Real>, 2>, 2>,
        threshold: T::Real,
    ) where
        T::Real: Default,
    {
        let eval_size = match eval_type {
            EvalType::Value => 1,
            EvalType::ValueDeriv => 4,
        };

        let leaf_idx = 0;
        let leaf = fmm.tree().target_tree().all_leaves().unwrap()[leaf_idx];

        let leaf_targets = fmm.tree().target_tree().coordinates(&leaf).unwrap();

        let ntargets = leaf_targets.len() / fmm.dim();

        let leaf_coordinates_row_major =
            rlst_array_from_slice2!(leaf_targets, [ntargets, fmm.dim()], [fmm.dim(), 1]);

        let mut leaf_coordinates_col_major = rlst_dynamic_array2!(T::Real, [ntargets, fmm.dim()]);
        leaf_coordinates_col_major.fill_from(leaf_coordinates_row_major.view());

        let [nsources, nmatvecs] = charges.shape();

        for i in 0..nmatvecs {
            let potential_i = fmm.potential(&leaf).unwrap()[i];
            let charges_i = &charges.data()[nsources * i..nsources * (i + 1)];
            let mut direct_i = vec![T::Real::zero(); ntargets * eval_size];
            fmm.kernel().evaluate_st(
                eval_type,
                sources.data(),
                leaf_coordinates_col_major.data(),
                charges_i,
                &mut direct_i,
            );

            println!(
                "i {:?} \n direct_i {:?}\n potential_i {:?}",
                i, direct_i, potential_i
            );
            direct_i.iter().zip(potential_i).for_each(|(&d, &p)| {
                let abs_error = RlstScalar::abs(d - p);
                let rel_error = abs_error / p;
                assert!(rel_error <= threshold)
            })
        }
    }

    fn test_single_node_helmholtz_fmm_vector_helper<T: RlstScalar<Complex = T> + Default>(
        fmm: Box<
            dyn Fmm<Scalar = T, Kernel = Helmholtz3dKernel<T>, Tree = SingleNodeFmmTree<T::Real>>,
        >,
        eval_type: EvalType,
        sources: &Array<T::Real, BaseArray<T::Real, VectorContainer<T::Real>, 2>, 2>,
        charges: &Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>,
        threshold: T::Real,
    ) where
        T::Real: Default,
    {
        let eval_size = match eval_type {
            EvalType::Value => 1,
            EvalType::ValueDeriv => 4,
        };

        let leaf_idx = 0;
        let leaf = fmm.tree().target_tree().all_leaves().unwrap()[leaf_idx];
        let potential = fmm.potential(&leaf).unwrap()[0];

        let leaf_targets = fmm.tree().target_tree().coordinates(&leaf).unwrap();

        let ntargets = leaf_targets.len() / fmm.dim();
        let mut direct = vec![T::zero(); ntargets * eval_size];

        let leaf_coordinates_row_major =
            rlst_array_from_slice2!(leaf_targets, [ntargets, fmm.dim()], [fmm.dim(), 1]);
        let mut leaf_coordinates_col_major = rlst_dynamic_array2!(T::Real, [ntargets, fmm.dim()]);
        leaf_coordinates_col_major.fill_from(leaf_coordinates_row_major.view());

        fmm.kernel().evaluate_st(
            eval_type,
            sources.data(),
            leaf_coordinates_col_major.data(),
            charges.data(),
            &mut direct,
        );

        direct.iter().zip(potential).for_each(|(&d, &p)| {
            let abs_error = (d - p).abs();
            let rel_error = abs_error / p.abs();

            println!("err {:?} \nd {:?} \np {:?}", rel_error, direct, potential);
            assert!(rel_error <= threshold)
        });
    }

    fn test_single_node_laplace_fmm_vector_helper<T: RlstScalar + Float + Default>(
        fmm: Box<
            dyn Fmm<
                Scalar = T::Real,
                Kernel = Laplace3dKernel<T::Real>,
                Tree = SingleNodeFmmTree<T::Real>,
            >,
        >,
        eval_type: EvalType,
        sources: &Array<T::Real, BaseArray<T::Real, VectorContainer<T::Real>, 2>, 2>,
        charges: &Array<T::Real, BaseArray<T::Real, VectorContainer<T::Real>, 2>, 2>,
        threshold: T::Real,
    ) where
        T::Real: Default,
    {
        let eval_size = match eval_type {
            EvalType::Value => 1,
            EvalType::ValueDeriv => 4,
        };

        let leaf_idx = 0;
        let leaf = fmm.tree().target_tree().all_leaves().unwrap()[leaf_idx];
        let potential = fmm.potential(&leaf).unwrap()[0];

        let leaf_targets = fmm.tree().target_tree().coordinates(&leaf).unwrap();

        let ntargets = leaf_targets.len() / fmm.dim();
        let mut direct = vec![T::Real::zero(); ntargets * eval_size];

        let leaf_coordinates_row_major =
            rlst_array_from_slice2!(leaf_targets, [ntargets, fmm.dim()], [fmm.dim(), 1]);
        let mut leaf_coordinates_col_major = rlst_dynamic_array2!(T::Real, [ntargets, fmm.dim()]);
        leaf_coordinates_col_major.fill_from(leaf_coordinates_row_major.view());

        fmm.kernel().evaluate_st(
            eval_type,
            sources.data(),
            leaf_coordinates_col_major.data(),
            charges.data(),
            &mut direct,
        );

        direct.iter().zip(potential).for_each(|(&d, &p)| {
            let abs_error = RlstScalar::abs(d - p);
            let rel_error = abs_error / p;
            println!("err {:?} \nd {:?} \np {:?}", rel_error, direct, potential);
            assert!(rel_error <= threshold)
        });
    }

    fn test_root_multipole_laplace_single_node<T: RlstScalar + Float + Default>(
        fmm: Box<
            dyn Fmm<
                Scalar = T::Real,
                Kernel = Laplace3dKernel<T::Real>,
                Tree = SingleNodeFmmTree<T::Real>,
            >,
        >,
        sources: &Array<T::Real, BaseArray<T::Real, VectorContainer<T::Real>, 2>, 2>,
        charges: &Array<T::Real, BaseArray<T::Real, VectorContainer<T::Real>, 2>, 2>,
        threshold: T::Real,
    ) where
        T::Real: Default,
    {
        let root = MortonKey::root();

        let multipole = fmm.multipole(&root).unwrap();
        let upward_equivalent_surface = root.surface_grid(
            fmm.expansion_order(),
            fmm.tree().domain(),
            T::from(ALPHA_INNER).unwrap().re(),
        );

        let test_point = vec![T::real(100000.), T::Real::zero(), T::Real::zero()];
        let mut expected = vec![T::Real::zero()];
        let mut found = vec![T::Real::zero()];

        fmm.kernel().evaluate_st(
            EvalType::Value,
            sources.data(),
            &test_point,
            charges.data(),
            &mut expected,
        );

        fmm.kernel().evaluate_st(
            EvalType::Value,
            &upward_equivalent_surface,
            &test_point,
            multipole,
            &mut found,
        );

        let abs_error = RlstScalar::abs(expected[0] - found[0]);
        let rel_error = abs_error / expected[0];

        assert!(rel_error <= threshold);
    }

    fn test_root_multipole_helmholtz_single_node<T: RlstScalar<Complex = T> + Default>(
        fmm: Box<
            dyn Fmm<Scalar = T, Kernel = Helmholtz3dKernel<T>, Tree = SingleNodeFmmTree<T::Real>>,
        >,
        sources: &Array<T::Real, BaseArray<T::Real, VectorContainer<T::Real>, 2>, 2>,
        charges: &Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>,
        threshold: T::Real,
    ) where
        T::Real: Default,
    {
        let root = MortonKey::<T::Real>::root();
        let multipole = fmm.multipole(&root).unwrap();

        let upward_equivalent_surface = root.surface_grid(
            fmm.expansion_order(),
            fmm.tree().domain(),
            T::from(ALPHA_INNER).unwrap().re(),
        );

        let test_point = vec![T::real(100000.), T::Real::zero(), T::Real::zero()];
        let mut expected = vec![T::zero()];
        let mut found = vec![T::zero()];

        fmm.kernel().evaluate_st(
            EvalType::Value,
            sources.data(),
            &test_point,
            charges.data(),
            &mut expected,
        );

        fmm.kernel().evaluate_st(
            EvalType::Value,
            &upward_equivalent_surface,
            &test_point,
            multipole,
            &mut found,
        );

        let abs_error = (expected[0] - found[0]).abs();
        let rel_error = abs_error / expected[0].abs();
        println!(
            "abs {:?} rel {:?} \n expected {:?} found {:?}",
            abs_error, rel_error, expected, found
        );
        assert!(rel_error <= threshold);
    }

    #[test]
    fn test_upward_pass_vector_laplace() {
        // Setup random sources and targets
        let nsources = 10000;
        let ntargets = 10000;
        let sources = points_fixture::<f64>(nsources, None, None, Some(1));
        let targets = points_fixture::<f64>(ntargets, None, None, Some(1));

        // FMM parameters
        let n_crit = Some(100);
        let expansion_order = 6;
        let sparse = true;

        // Charge data
        let nvecs = 1;
        let mut rng = StdRng::seed_from_u64(0);
        let mut charges = rlst_dynamic_array2!(f64, [nsources, nvecs]);
        charges.data_mut().iter_mut().for_each(|c| *c = rng.gen());

        let fmm_fft = SingleNodeBuilder::new()
            .tree(&sources, &targets, n_crit, sparse)
            .unwrap()
            .parameters(
                &charges,
                expansion_order,
                Laplace3dKernel::new(),
                EvalType::Value,
                FftFieldTranslation::new(),
            )
            .unwrap()
            .build()
            .unwrap();
        fmm_fft.evaluate();

        let svd_threshold = Some(1e-5);
        let fmm_svd = SingleNodeBuilder::new()
            .tree(&sources, &targets, n_crit, sparse)
            .unwrap()
            .parameters(
                &charges,
                expansion_order,
                Laplace3dKernel::new(),
                EvalType::Value,
                BlasFieldTranslation::new(svd_threshold),
            )
            .unwrap()
            .build()
            .unwrap();
        fmm_svd.evaluate();

        let fmm_fft = Box::new(fmm_fft);
        let fmm_svd = Box::new(fmm_svd);
        test_root_multipole_laplace_single_node::<f64>(fmm_fft, &sources, &charges, 1e-5);
        test_root_multipole_laplace_single_node::<f64>(fmm_svd, &sources, &charges, 1e-5);
    }

    // #[test]
    // fn test_upward_pass_vector_helmholtz() {
    //     // Setup random sources and targets
    //     let nsources = 10000;
    //     let ntargets = 10000;
    //     let sources = points_fixture::<f64>(nsources, None, None, Some(1));
    //     let targets = points_fixture::<f64>(ntargets, None, None, Some(1));

    //     // FMM parameters
    //     let n_crit = Some(100);
    //     let expansion_order = 6;
    //     let sparse = true;

    //     // Charge data
    //     let nvecs = 1;
    //     let tmp = vec![c64::one(); nsources * nvecs];
    //     let mut charges = rlst_dynamic_array2!(c64, [nsources, nvecs]);
    //     charges.data_mut().copy_from_slice(&tmp);
    //     let wavenumber = 0.0000001;

    //     let fmm_fft = SingleNodeBuilder::new()
    //         .tree(&sources, &targets, n_crit, sparse)
    //         .unwrap()
    //         .parameters(
    //             &charges,
    //             expansion_order,
    //             Helmholtz3dKernel::new(wavenumber),
    //             EvalType::Value,
    //             FftFieldTranslation::new(),
    //         )
    //         .unwrap()
    //         .build()
    //         .unwrap();
    //     fmm_fft.evaluate();

    //     let svd_threshold = Some(1e-5);
    //     let fmm_svd = SingleNodeBuilder::new()
    //         .tree(&sources, &targets, n_crit, sparse)
    //         .unwrap()
    //         .parameters(
    //             &charges,
    //             expansion_order,
    //             Helmholtz3dKernel::new(wavenumber),
    //             EvalType::Value,
    //             BlasFieldTranslation::new(svd_threshold),
    //         )
    //         .unwrap()
    //         .build()
    //         .unwrap();
    //     fmm_svd.evaluate();

    //     let fmm_fft = Box::new(fmm_fft);
    //     let fmm_svd = Box::new(fmm_svd);
    //     test_root_multipole_helmholtz_single_node::<c64>(fmm_fft, &sources, &charges, 1e-5);
    //     test_root_multipole_helmholtz_single_node::<c64>(fmm_svd, &sources, &charges, 1e-5);
    // }

    #[test]
    fn test_fmm_api() {
        // Setup random sources and targets
        let nsources = 9000;
        let ntargets = 10000;

        let min = None;
        let max = None;
        let sources = points_fixture::<f64>(nsources, min, max, Some(0));
        let targets = points_fixture::<f64>(ntargets, min, max, Some(1));

        // FMM parameters
        let n_crit = Some(100);
        let expansion_order = 6;
        let sparse = true;
        let threshold_pot = 1e-5;

        // Set charge data and evaluate an FMM
        let nvecs = 1;
        let mut rng = StdRng::seed_from_u64(0);
        let mut charges = rlst_dynamic_array2!(f64, [nsources, nvecs]);
        charges.data_mut().iter_mut().for_each(|c| *c = rng.gen());

        let mut fmm = SingleNodeBuilder::new()
            .tree(&sources, &targets, n_crit, sparse)
            .unwrap()
            .parameters(
                &charges,
                expansion_order,
                Laplace3dKernel::new(),
                EvalType::Value,
                FftFieldTranslation::new(),
            )
            .unwrap()
            .build()
            .unwrap();
        fmm.evaluate();

        // Reset Charge data and re-evaluate potential
        let mut rng = StdRng::seed_from_u64(1);
        charges.data_mut().iter_mut().for_each(|c| *c = rng.gen());

        fmm.clear(&charges);
        fmm.evaluate();

        let fmm = Box::new(fmm);
        test_single_node_laplace_fmm_vector_helper::<f64>(
            fmm,
            EvalType::Value,
            &sources,
            &charges,
            threshold_pot,
        );
    }

    #[test]
    fn test_laplace_fmm_vector() {
        // Setup random sources and targets
        let nsources = 9000;
        let ntargets = 10000;

        let min = None;
        let max = None;
        let sources = points_fixture::<f64>(nsources, min, max, Some(0));
        let targets = points_fixture::<f64>(ntargets, min, max, Some(1));

        // FMM parameters
        let n_crit = Some(100);
        let expansion_order = 6;
        let sparse = true;
        let threshold_pot = 1e-5;
        let threshold_deriv = 1e-4;
        let threshold_deriv_blas = 1e-3;
        let singular_value_threshold = Some(1e-2);

        // Charge data
        let nvecs = 1;
        let mut rng = StdRng::seed_from_u64(0);
        let mut charges = rlst_dynamic_array2!(f64, [nsources, nvecs]);
        charges.data_mut().iter_mut().for_each(|c| *c = rng.gen());

        // fmm with fft based field translation
        {
            // Evaluate potentials
            let fmm_fft = SingleNodeBuilder::new()
                .tree(&sources, &targets, n_crit, sparse)
                .unwrap()
                .parameters(
                    &charges,
                    expansion_order,
                    Laplace3dKernel::new(),
                    EvalType::Value,
                    FftFieldTranslation::new(),
                )
                .unwrap()
                .build()
                .unwrap();
            fmm_fft.evaluate();
            let eval_type = fmm_fft.kernel_eval_type;
            let fmm_fft = Box::new(fmm_fft);
            test_single_node_laplace_fmm_vector_helper::<f64>(
                fmm_fft,
                eval_type,
                &sources,
                &charges,
                threshold_pot,
            );

            // Evaluate potentials + derivatives
            let fmm_fft = SingleNodeBuilder::new()
                .tree(&sources, &targets, n_crit, sparse)
                .unwrap()
                .parameters(
                    &charges,
                    expansion_order,
                    Laplace3dKernel::new(),
                    EvalType::ValueDeriv,
                    FftFieldTranslation::new(),
                )
                .unwrap()
                .build()
                .unwrap();
            fmm_fft.evaluate();
            let eval_type = fmm_fft.kernel_eval_type;
            let fmm_fft = Box::new(fmm_fft);
            test_single_node_laplace_fmm_vector_helper::<f64>(
                fmm_fft,
                eval_type,
                &sources,
                &charges,
                threshold_deriv,
            );
        }

        // fmm with BLAS based field translation
        {
            // Evaluate potentials
            let eval_type = EvalType::Value;
            let fmm_blas = SingleNodeBuilder::new()
                .tree(&sources, &targets, n_crit, sparse)
                .unwrap()
                .parameters(
                    &charges,
                    expansion_order,
                    Laplace3dKernel::new(),
                    eval_type,
                    BlasFieldTranslation::new(singular_value_threshold),
                )
                .unwrap()
                .build()
                .unwrap();
            fmm_blas.evaluate();
            let fmm_blas = Box::new(fmm_blas);
            test_single_node_laplace_fmm_vector_helper::<f64>(
                fmm_blas,
                eval_type,
                &sources,
                &charges,
                threshold_pot,
            );

            // Evaluate potentials + derivatives
            let eval_type = EvalType::ValueDeriv;
            let fmm_blas = SingleNodeBuilder::new()
                .tree(&sources, &targets, n_crit, sparse)
                .unwrap()
                .parameters(
                    &charges,
                    expansion_order,
                    Laplace3dKernel::new(),
                    eval_type,
                    BlasFieldTranslation::new(singular_value_threshold),
                )
                .unwrap()
                .build()
                .unwrap();
            fmm_blas.evaluate();
            let fmm_blas = Box::new(fmm_blas);
            test_single_node_laplace_fmm_vector_helper::<f64>(
                fmm_blas,
                eval_type,
                &sources,
                &charges,
                threshold_deriv_blas,
            );
        }
    }

    #[test]
    fn test_upward_pass_helmholtz() {
        // Setup random sources and targets
        let nsources = 10000;
        let ntargets = 10000;
        let sources = points_fixture::<f64>(nsources, None, None, Some(1));
        let targets = points_fixture::<f64>(ntargets, None, None, Some(1));

        // FMM parameters
        let n_crit = Some(100);
        let expansion_order = 6;
        let sparse = true;

        // Charge data
        let nvecs = 1;
        let tmp = vec![c64::one(); nsources * nvecs];
        let mut charges = rlst_dynamic_array2!(c64, [nsources, nvecs]);
        charges.data_mut().copy_from_slice(&tmp);

        let wavenumber = 5.0;

        let fmm_fft = SingleNodeBuilder::new()
            .tree(&sources, &targets, n_crit, sparse)
            .unwrap()
            .parameters(
                &charges,
                expansion_order,
                Helmholtz3dKernel::new(wavenumber),
                EvalType::Value,
                FftFieldTranslation::new(),
            )
            .unwrap()
            .build()
            .unwrap();

        // Manual upward pass
        let depth = fmm_fft.tree.source_tree.depth;
        fmm_fft.p2m();
        for level in (1..=depth).rev() {
            fmm_fft.m2m(level)
        }

        // Test a random leaf expansion
        let leaf_idx = 0;
        let leaf = fmm_fft.tree().source_tree().all_leaves().unwrap()[leaf_idx];

        let upward_check_surface= leaf.surface_grid(
            fmm_fft.expansion_order(),
            fmm_fft.tree().domain(),
            f64::from(ALPHA_OUTER),
        );

        let upward_equivalent_surface = leaf.surface_grid(
            fmm_fft.expansion_order(),
            fmm_fft.tree().domain(),
            f64::from(ALPHA_INNER),
        );


        let found_multipole = fmm_fft.multipole(&leaf).unwrap();

        let &leaf_idx = fmm_fft.tree().source_tree().leaf_index(&leaf).unwrap();
        let index_pointer = fmm_fft.charge_index_pointer_sources[leaf_idx];
        let charges = &fmm_fft.charges[index_pointer.0..index_pointer.1];
        let sources = &fmm_fft.tree().source_tree().all_coordinates().unwrap()
            [index_pointer.0 * 3..index_pointer.1 * 3];

        let charge_index_pointer = fmm_fft.charge_index_pointer_sources[leaf_idx];
        let charges =
            &fmm_fft.charges[charge_index_pointer.0..charge_index_pointer.1];

        let coordinates_row_major = &fmm_fft.tree().source_tree().all_coordinates().unwrap()[charge_index_pointer.0
            * fmm_fft.dim
            ..charge_index_pointer.1 * fmm_fft.dim];

        let nsources = coordinates_row_major.len() / fmm_fft.dim;

        let coordinates_row_major = rlst_array_from_slice2!(
            coordinates_row_major,
            [nsources, fmm_fft.dim],
            [fmm_fft.dim, 1]
        );
        let mut coordinates_col_major =
            rlst_dynamic_array2!(f64, [nsources, fmm_fft.dim]);
        coordinates_col_major.fill_from(coordinates_row_major.view());

        println!("charges {:?} {:?}", charges.len(), sources.len());

        let mut check_potential = rlst_dynamic_array2!(c64, [upward_check_surface.len() / 3, 1]);

        fmm_fft.kernel().evaluate_st(
            EvalType::Value,
            coordinates_col_major.data(),
            &upward_check_surface,
            charges,
            check_potential.data_mut(),
        );

        let evaluated_multipole = empty_array::<c64, 2>().simple_mult_into_resize(
            fmm_fft.uc2e_inv_1[leaf.level() as usize].view(),
            empty_array::<c64, 2>()
                .simple_mult_into_resize(fmm_fft.uc2e_inv_2[leaf.level() as usize].view(), check_potential.view()),
        );

        let test_point = vec![100f64, 0f64, 0f64];

        let mut evaluated = vec![c64::zero()];
        let mut found = vec![c64::zero()];
        let mut direct = vec![c64::zero()];

        fmm_fft.kernel().evaluate_st(
            EvalType::Value,
            &upward_equivalent_surface,
            &test_point,
            &found_multipole,
            &mut found,
        );

        fmm_fft.kernel().evaluate_st(
            EvalType::Value,
            &upward_equivalent_surface,
            &test_point,
            evaluated_multipole.data(),
            &mut evaluated,
        );

        fmm_fft.kernel().evaluate_st(
            EvalType::Value,
            &coordinates_col_major.data(),
            &test_point,
            &charges,
            &mut direct,
        );

        let abs_error = (evaluated[0] - direct[0]).abs();
        let rel_error = abs_error / direct[0].abs();
        // println!("found multipole {:?}", &found_multipole[0..5]);
        // println!("evaluated multipole {:?}", &evaluated_multipole.data()[0..5]);
        println!(
            "here 2 abs {:?} rel {:?} \n evaluated {:?} found {:?} direct {:?}",
            abs_error, rel_error, evaluated, found, direct
        );

        assert!(false);

        // fmm_fft.evaluate();


    }

    // #[test]
    // fn test_helmholtz_fmm_vector() {
    //     // Setup random sources and targets
    //     let nsources = 10000;
    //     let ntargets = 10000;
    //     let sources = points_fixture::<f64>(nsources, None, None, Some(1));
    //     let targets = points_fixture::<f64>(ntargets, None, None, Some(1));

    //     // FMM parameters
    //     let n_crit = Some(100);
    //     let expansion_order = 6;
    //     let sparse = true;

    //     // Charge data
    //     let nvecs = 1;
    //     let tmp = vec![c64::one(); nsources * nvecs];
    //     let mut charges = rlst_dynamic_array2!(c64, [nsources, nvecs]);
    //     charges.data_mut().copy_from_slice(&tmp);

    //     let wavenumber = 0.0000001;

    //     let fmm_fft = SingleNodeBuilder::new()
    //         .tree(&sources, &targets, n_crit, sparse)
    //         .unwrap()
    //         .parameters(
    //             &charges,
    //             expansion_order,
    //             Helmholtz3dKernel::new(wavenumber),
    //             EvalType::Value,
    //             FftFieldTranslation::new(),
    //         )
    //         .unwrap()
    //         .build()
    //         .unwrap();
    //     fmm_fft.evaluate();

    //     let svd_threshold = Some(1e-5);
    //     let fmm_svd = SingleNodeBuilder::new()
    //         .tree(&sources, &targets, n_crit, sparse)
    //         .unwrap()
    //         .parameters(
    //             &charges,
    //             expansion_order,
    //             Helmholtz3dKernel::new(wavenumber),
    //             EvalType::Value,
    //             BlasFieldTranslation::new(svd_threshold),
    //         )
    //         .unwrap()
    //         .build()
    //         .unwrap();
    //     fmm_svd.evaluate();

    //     let fmm_fft = Box::new(fmm_fft);
    //     let eval_type = fmm_fft.kernel_eval_type;
    //     test_single_node_helmholtz_fmm_vector_helper::<c64>(
    //         fmm_fft, eval_type, &sources, &charges, 1e-5,
    //     );

    //     let fmm_svd = Box::new(fmm_svd);
    //     let eval_type = fmm_svd.kernel_eval_type;
    //     test_single_node_helmholtz_fmm_vector_helper::<c64>(
    //         fmm_svd, eval_type, &sources, &charges, 1e-5,
    //     );
    // }

    #[test]
    fn test_laplace_fmm_matrix() {
        // Setup random sources and targets
        let nsources = 9000;
        let ntargets = 10000;

        let min = None;
        let max = None;
        let sources = points_fixture::<f64>(nsources, min, max, Some(0));
        let targets = points_fixture::<f64>(ntargets, min, max, Some(1));
        // FMM parameters
        let n_crit = Some(10);
        let expansion_order = 6;
        let sparse = true;
        let threshold = 1e-5;
        let threshold_deriv = 1e-3;
        let singular_value_threshold = Some(1e-2);

        // Charge data
        let nvecs = 5;
        let mut charges = rlst_dynamic_array2!(f64, [nsources, nvecs]);
        let mut rng = StdRng::seed_from_u64(0);
        charges
            .data_mut()
            .chunks_exact_mut(nsources)
            .for_each(|chunk| chunk.iter_mut().for_each(|elem| *elem += rng.gen::<f64>()));

        // fmm with blas based field translation
        {
            // Evaluate potentials
            let eval_type = EvalType::Value;
            let fmm_blas = SingleNodeBuilder::new()
                .tree(&sources, &targets, n_crit, sparse)
                .unwrap()
                .parameters(
                    &charges,
                    expansion_order,
                    Laplace3dKernel::new(),
                    eval_type,
                    BlasFieldTranslation::new(singular_value_threshold),
                )
                .unwrap()
                .build()
                .unwrap();
            fmm_blas.evaluate();

            let fmm_blas = Box::new(fmm_blas);
            test_single_node_laplace_fmm_matrix_helper::<f64>(
                fmm_blas, eval_type, &sources, &charges, threshold,
            );

            // Evaluate potentials + derivatives
            let eval_type = EvalType::ValueDeriv;
            let fmm_blas = SingleNodeBuilder::new()
                .tree(&sources, &targets, n_crit, sparse)
                .unwrap()
                .parameters(
                    &charges,
                    expansion_order,
                    Laplace3dKernel::new(),
                    eval_type,
                    BlasFieldTranslation::new(singular_value_threshold),
                )
                .unwrap()
                .build()
                .unwrap();
            fmm_blas.evaluate();
            let fmm_blas = Box::new(fmm_blas);
            test_single_node_laplace_fmm_matrix_helper::<f64>(
                fmm_blas,
                eval_type,
                &sources,
                &charges,
                threshold_deriv,
            );
        }
    }
}
