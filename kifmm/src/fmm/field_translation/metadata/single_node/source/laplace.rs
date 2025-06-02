//! Implementation of traits to compute metadata for field translation operations.

use green_kernels::{
    laplace_3d::Laplace3dKernel, traits::Kernel as KernelTrait, types::GreenKernelEvalType,
};
use rlst::{
    empty_array, rlst_dynamic_array2, MatrixSvd, MultIntoResize, RawAccess, RawAccessMut,
    RlstScalar,
};

use crate::{
    fmm::helpers::single_node::ncoeffs_kifmm,
    linalg::pinv::pinv,
    traits::{
        field::{FieldTranslation as FieldTranslationTrait, SourceTranslationMetadata},
        fmm::{DataAccess, MetadataAccess},
        general::single_node::Epsilon,
        tree::{FmmTreeNode, SingleFmmTree, SingleTree},
    },
    tree::{
        constants::{ALPHA_INNER, ALPHA_OUTER},
        types::MortonKey,
    },
    KiFmm,
};

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
        let depth = self.tree.source_tree.depth();

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

        let level_iterator = if self.variable_expansion_order {
            0..depth
        } else {
            0..1
        };

        // Calculate M2M operator matrices on each level, if required
        for parent_level in level_iterator {
            let check_surface_order_parent = self.check_surface_order(parent_level);
            let equivalent_surface_order_parent = self.equivalent_surface_order(parent_level);
            let equivalent_surface_order_child = self.equivalent_surface_order(parent_level + 1);

            let parent_upward_check_surface =
                root.surface_grid(check_surface_order_parent, domain, alpha_outer);

            let children = root.children();
            let n_check_surface_parent = ncoeffs_kifmm(check_surface_order_parent);
            let n_equiv_surface_child = ncoeffs_kifmm(equivalent_surface_order_child);
            let n_equiv_surface_parent = ncoeffs_kifmm(equivalent_surface_order_parent);

            let mut m2m_level =
                rlst_dynamic_array2!(Scalar, [n_equiv_surface_parent, 8 * n_equiv_surface_child]);
            let mut m2m_vec_level = Vec::new();

            for (i, child) in children.iter().enumerate() {
                let child_upward_equivalent_surface =
                    child.surface_grid(equivalent_surface_order_child, domain, alpha_inner);

                let mut ce2pc =
                    rlst_dynamic_array2!(Scalar, [n_check_surface_parent, n_equiv_surface_child]);

                // Note, this way around due to calling convention of kernel, source/targets are 'swapped'
                self.kernel.assemble_st(
                    GreenKernelEvalType::Value,
                    &parent_upward_check_surface,
                    &child_upward_equivalent_surface,
                    ce2pc.data_mut(),
                );

                let tmp = empty_array::<Scalar, 2>().simple_mult_into_resize(
                    uc2e_inv_1[self.expansion_index(parent_level)].r(),
                    empty_array::<Scalar, 2>().simple_mult_into_resize(
                        uc2e_inv_2[self.expansion_index(parent_level)].r(),
                        ce2pc.r(),
                    ),
                );

                let l = i * n_equiv_surface_child * n_equiv_surface_parent;
                let r = l + n_equiv_surface_child * n_equiv_surface_parent;

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
