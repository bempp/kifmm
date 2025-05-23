//! Implementation of traits to compute metadata for field translation operations.

use green_kernels::{
    laplace_3d::Laplace3dKernel, traits::Kernel as KernelTrait, types::GreenKernelEvalType,
};
use rlst::{empty_array, rlst_dynamic_array2, MatrixSvd, MultIntoResize, RawAccessMut, RlstScalar};

use crate::{
    fmm::helpers::single_node::{homogenous_kernel_scale, ncoeffs_kifmm},
    linalg::pinv::pinv,
    traits::{
        field::{FieldTranslation as FieldTranslationTrait, TargetTranslationMetadata},
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
        let depth = self.tree.target_tree.depth();

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

        let level_iterator = if self.equivalent_surface_order.len() > 1 {
            0..depth
        } else {
            0..1
        };

        for parent_level in level_iterator {
            let equivalent_surface_order_parent = self.equivalent_surface_order(parent_level);
            let check_surface_order_child = self.check_surface_order(parent_level + 1);

            let parent_downward_equivalent_surface =
                root.surface_grid(equivalent_surface_order_parent, domain, alpha_outer);

            // Calculate L2L operator matrices
            let children = root.children();
            let n_coeffs_check_surface_child = ncoeffs_kifmm(check_surface_order_child);
            let n_coeffs_equivalent_surface_parent = ncoeffs_kifmm(equivalent_surface_order_parent);

            let mut l2l_level = Vec::new();

            for child in children.iter() {
                let child_downward_check_surface =
                    child.surface_grid(check_surface_order_child, domain, alpha_inner);

                // Note, this way around due to calling convention of kernel, source/targets are 'swapped'
                let mut pe2cc = rlst_dynamic_array2!(
                    Scalar,
                    [
                        n_coeffs_check_surface_child,
                        n_coeffs_equivalent_surface_parent
                    ]
                );
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
