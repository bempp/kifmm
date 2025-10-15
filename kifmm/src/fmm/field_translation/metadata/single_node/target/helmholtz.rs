//! Implementation of traits to compute metadata for field translation operations.

use green_kernels::{
    helmholtz_3d::Helmholtz3dKernel, traits::Kernel as KernelTrait, types::GreenKernelEvalType,
};
use itertools::Itertools;
use rlst::{empty_array, rlst_dynamic_array2, MatrixSvd, MultIntoResize, RawAccessMut, RlstScalar};

use crate::{
    fmm::helpers::single_node::ncoeffs_kifmm,
    linalg::pinv::pinv,
    traits::{
        field::{FieldTranslation as FieldTranslationTrait, TargetTranslationMetadata},
        general::single_node::Epsilon,
        tree::{FmmTreeNode, SingleFmmTree, SingleTree},
    },
    tree::{
        constants::{ALPHA_INNER, ALPHA_OUTER},
        types::MortonKey,
    },
    Evaluate, KiFmm,
};

impl<Scalar, FieldTranslation> TargetTranslationMetadata
    for KiFmm<Scalar, Helmholtz3dKernel<Scalar>, FieldTranslation>
where
    Scalar: RlstScalar<Complex = Scalar> + Default + Epsilon + MatrixSvd,
    FieldTranslation: FieldTranslationTrait + Send + Sync,
    <Scalar as RlstScalar>::Real: Default + Epsilon,
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

            let n_cols = ncoeffs_kifmm(equivalent_surface_order);
            let n_rows = ncoeffs_kifmm(check_surface_order);

            // Assemble matrix of kernel evaluations between upward check to equivalent, and downward check to equivalent matrices
            // As well as estimating their inverses using SVD
            let mut dc2e = rlst_dynamic_array2!(Scalar, [n_rows, n_cols]);
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
            let n_equiv_surface_parent = ncoeffs_kifmm(equivalent_surface_order_parent);

            // Calculate l2l operator matrices on each level
            let children = curr.children();
            let mut l2l = Vec::new();

            for child in children.iter() {
                let child_downward_check_surface =
                    child.surface_grid(check_surface_order_child, domain, alpha_inner);

                let mut pe2cc =
                    rlst_dynamic_array2!(Scalar, [ncheck_surface_child, n_equiv_surface_parent]);
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
