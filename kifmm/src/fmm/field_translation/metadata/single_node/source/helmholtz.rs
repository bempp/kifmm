//! Implementation of traits to compute metadata for field translation operations.

use green_kernels::{
    helmholtz_3d::Helmholtz3dKernel, traits::Kernel as KernelTrait, types::GreenKernelEvalType,
};
use itertools::Itertools;
use rlst::{
    empty_array, rlst_dynamic_array2, MatrixSvd, MultIntoResize, RawAccess, RawAccessMut,
    RlstScalar,
};

use crate::{
    fmm::helpers::single_node::ncoeffs_kifmm,
    fmm::types::PinvMode,
    linalg::pinv::pinv,
    traits::{
        field::{FieldTranslation as FieldTranslationTrait, SourceTranslationMetadata},
        fmm::DataAccess,
        general::single_node::Epsilon,
        tree::{FmmTreeNode, SingleFmmTree, SingleTree},
    },
    tree::{
        constants::{ALPHA_INNER, ALPHA_OUTER},
        types::MortonKey,
    },
    KiFmm,
};

impl<Scalar, FieldTranslation> SourceTranslationMetadata<Scalar, Helmholtz3dKernel<Scalar>>
    for KiFmm<Scalar, Helmholtz3dKernel<Scalar>, FieldTranslation>
where
    Scalar: RlstScalar<Complex = Scalar> + Default + Epsilon + MatrixSvd,
    FieldTranslation: FieldTranslationTrait + Send + Sync,
    <Scalar as RlstScalar>::Real: Default + Epsilon,
    Self: DataAccess,
{
    fn source(&mut self, pinv_mode: PinvMode<Scalar, Helmholtz3dKernel<Scalar>>) {
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

            let n_cols = ncoeffs_kifmm(equivalent_surface_order);
            let n_rows = ncoeffs_kifmm(check_surface_order);

            // Assemble matrix of kernel evaluations between upward check to equivalent, and downward check to equivalent matrices
            // As well as estimating their inverses using SVD
            let mut uc2e = rlst_dynamic_array2!(Scalar, [n_rows, n_cols]);
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

            let n_check_surface_parent = ncoeffs_kifmm(check_surface_order_parent);
            let n_equiv_surface_child = ncoeffs_kifmm(equivalent_surface_order_child);
            let n_equiv_surface_parent = ncoeffs_kifmm(equivalent_surface_order_parent);

            let children = curr.children();
            let mut m2m =
                rlst_dynamic_array2!(Scalar, [n_equiv_surface_parent, 8 * n_equiv_surface_child]);
            let mut m2m_vec = Vec::new();

            for (i, child) in children.iter().enumerate() {
                let child_upward_equivalent_surface =
                    child.surface_grid(equivalent_surface_order_child, domain, alpha_inner);

                let mut ce2pc =
                    rlst_dynamic_array2!(Scalar, [n_check_surface_parent, n_equiv_surface_child]);

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
                let l = i * n_equiv_surface_child * n_equiv_surface_parent;
                let r = l + n_equiv_surface_child * n_equiv_surface_parent;

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
