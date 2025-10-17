//! Implementation of traits to compute metadata for field translation operations.

use green_kernels::{
    laplace_3d::Laplace3dKernel, traits::Kernel as KernelTrait, types::GreenKernelEvalType,
};
use rlst::{
    empty_array, rlst_dynamic_array2, MatrixQr, MatrixSvd, MultIntoResize, RawAccess, RawAccessMut,
    RlstScalar,
};

use crate::{
    fmm::{helpers::single_node::ncoeffs_kifmm, types::PinvMode},
    linalg::pinv::{pinv, pinv_aca_plus},
    traits::{
        field::{FieldTranslation as FieldTranslationTrait, SourceTranslationMetadata},
        fmm::{DataAccess, MetadataAccess},
        general::single_node::{ArgmaxValue, Cast, Epsilon, Upcast},
        tree::{FmmTreeNode, SingleFmmTree, SingleTree},
    },
    tree::{
        constants::{ALPHA_INNER, ALPHA_OUTER},
        types::MortonKey,
    },
    KiFmm,
};

impl<Scalar, FieldTranslation> SourceTranslationMetadata<Scalar>
    for KiFmm<Scalar, Laplace3dKernel<Scalar>, FieldTranslation>
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
    FieldTranslation: FieldTranslationTrait + Send + Sync,
    Self: DataAccess,
{
    fn source(&mut self, pinv_mode: PinvMode<Scalar>) {
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

            let n_equiv_surface = ncoeffs_kifmm(equivalent_surface_order);
            let n_check_surface = ncoeffs_kifmm(check_surface_order);

            // Compute pseudo-inverse
            let s;
            let ut;
            let v;
            match pinv_mode {
                PinvMode::Svd { atol, rtol } => {
                    let mut uc2e = rlst_dynamic_array2!(Scalar, [n_check_surface, n_equiv_surface]);
                    self.kernel.assemble_st(
                        GreenKernelEvalType::Value,
                        &upward_check_surface[..],
                        &upward_equivalent_surface[..],
                        uc2e.data_mut(),
                    );
                    (s, ut, v) = pinv(&uc2e, atol, rtol).unwrap();
                }

                PinvMode::AcaPlus {
                    eps,
                    max_iter,
                    local_radius,
                    multithreaded,
                } => {
                    (s, ut, v) = pinv_aca_plus(
                        &upward_equivalent_surface,
                        &upward_check_surface,
                        self.kernel.clone(),
                        eps,
                        max_iter,
                        local_radius,
                        true,
                        multithreaded,
                        true,
                    )
                    .unwrap();
                }
            }

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

#[cfg(test)]
mod test {
    use rand::{thread_rng, Rng};
    use rlst::Shape;

    use super::*;
    use crate::tree::Domain;

    #[test]
    fn test_m2m() {
        let kernel = Laplace3dKernel::<f64>::new();
        let root = MortonKey::<f64>::root();

        // Cast surface parameters
        let alpha_outer = ALPHA_OUTER;
        let alpha_inner = ALPHA_INNER;
        let domain = Domain::new(&[0., 0., 0.], &[1., 1., 1.]);

        let equivalent_surface_order = 10;
        let check_surface_order = equivalent_surface_order;

        // let pinv_mode = PinvMode::<f64>::svd(None, None);
        let pinv_mode = PinvMode::<f64>::aca(Some(1e-6), None, None, true);

        // Compute required surfaces
        let upward_equivalent_surface =
            root.surface_grid(equivalent_surface_order, &domain, alpha_inner);
        let upward_check_surface = root.surface_grid(check_surface_order, &domain, alpha_outer);

        let n_equiv_surface = ncoeffs_kifmm(equivalent_surface_order);
        let n_check_surface = ncoeffs_kifmm(check_surface_order);

        // Compute pseudo-inverse
        let s;
        let ut;
        let v;

        match pinv_mode {
            PinvMode::Svd { atol, rtol } => {
                let mut uc2e = rlst_dynamic_array2!(f64, [n_check_surface, n_equiv_surface]);
                kernel.assemble_st(
                    GreenKernelEvalType::Value,
                    &upward_check_surface[..],
                    &upward_equivalent_surface[..],
                    uc2e.data_mut(),
                );
                (s, ut, v) = pinv(&uc2e, atol, rtol).unwrap();
            }

            PinvMode::AcaPlus {
                eps,
                max_iter,
                local_radius,
                multithreaded,
            } => {
                (s, ut, v) = pinv_aca_plus(
                    &upward_equivalent_surface,
                    &upward_check_surface,
                    kernel.clone(),
                    eps,
                    max_iter,
                    local_radius,
                    true,
                    multithreaded,
                    true,
                )
                .unwrap();
            }
        }

        let mut mat_s = rlst_dynamic_array2!(f64, [s.len(), s.len()]);
        for i in 0..s.len() {
            mat_s[[i, i]] = s[i];
        }

        let uc2e_inv_1 = empty_array::<f64, 2>().simple_mult_into_resize(v.r(), mat_s.r());
        let uc2e_inv_2 = ut;

        let check_surface_order_parent = check_surface_order;
        let equivalent_surface_order_parent = equivalent_surface_order;
        let equivalent_surface_order_child = equivalent_surface_order;

        let parent_upward_check_surface =
            root.surface_grid(check_surface_order_parent, &domain, alpha_outer);
        let parent_upward_equivalent_surface =
            root.surface_grid(equivalent_surface_order_parent, &domain, alpha_inner);

        let children = root.children();
        let n_check_surface_parent = ncoeffs_kifmm(check_surface_order_parent);
        let n_equiv_surface_child = ncoeffs_kifmm(equivalent_surface_order_child);

        let mut m2m_vec = Vec::new();

        let mut child_equivalent_surfaces = Vec::new();
        let mut ce2pc_vec = Vec::new();

        for (_i, child) in children.iter().enumerate().take(1) {
            let child_upward_equivalent_surface =
                child.surface_grid(equivalent_surface_order_child, &domain, alpha_inner);
            child_equivalent_surfaces.push(child_upward_equivalent_surface.clone());

            let mut ce2pc =
                rlst_dynamic_array2!(f64, [n_check_surface_parent, n_equiv_surface_child]);

            // Note, this way around due to calling convention of kernel, source/targets are 'swapped'
            kernel.assemble_st(
                GreenKernelEvalType::Value,
                &parent_upward_check_surface,
                &child_upward_equivalent_surface,
                ce2pc.data_mut(),
            );

            let tmp = empty_array::<f64, 2>().simple_mult_into_resize(
                uc2e_inv_1.r(),
                empty_array::<f64, 2>().simple_mult_into_resize(uc2e_inv_2.r(), ce2pc.r()),
            );

            ce2pc_vec.push(ce2pc);
            m2m_vec.push(tmp);
        }

        // Calculate truth at far field point
        let mut rng = thread_rng();
        let mut x = rlst_dynamic_array2![f64, [n_equiv_surface, 1]]; // random column vector, i.e. multipoles on child check surface
        x.data_mut().iter_mut().for_each(|x| *x = rng.gen());

        let far_field = vec![100., 0., 0.];
        let mut truth = vec![0.];
        kernel.evaluate_st(
            GreenKernelEvalType::Value,
            &child_equivalent_surfaces[0],
            &far_field,
            x.data(),
            &mut truth,
        );

        // Calculate the result from translated field
        let b = empty_array::<f64, 2>().simple_mult_into_resize(
            uc2e_inv_1.r(),
            empty_array::<f64, 2>().simple_mult_into_resize(
                uc2e_inv_2.r(),
                empty_array::<f64, 2>().simple_mult_into_resize(ce2pc_vec[0].r(), x.r()),
            ),
        );

        let mut found = vec![0.];
        kernel.evaluate_st(
            GreenKernelEvalType::Value,
            &parent_upward_equivalent_surface,
            &far_field,
            b.data(),
            &mut found,
        );

        println!("Truth = {:?}", truth);
        println!("Found = {:?} {:?}", found, b.shape());
        println!("Err = {:?}", (truth[0] - found[0]).abs());

        assert!((truth[0] - found[0]).abs() < 1e-6);
    }
}
