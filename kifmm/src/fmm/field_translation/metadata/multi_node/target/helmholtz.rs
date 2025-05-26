use green_kernels::{
    helmholtz_3d::Helmholtz3dKernel, traits::Kernel as KernelTrait, types::GreenKernelEvalType,
};
use itertools::Itertools;
use mpi::traits::{Communicator, Equivalence};
use num::Float;
use rlst::{
    empty_array, rlst_dynamic_array2, MatrixSvd, MultIntoResize, RawAccess, RawAccessMut,
    RlstScalar, Shape,
};

use crate::{
    fmm::{
        helpers::{
            multi_node::{
                all_gather_v_serialised, calculate_precomputation_load, deserialise_nested_array,
                serialise_nested_array,
            },
            single_node::ncoeffs_kifmm,
        },
        types::KiFmmMulti,
    },
    linalg::pinv::pinv,
    traits::{
        field::{FieldTranslation as FieldTranslationTrait, TargetTranslationMetadata},
        fmm::{DataAccessMulti, MetadataAccess},
        general::single_node::Epsilon,
        tree::{FmmTreeNode, MultiFmmTree, MultiTree},
    },
    tree::{
        constants::{ALPHA_INNER, ALPHA_OUTER},
        types::MortonKey,
    },
};

impl<Scalar, FieldTranslation> TargetTranslationMetadata
    for KiFmmMulti<Scalar, Helmholtz3dKernel<Scalar>, FieldTranslation>
where
    Scalar: RlstScalar<Complex = Scalar> + Default + Epsilon + MatrixSvd + Equivalence,
    <Scalar as RlstScalar>::Real: Default + Sync + Send + Default + Equivalence + Float,
    FieldTranslation: FieldTranslationTrait + Send + Sync + Default,
    Self: DataAccessMulti + MetadataAccess,
{
    fn target(&mut self) {
        let root = MortonKey::<Scalar::Real>::root();
        let size = self.communicator.size();
        let rank = self.communicator.rank();
        let total_depth = self.tree.source_tree().total_depth();

        // Cast surface parameters
        let alpha_outer = Scalar::from(ALPHA_OUTER).unwrap().re();
        let alpha_inner = Scalar::from(ALPHA_INNER).unwrap().re();
        let domain = self.tree.domain();

        // Parallelise matrix inversion by level
        let mut dc2e_inv_1_r = Vec::new();
        let mut dc2e_inv_2_r = Vec::new();

        // Calculate inverse upward check to equivalent matrices on each level
        let iterator = if self.equivalent_surface_order.len() > 1 {
            (0..=total_depth)
                .zip(self.equivalent_surface_order.iter().cloned())
                .zip(self.check_surface_order.iter().cloned())
                .collect_vec()
        } else {
            let eq_value = *self.equivalent_surface_order.last().unwrap();
            let ch_value = *self.check_surface_order.last().unwrap();
            let eq_vec = vec![eq_value; (total_depth + 1) as usize];
            let ch_vec = vec![ch_value; (total_depth + 1) as usize];

            (0..=total_depth).zip(eq_vec).zip(ch_vec).collect_vec()
        };

        // Distribute pseudo-inverse calculation by level
        // Number of pre-computations
        let n_precomputations = iterator.len() as i32;

        let (load_counts, load_displacement) =
            calculate_precomputation_load(n_precomputations, size).unwrap();

        // Compute mandated local portion
        let local_load_count = load_counts[rank as usize];
        let local_load_displacement = load_displacement[rank as usize];

        let iterator_r = &iterator[(local_load_displacement as usize)
            ..((local_load_displacement + local_load_count) as usize)];

        for &((level, equivalent_surface_order), check_surface_order) in iterator_r.iter() {
            let curr = root.first_child_at_level(level).unwrap();

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

            dc2e_inv_1_r.push(empty_array::<Scalar, 2>().simple_mult_into_resize(v.r(), mat_s.r()));
            dc2e_inv_2_r.push(ut);
        }

        // Need to gather results at all ranks
        let dc2e_inv_1_r_serialised = serialise_nested_array(&dc2e_inv_1_r);
        let dc2e_inv_2_r_serialised = serialise_nested_array(&dc2e_inv_2_r);

        let dc2e_inv_1_serialised =
            all_gather_v_serialised(&dc2e_inv_1_r_serialised, &self.communicator);
        let dc2e_inv_2_serialised =
            all_gather_v_serialised(&dc2e_inv_2_r_serialised, &self.communicator);

        // Reconstruct metadata
        let (mut dc2e_inv_1, mut rest) = deserialise_nested_array(&dc2e_inv_1_serialised);
        while !rest.is_empty() {
            let (mut t1, t2) = deserialise_nested_array::<Scalar>(rest);
            dc2e_inv_1.append(&mut t1);
            rest = t2;
        }

        let (mut dc2e_inv_2, mut rest) = deserialise_nested_array(&dc2e_inv_2_serialised);
        while !rest.is_empty() {
            let (mut t1, t2) = deserialise_nested_array::<Scalar>(rest);
            dc2e_inv_2.append(&mut t1);
            rest = t2;
        }

        let mut target_vec = Vec::new();
        let mut target_vec_global = Vec::new();

        let iterator = if self.equivalent_surface_order.len() > 1 {
            (0..total_depth)
                .zip(
                    self.equivalent_surface_order
                        .iter()
                        .cloned()
                        .take(total_depth as usize)
                        .zip(
                            self.check_surface_order
                                .iter()
                                .skip(1)
                                .cloned()
                                .take(total_depth as usize),
                        ),
                )
                .collect_vec()
        } else {
            (0..total_depth)
                .zip(
                    vec![*self.equivalent_surface_order.last().unwrap(); total_depth as usize]
                        .into_iter()
                        .zip(vec![
                            *self.check_surface_order.last().unwrap();
                            total_depth as usize
                        ]),
                )
                .collect_vec()
        };

        for (level, (equivalent_surface_order_parent, check_surface_order_child)) in iterator {
            // Compute required surfaces
            let curr = root.first_child_at_level(level).unwrap();

            let parent_downward_equivalent_surface =
                curr.surface_grid(equivalent_surface_order_parent, domain, alpha_outer);

            let ncheck_surface_child = ncoeffs_kifmm(check_surface_order_child);
            let n_equiv_surface_parent = ncoeffs_kifmm(equivalent_surface_order_parent);

            // Calculate l2l operator matrices on each level
            let children = curr.children();
            let mut l2l = Vec::new();
            let mut l2l_2 = Vec::new();

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

                let tmp_1 = empty_array::<Scalar, 2>().simple_mult_into_resize(
                    dc2e_inv_1[(level + 1) as usize].r(),
                    empty_array::<Scalar, 2>()
                        .simple_mult_into_resize(dc2e_inv_2[(level + 1) as usize].r(), pe2cc.r()),
                );

                let mut tmp_2 = rlst_dynamic_array2!(Scalar, tmp_1.shape());
                tmp_2.data_mut().copy_from_slice(tmp_1.data());

                l2l.push(tmp_1);
                l2l_2.push(tmp_2);
            }

            target_vec.push(l2l);
            target_vec_global.push(l2l_2);
        }

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

        self.global_fmm.target_vec = target_vec_global;
        self.global_fmm.dc2e_inv_1 = dc2e_inv_1_global;
        self.global_fmm.dc2e_inv_2 = dc2e_inv_2_global;

        self.target_vec = target_vec;
        self.dc2e_inv_1 = dc2e_inv_1;
        self.dc2e_inv_2 = dc2e_inv_2;
    }
}
