use green_kernels::{
    laplace_3d::Laplace3dKernel, traits::Kernel as KernelTrait, types::GreenKernelEvalType,
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
                all_gather_v_serialised, calculate_precomputation_load,
                deserialise_nested_array_2x2, serialise_nested_array_2x2,
            },
            single_node::ncoeffs_kifmm,
        },
        types::KiFmmMulti,
    },
    linalg::pinv::pinv,
    traits::{
        field::{FieldTranslation as FieldTranslationTrait, SourceTranslationMetadata},
        fmm::{DataAccessMulti, MetadataAccess},
        general::single_node::Epsilon,
        tree::{FmmTreeNode, MultiFmmTree, MultiTree},
    },
    tree::{
        constants::{ALPHA_INNER, ALPHA_OUTER},
        types::MortonKey,
    },
};

impl<Scalar, FieldTranslation> SourceTranslationMetadata
    for KiFmmMulti<Scalar, Laplace3dKernel<Scalar>, FieldTranslation>
where
    Scalar: RlstScalar + Default + Epsilon + MatrixSvd + Equivalence + Float,
    <Scalar as RlstScalar>::Real: Default + Equivalence + Float,
    FieldTranslation: FieldTranslationTrait + Send + Sync + Default,
    Self: DataAccessMulti + MetadataAccess,
{
    fn source(&mut self) {
        let root = MortonKey::<Scalar::Real>::root();
        let size = self.communicator.size();
        let rank = self.communicator.rank();
        let total_depth = self.tree.source_tree().total_depth();

        // Cast surface parameters
        let alpha_outer = Scalar::from(ALPHA_OUTER).unwrap().re();
        let alpha_inner = Scalar::from(ALPHA_INNER).unwrap().re();
        let domain = self.tree.domain();

        // Parallelise the matrix inversion over MPI ranks

        // Local buffers
        let mut uc2e_inv_1_r = Vec::new();
        let mut uc2e_inv_2_r = Vec::new();

        let iterator = if self.variable_expansion_order {
            self.equivalent_surface_order
                .iter()
                .cloned()
                .zip(self.check_surface_order.iter().cloned())
                .collect_vec()
        } else {
            vec![*self.equivalent_surface_order.last().unwrap(); (total_depth - 1) as usize]
                .iter()
                .cloned()
                .zip(vec![
                    *self.check_surface_order.last().unwrap();
                    (total_depth - 1) as usize
                ])
                .collect_vec()
        };

        // Distribute pseudo-inverse calculation, if have variable expansion order by level
        // Number of pre-computations
        let n_precomputations = if self.variable_expansion_order {
            self.equivalent_surface_order.len() as i32
        } else {
            1
        };

        let (load_counts, load_displacement) =
            calculate_precomputation_load(n_precomputations, size).unwrap();

        // Compute mandated local portion
        let local_load_count = load_counts[rank as usize];
        let local_load_displacement = load_displacement[rank as usize];

        let iterator_r = &iterator[(local_load_displacement as usize)
            ..((local_load_displacement + local_load_count) as usize)];

        for &(equivalent_surface_order, check_surface_order) in iterator_r.iter() {
            // Compute required surfaces
            let upward_equivalent_surface =
                root.surface_grid(equivalent_surface_order, domain, alpha_inner);
            let upward_check_surface = root.surface_grid(check_surface_order, domain, alpha_outer);

            let n_rows = ncoeffs_kifmm(check_surface_order);
            let n_cols = ncoeffs_kifmm(equivalent_surface_order);

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

            uc2e_inv_1_r.push(empty_array::<Scalar, 2>().simple_mult_into_resize(v.r(), mat_s.r()));
            uc2e_inv_2_r.push(ut);
        }

        // Need to gather results at all ranks
        let uc2e_inv_1_r_serialised = serialise_nested_array_2x2(&uc2e_inv_1_r);
        let uc2e_inv_2_r_serialised = serialise_nested_array_2x2(&uc2e_inv_2_r);

        let uc2e_inv_1_serialised =
            all_gather_v_serialised(&uc2e_inv_1_r_serialised, &self.communicator);
        let uc2e_inv_2_serialised =
            all_gather_v_serialised(&uc2e_inv_2_r_serialised, &self.communicator);

        // Reconstruct metadata
        let (mut uc2e_inv_1, mut rest) = deserialise_nested_array_2x2(&uc2e_inv_1_serialised);
        while !rest.is_empty() {
            let (mut t1, t2) = deserialise_nested_array_2x2::<Scalar>(rest);
            uc2e_inv_1.append(&mut t1);
            rest = t2;
        }

        let (mut uc2e_inv_2, mut rest) = deserialise_nested_array_2x2(&uc2e_inv_2_serialised);
        while !rest.is_empty() {
            let (mut t1, t2) = deserialise_nested_array_2x2::<Scalar>(rest);
            uc2e_inv_2.append(&mut t1);
            rest = t2;
        }

        // Calculate all M2M operator matrices
        let level_iterator = if self.variable_expansion_order {
            0..self.tree.source_tree().total_depth()
        } else {
            0..1
        };

        // The M2M operators are cheap to calculate on each node, as they are just matmuls
        let mut m2m = Vec::new();
        let mut m2m_vec = Vec::new();
        let mut m2m_global = Vec::new();
        let mut m2m_global_vec = Vec::new();

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

            let mut m2m_level_1 =
                rlst_dynamic_array2!(Scalar, [n_equiv_surface_parent, 8 * n_equiv_surface_child]);
            let mut m2m_vec_level_1 = Vec::new();

            let mut m2m_level_2 =
                rlst_dynamic_array2!(Scalar, [n_equiv_surface_parent, 8 * n_equiv_surface_child]);
            let mut m2m_vec_level_2 = Vec::new();

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

                let tmp_1 = empty_array::<Scalar, 2>().simple_mult_into_resize(
                    uc2e_inv_1[self.expansion_index(parent_level)].r(),
                    empty_array::<Scalar, 2>().simple_mult_into_resize(
                        uc2e_inv_2[self.expansion_index(parent_level)].r(),
                        ce2pc.r(),
                    ),
                );

                let mut tmp_2 = rlst_dynamic_array2!(Scalar, tmp_1.shape());
                tmp_2.data_mut().copy_from_slice(tmp_1.data());

                let l = i * n_equiv_surface_child * n_equiv_surface_parent;
                let r = l + n_equiv_surface_child * n_equiv_surface_parent;

                m2m_level_1.data_mut()[l..r].copy_from_slice(tmp_1.data());
                m2m_vec_level_1.push(tmp_1);

                m2m_level_2.data_mut()[l..r].copy_from_slice(tmp_2.data());
                m2m_vec_level_2.push(tmp_2);
            }

            m2m_vec.push(m2m_vec_level_1);
            m2m.push(m2m_level_1);
            m2m_global_vec.push(m2m_vec_level_2);
            m2m_global.push(m2m_level_2);
        }

        self.global_fmm.source = m2m_global;
        self.global_fmm.source_vec = m2m_global_vec;
        self.source = m2m;
        self.source_vec = m2m_vec;
        self.uc2e_inv_1 = uc2e_inv_1;
        self.uc2e_inv_2 = uc2e_inv_2;
    }
}
