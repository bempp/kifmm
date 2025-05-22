use bytemuck::cast_slice;
use bytemuck::Pod;
use green_kernels::laplace_3d::Laplace3dKernel;
use green_kernels::traits::Kernel as KernelTrait;
use green_kernels::types::GreenKernelEvalType;
use itertools::izip;
use itertools::Itertools;
use mpi::datatype::Partition;
use mpi::datatype::PartitionMut;
use mpi::datatype::Partitioned;
use mpi::topology::SimpleCommunicator;
use mpi::traits::{Communicator, CommunicatorCollectives, Equivalence};
use mpi::{Count, Rank};
use num::{Float, Zero};
use pulp::Scalar;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use rlst::{
    empty_array, rlst_array_from_slice2, rlst_dynamic_array2, rlst_dynamic_array3, Array,
    BaseArray, MatrixSvd, MultIntoResize, RawAccess, RawAccessMut, RlstScalar, Shape, SvdMode,
    UnsafeRandomAccessByRef, UnsafeRandomAccessMut, VectorContainer,
};
use std::collections::{HashMap, HashSet};
use std::sync::{Mutex, RwLock};

use crate::fmm::field_translation::source_to_target::transfer_vector::compute_transfer_vectors_at_level;
use crate::fmm::helpers::multi_node::all_gather_v_serialised;
use crate::fmm::helpers::multi_node::deserialise_nested_vec;
use crate::fmm::helpers::multi_node::deserialise_vec;
use crate::fmm::helpers::multi_node::deserialise_vec_blas_metadata_sarcmp;
use crate::fmm::helpers::multi_node::deserialise_vec_fft_metadata;
use crate::fmm::helpers::multi_node::serialise_blas_metadata_sarcmp;
use crate::fmm::helpers::multi_node::serialise_nested_vec;
use crate::fmm::helpers::multi_node::serialise_vec;
use crate::fmm::helpers::multi_node::serialise_vec_blas_metadata_sarcmp;
use crate::fmm::helpers::multi_node::serialise_vec_fft_metadata;
use crate::fmm::helpers::multi_node::{
    coordinate_index_pointer_multi_node, leaf_expansion_pointers_multi_node,
    leaf_surfaces_multi_node, level_expansion_pointers_multi_node, level_index_pointer_multi_node,
    potential_pointers_multi_node,
};
use crate::fmm::helpers::single_node::ncoeffs_kifmm;
use crate::fmm::helpers::single_node::{flip3, homogenous_kernel_scale, optionally_time};
use crate::fmm::types::{BlasMetadataSaRcmp, FftMetadata};
use crate::linalg::pinv::pinv;
use crate::traits::fftw::{Dft, DftType};
use crate::traits::fmm::{DataAccess, DataAccessMulti, Metadata, MetadataAccess};
use crate::traits::general::{
    multi_node::GhostExchange,
    single_node::{AsComplex, Epsilon},
};
use crate::traits::tree::{Domain, FmmTreeNode, MultiFmmTree, MultiTree};
use crate::traits::types::{CommunicationTime, CommunicationType};
use crate::tree::constants::{NHALO, NSIBLINGS, NSIBLINGS_SQUARED, NTRANSFER_VECTORS_KIFMM};
use crate::tree::helpers::find_corners;
use crate::tree::types::MortonKey;
use crate::KiFmm;
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

/// Calculate load for precomputation based on a block distribution strategy
pub(crate) fn calculate_precomputation_load(
    n_precomputations: i32,
    size: i32,
) -> Option<(Vec<Count>, Vec<Count>)> {
    if n_precomputations > 0 {
        let mut counts;

        if n_precomputations > 1 {
            // Distributed pre-computation
            let q = n_precomputations / size; // Base number of calculations per processor
            let r = n_precomputations % size; // Extra calculations to distribute evenly among ranks

            // Block distribution strategy
            counts = (0..size)
                .map(|i| if i < r { q + 1 } else { q })
                .collect_vec();
        } else {
            // If only have one pre-computation, carry out on a single rank (root rank)
            counts = vec![0; size as usize];
            counts[0] = n_precomputations;
        }

        let mut curr = 0;
        let mut displacements = Vec::new();
        for &count in counts.iter() {
            displacements.push(curr);
            curr += count;
        }

        Some((counts, displacements))
    } else {
        None
    }
}

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

        // Cast surface parameters
        let alpha_outer = Scalar::from(ALPHA_OUTER).unwrap().re();
        let alpha_inner = Scalar::from(ALPHA_INNER).unwrap().re();
        let domain = self.tree.domain();

        // Local buffers
        let mut local_uc2e_inv_1 = Vec::new();
        let mut local_uc2e_inv_2 = Vec::new();

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
        let equivalent_surface_order = &self.equivalent_surface_order[(local_load_displacement
            as usize)
            ..((local_load_displacement + local_load_count) as usize)];
        let check_surface_order = &self.check_surface_order[(local_load_displacement as usize)
            ..((local_load_displacement + local_load_count) as usize)];
        let mut local_shared_dim = Vec::new();

        for (&equivalent_surface_order, &check_surface_order) in equivalent_surface_order
            .iter()
            .zip(check_surface_order.iter())
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
            local_shared_dim.push(s.len());

            let mut mat_s = rlst_dynamic_array2!(Scalar, [s.len(), s.len()]);
            for i in 0..s.len() {
                mat_s[[i, i]] = Scalar::from_real(s[i]);
            }

            local_uc2e_inv_1
                .push(empty_array::<Scalar, 2>().simple_mult_into_resize(v.r(), mat_s.r()));
            local_uc2e_inv_2.push(ut);
        }

        // Need to gather results at all ranks

        // Gather shared dimension of uc2e_inv_1 and uc2e_inv_2
        let mut shared_dim = vec![0usize; n_precomputations as usize];
        {
            let mut partition =
                PartitionMut::new(&mut shared_dim, &load_counts[..], &load_displacement[..]);
            self.communicator
                .all_gather_varcount_into(&local_shared_dim[..], &mut partition);
        }

        // If only a single precomputation have to update shared dim to match size of equivalent/check surface order vecs for buffer calcs
        if n_precomputations == 1 {
            shared_dim = vec![shared_dim[0]; self.equivalent_surface_order.len()]
        }

        // Compute message sizes of local buffers
        let mut local_msg_size_uc2e_inv_1 = 0;
        let mut local_msg_size_uc2e_inv_2 = 0;

        for (mat_1, mat_2) in izip!(local_uc2e_inv_1.iter(), local_uc2e_inv_2.iter()) {
            local_msg_size_uc2e_inv_1 += mat_1.shape()[0] * mat_1.shape()[1];
            local_msg_size_uc2e_inv_2 += mat_2.shape()[0] * mat_2.shape()[1];
        }

        // Flatten buffers for sending
        let mut local_msg_uc2e_inv_1 = vec![Scalar::zero(); local_msg_size_uc2e_inv_1];
        let mut local_msg_uc2e_inv_2 = vec![Scalar::zero(); local_msg_size_uc2e_inv_2];

        let mut curr1 = 0;
        let mut curr2 = 0;
        for (mat_1, mat_2) in izip!(local_uc2e_inv_1.iter(), local_uc2e_inv_2.iter()) {
            let mat_1_size = mat_1.shape()[0] * mat_1.shape()[1];
            local_msg_uc2e_inv_1[curr1..(curr1 + mat_1_size)].copy_from_slice(mat_1.data());
            curr1 += mat_1_size;

            let mat_2_size = mat_1.shape()[0] * mat_1.shape()[1];
            local_msg_uc2e_inv_2[curr2..(curr2 + mat_2_size)].copy_from_slice(mat_2.data());
            curr2 += mat_1_size;
        }

        // Setup buffers to receive data
        let mut msg_size_uc2e_inv_1 = 0;
        let mut msg_size_uc2e_inv_2 = 0;
        let mut msg_counts_uc2e_inv_1 = Vec::new();
        let mut msg_counts_uc2e_inv_2 = Vec::new();

        for (_rank, &load_r, &load_displacement_r) in
            izip!((0..size), load_counts.iter(), load_displacement.iter())
        {
            let equivalent_surface_order = &self.equivalent_surface_order
                [(load_displacement_r as usize)..((load_displacement_r + load_r) as usize)];
            let check_surface_order = &self.check_surface_order
                [(load_displacement_r as usize)..((load_displacement_r + load_r) as usize)];
            let shared_dim = &shared_dim
                [(load_displacement_r as usize)..((load_displacement_r + load_r) as usize)];

            let mut count_uc2e_inv_1 = 0;
            let mut count_uc2e_inv_2 = 0;

            for (&e_order, &c_order, &shared_dim) in
                izip!(equivalent_surface_order, check_surface_order, shared_dim)
            {
                count_uc2e_inv_1 += ncoeffs_kifmm(c_order) * shared_dim;
                count_uc2e_inv_2 += ncoeffs_kifmm(e_order) * shared_dim;
            }

            msg_counts_uc2e_inv_1.push(count_uc2e_inv_1 as Count);
            msg_counts_uc2e_inv_2.push(count_uc2e_inv_2 as Count);

            msg_size_uc2e_inv_1 += count_uc2e_inv_1;
            msg_size_uc2e_inv_2 += count_uc2e_inv_2;
        }

        let msg_displs_uc2e_inv_1 = msg_counts_uc2e_inv_1
            .iter()
            .scan(0, |acc, &x| {
                let tmp = *acc;
                *acc += x;
                Some(tmp)
            })
            .collect_vec();

        let msg_displs_uc2e_inv_2 = msg_counts_uc2e_inv_2
            .iter()
            .scan(0, |acc, &x| {
                let tmp = *acc;
                *acc += x;
                Some(tmp)
            })
            .collect_vec();

        let mut buf_uc2e_inv_1 = vec![Scalar::zero(); msg_size_uc2e_inv_1];
        let mut buf_uc2e_inv_2 = vec![Scalar::zero(); msg_size_uc2e_inv_2];

        // Communicate pre-computations
        {
            let mut partition = PartitionMut::new(
                &mut buf_uc2e_inv_1,
                &msg_counts_uc2e_inv_1[..],
                &msg_displs_uc2e_inv_1[..],
            );
            self.communicator
                .all_gather_varcount_into(&local_msg_uc2e_inv_1, &mut partition);

            let mut partition = PartitionMut::new(
                &mut buf_uc2e_inv_2,
                &msg_counts_uc2e_inv_2[..],
                &msg_displs_uc2e_inv_2[..],
            );
            self.communicator
                .all_gather_varcount_into(&local_msg_uc2e_inv_2, &mut partition);
        }

        // Re-structure messages using shared dimensions
        let mut uc2e_inv_1_vec = Vec::new();
        let mut uc2e_inv_2_vec = Vec::new();

        if n_precomputations > 1 {
            let mut curr1 = 0;
            let mut curr2 = 0;
            for (&e_order, &c_order, &shared_dim) in izip!(
                &self.equivalent_surface_order,
                &self.check_surface_order,
                shared_dim.iter()
            ) {
                let l = curr1;
                let r = curr1 + ncoeffs_kifmm(c_order) * shared_dim;
                let mut uc2e_inv_1 =
                    rlst_dynamic_array2!(Scalar, [ncoeffs_kifmm(c_order), shared_dim]);
                uc2e_inv_1.data_mut().copy_from_slice(&buf_uc2e_inv_1[l..r]);
                curr1 = r;

                let l = curr2;
                let r = curr2 + ncoeffs_kifmm(e_order) * shared_dim;
                let mut uc2e_inv_2 =
                    rlst_dynamic_array2!(Scalar, [shared_dim, ncoeffs_kifmm(e_order)]);
                uc2e_inv_2.data_mut().copy_from_slice(&buf_uc2e_inv_2[l..r]);
                curr2 = r;

                uc2e_inv_1_vec.push(uc2e_inv_1);
                uc2e_inv_2_vec.push(uc2e_inv_2);
            }
        } else {
            let c_order = self.check_surface_order[0];
            let e_order = self.equivalent_surface_order[0];
            let shared_dim = shared_dim[0];
            let l = 0;
            let r = ncoeffs_kifmm(c_order) * shared_dim;
            let mut uc2e_inv_1 = rlst_dynamic_array2!(Scalar, [ncoeffs_kifmm(c_order), shared_dim]);
            uc2e_inv_1.data_mut().copy_from_slice(&buf_uc2e_inv_1[l..r]);

            let l = 0;
            let r = ncoeffs_kifmm(e_order) * shared_dim;
            let mut uc2e_inv_2 = rlst_dynamic_array2!(Scalar, [shared_dim, ncoeffs_kifmm(e_order)]);
            uc2e_inv_2.data_mut().copy_from_slice(&buf_uc2e_inv_2[l..r]);

            uc2e_inv_1_vec.push(uc2e_inv_1);
            uc2e_inv_2_vec.push(uc2e_inv_2);
        }

        // Calculate all M2M operator matrices
        let level_iterator = if self.variable_expansion_order {
            0..self.tree.source_tree().total_depth()
        } else {
            0..1
        };

        let mut m2m = Vec::new();
        let mut m2m_vec = Vec::new();
        let mut m2m_global = Vec::new();
        let mut m2m_global_vec = Vec::new();

        // Calculate M2M operator matrices on each level, if required
        for parent_level in level_iterator {
            let check_surface_order_parent = self.check_surface_order(parent_level as u64);
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
                    uc2e_inv_1_vec[self.expansion_index(parent_level as u64)].r(),
                    empty_array::<Scalar, 2>().simple_mult_into_resize(
                        uc2e_inv_2_vec[self.expansion_index(parent_level as u64)].r(),
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
        self.uc2e_inv_1 = uc2e_inv_1_vec;
        self.uc2e_inv_2 = uc2e_inv_2_vec;
    }
}

impl<Scalar, FieldTranslation> TargetTranslationMetadata
    for KiFmmMulti<Scalar, Laplace3dKernel<Scalar>, FieldTranslation>
where
    Scalar: RlstScalar + Default + Epsilon + MatrixSvd + Equivalence + Float,
    <Scalar as RlstScalar>::Real: Default + Equivalence + Float,
    FieldTranslation: FieldTranslationTrait + Send + Sync,
    Self: MetadataAccess,
{
    fn target(&mut self) {
        let root = MortonKey::<Scalar::Real>::root();
        let size = self.communicator.size();
        let rank = self.communicator.rank();

        // Cast surface parameters
        let alpha_outer = Scalar::from(ALPHA_OUTER).unwrap().re();
        let alpha_inner = Scalar::from(ALPHA_INNER).unwrap().re();
        let domain = self.tree.domain();

        // Local buffers
        let mut local_dc2e_inv_1 = Vec::new();
        let mut local_dc2e_inv_2 = Vec::new();

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
        let equivalent_surface_order = &self.equivalent_surface_order[(local_load_displacement
            as usize)
            ..((local_load_displacement + local_load_count) as usize)];
        let check_surface_order = &self.check_surface_order[(local_load_displacement as usize)
            ..((local_load_displacement + local_load_count) as usize)];
        let mut local_shared_dim = Vec::new();

        for (&equivalent_surface_order, &check_surface_order) in equivalent_surface_order
            .iter()
            .zip(check_surface_order.iter())
        {
            // Compute required surfaces
            let downward_equivalent_surface =
                root.surface_grid(equivalent_surface_order, domain, alpha_outer);
            let downward_check_surface =
                root.surface_grid(check_surface_order, domain, alpha_inner);

            let n_coeffs_equivalent_surface = ncoeffs_kifmm(equivalent_surface_order);
            let n_coeffs_check_surface = ncoeffs_kifmm(check_surface_order);

            // Assemble matrix of kernel evaluations between upward check to equivalent, and downward check to equivalent matrices
            // As well as estimating their inverses using SVD
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

            let (s, ut, v) = pinv(&dc2e, None, None).unwrap();
            local_shared_dim.push(s.len());

            let mut mat_s = rlst_dynamic_array2!(Scalar, [s.len(), s.len()]);
            for i in 0..s.len() {
                mat_s[[i, i]] = Scalar::from_real(s[i]);
            }

            local_dc2e_inv_1
                .push(empty_array::<Scalar, 2>().simple_mult_into_resize(v.r(), mat_s.r()));
            local_dc2e_inv_2.push(ut);
        }

        // Need to gather results at all ranks

        // Gather shared dimension of dc2e_inv_1 and dc2e_inv_2
        let mut shared_dim = vec![0usize; n_precomputations as usize];
        {
            let mut partition =
                PartitionMut::new(&mut shared_dim, &load_counts[..], &load_displacement[..]);
            self.communicator
                .all_gather_varcount_into(&local_shared_dim[..], &mut partition);
        }

        // If only a single precomputation have to update shared dim to match size of equivalent/check surface order vecs for buffer calcs
        if n_precomputations == 1 {
            shared_dim = vec![shared_dim[0]; self.equivalent_surface_order.len()]
        }

        // Compute message sizes of local buffers
        let mut local_msg_size_dc2e_inv_1 = 0;
        let mut local_msg_size_dc2e_inv_2 = 0;

        for (mat_1, mat_2) in izip!(local_dc2e_inv_1.iter(), local_dc2e_inv_2.iter()) {
            local_msg_size_dc2e_inv_1 += mat_1.shape()[0] * mat_1.shape()[1];
            local_msg_size_dc2e_inv_2 += mat_2.shape()[0] * mat_2.shape()[1];
        }

        // Flatten buffers for sending
        let mut local_msg_dc2e_inv_1 = vec![Scalar::zero(); local_msg_size_dc2e_inv_1];
        let mut local_msg_dc2e_inv_2 = vec![Scalar::zero(); local_msg_size_dc2e_inv_2];

        let mut curr1 = 0;
        let mut curr2 = 0;
        for (mat_1, mat_2) in izip!(local_dc2e_inv_1.iter(), local_dc2e_inv_2.iter()) {
            let mat_1_size = mat_1.shape()[0] * mat_1.shape()[1];
            local_msg_dc2e_inv_1[curr1..(curr1 + mat_1_size)].copy_from_slice(mat_1.data());
            curr1 += mat_1_size;

            let mat_2_size = mat_1.shape()[0] * mat_1.shape()[1];
            local_msg_dc2e_inv_2[curr2..(curr2 + mat_2_size)].copy_from_slice(mat_2.data());
            curr2 += mat_1_size;
        }

        // Setup buffers to receive data
        let mut msg_size_dc2e_inv_1 = 0;
        let mut msg_size_dc2e_inv_2 = 0;
        let mut msg_counts_dc2e_inv_1 = Vec::new();
        let mut msg_counts_dc2e_inv_2 = Vec::new();

        for (_rank, &load_r, &load_displacement_r) in
            izip!((0..size), load_counts.iter(), load_displacement.iter())
        {
            let equivalent_surface_order = &self.equivalent_surface_order
                [(load_displacement_r as usize)..((load_displacement_r + load_r) as usize)];
            let check_surface_order = &self.check_surface_order
                [(load_displacement_r as usize)..((load_displacement_r + load_r) as usize)];
            let shared_dim = &shared_dim
                [(load_displacement_r as usize)..((load_displacement_r + load_r) as usize)];

            let mut count_dc2e_inv_1 = 0;
            let mut count_dc2e_inv_2 = 0;

            for (&e_order, &c_order, &shared_dim) in
                izip!(equivalent_surface_order, check_surface_order, shared_dim)
            {
                count_dc2e_inv_1 += ncoeffs_kifmm(c_order) * shared_dim;
                count_dc2e_inv_2 += ncoeffs_kifmm(e_order) * shared_dim;
            }

            msg_counts_dc2e_inv_1.push(count_dc2e_inv_1 as Count);
            msg_counts_dc2e_inv_2.push(count_dc2e_inv_2 as Count);

            msg_size_dc2e_inv_1 += count_dc2e_inv_1;
            msg_size_dc2e_inv_2 += count_dc2e_inv_2;
        }

        let msg_displs_dc2e_inv_1 = msg_counts_dc2e_inv_1
            .iter()
            .scan(0, |acc, &x| {
                let tmp = *acc;
                *acc += x;
                Some(tmp)
            })
            .collect_vec();

        let msg_displs_dc2e_inv_2 = msg_counts_dc2e_inv_2
            .iter()
            .scan(0, |acc, &x| {
                let tmp = *acc;
                *acc += x;
                Some(tmp)
            })
            .collect_vec();

        let mut buf_dc2e_inv_1 = vec![Scalar::zero(); msg_size_dc2e_inv_1];
        let mut buf_dc2e_inv_2 = vec![Scalar::zero(); msg_size_dc2e_inv_2];

        // Communicate pre-computations
        {
            let mut partition = PartitionMut::new(
                &mut buf_dc2e_inv_1,
                &msg_counts_dc2e_inv_1[..],
                &msg_displs_dc2e_inv_1[..],
            );
            self.communicator
                .all_gather_varcount_into(&local_msg_dc2e_inv_1, &mut partition);

            let mut partition = PartitionMut::new(
                &mut buf_dc2e_inv_2,
                &msg_counts_dc2e_inv_2[..],
                &msg_displs_dc2e_inv_2[..],
            );
            self.communicator
                .all_gather_varcount_into(&local_msg_dc2e_inv_2, &mut partition);
        }

        // Re-structure messages using shared dimensions
        let mut dc2e_inv_1_vec = Vec::new();
        let mut dc2e_inv_2_vec = Vec::new();

        if n_precomputations > 1 {
            let mut curr1 = 0;
            let mut curr2 = 0;
            for (&e_order, &c_order, &shared_dim) in izip!(
                &self.equivalent_surface_order,
                &self.check_surface_order,
                shared_dim.iter()
            ) {
                let l = curr1;
                let r = curr1 + ncoeffs_kifmm(c_order) * shared_dim;
                let mut dc2e_inv_1 =
                    rlst_dynamic_array2!(Scalar, [ncoeffs_kifmm(c_order), shared_dim]);
                dc2e_inv_1.data_mut().copy_from_slice(&buf_dc2e_inv_1[l..r]);
                curr1 = r;

                let l = curr2;
                let r = curr2 + ncoeffs_kifmm(e_order) * shared_dim;
                let mut dc2e_inv_2 =
                    rlst_dynamic_array2!(Scalar, [shared_dim, ncoeffs_kifmm(e_order)]);
                dc2e_inv_2.data_mut().copy_from_slice(&buf_dc2e_inv_2[l..r]);
                curr2 = r;

                dc2e_inv_1_vec.push(dc2e_inv_1);
                dc2e_inv_2_vec.push(dc2e_inv_2);
            }
        } else {
            let c_order = self.check_surface_order[0];
            let e_order = self.equivalent_surface_order[0];
            let shared_dim = shared_dim[0];
            let l = 0;
            let r = ncoeffs_kifmm(c_order) * shared_dim;
            let mut dc2e_inv_1 = rlst_dynamic_array2!(Scalar, [ncoeffs_kifmm(c_order), shared_dim]);
            dc2e_inv_1.data_mut().copy_from_slice(&buf_dc2e_inv_1[l..r]);

            let l = 0;
            let r = ncoeffs_kifmm(e_order) * shared_dim;
            let mut dc2e_inv_2 = rlst_dynamic_array2!(Scalar, [shared_dim, ncoeffs_kifmm(e_order)]);
            dc2e_inv_2.data_mut().copy_from_slice(&buf_dc2e_inv_2[l..r]);

            dc2e_inv_1_vec.push(dc2e_inv_1);
            dc2e_inv_2_vec.push(dc2e_inv_2);
        }

        // Calculate all L2L operator matrices
        let level_iterator = if self.variable_expansion_order {
            0..self.tree.target_tree().total_depth()
        } else {
            0..1
        };

        let mut l2l_vec = Vec::new();
        let mut l2l_global_vec = Vec::new();

        for parent_level in level_iterator {
            let equivalent_surface_order_parent = self.equivalent_surface_order(parent_level);
            let check_surface_order_child = self.check_surface_order(parent_level + 1);

            let parent_downward_equivalent_surface =
                root.surface_grid(equivalent_surface_order_parent, domain, alpha_outer);

            // Calculate L2L operator matrices
            let children = root.children();
            let n_coeffs_check_surface_child = ncoeffs_kifmm(check_surface_order_child);
            let n_coeffs_equivalent_surface_parent = ncoeffs_kifmm(equivalent_surface_order_parent);

            let mut l2l_level_1 = Vec::new();
            let mut l2l_level_2 = Vec::new();

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

                let mut tmp_1 = empty_array::<Scalar, 2>().simple_mult_into_resize(
                    dc2e_inv_1_vec[self.expansion_index(parent_level + 1)].r(),
                    empty_array::<Scalar, 2>().simple_mult_into_resize(
                        dc2e_inv_2_vec[self.expansion_index(parent_level + 1)].r(),
                        pe2cc.r(),
                    ),
                );

                tmp_1
                    .data_mut()
                    .iter_mut()
                    .for_each(|d| *d *= homogenous_kernel_scale(child.level()));

                let mut tmp_2 = rlst_dynamic_array2!(Scalar, tmp_1.shape());
                tmp_2.data_mut().copy_from_slice(tmp_1.data());

                l2l_level_1.push(tmp_1);
                l2l_level_2.push(tmp_2);
            }

            l2l_vec.push(l2l_level_1);
            l2l_global_vec.push(l2l_level_2);
        }

        let mut dc2e_inv_1_global = dc2e_inv_1_vec
            .iter()
            .map(|x| rlst_dynamic_array2!(Scalar, x.shape()))
            .collect_vec();
        let mut dc2e_inv_2_global = dc2e_inv_2_vec
            .iter()
            .map(|x| rlst_dynamic_array2!(Scalar, x.shape()))
            .collect_vec();

        dc2e_inv_1_global
            .iter_mut()
            .enumerate()
            .for_each(|(i, x)| x.data_mut().copy_from_slice(dc2e_inv_1_vec[i].data()));
        dc2e_inv_2_global
            .iter_mut()
            .enumerate()
            .for_each(|(i, x)| x.data_mut().copy_from_slice(dc2e_inv_2_vec[i].data()));

        self.global_fmm.target_vec = l2l_global_vec;
        self.global_fmm.dc2e_inv_1 = dc2e_inv_1_global;
        self.global_fmm.dc2e_inv_2 = dc2e_inv_2_global;

        self.target_vec = l2l_vec;
        self.dc2e_inv_1 = dc2e_inv_1_vec;
        self.dc2e_inv_2 = dc2e_inv_2_vec;
    }
}

impl<Scalar> SourceToTargetTranslationMetadata
    for KiFmmMulti<Scalar, Laplace3dKernel<Scalar>, BlasFieldTranslationSaRcmp<Scalar>>
where
    Scalar: RlstScalar + Default + MatrixRsvd + Equivalence + Float,
    <Scalar as RlstScalar>::Real: Default + Equivalence + Float,
{
    fn displacements(&mut self, start_level: Option<u64>) {
        let mut displacements = Vec::new();

        let start_level =
            start_level.unwrap_or_else(|| std::cmp::max(2, self.tree.source_tree().global_depth()));

        for level in start_level..=self.tree.source_tree().total_depth() {
            let sources = self.tree.source_tree().keys(level).unwrap_or_default();
            let n_sources = sources.len();

            let sentinel = -1_i32;
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
                            result_lock[source_idx] = target_idx as i32;
                        }
                    }
                });

            displacements.push(result);
        }

        self.source_to_target.displacements = displacements;
    }

    fn source_to_target(&mut self) {
        let size = self.communicator.size();
        let rank = self.communicator.rank();
        let total_depth = self.tree.source_tree().total_depth();
        let alpha = Scalar::real(ALPHA_INNER);

        // Distribute SVD by level
        // Number of pre-computations
        // Distribute the SVD by level
        let iterator = if self.variable_expansion_order {
            (2..=total_depth)
                .zip(self.equivalent_surface_order.iter().skip(2).cloned())
                .zip(self.check_surface_order.iter().skip(2).cloned())
                .collect_vec()
        } else {
            (2..=total_depth)
                .zip(vec![
                    *self.equivalent_surface_order.last().unwrap();
                    (total_depth - 1) as usize
                ])
                .zip(vec![
                    *self.check_surface_order.last().unwrap();
                    (total_depth - 1) as usize
                ])
                .collect_vec()
        };

        let n_precomputations;
        if self.variable_expansion_order {
            n_precomputations = iterator.len() as i32;
        } else {
            n_precomputations = 1;
        }

        let (load_counts, load_displacement) =
            calculate_precomputation_load(n_precomputations, size).unwrap();

        // Compute mandated local portion
        let local_load_count = load_counts[rank as usize];
        let local_load_displacement = load_displacement[rank as usize];

        let iterator_r = &iterator[(local_load_displacement as usize)
            ..((local_load_displacement + local_load_count) as usize)];

        let mut cutoff_rank_r = Vec::new();
        let mut directional_cutoff_ranks_r = Vec::new();
        let mut metadata_r = Vec::new();

        for &((level, equivalent_surface_order), check_surface_order) in iterator_r.iter() {
            let transfer_vectors = compute_transfer_vectors_at_level::<Scalar::Real>(3).unwrap();

            let n_rows = ncoeffs_kifmm(check_surface_order);
            let n_cols = ncoeffs_kifmm(equivalent_surface_order);

            let mut se2tc_fat =
                rlst_dynamic_array2!(Scalar, [n_rows, n_cols * NTRANSFER_VECTORS_KIFMM]);
            let mut se2tc_thin =
                rlst_dynamic_array2!(Scalar, [n_rows * NTRANSFER_VECTORS_KIFMM, n_cols]);

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

                let mut tmp_gram = rlst_dynamic_array2!(Scalar, [n_rows, n_cols]);

                self.kernel.assemble_st(
                    GreenKernelEvalType::Value,
                    &target_check_surface[..],
                    &source_equivalent_surface[..],
                    tmp_gram.data_mut(),
                );

                let mut block = se2tc_fat
                    .r_mut()
                    .into_subview([0, i * n_cols], [n_rows, n_cols]);
                block.fill_from(tmp_gram.r());

                let mut block_column = se2tc_thin
                    .r_mut()
                    .into_subview([i * n_rows, 0], [n_rows, n_cols]);
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
                find_cutoff_rank(&sigma, self.source_to_target.threshold, n_cols).min(target_rank);

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
            let directional_cutoff_ranks = Mutex::new(vec![0i32; NTRANSFER_VECTORS_KIFMM]);

            for _ in 0..NTRANSFER_VECTORS_KIFMM {
                c_u.lock()
                    .unwrap()
                    .push(rlst_dynamic_array2!(Scalar, [1, 1]));
                c_vt.lock()
                    .unwrap()
                    .push(rlst_dynamic_array2!(Scalar, [1, 1]));
            }

            (0..NTRANSFER_VECTORS_KIFMM).into_par_iter().for_each(|i| {
                let vt_block = vt.r().into_subview([0, i * n_cols], [cutoff_rank, n_cols]);

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

                directional_cutoff_ranks.lock().unwrap()[i] = directional_cutoff_rank as i32;
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

            metadata_r.push(result);
            cutoff_rank_r.push(cutoff_rank as i32);
            directional_cutoff_ranks_r.push(directional_cutoff_ranks);
        }

        // Communicate metadata
        let metadata_r_serialised = serialise_vec_blas_metadata_sarcmp(&metadata_r);
        let global_metadata_serialised =
            all_gather_v_serialised(&metadata_r_serialised, &self.communicator);

        let cutoff_rank_r_serialised = serialise_vec(&cutoff_rank_r);
        let global_cutoff_rank_serialised =
            all_gather_v_serialised(&cutoff_rank_r_serialised, &self.communicator);

        let directional_cutoff_rank_r_serialised =
            serialise_nested_vec(&directional_cutoff_ranks_r);
        let global_directional_cutoff_rank_serialised =
            all_gather_v_serialised(&directional_cutoff_rank_r_serialised, &self.communicator);

        // Reconstruct metadata
        let (mut global_metadata, mut rest) =
            deserialise_vec_blas_metadata_sarcmp::<Scalar>(&global_metadata_serialised);
        while rest.len() > 0 {
            let (mut t1, t2) = deserialise_vec_blas_metadata_sarcmp::<Scalar>(rest);
            global_metadata.append(&mut t1);
            rest = t2;
        }

        let (mut global_metadata_clone, mut rest) =
            deserialise_vec_blas_metadata_sarcmp::<Scalar>(&global_metadata_serialised);
        while rest.len() > 0 {
            let (mut t1, t2) = deserialise_vec_blas_metadata_sarcmp::<Scalar>(rest);
            global_metadata_clone.append(&mut t1);
            rest = t2;
        }

        let mut buffer = Vec::new();
        let (global_cutoff_ranks, mut rest) =
            deserialise_vec::<i32>(&global_cutoff_rank_serialised);
        buffer.extend_from_slice(global_cutoff_ranks);
        while rest.len() > 0 {
            let (t1, t2) = deserialise_vec::<i32>(rest);
            buffer.extend_from_slice(t1);
            rest = t2;
        }

        let (mut global_directional_cutoff_ranks, mut rest) =
            deserialise_nested_vec::<i32>(&global_directional_cutoff_rank_serialised);
        while rest.len() > 0 {
            let (mut t1, t2) = deserialise_nested_vec::<i32>(&rest);
            global_directional_cutoff_ranks.append(&mut t1);
            rest = t2;
        }

        let global_cutoff_ranks = buffer.iter().map(|&x| x as usize).collect_vec();
        let global_directional_cutoff_ranks = global_directional_cutoff_ranks
            .iter()
            .map(|vec| vec.iter().map(|&x| x as usize).collect_vec())
            .collect_vec();

        for (metadata, metadata_clone) in izip!(global_metadata, global_metadata_clone) {
            self.global_fmm.source_to_target.metadata.push(metadata);
            self.source_to_target.metadata.push(metadata_clone);
        }

        self.source_to_target.cutoff_rank = global_cutoff_ranks.clone();
        self.source_to_target.directional_cutoff_ranks = global_directional_cutoff_ranks.clone();

        self.global_fmm.source_to_target.cutoff_rank = global_cutoff_ranks.clone();
        self.global_fmm.source_to_target.directional_cutoff_ranks =
            global_directional_cutoff_ranks.clone();
        self.global_fmm.source_to_target.transfer_vectors =
            self.source_to_target.transfer_vectors.clone();

        self.ghost_fmm_v.source_to_target.transfer_vectors =
            self.source_to_target.transfer_vectors.clone();
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
    // TODO: the displacements by level must also be parallelised
    fn displacements(&mut self, start_level: Option<u64>) {
        let mut displacements = Vec::new();
        let start_level =
            start_level.unwrap_or_else(|| std::cmp::max(2, self.tree.source_tree().global_depth()));

        for level in start_level..=self.tree.source_tree.total_depth() {
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

    // TODO
    fn source_to_target(&mut self) {
        // Compute the field translation operators
        let size = self.communicator.size();
        let rank = self.communicator.rank();

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

        let iterator = if self.variable_expansion_order {
            self.equivalent_surface_order
                .iter()
                .skip(2)
                .cloned()
                .collect_vec()
        } else {
            self.equivalent_surface_order.clone()
        };

        let n_precomputations;
        if self.variable_expansion_order {
            n_precomputations = iterator.len() as i32;
        } else {
            n_precomputations = 1;
        }

        let (load_counts, load_displacement) =
            calculate_precomputation_load(n_precomputations, size).unwrap();

        // Compute mandated local portion
        let local_load_count = load_counts[rank as usize];
        let local_load_displacement = load_displacement[rank as usize];

        let iterator_r = &iterator[(local_load_displacement as usize)
            ..((local_load_displacement + local_load_count) as usize)];

        let mut metadata_r: Vec<FftMetadata<<Scalar as AsComplex>::ComplexType>> = Vec::new();

        for &equivalent_surface_order in iterator_r.iter() {
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

            // // Set operator data
            // self.source_to_target.metadata.push(metadata.clone());

            // // Copy for global FMM
            // self.global_fmm
            //     .source_to_target
            //     .metadata
            //     .push(metadata.clone());

            metadata_r.push(metadata);
        }

        // Communicate metadata
        let metadata_r_serialised = serialise_vec_fft_metadata(&metadata_r);
        let global_metadata_serialised =
            all_gather_v_serialised(&metadata_r_serialised, &self.communicator);

        // Reconstruct metadata
        let (mut global_metadata, mut rest) =
            deserialise_vec_fft_metadata(&global_metadata_serialised);
        while rest.len() > 0 {
            let (mut t1, t2) =
                deserialise_vec_fft_metadata::<<Scalar as AsComplex>::ComplexType>(rest);
            global_metadata.append(&mut t1);
            rest = t2;
        }

        for metadata in global_metadata.iter() {
            // Copy for global FMM
            self.global_fmm
                .source_to_target
                .metadata
                .push(metadata.clone());

            // Set operator data
            self.source_to_target.metadata.push(metadata.clone());
        }

        // Compute and attach maps
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

        // Copy for global FMM
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

    fn metadata(&mut self, eval_type: GreenKernelEvalType, charges: &[Self::Scalar]) {
        let alpha_outer = Scalar::from(ALPHA_OUTER).unwrap().re();
        let alpha_inner = Scalar::from(ALPHA_INNER).unwrap().re();

        // Check if computing potentials, or potentials and derivatives
        let kernel_eval_size = match eval_type {
            GreenKernelEvalType::Value => 1,
            GreenKernelEvalType::ValueDeriv => 4,
        };

        // TODO Add real matrix input
        let n_matvecs = 1;
        let n_target_points = self.tree.target_tree.n_coordinates_tot().unwrap();
        let n_source_keys = self.tree.source_tree.n_keys_tot().unwrap();
        let n_target_keys = self.tree.target_tree.n_keys_tot().unwrap();

        let local_depth = self.tree.source_tree().local_depth();
        let global_depth = self.tree.source_tree().global_depth();
        let total_depth = self.tree.source_tree().total_depth();

        // ncoeffs in the local part of the tree
        let local_ncoeffs_equivalent_surface: Vec<usize> = self
            .n_coeffs_equivalent_surface
            .iter()
            .skip(global_depth as usize)
            .cloned()
            .collect_vec();

        // Buffers to store all multipole and local data
        let n_multipole_coeffs;
        let n_local_coeffs;
        if self.equivalent_surface_order.len() > 1 {
            n_multipole_coeffs = (global_depth..=total_depth)
                .zip(local_ncoeffs_equivalent_surface.iter())
                .fold(0usize, |acc, (level, &ncoeffs)| {
                    acc + self.tree.source_tree().n_keys(level).unwrap_or(0) * ncoeffs
                });

            n_local_coeffs = (global_depth..=total_depth)
                .zip(local_ncoeffs_equivalent_surface.iter())
                .fold(0usize, |acc, (level, &ncoeffs)| {
                    acc + self.tree.target_tree().n_keys(level).unwrap_or(0) * ncoeffs
                })
        } else {
            n_multipole_coeffs = n_source_keys * self.n_coeffs_equivalent_surface.last().unwrap();
            n_local_coeffs = n_target_keys * self.n_coeffs_equivalent_surface.last().unwrap();
        }

        let multipoles = vec![Scalar::default(); n_multipole_coeffs * n_matvecs];
        let locals = vec![Scalar::default(); n_local_coeffs * n_matvecs];

        // Index pointers of multipole and local data, indexed by level
        let level_index_pointer_multipoles = level_index_pointer_multi_node(&self.tree.source_tree);
        let level_index_pointer_locals = level_index_pointer_multi_node(&self.tree.target_tree);

        // Allocate buffers for local potential data
        let potentials = vec![Scalar::default(); n_target_points * kernel_eval_size];

        // Pre compute check surfaces
        let leaf_upward_equivalent_surfaces_sources = leaf_surfaces_multi_node(
            &self.tree.source_tree,
            *self.n_coeffs_equivalent_surface.last().unwrap(),
            alpha_inner,
            *self.equivalent_surface_order.last().unwrap(),
        );

        let leaf_upward_check_surfaces_sources = leaf_surfaces_multi_node(
            &self.tree.source_tree,
            *self.n_coeffs_check_surface.last().unwrap(),
            alpha_outer,
            *self.check_surface_order.last().unwrap(),
        );

        let leaf_downward_equivalent_surfaces_targets = leaf_surfaces_multi_node(
            &self.tree.target_tree,
            *self.n_coeffs_equivalent_surface.last().unwrap(),
            alpha_outer,
            *self.equivalent_surface_order.last().unwrap(),
        );

        // Mutable pointers to multipole and local data, indexed by level
        let level_multipoles = level_expansion_pointers_multi_node(
            &self.tree.source_tree,
            &self.n_coeffs_equivalent_surface,
            n_matvecs,
            &multipoles,
        );

        let level_locals = level_expansion_pointers_multi_node(
            &self.tree.target_tree,
            &self.n_coeffs_equivalent_surface,
            n_matvecs,
            &locals,
        );

        // Mutable pointers to multipole and local data only at leaf level, for utility
        let leaf_multipoles = leaf_expansion_pointers_multi_node(
            &self.tree.source_tree,
            &self.n_coeffs_equivalent_surface,
            n_matvecs,
            &multipoles,
        );

        let leaf_locals = leaf_expansion_pointers_multi_node(
            &self.tree.target_tree,
            &self.n_coeffs_equivalent_surface,
            n_matvecs,
            &locals,
        );

        // Mutable pointers to potential data at each target leaf
        let potential_send_pointers =
            potential_pointers_multi_node(&self.tree.target_tree, kernel_eval_size, &potentials);

        // Setup neighbourhood communication for charges
        let n_sources = charges.len() as u64;
        let size = self.tree.source_tree.communicator.size();
        let mut counts = vec![0u64; size as usize];
        self.communicator
            .all_gather_into(&n_sources, &mut counts[..]);

        let mut displacements = Vec::new();
        let mut curr = 0;
        for count in counts.iter() {
            displacements.push((curr, curr + count - 1));
            curr += count;
        }

        let rank = self.communicator.rank();
        let mut local_displacement = 0;
        for count in counts.iter().take(rank as usize) {
            local_displacement += count;
        }

        self.local_count_charges = charges.len() as u64;
        self.local_displacement_charges = local_displacement;

        let global_indices = &self.tree.source_tree.global_indices;

        let mut ranks = Vec::new();
        let mut send_counts = vec![0 as Count; self.tree.source_tree.communicator.size() as usize];
        let mut send_marker = vec![0 as Rank; self.tree.source_tree.communicator.size() as usize];

        for &global_index in global_indices.iter() {
            let rank = displacements
                .iter()
                .position(|&(start, end)| {
                    global_index >= (start as usize) && global_index <= (end as usize)
                })
                .unwrap();
            ranks.push(rank as i32);
            send_counts[rank] += 1;
        }

        for (rank, &send_count) in send_counts.iter().enumerate() {
            if send_count > 0 {
                send_marker[rank] = 1;
            }
        }

        // Sort queries by destination rank
        let queries = {
            let mut indices = (0..global_indices.len()).collect_vec();
            indices.sort_by_key(|&i| ranks[i]);

            let mut sorted_queries_ = Vec::with_capacity(global_indices.len());
            for i in indices {
                sorted_queries_.push(global_indices[i])
            }
            sorted_queries_
        };

        // Sort ranks of queries into rank order
        ranks.sort();

        // Compute the receive counts, and mark again processes involved
        let mut receive_counts = vec![0i32; self.tree.source_tree.communicator.size() as usize];
        let mut receive_marker = vec![0i32; self.tree.source_tree.communicator.size() as usize];

        self.communicator
            .all_to_all_into(&send_counts, &mut receive_counts);

        for (rank, &receive_count) in receive_counts.iter().enumerate() {
            if receive_count > 0 {
                receive_marker[rank] = 1
            }
        }

        let neighbourhood_communicator_charge =
            NeighbourhoodCommunicator::new(&self.communicator, &send_marker, &receive_marker);

        // Communicate ghost queries and receive from foreign ranks
        let mut neighbourhood_send_counts = Vec::new();
        let mut neighbourhood_receive_counts = Vec::new();
        let mut neighbourhood_send_displacements = Vec::new();
        let mut neighbourhood_receive_displacements = Vec::new();

        // Now can calculate displacements
        let mut send_counter = 0;
        let mut receive_counter = 0;

        // Remember these queries are constructed over the global communicator, so have
        // to filter for relevant queries in local communicator
        for (&send_count, &receive_count) in izip!(&send_counts, &receive_counts) {
            // Note this checks if any communication is happening between these ranks
            if send_count != 0 || receive_count != 0 {
                neighbourhood_send_counts.push(send_count);
                neighbourhood_receive_counts.push(receive_count);
                neighbourhood_send_displacements.push(send_counter);
                neighbourhood_receive_displacements.push(receive_counter);
                send_counter += send_count;
                receive_counter += receive_count;
            }
        }

        let total_receive_count = receive_counter as usize;

        // Setup buffers for queries received at this process and to handle
        // filtering for available queries to be sent back to partners
        let mut received_queries = vec![0u64; total_receive_count];

        // Available keys
        let mut available_queries = Vec::new();
        let mut available_queries_counts = Vec::new();
        let mut available_queries_displacements = Vec::new();

        {
            // Communicate queries
            let partition_send = Partition::new(
                &queries,
                neighbourhood_send_counts,
                neighbourhood_send_displacements,
            );

            let mut partition_receive = PartitionMut::new(
                &mut received_queries,
                neighbourhood_receive_counts,
                neighbourhood_receive_displacements,
            );

            neighbourhood_communicator_charge
                .all_to_all_varcount_into(&partition_send, &mut partition_receive);

            // Filter for locally available queries to send back
            let receive_counts_ = partition_receive.counts().iter().cloned().collect_vec();
            let receive_displacements_ = partition_receive.displs().iter().cloned().collect_vec();

            self.ghost_received_queries_charge = received_queries.iter().cloned().collect_vec();
            self.ghost_received_queries_charge_counts =
                receive_counts_.iter().cloned().collect_vec();
            self.ghost_received_queries_charge_displacements =
                receive_displacements_.iter().cloned().collect_vec();

            let mut counter = 0;

            // Iterate over received data rank by rank
            for (count, displacement) in izip!(receive_counts_, receive_displacements_) {
                let l = displacement as usize;
                let r = l + (count as usize);

                // Received queries from this rank
                let received_queries_rank = &received_queries[l..r];

                // Filter for available data corresponding to this request
                let mut available_queries_rank = Vec::new();

                let mut counter_rank = 0i32;

                // Only communicate back queries and associated data if particle data is found
                for &query in received_queries_rank.iter() {
                    available_queries_rank.push(charges[(query - local_displacement) as usize]);
                    // Update counters
                    counter_rank += 1;
                }

                // Update return buffers
                available_queries.extend(available_queries_rank);
                available_queries_counts.push(counter_rank);
                available_queries_displacements.push(counter);

                // Update counters
                counter += counter_rank;
            }
        }

        // Communicate expected query sizes
        let mut requested_queries_counts =
            vec![0 as Count; neighbourhood_communicator_charge.neighbours.len()];
        {
            let send_counts_ = vec![1i32; neighbourhood_communicator_charge.neighbours.len()];
            let send_displacements_ = send_counts_
                .iter()
                .scan(0, |acc, &x| {
                    let tmp = *acc;
                    *acc += x;
                    Some(tmp)
                })
                .collect_vec();

            let partition_send =
                Partition::new(&available_queries_counts, send_counts_, send_displacements_);

            let recv_counts_ = vec![1i32; neighbourhood_communicator_charge.neighbours.len()];
            let recv_displacements_ = recv_counts_
                .iter()
                .scan(0, |acc, &x| {
                    let tmp = *acc;
                    *acc += x;
                    Some(tmp)
                })
                .collect_vec();

            let mut partition_receive = PartitionMut::new(
                &mut requested_queries_counts,
                recv_counts_,
                recv_displacements_,
            );

            neighbourhood_communicator_charge
                .all_to_all_varcount_into(&partition_send, &mut partition_receive);
        }

        // Create buffers to receive charge data
        let total_receive_count_available_queries =
            requested_queries_counts.iter().sum::<i32>() as usize;
        let mut requested_queries = vec![Scalar::default(); total_receive_count_available_queries];

        let mut requested_queries_displacements = Vec::new();
        let mut counter = 0;
        for &count in requested_queries_counts.iter() {
            requested_queries_displacements.push(counter);
            counter += count;
        }

        self.charge_send_queries_counts = available_queries_counts.iter().cloned().collect_vec();
        self.charge_send_queries_displacements = available_queries_displacements
            .iter()
            .cloned()
            .collect_vec();
        self.charge_receive_queries_counts = requested_queries_counts.iter().cloned().collect_vec();
        self.charge_receive_queries_displacements = requested_queries_displacements
            .iter()
            .cloned()
            .collect_vec();

        // Communicate ghost charges
        {
            let partition_send = Partition::new(
                &available_queries,
                &available_queries_counts[..],
                &available_queries_displacements[..],
            );

            let mut partition_receive = PartitionMut::new(
                &mut requested_queries,
                &requested_queries_counts[..],
                &requested_queries_displacements[..],
            );

            neighbourhood_communicator_charge
                .all_to_all_varcount_into(&partition_send, &mut partition_receive);
        }

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

        self.neighbourhood_communicator_charge = neighbourhood_communicator_charge;

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
        self.charge_index_pointer_sources = charge_index_pointer_sources;
        self.charge_index_pointer_targets = charge_index_pointer_targets;
        self.kernel_eval_size = kernel_eval_size;
        self.charges = requested_queries;

        // Can perform U list exchange now
        let (_, duration) = optionally_time(self.timed, || {
            self.u_list_exchange();
        });

        if let Some(d) = duration {
            self.communication_times
                .push(CommunicationTime::from_duration(
                    CommunicationType::GhostExchangeU,
                    d,
                ))
        }

        let (_, duration) = optionally_time(self.timed, || {
            self.v_list_exchange();
        });

        if let Some(d) = duration {
            self.communication_times
                .push(CommunicationTime::from_duration(
                    CommunicationType::GhostExchangeV,
                    d,
                ))
        }
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
    FieldTranslation: FieldTranslationTrait + Send + Sync + Default,
    KiFmm<Scalar, Laplace3dKernel<Scalar>, FieldTranslation>: DataAccess<Scalar = Scalar, Tree = SingleNodeFmmTree<Scalar::Real>>
        + SourceToTargetTranslationMetadata
        + SourceTranslationMetadata
        + TargetTranslationMetadata,
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
        let start_level = std::cmp::max(2, self.tree.source_tree.global_depth);
        (level - start_level) as usize
    }
}
