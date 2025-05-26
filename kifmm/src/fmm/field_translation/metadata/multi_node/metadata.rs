use green_kernels::{
    helmholtz_3d::Helmholtz3dKernel, laplace_3d::Laplace3dKernel, traits::Kernel as KernelTrait,
    types::GreenKernelEvalType,
};
use itertools::{izip, Itertools};
use mpi::{
    datatype::{Partition, PartitionMut, Partitioned},
    traits::{Communicator, CommunicatorCollectives, Equivalence},
    Count, Rank,
};
use num::Float;
use rlst::{rlst_dynamic_array3, Array, BaseArray, RawAccessMut, RlstScalar, VectorContainer};

use crate::{
    fmm::{
        helpers::{
            multi_node::{
                coordinate_index_pointer_multi_node, leaf_expansion_pointers_multi_node,
                leaf_surfaces_multi_node, level_expansion_pointers_multi_node,
                level_index_pointer_multi_node, potential_pointers_multi_node,
            },
            single_node::optionally_time,
        },
        types::{KiFmmMulti, NeighbourhoodCommunicator},
    },
    traits::{
        fftw::Dft,
        field::{
            FieldTranslation as FieldTranslationTrait, SourceToTargetTranslationMetadata,
            SourceTranslationMetadata, TargetTranslationMetadata,
        },
        fmm::{DataAccess, DataAccessMulti, HomogenousKernel, Metadata, MetadataAccess},
        general::{multi_node::GhostExchange, single_node::AsComplex},
        tree::{MultiFmmTree, MultiTree},
        types::{CommunicationTime, CommunicationType},
    },
    tree::constants::{ALPHA_INNER, ALPHA_OUTER},
    FftFieldTranslation, KiFmm, SingleNodeFmmTree,
};

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

impl<Scalar, FieldTranslation> MetadataAccess
    for KiFmmMulti<Scalar, Helmholtz3dKernel<Scalar>, FieldTranslation>
where
    Scalar: RlstScalar<Complex = Scalar> + Default + Equivalence + Float,
    <Scalar as RlstScalar>::Real: RlstScalar + Default + Equivalence + Float,
    FieldTranslation: FieldTranslationTrait + Send + Sync + Default,
    KiFmm<Scalar, Helmholtz3dKernel<Scalar>, FieldTranslation>: DataAccess<Scalar = Scalar, Tree = SingleNodeFmmTree<Scalar::Real>>
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
        level as usize
    }

    fn m2m_operator_index(&self, level: u64) -> usize {
        (level - 1) as usize
    }

    fn l2l_operator_index(&self, level: u64) -> usize {
        (level - 1) as usize
    }

    fn m2l_operator_index(&self, level: u64) -> usize {
        (level - 2) as usize
    }

    fn displacement_index(&self, level: u64) -> usize {
        let start_level = std::cmp::max(2, self.tree.source_tree.global_depth);
        (level - start_level) as usize
    }
}
