use std::collections::HashMap;
use std::collections::HashSet;

use green_kernels::traits::Kernel as KernelTrait;
use itertools::izip;
use itertools::Itertools;
use mpi::datatype::Partition;
use mpi::datatype::PartitionMut;
use mpi::topology::Communicator;
use mpi::topology::SimpleCommunicator;
use mpi::traits::Equivalence;
use mpi::traits::Partitioned;
use mpi::traits::Root;
use mpi::Count;
use mpi::Rank;
use num::Float;
use rlst::RlstScalar;

use crate::traits::field::FieldTranslation as FieldTranslationTrait;
use crate::traits::field::SourceToTargetTranslationMetadata;
use crate::traits::fmm::DataAccess;
use crate::traits::fmm::DataAccessMulti;
use crate::traits::fmm::HomogenousKernel;
use crate::traits::general::multi_node::GhostExchange;
use crate::traits::general::multi_node::GlobalFmmMetadata;
use crate::traits::tree::MultiFmmTree;
use crate::traits::tree::MultiTree;
use crate::traits::tree::SingleFmmTree;
use crate::traits::tree::SingleTree;
use crate::tree::types::Domain;
use crate::tree::types::MortonKey;
use crate::tree::SingleNodeTree;
use crate::MultiNodeFmmTree;
use crate::SingleNodeFmmTree;

use super::helpers::single_node::coordinate_index_pointer_single_node;
use super::helpers::single_node::leaf_expansion_pointers_single_node;
use super::helpers::single_node::level_expansion_pointers_single_node;
use super::helpers::single_node::level_index_pointer_single_node;
use super::types::KiFmmMulti;
use super::KiFmm;

impl<Scalar, Kernel, FieldTranslation> GhostExchange
    for KiFmmMulti<Scalar, Kernel, FieldTranslation>
where
    Scalar: RlstScalar + Default + Equivalence + Float,
    <Scalar as RlstScalar>::Real: RlstScalar + Default + Equivalence + Float,
    Kernel: KernelTrait<T = Scalar> + HomogenousKernel + Default + Send + Sync,
    FieldTranslation: FieldTranslationTrait + Send + Sync + Default,
    Self: DataAccessMulti<
        Scalar = Scalar,
        Kernel = Kernel,
        Tree = MultiNodeFmmTree<Scalar::Real, SimpleCommunicator>,
    >,
    KiFmm<Scalar, Kernel, FieldTranslation>: DataAccess<Scalar = Scalar, Tree = SingleNodeFmmTree<Scalar::Real>>
        + SourceToTargetTranslationMetadata,
{
    fn gather_global_fmm_at_root(&mut self) {
        let size = self.communicator.size();
        let rank = self.communicator.rank();

        // Nominated rank chosen to run the global upward pass
        let root_rank = 0;
        let root_process = self.communicator.process_at_rank(root_rank);

        // Gather multipole data
        if rank == root_rank {
            // Allocate memory for locally contained data that will be communicated
            let n_roots = self.tree.source_tree().n_trees() as Count;
            let mut roots = Vec::new();
            let mut multipoles =
                vec![Scalar::default(); (n_roots as usize) * self.n_coeffs_equivalent_surface];

            for (i, tree) in self.tree.source_tree().trees().iter().enumerate() {
                let tmp = self.multipole(&tree.root()).unwrap();
                multipoles[i * self.n_coeffs_equivalent_surface
                    ..(i + 1) * self.n_coeffs_equivalent_surface]
                    .copy_from_slice(tmp);
                roots.push(tree.root())
            }

            // Gather multipole data from root processes on all ranks
            // Begin by counting the number of global leaves/local roots there are for source trees
            let mut counts = vec![0 as Count; size as usize];
            root_process.gather_into_root(&n_roots, &mut counts);

            // Calculate associated displacements from the counts
            let mut displacements = Vec::new();
            let mut displacement = 0;
            for &count in counts.iter() {
                displacements.push(displacement);
                displacement += count;
            }

            // Calculate displacements and counts for multipole data from that for keys
            let counts_buffers = counts
                .iter()
                .map(|c| c * self.n_coeffs_equivalent_surface as i32)
                .collect_vec();

            let mut displacements_buffers = Vec::new();
            let mut displacement = 0;
            for &count in counts_buffers.iter() {
                displacements_buffers.push(displacement);
                displacement += count
            }

            // Allocate memory to store received multipoles
            let total_receive_buffers = counts_buffers.iter().sum::<i32>() as usize;
            let mut global_multipoles = vec![Scalar::default(); total_receive_buffers];

            // Gather all multipole data
            let mut partition = PartitionMut::new(
                &mut global_multipoles,
                &counts_buffers[..],
                &displacements_buffers[..],
            );

            root_process.gather_varcount_into_root(&multipoles, &mut partition);

            // Gather all associated keys
            let total_receive = counts.iter().sum::<i32>();
            let mut global_roots =
                vec![MortonKey::<Scalar::Real>::default(); total_receive as usize];
            let mut partition =
                PartitionMut::new(&mut global_roots, &counts[..], &displacements[..]);

            root_process.gather_varcount_into_root(&roots, &mut partition);

            // Need to set metadata for global FMM
            self.global_fmm.global_fmm_multipole_metadata(
                self.tree.domain(),
                self.tree.source_tree().global_depth(),
                global_roots,
                global_multipoles,
            );
        } else {
            // Send number of roots of local source trees
            let n_roots = self.tree.source_tree().n_trees() as Count;
            root_process.gather_into(&n_roots);

            // Create buffers of multipole data to be sent
            let mut multipoles =
                vec![Scalar::default(); (n_roots as usize) * self.n_coeffs_equivalent_surface];
            let mut roots = Vec::new();
            for (i, tree) in self.tree.source_tree().trees().iter().enumerate() {
                let tmp = self.multipole(&tree.root()).unwrap();
                multipoles[i * self.n_coeffs_equivalent_surface
                    ..(i + 1) * self.n_coeffs_equivalent_surface]
                    .copy_from_slice(tmp);
                roots.push(tree.root())
            }

            root_process.gather_varcount_into(&multipoles);
            root_process.gather_varcount_into(&roots);
        }

        // Gather local data
        if rank == root_rank {
            let n_roots = self.tree.target_tree().n_trees() as i32;

            // Begin by counting the number of global leaves/local roots there are for target trees
            let mut counts = vec![0 as Count; size as usize];
            root_process.gather_into_root(&n_roots, &mut counts);

            // Calculate displacements and counts for associated morton keys
            let mut displacements = Vec::new();
            let mut displacement = 0;
            for &count in counts.iter() {
                displacements.push(displacement);
                displacement += count;
            }

            // Allocate memory for locally contained data
            let mut roots = Vec::new();

            for tree in self.tree.target_tree().trees().iter() {
                roots.push(tree.root())
            }

            // Store origin ranks of each requested local
            let total_receive = counts.iter().sum::<i32>();
            let mut global_roots =
                vec![MortonKey::<Scalar::Real>::default(); total_receive as usize];
            let mut partition =
                PartitionMut::new(&mut global_roots, &counts[..], &displacements[..]);

            root_process.gather_varcount_into_root(&roots, &mut partition);

            let mut global_roots_ranks = Vec::new();
            for (rank, &count) in counts.iter().enumerate() {
                for _ in 0..(count as usize) {
                    global_roots_ranks.push(rank as Rank)
                }
            }

            // Set metadata
            self.global_fmm.global_fmm_local_metadata(
                self.tree.domain(),
                self.tree.target_tree().global_depth(),
                global_roots.iter().cloned().collect_vec(),
            );

            // Local roots, which need to be broadcasted back to at the end of global FMM evaluation
            // to contain local data
            self.local_roots = global_roots;
            self.local_roots_counts = counts;
            self.local_roots_displacements = displacements;
            self.local_roots_ranks = global_roots_ranks;
        } else {
            let n_roots = self.tree.target_tree().n_trees() as i32;
            root_process.gather_into(&n_roots);

            let mut local_roots = Vec::new();
            for tree in self.tree.target_tree().trees().iter() {
                local_roots.push(tree.root())
            }

            root_process.gather_varcount_into(&local_roots);
        }

        // TEST: Test displacements
        // Calculate displacements required for M2L
        if rank == root_rank {
            self.global_fmm.displacements(None)
        }
    }

    fn scatter_global_fmm_from_root(&mut self) {
        let rank = self.communicator.rank();
        let n_roots = self.tree.target_tree.trees.len();

        let mut expected_roots = vec![MortonKey::<Scalar::Real>::default(); n_roots];

        // Allocate buffers to receive local data
        let receive_buffer_size = n_roots * self.n_coeffs_equivalent_surface;
        let mut receive_buffer = vec![Scalar::default(); receive_buffer_size];

        // Nominated rank chosen to run global upward pass
        let root_rank = 0;
        let root_process = self.communicator.process_at_rank(root_rank);

        if rank == root_rank {
            // Lookup local data to be sent back from global FMM
            let send_buffer_size = self.local_roots.len() * self.n_coeffs_equivalent_surface;
            let mut send_buffer = vec![Scalar::default(); send_buffer_size];

            for (root_idx, root) in self.local_roots.iter().enumerate() {
                if let Some(local) = self.global_fmm.local(root) {
                    send_buffer[root_idx * self.n_coeffs_equivalent_surface
                        ..(root_idx + 1) * self.n_coeffs_equivalent_surface]
                        .copy_from_slice(local);
                }
            }

            // Displace items to send back by number of coefficients
            let counts = self
                .local_roots_counts
                .iter()
                .map(|&c| c * (self.n_coeffs_equivalent_surface as i32))
                .collect_vec();

            let displacements = self
                .local_roots_displacements
                .iter()
                .map(|&d| d * (self.n_coeffs_equivalent_surface as i32))
                .collect_vec();

            // Send back coefficient data
            let partition = Partition::new(&send_buffer, counts, &displacements[..]);
            root_process.scatter_varcount_into_root(&partition, &mut receive_buffer);

            // Send back associated keys
            let partition = Partition::new(
                &self.local_roots,
                &self.local_roots_counts[..],
                &self.local_roots_displacements[..],
            );
            root_process.scatter_varcount_into_root(&partition, &mut expected_roots);
        } else {
            root_process.scatter_varcount_into(&mut receive_buffer);
            root_process.scatter_varcount_into(&mut expected_roots);
        }

        // Insert received local data into target tree
        for (i, root) in expected_roots.iter().enumerate() {
            let l = i * self.n_coeffs_equivalent_surface;
            let r = l + self.n_coeffs_equivalent_surface;
            self.local_mut(root)
                .unwrap()
                .copy_from_slice(&receive_buffer[l..r]);
        }
    }

    fn u_list_exchange(&mut self) {
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
        for (&send_count, &receive_count) in izip!(
            &self.tree.u_list_query.send_counts,
            &self.tree.u_list_query.receive_counts
        ) {
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

        // Associated coordinate data
        let mut available_coordinates = Vec::new();
        let mut available_coordinates_counts = Vec::new();
        let mut available_coordinates_displacements = Vec::new();

        // Associated charge data
        let mut available_charges = Vec::new();
        let mut available_charges_counts = Vec::new();
        let mut available_charges_displacements = Vec::new();

        {
            // Communicate queries
            let partition_send = Partition::new(
                &self.tree.u_list_query.queries,
                neighbourhood_send_counts,
                neighbourhood_send_displacements,
            );

            let mut partition_receive = PartitionMut::new(
                &mut received_queries,
                neighbourhood_receive_counts,
                neighbourhood_receive_displacements,
            );

            self.neighbourhood_communicator_u
                .all_to_all_varcount_into(&partition_send, &mut partition_receive);

            // Filter for locally available queries to send back
            let receive_counts_ = partition_receive.counts().iter().cloned().collect_vec();
            let receive_displacements_ = partition_receive.displs().iter().cloned().collect_vec();

            let mut counter = 0;
            let mut counter_coordinates = 0;
            let mut counter_charges = 0;

            // Iterate over received data rank by rank
            for (count, displacement) in izip!(receive_counts_, receive_displacements_) {
                let l = displacement as usize;
                let r = l + (count as usize);

                // Received queries from this rank
                let received_queries_rank = &received_queries[l..r];

                // Filter for available data corresponding to this request
                let mut available_queries_rank = Vec::new();
                let mut available_coordinates_rank = Vec::new();
                let mut available_coordinates_counts_rank = Vec::new();
                let mut available_charges_rank = Vec::new();
                let mut available_charges_counts_rank = Vec::new();

                let mut counter_rank = 0i32;
                let mut counter_coordinates_rank = 0i32;
                let mut counter_charges_rank = 0i32;

                // Only communicate back queries and associated data if particle data is found
                for &query in received_queries_rank.iter() {
                    let key = MortonKey::from_morton(query);
                    if let Some(coordinates) = self.tree.source_tree.coordinates(&key) {
                        available_queries_rank.push(key);
                        available_coordinates_rank.push(coordinates);
                        available_coordinates_counts_rank.push(coordinates.len());

                        // Lookup associated charges
                        let &leaf_idx = self.tree.source_tree().leaf_index(&key).unwrap();
                        let index_pointer = &self.charge_index_pointer_sources[leaf_idx];
                        let charges = &self.charges[index_pointer.0..index_pointer.1];
                        available_charges_rank.push(charges);
                        available_charges_counts_rank.push(charges.len());

                        // Update counters
                        counter_rank += 1;
                        counter_coordinates_rank += coordinates.len() as i32;
                        counter_charges_rank += charges.len() as i32;
                    }
                }

                // Update return buffers
                available_queries.extend(available_queries_rank);
                available_queries_counts.push(counter_rank);
                available_queries_displacements.push(counter);

                let available_coordinates_rank = available_coordinates_rank.concat();
                available_coordinates.extend(available_coordinates_rank);
                available_coordinates_counts.push(counter_coordinates_rank);
                available_coordinates_displacements.push(counter_coordinates);

                let available_charges_rank = available_charges_rank.concat();
                available_charges.extend(available_charges_rank);
                available_charges_counts.push(counter_charges_rank);
                available_charges_displacements.push(counter_charges);

                // Update counters
                counter += counter_rank;
                counter_coordinates += counter_coordinates_rank;
                counter_charges += counter_charges_rank;
            }
        }

        // Communicate expected query sizes
        let mut requested_coordinates_counts =
            vec![0 as Count; self.neighbourhood_communicator_u.neighbours.len()];
        {
            let send_counts_ = vec![1i32; self.neighbourhood_communicator_u.neighbours.len()];
            let send_displacements_ = send_counts_
                .iter()
                .scan(0, |acc, &x| {
                    let tmp = *acc;
                    *acc += x;
                    Some(tmp)
                })
                .collect_vec();

            let partition_send = Partition::new(
                &available_coordinates_counts,
                send_counts_,
                send_displacements_,
            );

            let recv_counts_ = vec![1i32; self.neighbourhood_communicator_u.neighbours.len()];
            let recv_displacements_ = recv_counts_
                .iter()
                .scan(0, |acc, &x| {
                    let tmp = *acc;
                    *acc += x;
                    Some(tmp)
                })
                .collect_vec();

            let mut partition_receive = PartitionMut::new(
                &mut requested_coordinates_counts,
                recv_counts_,
                recv_displacements_,
            );

            // TODO: Investigate why all to all failing, and require all to all v
            self.neighbourhood_communicator_u
                .all_to_all_varcount_into(&partition_send, &mut partition_receive);
        }

        // Communicate expected query sizes
        let mut requested_queries_counts =
            vec![0 as Count; self.neighbourhood_communicator_u.neighbours.len()];
        {
            let send_counts_ = vec![1i32; self.neighbourhood_communicator_u.neighbours.len()];
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

            let recv_counts_ = vec![1i32; self.neighbourhood_communicator_u.neighbours.len()];
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

            // TODO: Investigate why all to all failing, and require all to all v
            self.neighbourhood_communicator_u
                .all_to_all_varcount_into(&partition_send, &mut partition_receive);
        }

        // Create buffers to receive charge and coordinate data
        let total_receive_count_available_coordinates =
            requested_coordinates_counts.iter().sum::<i32>() as usize;
        let total_receive_count_available_charges = total_receive_count_available_coordinates / 3;
        let total_receive_count_available_queries =
            requested_queries_counts.iter().sum::<i32>() as usize;

        let mut requested_coordinates =
            vec![Scalar::Real::default(); total_receive_count_available_coordinates];
        let mut requested_charges = vec![Scalar::default(); total_receive_count_available_charges];
        let mut requested_queries =
            vec![MortonKey::<Scalar::Real>::default(); total_receive_count_available_queries];

        // Calculate counts for requested charges
        let mut requested_charges_counts = Vec::new();
        for &count in requested_coordinates_counts.iter() {
            requested_charges_counts.push(count / 3);
        }

        // Create displacements for coordinate and charge data from expected count
        let mut requested_coordinates_displacements = Vec::new();
        let mut requested_charges_displacements = Vec::new();
        let mut requested_queries_displacements = Vec::new();

        let mut counter = 0;
        for &count in requested_coordinates_counts.iter() {
            requested_coordinates_displacements.push(counter);
            requested_charges_displacements.push(counter / 3);
            counter += count;
        }

        let mut counter = 0;
        for &count in requested_queries_counts.iter() {
            requested_queries_displacements.push(counter);
            counter += count;
        }

        // Communicate ghost queries, to remove
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

            self.neighbourhood_communicator_u
                .all_to_all_varcount_into(&partition_send, &mut partition_receive);
        }

        // Communicate ghost coordinate data
        {
            let partition_send = Partition::new(
                &available_coordinates,
                &available_coordinates_counts[..],
                &available_coordinates_displacements[..],
            );

            let mut partition_receive = PartitionMut::new(
                &mut requested_coordinates,
                &requested_coordinates_counts[..],
                &requested_coordinates_displacements[..],
            );

            self.neighbourhood_communicator_u
                .all_to_all_varcount_into(&partition_send, &mut partition_receive);
        }

        // Communicate ghost charge data
        {
            let partition_send = Partition::new(
                &available_charges,
                available_charges_counts,
                available_charges_displacements,
            );

            let mut partition_receive = PartitionMut::new(
                &mut requested_charges,
                requested_charges_counts,
                requested_charges_displacements,
            );

            self.neighbourhood_communicator_u
                .all_to_all_varcount_into(&partition_send, &mut partition_receive);
        }

        // Set metadata for Ghost FMM object
        let sort_indices;
        (self.ghost_fmm_u.tree.source_tree, sort_indices) = SingleNodeTree::from_ghost_octants_u(
            self.tree.domain(),
            self.tree.source_tree().total_depth(),
            requested_coordinates,
        );

        let ghost_charges = sort_indices
            .iter()
            .map(|&i| requested_charges[i])
            .collect_vec();

        self.ghost_fmm_u.charges = ghost_charges;
        self.ghost_fmm_u.charge_index_pointer_sources =
            coordinate_index_pointer_single_node(&self.ghost_fmm_u.tree.source_tree);
    }

    fn v_list_exchange(&mut self) {
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
        for (&send_count, &receive_count) in izip!(
            &self.tree.v_list_query.send_counts,
            &self.tree.v_list_query.receive_counts
        ) {
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

        // Available multipole data
        let mut available_multipoles = Vec::new();
        let mut available_multipoles_counts = Vec::new();
        let mut available_multipoles_displacements = Vec::new();

        {
            // Communicate queries
            let partition_send = Partition::new(
                &self.tree.v_list_query.queries,
                neighbourhood_send_counts,
                neighbourhood_send_displacements,
            );

            let mut partition_receive = PartitionMut::new(
                &mut received_queries,
                neighbourhood_receive_counts,
                neighbourhood_receive_displacements,
            );

            self.neighbourhood_communicator_v
                .all_to_all_varcount_into(&partition_send, &mut partition_receive);

            // Filter for locally available queries to send back
            let receive_counts = partition_receive.counts().iter().cloned().collect_vec();
            let receive_displacements = partition_receive.displs().iter().cloned().collect_vec();

            let mut counter = 0;

            // Iterate over received data rank by rank
            for (count, displacement) in izip!(receive_counts, receive_displacements) {
                let l = displacement as usize;
                let r = l + (count as usize);

                // Received queries from this rank
                let received_queries_rank = &received_queries[l..r]; // received queries from this rank

                // Filter for available data corresponding to this request
                let mut available_queries_rank = Vec::new();
                let mut available_multipoles_rank = Vec::new();

                let mut counter_rank = 0i32;

                // Only communicate back queries and associated data if multipole data is found
                for &query in received_queries_rank.iter() {
                    let key = MortonKey::from_morton(query);
                    if let Some(multipole) = self.multipole(&key) {
                        available_queries_rank.push(key.morton);
                        available_multipoles_rank.push(multipole);
                        counter_rank += 1;
                    }
                }

                // Update return buffers
                available_queries.extend(available_queries_rank);
                available_queries_counts.push(counter_rank);
                available_queries_displacements.push(counter);

                let available_multipoles_rank = available_multipoles_rank.concat();
                available_multipoles.extend(available_multipoles_rank);
                available_multipoles_counts
                    .push(counter_rank * (self.n_coeffs_equivalent_surface as i32));
                available_multipoles_displacements
                    .push(counter * (self.n_coeffs_equivalent_surface as i32));

                counter += counter_rank;
            }
        }

        // Communicate expected query sizes
        let mut requested_queries_counts =
            vec![0 as Count; self.neighbourhood_communicator_v.neighbours.len()];
        {
            let send_counts_ = vec![1i32; self.neighbourhood_communicator_v.neighbours.len()];
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

            let recv_counts_ = vec![1i32; self.neighbourhood_communicator_v.neighbours.len()];
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

            self.neighbourhood_communicator_v
                .all_to_all_varcount_into(&partition_send, &mut partition_receive);
        }

        // Create buffers to receive charge and coordinate data
        let total_receive_count_requested_queries =
            requested_queries_counts.iter().sum::<i32>() as usize;
        let total_receive_count_requested_multipoles =
            total_receive_count_requested_queries * self.n_coeffs_equivalent_surface;
        let mut requested_queries = vec![0u64; total_receive_count_requested_queries];
        let mut requested_multipoles =
            vec![Scalar::default(); total_receive_count_requested_multipoles];

        // Calculate counts for requested multipoles
        let mut requested_multipoles_counts = Vec::new();
        for &count in requested_queries_counts.iter() {
            requested_multipoles_counts.push(count * (self.n_coeffs_equivalent_surface as i32));
        }

        // Calculate displacements for query and multipole data from expected count
        let mut requested_queries_displacements = Vec::new();
        let mut requested_multipoles_displacements = Vec::new();

        let mut counter = 0;
        for &count in requested_queries_counts.iter() {
            requested_queries_displacements.push(counter);
            requested_multipoles_displacements
                .push(counter * (self.n_coeffs_equivalent_surface as i32));
            counter += count
        }

        // Communicate ghost morton keys
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

            self.neighbourhood_communicator_v
                .all_to_all_varcount_into(&partition_send, &mut partition_receive);
        }

        // Communicate ghost multipoles
        {
            let partition_send = Partition::new(
                &available_multipoles,
                &available_multipoles_counts[..],
                &available_multipoles_displacements[..],
            );

            let mut partition_receive = PartitionMut::new(
                &mut requested_multipoles,
                &requested_multipoles_counts[..],
                &requested_multipoles_displacements[..],
            );

            self.neighbourhood_communicator_v
                .all_to_all_varcount_into(&partition_send, &mut partition_receive);
        }

        // Store original ordering of received data temporarily
        let ghost_keys = requested_queries
            .into_iter()
            .map(MortonKey::<Scalar::Real>::from_morton)
            .collect_vec();

        // Create a mapping from keys to index as received
        let mut key_to_index = HashMap::new();
        for (i, &key) in ghost_keys.iter().enumerate() {
            key_to_index.insert(key, i);
        }

        // Create a set of all keys, ensure all sibling keys are included too
        // as this is required for efficient kernel application
        let mut ghost_keys_set = HashSet::new();

        for &key in ghost_keys.iter() {
            let siblings = key.siblings();
            ghost_keys_set.extend(siblings);
        }

        let ghost_keys = ghost_keys_set.iter().cloned().collect_vec();

        // Set metadata for V Ghost FMM
        // The tree construction sorts keys by level and then by Morton key
        self.ghost_fmm_v.tree.source_tree = SingleNodeTree::from_ghost_octants_v(
            self.tree.source_tree.global_depth(),
            self.tree.source_tree().total_depth(),
            ghost_keys, // doesn't matter at this point if they're unsorted, as method sorts them into 'tree' order (level, then Morton)
            ghost_keys_set,
        );

        // Allocate ghost multipoles including sibling data, ordering dictated by tree construction
        let mut ghost_multipoles_with_siblings = vec![
            Scalar::default();
            self.ghost_fmm_v.tree.source_tree.keys.len()
                * self.n_coeffs_equivalent_surface
        ];

        // Re-order with zeros added for sibling data that isn't included in the request
        for (new_idx, key) in self.ghost_fmm_v.tree.source_tree.keys.iter().enumerate() {
            if let Some(&old_idx) = key_to_index.get(key) {
                let tmp = &requested_multipoles[old_idx * self.n_coeffs_equivalent_surface
                    ..(old_idx + 1) * self.n_coeffs_equivalent_surface];

                ghost_multipoles_with_siblings[new_idx * self.n_coeffs_equivalent_surface
                    ..(new_idx + 1) * self.n_coeffs_equivalent_surface]
                    .copy_from_slice(tmp);
            }
        }

        self.ghost_fmm_v.multipoles = ghost_multipoles_with_siblings;

        self.ghost_fmm_v.level_multipoles = level_expansion_pointers_single_node(
            &self.ghost_fmm_v.tree.source_tree, // relies on above method call
            &[self.n_coeffs_equivalent_surface],
            1,
            &self.ghost_fmm_v.multipoles,
        );

        self.ghost_fmm_v.level_index_pointer_multipoles = // relies on above method call
            level_index_pointer_single_node(&self.ghost_fmm_v.tree.source_tree);

        // Required to create displacements
        self.ghost_fmm_v.tree.target_tree.keys = self
            .tree
            .target_tree
            .keys
            .iter()
            .cloned()
            .collect_vec()
            .into();

        self.ghost_fmm_v.tree.target_tree.keys_set =
            self.tree.target_tree.keys_set.iter().cloned().collect();

        self.ghost_fmm_v.tree.target_tree.depth = self.tree.target_tree.total_depth();
        self.ghost_fmm_v.tree.target_tree.levels_to_keys =
            self.tree.target_tree.levels_to_keys.clone();
        self.ghost_fmm_v.level_index_pointer_locals = // relies on above method call
            level_index_pointer_single_node(&self.ghost_fmm_v.tree.target_tree);

        // TODO: this method should be more flexible to avoid copy above
        KiFmm::displacements(
            &mut self.ghost_fmm_v,
            Some(self.tree.source_tree.global_depth()),
        );
    }
}

impl<Scalar, Kernel, FieldTranslation> GlobalFmmMetadata for KiFmm<Scalar, Kernel, FieldTranslation>
where
    Scalar: RlstScalar + Default + Float,
    <Scalar as RlstScalar>::Real: RlstScalar + Default + Float,
    Kernel: KernelTrait<T = Scalar> + HomogenousKernel + Default + Send + Sync,
    FieldTranslation: FieldTranslationTrait + Send + Sync,
    Self: DataAccess<Scalar = Scalar, Tree = SingleNodeFmmTree<Scalar::Real>>
        + SourceToTargetTranslationMetadata,
{
    fn global_fmm_multipole_metadata(
        &mut self,
        domain: &Domain<<Self::Scalar as RlstScalar>::Real>,
        depth: u64,
        leaves: Vec<<<<Self as DataAccess>::Tree as SingleFmmTree>::Tree as SingleTree>::Node>,
        multipoles: Vec<Self::Scalar>,
    ) {
        // Find mapping between received leaves and their index
        // for mapping multipole data
        let mut old_leaf_to_index = HashMap::new();
        for (i, &root) in leaves.iter().enumerate() {
            old_leaf_to_index.insert(root, i);
        }

        // Set metadata for tree
        self.tree.source_tree = SingleNodeTree::from_leaves(leaves, domain, depth);

        // Global multipole data with missing siblings if they don't exist globally with zeros for coefficients
        let mut multipoles_with_ancestors = vec![
            Scalar::zero();
            self.tree.source_tree.keys.len()
                * self.n_coeffs_equivalent_surface[0]
        ];

        for (new_idx, key) in self.tree.source_tree.keys.iter().enumerate() {
            if let Some(old_idx) = old_leaf_to_index.get(key) {
                let multipole = &multipoles[old_idx * self.n_coeffs_equivalent_surface[0]
                    ..(old_idx + 1) * self.n_coeffs_equivalent_surface[0]];
                multipoles_with_ancestors[new_idx * self.n_coeffs_equivalent_surface[0]
                    ..(new_idx + 1) * self.n_coeffs_equivalent_surface[0]]
                    .copy_from_slice(multipole);
            }
        }

        // Set metadata for FMM
        self.multipoles = multipoles_with_ancestors;
        self.level_multipoles = level_expansion_pointers_single_node(
            &self.tree.source_tree,
            &self.n_coeffs_equivalent_surface,
            1,
            &self.multipoles,
        );
        self.leaf_multipoles = leaf_expansion_pointers_single_node(
            &self.tree.source_tree,
            &self.n_coeffs_equivalent_surface,
            1,
            &self.multipoles,
        );
        self.level_index_pointer_multipoles =
            level_index_pointer_single_node(&self.tree.source_tree);
    }

    fn global_fmm_local_metadata(
        &mut self,
        domain: &Domain<<Self::Scalar as RlstScalar>::Real>,
        depth: u64,
        leaves: Vec<<<<Self as DataAccess>::Tree as SingleFmmTree>::Tree as SingleTree>::Node>,
    ) {
        // Set metadata for tree
        self.tree.target_tree = SingleNodeTree::from_leaves(leaves, domain, depth);

        let locals = vec![
            Scalar::default();
            self.tree.target_tree.keys.len() * self.n_coeffs_equivalent_surface[0]
        ];

        // Set metadata for FMM
        self.locals = locals;

        self.level_locals = level_expansion_pointers_single_node(
            &self.tree.target_tree,
            &self.n_coeffs_equivalent_surface,
            1,
            &self.locals,
        );

        self.leaf_locals = leaf_expansion_pointers_single_node(
            &self.tree.target_tree,
            &self.n_coeffs_equivalent_surface,
            1,
            &self.locals,
        );

        self.level_index_pointer_locals = level_index_pointer_single_node(&self.tree.target_tree);
    }
}
