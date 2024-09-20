use std::collections::HashMap;
use std::collections::HashSet;

use green_kernels::traits::Kernel as KernelTrait;
use itertools::Itertools;
use mpi::datatype::Partition;
use mpi::datatype::PartitionMut;
use mpi::topology::Communicator;
use mpi::traits::Equivalence;
use mpi::traits::Root;
use mpi::Count;
use mpi::Rank;
use num::Float;
use rlst::RlstScalar;

use crate::traits::field::SourceToTargetData as SourceToTargetDataTrait;
use crate::traits::field::SourceToTargetTranslationMetadata;
use crate::traits::fmm::HomogenousKernel;
use crate::traits::fmm::MultiFmm;
use crate::traits::fmm::SourceToTargetTranslation;
use crate::traits::general::multi_node::GhostExchange;
use crate::traits::general::multi_node::GlobalFmmMetadata;
use crate::traits::tree::MultiFmmTree;
use crate::traits::tree::MultiTree;
use crate::traits::tree::SingleFmmTree;
use crate::traits::tree::SingleTree;
use crate::tree::types::Domain;
use crate::tree::types::Point;
use crate::tree::types::{MortonKey, MortonKeys};
use crate::tree::SingleNodeTree;
use crate::SingleFmm;
use crate::SingleNodeFmmTree;

use super::helpers::single_node::coordinate_index_pointer_single_node;
use super::helpers::single_node::leaf_expansion_pointers_single_node;
use super::helpers::single_node::level_expansion_pointers_single_node;
use super::helpers::single_node::level_index_pointer_single_node;
use super::types::{KiFmmMulti, Layout};
use super::KiFmm;

impl<Scalar, Kernel, SourceToTargetData> GhostExchange
    for KiFmmMulti<Scalar, Kernel, SourceToTargetData>
where
    Scalar: RlstScalar + Default + Equivalence + Float,
    <Scalar as RlstScalar>::Real: RlstScalar + Default + Equivalence + Float,
    Kernel: KernelTrait<T = Scalar> + HomogenousKernel + Default + Send + Sync,
    SourceToTargetData: SourceToTargetDataTrait + Send + Sync,
    Self: SourceToTargetTranslation,
    KiFmm<Scalar, Kernel, SourceToTargetData>: SingleFmm<Scalar = Scalar, Tree = SingleNodeFmmTree<Scalar::Real>>
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
            // Gather multipole data from root processes on all ranks
            let n_root_multipoles = self.tree.source_tree().n_trees();
            let mut global_multipoles_counts = vec![0 as Count; size as usize];
            root_process.gather_into_root(&n_root_multipoles, &mut global_multipoles_counts);

            // Calculate displacements and counts for associated morton keys
            let mut global_multipoles_displacements = Vec::new();
            let mut displacement = 0;
            for &count in global_multipoles_counts.iter() {
                global_multipoles_displacements.push(displacement);
                displacement += count;
            }

            // Allocate memory for locally contained data
            let mut multipole_roots = Vec::new();
            let mut multipoles =
                vec![Scalar::default(); n_root_multipoles * self.n_coeffs_equivalent_surface];

            for (i, tree) in self.tree.source_tree().trees().iter().enumerate() {
                let tmp = self.multipole(&tree.root()).unwrap();
                multipoles[i * self.n_coeffs_equivalent_surface
                    ..(i + 1) * self.n_coeffs_equivalent_surface]
                    .copy_from_slice(tmp);
                multipole_roots.push(tree.root())
            }

            // Calculate displacements and counts for multipole data
            let global_multipoles_bufs_counts = global_multipoles_counts
                .iter()
                .map(|c| c * self.n_coeffs_equivalent_surface as i32)
                .collect_vec();
            let mut global_multipoles_bufs_displacements = Vec::new();
            let mut displacement = 0;
            for &count in global_multipoles_bufs_counts.iter() {
                global_multipoles_bufs_displacements.push(displacement);
                displacement += count
            }

            // Allocate memory to store received multipoles
            let n = global_multipoles_bufs_counts.iter().sum::<i32>() as usize;
            let mut global_multipoles =
                vec![Scalar::default(); n * self.n_coeffs_equivalent_surface];

            let mut partition = PartitionMut::new(
                &mut global_multipoles,
                &global_multipoles_bufs_counts[..],
                &global_multipoles_bufs_displacements[..],
            );
            root_process.gather_varcount_into_root(&multipoles, &mut partition);

            let n = global_multipoles_counts.iter().sum::<i32>();
            let mut global_multipole_roots = vec![MortonKey::<Scalar::Real>::default(); n as usize];
            let mut partition = PartitionMut::new(
                &mut global_multipole_roots,
                &global_multipoles_counts[..],
                &global_multipoles_displacements[..],
            );

            root_process.gather_varcount_into_root(&multipole_roots, &mut partition);

            // Find mapping between received keys and their index
            let mut global_keys_to_index = HashMap::new();
            for (i, &global_root) in global_multipole_roots.iter().enumerate() {
                global_keys_to_index.insert(global_root, i);
            }

            // Need to insert sibling and their ancestors data if its missing in multipole data received
            // also need to ensure that siblings of ancestors are included
            let mut global_leaves_set: HashSet<_> =
                global_multipole_roots.iter().cloned().collect();
            let mut global_keys_set = HashSet::new();
            for leaf in global_multipole_roots.iter() {
                let siblings = leaf.siblings();
                let ancestors_siblings = leaf
                    .ancestors()
                    .iter()
                    .flat_map(|a| if a.level() != 0 { a.siblings() } else { vec![] })
                    .collect_vec();

                for &key in siblings.iter() {
                    global_keys_set.insert(key);
                    global_leaves_set.insert(key);
                }

                for &key in ancestors_siblings.iter() {
                    global_keys_set.insert(key);
                }
            }

            let mut global_keys = global_keys_set.iter().cloned().collect_vec();
            global_keys.sort(); // Store in Morton order

            let mut global_leaves = global_leaves_set.iter().cloned().collect_vec();
            global_leaves.sort(); // Store in Morton order

            // Global multipole data with missing siblings if they don't exist globally with zeros for coefficients
            let mut global_multipoles_with_ancestors =
                vec![Scalar::zero(); global_keys.len() * self.n_coeffs_equivalent_surface];

            for (new_idx, key) in global_keys.iter().enumerate() {
                if let Some(old_idx) = global_keys_to_index.get(key) {
                    let multipole = &global_multipoles[old_idx * self.n_coeffs_equivalent_surface
                        ..(old_idx + 1) * self.n_coeffs_equivalent_surface];
                    global_multipoles_with_ancestors[new_idx * self.n_coeffs_equivalent_surface
                        ..(new_idx + 1) * self.n_coeffs_equivalent_surface]
                        .copy_from_slice(multipole);
                }
            }

            // Need to set metadata for global FMM
            self.global_fmm.global_fmm_multipole_metadata(
                self.tree.domain(),
                self.tree.source_tree().global_depth(),
                global_keys,
                global_keys_set,
                global_leaves,
                global_leaves_set,
                global_multipoles_with_ancestors,
            );
        } else {
            // 1. Send multipole roots, if they exist, and associated coeffient data
            let n_root_multipoles = self.tree.source_tree().n_trees();
            root_process.gather_into(&n_root_multipoles);

            // 1.1 Create buffers of multipole data to be sent
            let mut multipole_roots = Vec::new();
            let mut multipoles =
                vec![Scalar::default(); n_root_multipoles * self.n_coeffs_equivalent_surface];
            for (i, tree) in self.tree.source_tree().trees().iter().enumerate() {
                let tmp = self.multipole(&tree.root()).unwrap();
                multipoles[i * self.n_coeffs_equivalent_surface
                    ..(i + 1) * self.n_coeffs_equivalent_surface]
                    .copy_from_slice(tmp);
                multipole_roots.push(tree.root())
            }

            root_process.gather_varcount_into(&multipoles);
            root_process.gather_varcount_into(&multipole_roots);
        }

        // Gather local data
        if rank == root_rank {
            let n_root_locals = self.tree.target_tree().n_trees();
            let mut global_locals_counts = vec![0 as Count; size as usize];
            root_process.gather_into_root(&n_root_locals, &mut global_locals_counts);

            // Calculate displacements and counts for associated morton keys
            let mut global_locals_displacements = Vec::new();
            let mut displacement = 0;
            for &count in global_locals_counts.iter() {
                global_locals_displacements.push(displacement);
                displacement += count;
            }

            // Allocate memory for locally contained data
            let mut local_roots = Vec::new();

            for tree in self.tree.target_tree().trees().iter() {
                local_roots.push(tree.root())
            }

            // Store origin ranks of each requested local
            let n = global_locals_counts.iter().sum::<i32>();
            let mut global_local_roots = vec![MortonKey::<Scalar::Real>::default(); n as usize];
            let mut partition = PartitionMut::new(
                &mut global_local_roots,
                &global_locals_counts[..],
                &global_locals_displacements[..],
            );

            root_process.gather_varcount_into_root(&local_roots, &mut partition);

            let mut global_roots_ranks = Vec::new();
            for (rank, &count) in global_locals_counts.iter().enumerate() {
                for _ in 0..(count as usize) {
                    global_roots_ranks.push(rank as Rank)
                }
            }

            // Need to insert sibling and ancestor data
            let mut global_leaves_set: HashSet<_> = global_local_roots.iter().cloned().collect();
            let mut global_keys_set = HashSet::new();
            for leaf in global_local_roots.iter() {
                let siblings = leaf.siblings();
                let ancestors_siblings = leaf
                    .ancestors()
                    .iter()
                    .flat_map(|a| if a.level() != 0 { a.siblings() } else { vec![] })
                    .collect_vec();

                for &key in siblings.iter() {
                    global_keys_set.insert(key);
                    global_leaves_set.insert(key);
                }

                for &key in ancestors_siblings.iter() {
                    global_keys_set.insert(key);
                }
            }

            let mut global_keys = global_keys_set.iter().cloned().collect_vec();
            global_keys.sort(); // Store in Morton order

            let mut global_leaves = global_leaves_set.iter().cloned().collect_vec();
            global_leaves.sort(); // Store in Morton order

            // Global locals data with missing siblings if they don't exist globally with zeros for coefficients
            let global_locals_with_ancestors =
                vec![Scalar::zero(); global_keys.len() * self.n_coeffs_equivalent_surface];

            // Set metadata
            self.global_fmm.global_fmm_local_metadata(
                self.tree.domain(),
                self.tree.target_tree().global_depth(),
                global_keys,
                global_keys_set,
                global_leaves,
                global_leaves_set,
                global_locals_with_ancestors,
            );

            // Local roots, which need to be broadcasted back to
            self.local_roots = global_local_roots;
            self.local_roots_counts = global_locals_counts;
            self.local_roots_displacements = global_locals_displacements;
            self.local_roots_ranks = global_roots_ranks;
        } else {
            let n_root_locals = self.tree.target_tree().n_trees();
            root_process.gather_into(&n_root_locals);

            let mut local_roots = Vec::new();
            for tree in self.tree.target_tree().trees().iter() {
                local_roots.push(tree.root())
            }

            root_process.gather_varcount_into(&local_roots);
        }

        // Calculate displacements required for M2L
        if rank == root_rank {
            self.global_fmm.displacements()
        }
    }

    fn scatter_global_fmm_from_root(&mut self) {
        let rank = self.communicator.rank();
        let nroots = self.tree.target_tree.trees.len();
        let receive_buffer_size = nroots * self.n_coeffs_equivalent_surface;
        let mut receive_buffer = vec![Scalar::default(); receive_buffer_size];

        // Nominated rank chosen to run global upward pass
        let root_rank = 0;
        let root_process = self.communicator.process_at_rank(root_rank);

        if rank == root_rank {
            // Lookup local data to be sent back from global FMM
            let send_buffer_size = self.local_roots.len() * self.n_coeffs_equivalent_surface;
            let mut send_buffer = vec![Scalar::default(); send_buffer_size];

            let mut root_idx = 0;
            for root in self.local_roots.iter() {
                if let Some(local) = self.global_fmm.local(root) {
                    send_buffer[root_idx * self.n_coeffs_equivalent_surface
                        ..(root_idx + 1) * self.n_coeffs_equivalent_surface]
                        .copy_from_slice(local);
                    root_idx += 1;
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

            let partition = Partition::new(&send_buffer, counts, &displacements[..]);

            root_process.scatter_varcount_into_root(&partition, &mut receive_buffer);
        } else {
            root_process.scatter_varcount_into(&mut receive_buffer);
        }
    }

    fn u_list_exchange(&mut self) {
        // Begin by calculating the receive counts, this requires an all to all over the neighbourhood communicators
        let size = self.neighbourhood_communicator_u.size();

        let mut send_counts = vec![0 as Count; size as usize];

        for (global_rank, &send_count) in self.tree.u_list_query.send_counts.iter().enumerate() {
            if let Some(local_rank) = self
                .neighbourhood_communicator_v
                .global_to_local_rank(global_rank as Rank)
            {
                send_counts[local_rank as usize] = send_count;
            }
        }

        let mut receive_counts = vec![0i32; size as usize];
        self.neighbourhood_communicator_u
            .all_to_all_into(&send_counts, &mut receive_counts);

        // Now can create displacements
        let mut neighbourhood_send_displacements = Vec::new();
        let mut neighbourhood_receive_displacements = Vec::new();

        let mut send_counter = 0;
        let mut receive_counter = 0;

        for (send_count, receive_count) in send_counts.iter().zip(receive_counts.iter()) {
            neighbourhood_send_displacements.push(send_counter);
            neighbourhood_receive_displacements.push(receive_counter);
            send_counter += send_count;
            receive_counter += receive_count;
        }

        let total_receive_count = receive_counter;

        // Assign origin ranks to all received queries
        let mut receive_counter = 0;
        let mut received_queries_source_ranks = vec![0i32; total_receive_count as usize];
        for (i, &receive_count) in receive_counts.iter().enumerate() {
            let curr_receive_counter = receive_counter;
            receive_counter += receive_count as usize;

            let rank = self.neighbourhood_communicator_u.neighbours[i];
            let tmp = vec![rank as i32; receive_count as usize];
            received_queries_source_ranks[curr_receive_counter..receive_counter]
                .copy_from_slice(tmp.as_slice());
        }

        // Communicate queries for coordinate data
        let mut received_queries = vec![0u64; total_receive_count as usize];
        {
            let partition_send = Partition::new(
                &self.tree.u_list_query.queries,
                send_counts,
                neighbourhood_send_displacements,
            );
            let mut partition_receive = PartitionMut::new(
                &mut received_queries,
                receive_counts,
                &neighbourhood_receive_displacements[..],
            );
            self.neighbourhood_communicator_u
                .all_to_all_varcount_into(&partition_send, &mut partition_receive);
        }

        // Now have to check for locally contained keys from received queries
        let mut available_queries_source_ranks = Vec::new();
        let mut available_queries_sizes = Vec::new();
        let mut available_queries_bufs = Vec::new();

        let mut available_queries_charges_source_ranks = Vec::new();
        let mut available_queries_charges_sizes = Vec::new();
        let mut available_queries_charges_bufs = Vec::new();

        for (&query, &source_rank) in received_queries
            .iter()
            .zip(received_queries_source_ranks.iter())
        {
            let key = MortonKey::from_morton(query);

            // Lookup corresponding source tree at this rank

            // Only send back if it contains coordinate data
            if let Some(coordinates) = self.tree.source_tree.coordinates(&key) {
                available_queries_source_ranks.push(source_rank);
                available_queries_sizes.push(coordinates.len() as i32);
                available_queries_bufs.push(coordinates);

                // Lookup associated charge data
                let &index = self.tree.source_tree.leaf_index(&key).unwrap();
                let index_pointer = &self.charge_index_pointer_sources[index];
                let charges = &self.charges[index_pointer.0..index_pointer.1];
                available_queries_charges_source_ranks.push(source_rank);
                available_queries_charges_sizes.push(charges.len() as i32);
                available_queries_charges_bufs.push(charges);
            }
        }

        // Calculate send counts of available queries to send back
        let mut received_queries_sizes_send_counts_ = HashMap::new(); // particle data counts
        for (i, global_rank) in available_queries_source_ranks.iter().enumerate() {
            *received_queries_sizes_send_counts_
                .entry(*global_rank)
                .or_insert(0) += available_queries_sizes[i];
        }

        let mut received_queries_charges_sizes_send_counts_ = HashMap::new(); // particle data counts
        for (i, global_rank) in available_queries_charges_source_ranks.iter().enumerate() {
            *received_queries_charges_sizes_send_counts_
                .entry(*global_rank)
                .or_insert(0) += available_queries_charges_sizes[i];
        }

        let mut received_queries_sizes_send_counts = vec![0i32; size as usize];

        for (&global_rank, &send_count) in received_queries_sizes_send_counts_.iter() {
            if let Some(local_rank) = self
                .neighbourhood_communicator_v
                .global_to_local_rank(global_rank)
            {
                received_queries_sizes_send_counts[local_rank as usize] = send_count;
            }
        }

        let mut received_queries_charges_sizes_send_counts = vec![0i32; size as usize];

        for (&global_rank, &send_count) in received_queries_charges_sizes_send_counts_.iter() {
            if let Some(local_rank) = self
                .neighbourhood_communicator_v
                .global_to_local_rank(global_rank)
            {
                received_queries_charges_sizes_send_counts[local_rank as usize] = send_count;
            }
        }

        // Calculate counts for requested queries associated coordinate data
        let mut requested_queries_sizes_receive_counts = vec![0i32; size as usize];
        self.neighbourhood_communicator_u.all_to_all_into(
            &received_queries_sizes_send_counts,
            &mut requested_queries_sizes_receive_counts,
        );

        let mut requested_queries_charges_sizes_receive_counts = vec![0i32; size as usize];
        self.neighbourhood_communicator_u.all_to_all_into(
            &received_queries_charges_sizes_send_counts,
            &mut requested_queries_charges_sizes_receive_counts,
        );

        // Can now create displacements for coordinate data and charge data
        let mut received_queries_sizes_send_displacements = Vec::new();
        let mut requested_queries_sizes_receive_displacements = Vec::new();
        let mut received_queries_charges_sizes_send_displacements = Vec::new();
        let mut requested_queries_charges_sizes_receive_displacements = Vec::new();

        let mut send_counter = 0;
        let mut receive_counter = 0;

        for (send_count, receive_count) in received_queries_sizes_send_counts
            .iter()
            .zip(requested_queries_sizes_receive_counts.iter())
        {
            received_queries_sizes_send_displacements.push(send_counter);
            requested_queries_sizes_receive_displacements.push(receive_counter);
            send_counter += send_count;
            receive_counter += receive_count;
        }

        let total_receive_sizes_count = receive_counter;
        let total_send_sizes_count = send_counter;

        for (send_count, receive_count) in received_queries_charges_sizes_send_counts
            .iter()
            .zip(requested_queries_charges_sizes_receive_counts.iter())
        {
            received_queries_charges_sizes_send_displacements.push(send_counter);
            requested_queries_charges_sizes_receive_displacements.push(receive_counter);
            send_counter += send_count;
            receive_counter += receive_count;
        }

        let total_receive_charges_sizes_count = receive_counter;
        let total_send_charges_sizes_count = send_counter;

        // Allocate buffers to store requested coordinate and charge data to be sent
        let mut send_coordinates = vec![Scalar::Real::default(); total_send_sizes_count as usize];
        let mut displacement = 0;
        for buf in available_queries_bufs {
            let new_displacement = displacement + buf.len();
            send_coordinates[displacement..new_displacement].copy_from_slice(buf);
            displacement = new_displacement;
        }

        let mut send_charges = vec![Scalar::default(); total_send_charges_sizes_count as usize];
        let mut displacement = 0;
        for buf in available_queries_charges_bufs {
            let new_displacement = displacement + buf.len();
            send_charges[displacement..new_displacement].copy_from_slice(buf);
            displacement = new_displacement;
        }

        // Allocate buffers to store requested coordinate and charge data at this proc
        let mut ghost_coordinates =
            vec![Scalar::Real::default(); total_receive_sizes_count as usize];

        let mut ghost_charges = vec![Scalar::default(); total_receive_charges_sizes_count as usize];

        // Communicate ghost coordinate data
        {
            let partition_send = Partition::new(
                &send_coordinates,
                received_queries_sizes_send_counts,
                received_queries_sizes_send_displacements,
            );
            let mut partition_receive = PartitionMut::new(
                &mut ghost_coordinates,
                &requested_queries_sizes_receive_counts[..],
                &requested_queries_sizes_receive_displacements[..],
            );
            self.neighbourhood_communicator_u
                .all_to_all_varcount_into(&partition_send, &mut partition_receive);
        }

        // Communicate ghost charge data
        {
            let partition_send = Partition::new(
                &send_charges,
                received_queries_charges_sizes_send_counts,
                received_queries_charges_sizes_send_displacements,
            );
            let mut partition_receive = PartitionMut::new(
                &mut ghost_charges,
                &requested_queries_charges_sizes_receive_counts[..],
                &requested_queries_charges_sizes_receive_displacements[..],
            );
            self.neighbourhood_communicator_u
                .all_to_all_varcount_into(&partition_send, &mut partition_receive);
        }

        // Set tree
        let sort_indices;
        (self.ghost_tree_u, sort_indices) = SingleNodeTree::from_ghost_octants_u(
            &self.tree.domain(),
            self.tree.source_tree().total_depth(),
            ghost_coordinates,
        );

        // Set metadata, TODO: real charges need to be set eventually
        let ghost_charges = sort_indices
            .iter()
            .map(|&i| ghost_charges[i].clone())
            .collect_vec();
        self.ghost_charges = ghost_charges;
        self.charge_index_pointer_ghost_sources =
            coordinate_index_pointer_single_node(&self.ghost_tree_u);
    }

    fn v_list_exchange(&mut self) {
        // Begin by calculating the receive counts, this requires an all to all over the neighbourhood communicators
        let size = self.neighbourhood_communicator_v.size();

        let mut send_counts = vec![0 as Count; size as usize];

        for (global_rank, &send_count) in self.tree.v_list_query.send_counts.iter().enumerate() {
            if let Some(local_rank) = self
                .neighbourhood_communicator_v
                .global_to_local_rank(global_rank as Rank)
            {
                send_counts[local_rank as usize] = send_count;
            }
        }

        let mut receive_counts = vec![0 as Rank; size as usize];
        self.neighbourhood_communicator_v
            .all_to_all_into(&send_counts, &mut receive_counts);

        // Now can calculate displacements
        let mut send_displacements = Vec::new();
        let mut receive_displacements = Vec::new();
        let mut send_counter = 0;
        let mut receive_counter = 0;

        for (send_count, receive_count) in send_counts.iter().zip(receive_counts.iter()) {
            send_displacements.push(send_counter);
            receive_displacements.push(receive_counter);
            send_counter += send_count;
            receive_counter += receive_count;
        }

        let total_receive_count = receive_counter;

        // Assign origin ranks to all received queries
        let mut receive_counter = 0;
        let mut received_queries_source_ranks = vec![0i32; total_receive_count as usize];
        for (i, &receive_count) in receive_counts.iter().enumerate() {
            let curr_receive_counter = receive_counter;
            receive_counter += receive_count as usize;

            let rank = self.neighbourhood_communicator_v.neighbours[i];
            let tmp = vec![rank as i32; receive_count as usize];
            received_queries_source_ranks[curr_receive_counter..receive_counter]
                .copy_from_slice(tmp.as_slice());
        }

        // Communicate queries for multipole data
        let mut received_queries = vec![0u64; total_receive_count as usize];
        {
            let partition_send = Partition::new(
                &self.tree.v_list_query.queries,
                send_counts,
                send_displacements,
            );
            let mut partition_receive = PartitionMut::new(
                &mut received_queries,
                receive_counts,
                &receive_displacements[..],
            );

            self.neighbourhood_communicator_v
                .all_to_all_varcount_into(&partition_send, &mut partition_receive);
        }

        // Now have to check for locally contained keys from received queries
        let mut available_queries = Vec::new();
        let mut available_queries_source_ranks = Vec::new();
        for (&query, &source_rank) in received_queries
            .iter()
            .zip(received_queries_source_ranks.iter())
        {
            let key = MortonKey::from_morton(query);
            if self.tree.source_tree.keys_set.contains(&key) {
                available_queries.push(key.morton);
                available_queries_source_ranks.push(source_rank);
            }
        }

        // Calculate send counts of available queries to send back
        let mut received_queries_send_counts_ = HashMap::new();
        for global_rank in available_queries_source_ranks.iter() {
            *received_queries_send_counts_
                .entry(*global_rank)
                .or_insert(0) += 1;
        }

        let mut received_queries_send_counts = vec![0i32; size as usize];

        for (&global_rank, &send_count) in received_queries_send_counts_.iter() {
            if let Some(local_rank) = self
                .neighbourhood_communicator_v
                .global_to_local_rank(global_rank)
            {
                received_queries_send_counts[local_rank as usize] = send_count;
            }
        }

        // Calculate counts for requested queries
        let mut requested_queries_receive_counts = vec![0i32; size as usize];
        self.neighbourhood_communicator_v.all_to_all_into(
            &received_queries_send_counts,
            &mut requested_queries_receive_counts,
        );

        // Now can create displacements for morton data and multipole data
        let mut received_queries_send_displacements = Vec::new();
        let mut requested_queries_receive_displacements = Vec::new();
        let mut send_multipoles_counts = Vec::new();
        let mut receive_multipoles_counts = Vec::new();
        let mut send_multipoles_displacements = Vec::new();
        let mut receive_multipoles_displacements = Vec::new();

        let mut send_counter = 0;
        let mut receive_counter = 0;

        for (send_count, receive_count) in received_queries_send_counts
            .iter()
            .zip(requested_queries_receive_counts.iter())
        {
            received_queries_send_displacements.push(send_counter);
            requested_queries_receive_displacements.push(receive_counter);

            send_multipoles_counts.push(send_count * self.n_coeffs_equivalent_surface as i32);
            receive_multipoles_counts.push(receive_count * self.n_coeffs_equivalent_surface as i32);
            send_multipoles_displacements
                .push(send_counter * self.n_coeffs_equivalent_surface as i32);
            receive_multipoles_displacements
                .push(send_counter * self.n_coeffs_equivalent_surface as i32);

            send_counter += send_count;
            receive_counter += receive_count;
        }

        let total_receive_count = receive_counter;
        let total_send_count = send_counter;

        // Create partition and get back requests
        let mut requested_queries = vec![0u64; total_receive_count as usize];
        {
            let partition_send = Partition::new(
                &available_queries,
                received_queries_send_counts,
                received_queries_send_displacements,
            );
            let mut partition_receive = PartitionMut::new(
                &mut requested_queries,
                requested_queries_receive_counts,
                &requested_queries_receive_displacements[..],
            );
            self.neighbourhood_communicator_v
                .all_to_all_varcount_into(&partition_send, &mut partition_receive);
        }

        // Allocate buffers to store requested multipole data
        let mut send_multipoles =
            vec![Scalar::default(); (total_send_count as usize) * self.n_coeffs_equivalent_surface];
        let mut multipole_idx = 0;
        for &query in available_queries.iter() {
            let key = MortonKey::from_morton(query);

            let multipole = self.multipole(&key).unwrap();

            send_multipoles[multipole_idx * self.n_coeffs_equivalent_surface
                ..(multipole_idx + 1) * self.n_coeffs_equivalent_surface]
                .copy_from_slice(multipole);

            multipole_idx += 1;
        }

        // Allocate buffers to store requested multipole data
        let mut ghost_multipoles = vec![
            Scalar::default();
            (total_receive_count as usize)
                * self.n_coeffs_equivalent_surface
        ];

        // Communicate requested multipole data
        {
            let partition_send = Partition::new(
                &send_multipoles,
                send_multipoles_counts,
                send_multipoles_displacements,
            );
            let mut partition_receive = PartitionMut::new(
                &mut ghost_multipoles,
                receive_multipoles_counts,
                &receive_multipoles_displacements[..],
            );
            self.neighbourhood_communicator_v
                .all_to_all_varcount_into(&partition_send, &mut partition_receive);
        }

        let ghost_keys = requested_queries
            .into_iter()
            .map(|q| MortonKey::<Scalar::Real>::from_morton(q))
            .collect_vec();

        // Create a mapping from keys to index as received
        let mut key_to_index = HashMap::new();
        for (i, &key) in ghost_keys.iter().enumerate() {
            key_to_index.insert(key, i);
        }

        let mut ghost_keys_set = HashSet::new();

        // Ensure sibling data is included
        for &key in ghost_keys.iter() {
            let siblings = key.siblings();
            ghost_keys_set.extend(siblings);
        }

        // Create ghost keys
        let ghost_keys = ghost_keys_set.iter().cloned().collect_vec();

        // Set tree
        self.ghost_tree_v = SingleNodeTree::from_ghost_octants_v(
            self.tree.source_tree().total_depth(),
            ghost_keys,
            ghost_keys_set,
        );

        // Allocate ghost multipoles including sibling data, ordering dictated by tree
        let mut ghost_multipoles_with_siblings = vec![
            Scalar::default();
            self.ghost_tree_v.keys.len()
                * self.n_coeffs_equivalent_surface
        ];

        for (new_idx, key) in self.ghost_tree_v.keys.iter().enumerate() {
            if let Some(&old_idx) = key_to_index.get(key) {
                let tmp = &ghost_multipoles[old_idx * self.n_coeffs_equivalent_surface
                    ..(old_idx + 1) * self.n_coeffs_equivalent_surface];

                ghost_multipoles_with_siblings[new_idx * self.n_coeffs_equivalent_surface
                    ..(new_idx + 1) * self.n_coeffs_equivalent_surface]
                    .copy_from_slice(tmp);
            }
        }

        // Set metadata
        self.ghost_multipoles = ghost_multipoles_with_siblings;
        let mut tmp = level_expansion_pointers_single_node(
            &self.ghost_tree_v,
            &[self.n_coeffs_equivalent_surface],
            1,
            &self.ghost_multipoles,
        );
        self.ghost_level_multipoles = std::mem::replace(&mut tmp[0], Vec::new());
    }
}

impl<Scalar, Kernel, SourceToTargetData> GlobalFmmMetadata
    for KiFmm<Scalar, Kernel, SourceToTargetData>
where
    Scalar: RlstScalar + Default + Float,
    <Scalar as RlstScalar>::Real: RlstScalar + Default + Float,
    Kernel: KernelTrait<T = Scalar> + HomogenousKernel + Default + Send + Sync,
    SourceToTargetData: SourceToTargetDataTrait + Send + Sync,
    Self: SingleFmm<Scalar = Scalar, Tree = SingleNodeFmmTree<Scalar::Real>>
        + SourceToTargetTranslationMetadata,
{
    fn global_fmm_multipole_metadata(
        &mut self,
        domain: &Domain<<Self::Scalar as RlstScalar>::Real>,
        depth: u64,
        keys: Vec<<<<Self as SingleFmm>::Tree as SingleFmmTree>::Tree as SingleTree>::Node>,
        keys_set: HashSet<<<<Self as SingleFmm>::Tree as SingleFmmTree>::Tree as SingleTree>::Node>,
        leaves: Vec<<<<Self as SingleFmm>::Tree as SingleFmmTree>::Tree as SingleTree>::Node>,
        leaves_set: HashSet<
            <<<Self as SingleFmm>::Tree as SingleFmmTree>::Tree as SingleTree>::Node,
        >,
        multipoles: Vec<Self::Scalar>,
    ) {
        // Set metadata for tree
        self.tree.source_tree =
            SingleNodeTree::from_keys(keys, keys_set, leaves, leaves_set, None, domain, depth);

        // Set metadata for FMM
        self.multipoles = multipoles;
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
        keys: Vec<<<<Self as SingleFmm>::Tree as SingleFmmTree>::Tree as SingleTree>::Node>,
        keys_set: HashSet<<<<Self as SingleFmm>::Tree as SingleFmmTree>::Tree as SingleTree>::Node>,
        leaves: Vec<<<<Self as SingleFmm>::Tree as SingleFmmTree>::Tree as SingleTree>::Node>,
        leaves_set: HashSet<
            <<<Self as SingleFmm>::Tree as SingleFmmTree>::Tree as SingleTree>::Node,
        >,
        locals: Vec<Self::Scalar>,
    ) {
        // Set metadata for tree
        self.tree.target_tree =
            SingleNodeTree::from_keys(keys, keys_set, leaves, leaves_set, None, domain, depth);

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
