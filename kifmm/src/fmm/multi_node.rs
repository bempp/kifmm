//! Multi Node FMM
//! Single Node FMM
use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
    time::Instant,
};

use green_kernels::traits::Kernel as KernelTrait;

use itertools::Itertools;
use mpi::{
    collective::SystemOperation,
    ffi::RSMPI_SUM,
    raw::AsRaw,
    request::WaitGuard,
    topology::{Color, SimpleCommunicator},
    traits::{Communicator, Destination, Equivalence, Group, Root, Source},
};
use num::Float;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use rlst::RlstScalar;

use mpi::collective::CommunicatorCollectives;

use crate::{
    fmm::{
        helpers::sparse_point_to_point,
        types::{FmmEvalType, KiFmm},
    },
    traits::{
        field::SourceToTargetData as SourceToTargetDataTrait,
        fmm::{
            FmmOperatorData, GhostExchange, HomogenousKernel, MultiNodeFmm,
            SourceToTargetTranslation, SourceTranslation, TargetTranslation,
        },
        tree::SingleNodeTreeTrait,
        types::{FmmError, FmmOperatorTime, FmmOperatorType},
    },
    tree::{
        helpers::all_to_allv_sparse,
        types::{GhostTreeU, GhostTreeV, MortonKey},
    },
    Fmm, MultiNodeFmmTree,
};

use super::{
    helpers::{expected_queries, sparse_point_to_point_v},
    types::{CommunicationMode, IndexPointer, KiFmmMultiNode},
};

impl<Scalar, Kernel, SourceToTargetData> MultiNodeFmm
    for KiFmmMultiNode<Scalar, Kernel, SourceToTargetData>
where
    Scalar: RlstScalar + Default + Equivalence + Float,
    Kernel: KernelTrait<T = Scalar> + HomogenousKernel + Default + Send + Sync,
    SourceToTargetData: SourceToTargetDataTrait + Send + Sync,
    <Scalar as RlstScalar>::Real: Default + Equivalence + Float,
    Self: SourceTranslation + GhostExchange,
{
    type Scalar = Scalar;
    type Kernel = Kernel;
    type Tree = MultiNodeFmmTree<Scalar::Real, SimpleCommunicator>;

    fn dim(&self) -> usize {
        3
    }

    fn evaluate(&mut self, timed: bool) -> Result<(), FmmError> {
        // Run upward pass on local trees, up to local depth
        {
            let s = Instant::now();
            self.p2m()?;
            self.times
                .push(FmmOperatorTime::from_instant(FmmOperatorType::P2M, s));

            let local_depth = self.tree.source_tree.local_depth;
            let global_depth = self.tree.source_tree.global_depth;
            for level in ((global_depth + 1)..=(local_depth + global_depth)).rev() {
                let s = Instant::now();
                self.m2m(level)?;
                self.times.push(FmmOperatorTime::from_instant(
                    FmmOperatorType::M2M(level),
                    s,
                ));
            }
        }

        // At this point the exchange needs to happen of multipole data
        {
            self.v_list_exchange()
        }

        // Build up local data structures containing ghost data

        // Gather root multipoles at nominated node
        self.gather_root_multipoles();

        // Now can proceed with remainder of the upward pass on chosen node, and some of the downward pass
        {}

        // Scatter root locals back to local trees
        self.scatter_root_locals();

        // Now remainder of downward pass can happen in parallel on each process, similar to how I've written the local upward passes
        // new kernels have to reflect ghost data, and potentially multiple local source trees

        Ok(())
    }

    fn kernel(&self) -> &Self::Kernel {
        &self.kernel
    }

    fn check_surface_order(&self, level: u64) -> usize {
        self.check_surface_order
    }

    fn equivalent_surface_order(&self, level: u64) -> usize {
        self.equivalent_surface_order
    }

    fn ncoeffs_check_surface(&self, level: u64) -> usize {
        self.ncoeffs_check_surface
    }

    fn ncoeffs_equivalent_surface(&self, level: u64) -> usize {
        self.ncoeffs_equivalent_surface
    }

    fn multipole(
        &self,
        fmm_idx: usize,
        key: &<<<Self::Tree as crate::traits::tree::MultiNodeFmmTreeTrait>::Tree as crate::traits::tree::MultiNodeTreeTrait>::Tree as crate::traits::tree::SingleNodeTreeTrait>::Node,
    ) -> Option<&[Self::Scalar]> {
        if fmm_idx < self.nsource_trees {
            if let Some(&key_idx) = self.tree.source_tree.trees[fmm_idx].level_index(key) {
                let multipole_ptr = self.level_multipoles[fmm_idx][key.level() as usize][key_idx];
                unsafe {
                    Some(std::slice::from_raw_parts(
                        multipole_ptr.raw,
                        self.ncoeffs_equivalent_surface,
                    ))
                }
            } else {
                None
            }
        } else {
            None
        }
    }

    fn multipoles(&self, fmm_idx: usize, level: u64) -> Option<&[Self::Scalar]> {
        if fmm_idx < self.nsource_trees {
            let multipole_ptr = &self.level_multipoles[fmm_idx][level as usize][0];
            let nsources = self.tree.source_tree.trees[fmm_idx].n_keys(level).unwrap();
            unsafe {
                Some(std::slice::from_raw_parts(
                    multipole_ptr.raw,
                    self.ncoeffs_equivalent_surface * nsources,
                ))
            }
        } else {
            None
        }
    }
}

impl<Scalar, Kernel, SourceToTargetData> GhostExchange
    for KiFmmMultiNode<Scalar, Kernel, SourceToTargetData>
where
    Scalar: RlstScalar + Default + Equivalence + Float,
    Kernel: KernelTrait<T = Scalar> + HomogenousKernel + Default + Send + Sync,
    SourceToTargetData: SourceToTargetDataTrait + Send + Sync,
    <Scalar as RlstScalar>::Real: Default + Equivalence + Float,
    Self: SourceTranslation,
{
    fn gather_root_multipoles(&mut self) {}

    fn scatter_root_locals(&mut self) {}

    fn gather_ranges(&mut self) {
        let size = self.communicator.size();

        // 1. Gather ranges_min on all processes (should be defined also by interaction lists)
        let mut ranges_min = Vec::new();
        let mut ranges_max = Vec::new();
        for fmm_idx in 0..self.nsource_trees {
            ranges_min.push(
                self.tree.source_tree.trees[fmm_idx]
                    .owned_range()
                    .finest_first_child(),
            );
            ranges_max.push(
                self.tree.source_tree.trees[fmm_idx]
                    .owned_range()
                    .finest_last_child(),
            );
        }

        let range = [
            *ranges_min.iter().min().unwrap(),
            *ranges_max.iter().max().unwrap(),
        ];

        let mut all_ranges = vec![MortonKey::<Scalar::Real>::default(); (size * 2) as usize];
        self.communicator.all_gather_into(&range, &mut all_ranges);

        self.all_ranges = all_ranges;
    }

    fn u_list_exchange(&mut self) {
        let world_rank = self.rank;
        let size = self.communicator.size();

        match self.communication_mode {
            CommunicationMode::P2P => {
                // Use query packet to figure out where to send each query physically
                let mut queries = vec![Vec::new(); size as usize];

                for &query in self.particle_query_packet.iter() {
                    for (destination_rank, range) in self.all_ranges.chunks(2).enumerate() {
                        if range[0] <= query.finest_first_child()
                            && query.finest_first_child() <= range[1]
                        {
                            queries[destination_rank].push(query.morton)
                        }
                    }
                }

                let mut queries_to_send = Vec::new();
                let mut nqueries_to_send = Vec::new();
                let mut queries_to_send_sizes = Vec::new();
                let mut queries_to_send_destination_ranks = Vec::new();
                for (destination_rank, query) in queries.into_iter().enumerate() {
                    if !query.is_empty() {
                        queries_to_send_destination_ranks.push(destination_rank as i32);
                        queries_to_send_sizes.push(query.len() as i32);
                        queries_to_send.push(query);
                        nqueries_to_send.push(1)
                    } else {
                        nqueries_to_send.push(0)
                    }
                }

                // Mark ranks we expect to receive from with the number of expected messages
                let mut nqueries_to_receive = vec![0i32; size as usize];
                self.communicator.all_reduce_into(
                    &nqueries_to_send,
                    &mut nqueries_to_receive,
                    SystemOperation::sum(),
                );

                // Number of expected queries at this rank
                let recv_count = nqueries_to_receive[world_rank as usize];

                // Number of queries being send to other ranks
                let send_count = queries_to_send.len() as i32;

                // total number of requests being handled at this rank
                let nreqs = recv_count + send_count;

                // Communicate query sizes, and with whom to communicate
                // This is the size and origin of queries received from source ranks, and expected at this rank
                let (queries_received_sizes, queries_received_source_ranks) = expected_queries(
                    &self.communicator,
                    &queries_to_send_sizes,
                    &queries_to_send_destination_ranks,
                    recv_count,
                    nreqs,
                );

                // Allocate buffers to receive queries themselves
                let mut queries_received = Vec::new();
                for query_size in queries_received_sizes {
                    queries_received.push(vec![0u64; query_size as usize])
                }

                sparse_point_to_point_v(
                    &self.communicator,
                    &queries_to_send,
                    &queries_to_send_destination_ranks,
                    &mut queries_received,
                    &queries_received_source_ranks,
                    nreqs,
                );

                // Lookup locally available particle data
                let mut available_queries_received_reconstructed =
                    vec![Vec::new(); queries_received.len()];
                let mut available_queries_received = Vec::new();
                let mut available_queries_received_sizes = Vec::new(); // number of U list octants
                let mut available_queries_received_buffers_sizes = Vec::new(); // sizes of coordinate data
                let mut available_queries_received_index_pointers = Vec::new(); // index pointers of coordinate data

                // Check for inclusion in local tree
                for recv_buffer in queries_received.iter() {
                    let mut raw_morton_i = Vec::new();
                    let mut morton_i = Vec::new();
                    let mut index_pointers_i = Vec::new();

                    let mut query_size_i = 0;
                    let mut index_pointer = 0;

                    for &raw_morton in recv_buffer.iter() {
                        let key = MortonKey::from_morton(raw_morton, None);

                        // First check if this request is contained locally at all
                        if self.tree.source_tree.keys_set.contains(&key) {
                            // If it is, find associated source tree
                            let mut tree_idx = 0;
                            for (i, tree) in self.tree.source_tree.trees.iter().enumerate() {
                                if tree.keys_set.contains(&key) {
                                    tree_idx = i;
                                    break;
                                }
                            }

                            // Lookup associated coordinates for this key
                            if let Some(coords) =
                                self.tree.source_tree.trees[tree_idx].coordinates(&key)
                            {
                                raw_morton_i.push(raw_morton);
                                morton_i.push(key);
                                query_size_i += coords.len() as i32;

                                index_pointers_i.push(IndexPointer::new(
                                    index_pointer,
                                    index_pointer + query_size_i,
                                ));
                                index_pointer += query_size_i;
                            }
                        }
                    }

                    available_queries_received_sizes.push(raw_morton_i.len() as i32);
                    available_queries_received.push(raw_morton_i);
                    available_queries_received_reconstructed.push(morton_i);
                    available_queries_received_buffers_sizes.push(query_size_i);
                    available_queries_received_index_pointers.push(index_pointers_i);
                }

                // Allocate coordinate buffer to be sent that were requested and available
                let mut available_queries_received_buffers = Vec::new();
                for (query, &query_size) in available_queries_received_reconstructed
                    .iter()
                    .zip(available_queries_received_buffers_sizes.iter())
                {
                    let mut tmp = vec![Scalar::Real::default(); query_size as usize];

                    let mut index_pointer = 0;

                    for &key in query.iter() {
                        let mut tree_index = 0;
                        for (i, tree) in self.tree.source_tree.trees.iter().enumerate() {
                            if tree.keys_set.contains(&key) {
                                tree_index = i;
                                break;
                            }
                        }

                        if let Some(coords) =
                            self.tree.source_tree.trees[tree_index].coordinates(&key)
                        {
                            let ncoords = coords.len();
                            tmp[index_pointer..index_pointer + ncoords].copy_from_slice(coords);
                            index_pointer += ncoords
                        } else {
                            index_pointer += 0
                        }
                    }

                    available_queries_received_buffers.push(tmp);
                }

                // Send back how many of requested U list are actually available, and what they are and
                // the associated index pointers
                let mut available_queries_requested_sizes = vec![0i32; recv_count as usize];

                sparse_point_to_point(
                    &self.communicator,
                    &available_queries_received_sizes,
                    &queries_received_source_ranks,
                    &mut available_queries_requested_sizes,
                    &queries_to_send_destination_ranks,
                    nreqs,
                );

                // Now send back morton keys that were requested that actually exist
                let mut available_queries_requested = Vec::new();
                let mut available_queries_requested_index_pointers = Vec::new();
                for (_i, &size) in available_queries_requested_sizes.iter().enumerate() {
                    available_queries_requested.push(vec![0u64; size as usize]);
                    available_queries_requested_index_pointers
                        .push(vec![IndexPointer::default(); size as usize])
                }

                sparse_point_to_point_v(
                    &self.communicator,
                    &available_queries_received,
                    &queries_received_source_ranks,
                    &mut available_queries_requested,
                    &queries_to_send_destination_ranks,
                    nreqs,
                );

                // Send back index pointers for coordinate data
                sparse_point_to_point_v(
                    &self.communicator,
                    &available_queries_received_index_pointers,
                    &queries_received_source_ranks,
                    &mut available_queries_requested_index_pointers,
                    &queries_to_send_destination_ranks,
                    nreqs,
                );

                // Now send packet sizes for coordinate data that actually exists
                let mut available_queries_requested_buffers_sizes = vec![0i32; recv_count as usize];

                sparse_point_to_point(
                    &self.communicator,
                    &available_queries_received_buffers_sizes,
                    &queries_received_source_ranks,
                    &mut available_queries_requested_buffers_sizes,
                    &queries_to_send_destination_ranks,
                    nreqs,
                );

                // Allocate receive buffers for coordinate data
                let mut available_queries_requested_buffers = Vec::new();
                for &query_size in available_queries_requested_buffers_sizes.iter() {
                    let tmp = vec![Scalar::Real::default(); query_size as usize];
                    available_queries_requested_buffers.push(tmp)
                }

                // Can now send actual coordinate data
                sparse_point_to_point_v(
                    &self.communicator,
                    &available_queries_received_buffers,
                    &queries_received_source_ranks,
                    &mut available_queries_requested_buffers,
                    &queries_to_send_destination_ranks,
                    nreqs,
                );

                // Insert ghost particle data into special ghost tree
                let mut ghost_leaves = Vec::new();
                for query in available_queries_received.iter() {
                    for &raw_morton in query.iter() {
                        ghost_leaves.push(MortonKey::<Scalar::Real>::from_morton(raw_morton, None))
                    }
                }

                let ghost_coordinates = available_queries_received_buffers
                    .into_iter()
                    .flatten()
                    .collect_vec();

                // Flattening index pointers from all received queries requires to use displacement from each query
                // in received order
                let mut displacement = 0;
                for index_pointers in available_queries_received_index_pointers.iter_mut() {
                    index_pointers.iter_mut().for_each(|i| {
                        i.0 += displacement;
                        i.1 += displacement;
                    });

                    displacement += index_pointers.last().unwrap().1;
                }

                let ghost_index_pointers = available_queries_received_index_pointers
                    .into_iter()
                    .flatten()
                    .collect_vec();

                self.ghost_tree_u = GhostTreeU::from_ghost_data(
                    ghost_leaves,
                    ghost_index_pointers,
                    ghost_coordinates,
                )
                .unwrap_or_default();
            }

            CommunicationMode::Subcomm => {}
        }
    }

    fn v_list_exchange(&mut self) {
        let world_rank = self.rank;
        let size = self.communicator.size();

        match self.communication_mode {
            CommunicationMode::P2P => {
                // This can be multithreaded
                // Allocate queries from range in terms of Morton keys
                let mut queries = vec![Vec::new(); size as usize];

                for &query in self.multipole_query_packet.iter() {
                    for (destination_rank, range) in self.all_ranges.chunks(2).enumerate() {
                        if range[0] <= query.finest_first_child()
                            && query.finest_first_child() <= range[1]
                        {
                            queries[destination_rank].push(query.morton)
                        }
                    }
                }

                // These are queries to be sent to appropriate processors based on the range
                let mut queries_to_send = Vec::new();
                let mut nqueries_to_send = Vec::new();
                let mut queries_to_send_sizes = Vec::new();
                let mut queries_to_send_destination_ranks = Vec::new();
                for (destination_rank, query) in queries.into_iter().enumerate() {
                    if !query.is_empty() {
                        queries_to_send_destination_ranks.push(destination_rank as i32);
                        queries_to_send_sizes.push(query.len() as i32);
                        queries_to_send.push(query);
                        nqueries_to_send.push(1)
                    } else {
                        nqueries_to_send.push(0)
                    }
                }

                // Mark ranks we expect to receive from with the number of expected messages
                let mut nqueries_to_receive = vec![0i32; size as usize];
                self.communicator.all_reduce_into(
                    &nqueries_to_send,
                    &mut nqueries_to_receive,
                    SystemOperation::sum(),
                );

                // Number of expected queries at this rank
                let recv_count = nqueries_to_receive[world_rank as usize];

                // Number of queries being send to other ranks
                let send_count = queries_to_send.len() as i32;

                // total number of requests being handled at this rank
                let nreqs = recv_count + send_count;

                // Communicate query sizes, and with whom to communicate
                // This is the size and origin of queries received from source ranks, and expected at this rank
                let (queries_received_sizes, queries_received_source_ranks) = expected_queries(
                    &self.communicator,
                    &queries_to_send_sizes,
                    &queries_to_send_destination_ranks,
                    recv_count,
                    nreqs,
                );

                // Allocate buffers to receive queries themselves
                let mut queries_received = Vec::new();
                for query_size in queries_received_sizes {
                    queries_received.push(vec![0u64; query_size as usize])
                }

                // Communicate queries in sparse p2p
                sparse_point_to_point_v(
                    &self.communicator,
                    &queries_to_send, // queries to send to destinations
                    &queries_to_send_destination_ranks,
                    &mut queries_received, // expected queries from source ranks
                    &queries_received_source_ranks,
                    nreqs,
                );

                // Lookup locally available multipole data requested in recv queries
                // Also send back sibling data, as will be needed for M2L kernels
                let mut available_queries_received = Vec::new();
                let mut available_queries_received_sizes = Vec::new();

                for recv_buffer in queries_received.iter() {
                    // Check for inclusion in local source trees
                    let mut tmp = Vec::new();
                    for &raw_morton in recv_buffer.iter() {
                        let key = MortonKey::from_morton(raw_morton, None);
                        if self.tree.source_tree.keys_set.contains(&key) {
                            let raw_siblings = key.siblings().into_iter().map(|s| s.morton);
                            for s in raw_siblings {
                                tmp.push(s)
                            }
                        }
                    }

                    available_queries_received_sizes.push(tmp.len() as i32);
                    available_queries_received.push(tmp);
                }

                // Allocate multipole buffer to be sent that were requested and available
                let mut available_queries_received_buffers = Vec::new();
                for query in available_queries_received.iter() {
                    let mut multipole_idx = 0;
                    let mut tmp =
                        vec![Scalar::default(); query.len() * self.ncoeffs_equivalent_surface];

                    for &raw_morton in query.iter() {
                        let key = MortonKey::from_morton(raw_morton, None);

                        // Lookup corresponding source tree at this rank
                        let mut tree_idx = 0;
                        for (i, tree) in self.tree.source_tree.trees.iter().enumerate() {
                            if tree.keys_set.contains(&key) {
                                tree_idx = i;
                                break;
                            }
                        }

                        // Copy multipole data to send buffer
                        let multipole = self.multipole(tree_idx, &key).unwrap();
                        tmp[multipole_idx * self.ncoeffs_equivalent_surface
                            ..(multipole_idx + 1) * self.ncoeffs_equivalent_surface]
                            .copy_from_slice(multipole);
                        multipole_idx += 1;
                    }

                    available_queries_received_buffers.push(tmp);
                }

                // Send back how many are actually available, and expected message size
                let mut available_queries_requested_sizes = vec![0i32; recv_count as usize];

                sparse_point_to_point(
                    &self.communicator,
                    &available_queries_received_sizes,
                    &queries_received_source_ranks,
                    &mut available_queries_requested_sizes,
                    &queries_to_send_destination_ranks,
                    nreqs,
                );

                // Now send morton keys of query that actually exist
                let mut available_queries_requested = Vec::new();
                for (_i, &size) in available_queries_requested_sizes.iter().enumerate() {
                    available_queries_requested.push(vec![0u64; size as usize])
                }

                sparse_point_to_point_v(
                    &self.communicator,
                    &available_queries_received,
                    &queries_received_source_ranks,
                    &mut available_queries_requested,
                    &queries_to_send_destination_ranks,
                    nreqs,
                );

                // Allocate receive buffers for multipole data
                let mut available_queries_requested_buffers = Vec::new();
                for &query_size in available_queries_requested_sizes.iter() {
                    let tmp = vec![
                        Scalar::default();
                        (query_size as usize) * self.ncoeffs_equivalent_surface
                    ];
                    available_queries_requested_buffers.push(tmp);
                }

                // Communicate ghost multipole data
                sparse_point_to_point_v(
                    &self.communicator,
                    &available_queries_received_buffers,
                    &queries_received_source_ranks,
                    &mut available_queries_requested_buffers,
                    &queries_to_send_destination_ranks,
                    nreqs,
                );

                // Convert queries back to octants locally
                let mut ghost_keys = Vec::new();

                for query in available_queries_requested.iter() {
                    for &raw_morton in query.iter() {
                        ghost_keys.push(MortonKey::<Scalar::Real>::from_morton(raw_morton, None))
                    }
                }
                let ghost_multipoles = available_queries_received_buffers
                    .into_iter()
                    .flatten()
                    .collect_vec();
                let depth = self.tree.source_tree.local_depth + self.tree.source_tree.global_depth;

                self.ghost_tree_v = GhostTreeV::<Scalar>::from_ghost_data(
                    ghost_keys,
                    ghost_multipoles,
                    depth,
                    self.ncoeffs_equivalent_surface,
                )
                .unwrap();
            }

            CommunicationMode::Subcomm => {
                // fn v_list_subcomm(&mut self) {
                // // This can be multithreaded
                // let mut packets = vec![Vec::new(); size as usize];

                // for &query in self.multipole_query_packet.iter() {
                //     for (destination_rank, range) in self.all_ranges.chunks(2).enumerate() {
                //         if range[0] <= query.finest_first_child()
                //             && query.finest_first_child() <= range[1]
                //         {
                //             packets[destination_rank].push(query)
                //         }
                //     }
                // }

                // let messages = packets
                //     .iter()
                //     .map(|p| if !p.is_empty() { 1 } else { 0 })
                //     .collect_vec();

                // let packet_destinations = packets
                //     .iter()
                //     .enumerate()
                //     .filter_map(|(rank, packet)| {
                //         if packet.len() > 0 {
                //             Some(rank as i32)
                //         } else {
                //             None
                //         }
                //     })
                //     .collect_vec();

                // let packets = packets
                //     .into_iter()
                //     .filter_map(|p| if p.len() > 0 { Some(p) } else { None })
                //     .collect_vec();

                // let packet_sizes = packets.iter().map(|p| p.len() as i32).collect_vec();

                // let mut total_messages = vec![0i32; size as usize];

                // // All to all to figure out with whom to communicate
                // self.communicator.all_reduce_into(
                //     &messages,
                //     &mut total_messages,
                //     SystemOperation::sum(),
                // );

                // // Break into subcommunicators so multipoles can be broadcast
                // let recv_count = total_messages[world_rank as usize];
                // let send_count = packets.len() as i32;
                // let nreqs = recv_count + send_count;

                // Can now set up point to point, or break up into subcommunicators

                // Attempt at subcomm coloring
                // {
                // let mut received_packet_sizes = vec![0 as i32; recv_count as usize];
                //    let mut received_packet_sources = vec![0 as i32; recv_count as usize];
                // let color = self.communicator.rank();
                // // println!("ABOUT TO START {:?} {:?} {:?}", rank, send_count, recv_count);
                // mpi::request::multiple_scope(nreqs as usize, |scope, coll| {
                //     for (i, &destination_rank) in packet_destinations.iter().enumerate() {
                //         let tag = destination_rank;
                //         let sreq = self.communicator.process_at_rank(destination_rank).immediate_send_with_tag(
                //             scope,
                //             &packet_sizes[i],
                //             tag,
                //         );
                //         coll.add(sreq);
                //     }

                //     for (i, size) in received_packet_sizes.iter_mut().enumerate() {
                //         let (msg, status) = loop {
                //             // Spin for message availability. There is no guarantee that
                //             // immediate sends, even to the same process, will be immediately
                //             // visible to an immediate probe.
                //             let preq = self.communicator.any_process().immediate_matched_probe_with_tag(world_rank);
                //             if let Some(p) = preq {
                //                 break p;
                //             }
                //         };

                //         let rreq = msg.immediate_matched_receive_into(scope, size);
                //         received_packet_sources[i] = status.source_rank();

                //         coll.add(rreq);
                //     }

                //     let mut complete = vec![];
                //     coll.wait_all(&mut complete);
                // });

                // let mut split_comms = HashMap::new();

                // let all_colors = (0..size).collect_vec();
                // let mut relevant_colors: HashSet<i32> = received_packet_sources.iter().cloned().collect();
                // relevant_colors.insert(world_rank);
                // let mut all_colors_mapped = Vec::new();

                // for color in all_colors.iter() {
                //     let c = if relevant_colors.contains(color) {
                //         Color::with_value(*color)
                //     } else {
                //         Color::undefined()
                //     };
                //     all_colors_mapped.push(c);
                // };

                // for (color_raw, &color) in all_colors_mapped.iter().enumerate() {
                //     if let Some(new_comm) = self.communicator.split_by_color(color) {
                //         split_comms.insert(color_raw as i32, new_comm);
                //     }
                // }

                // // Broadcasting a message from global root rank, can loop over all root ranks and their respective subcommunicators
                // // scattering depends on original rank order, so packets have to be arranged in rank order at this point.
                // let broadcast_world_rank = 0;

                // // This serialises over each overlapping set of subcommunicators
                // (0..size).into_iter().for_each(|broadcast_world_rank: i32|{
                //     if split_comms.keys().contains(&broadcast_world_rank) {
                //         let comm = split_comms.get(&broadcast_world_rank).unwrap();

                //         let world_group = self.communicator.group();
                //         let split_group = comm.group();
                //         let split_rank = world_group.translate_rank(broadcast_world_rank, &split_group).unwrap();

                //         let root_process = comm.process_at_rank(split_rank);

                //         let mut received = 0i32;
                //         if comm.rank() == split_rank {
                //             let msg: Vec<i32> = vec![broadcast_world_rank; size as usize];
                //             mpi::request::scope(|scope| {
                //                 let req = root_process.immediate_scatter_into_root(scope, &msg[..], &mut received);
                //                 req.wait();
                //             });
                //         } else {
                //             mpi::request::scope(|scope| {
                //                 let req = root_process.immediate_scatter_into(scope, &mut received);
                //                 req.wait();
                //             });
                //         }

                //         println!("RECEIVING {:?} at rank {:?} from {:?}", received, world_rank, broadcast_world_rank)
                //     }
                // });
                // println!("");
                // }
            }
        }
        // Using query packet figure out where to send each query physically, using the computed ranges_min.
    }
}
