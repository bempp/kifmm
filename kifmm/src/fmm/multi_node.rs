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
    fmm::types::{FmmEvalType, KiFmm},
    traits::{
        field::SourceToTargetData as SourceToTargetDataTrait,
        fmm::{
            ExchangeGhostData, FmmOperatorData, HomogenousKernel, MultiNodeFmm,
            SourceToTargetTranslation, SourceTranslation, TargetTranslation,
        },
        tree::SingleNodeTreeTrait,
        types::{FmmError, FmmOperatorTime, FmmOperatorType},
    },
    tree::{helpers::all_to_allv_sparse, types::MortonKey},
    Fmm, MultiNodeFmmTree,
};

use super::types::{CommunicationMode, KiFmmMultiNode};

impl<Scalar, Kernel, SourceToTargetData> MultiNodeFmm
    for KiFmmMultiNode<Scalar, Kernel, SourceToTargetData>
where
    Scalar: RlstScalar + Default + Equivalence + Float,
    Kernel: KernelTrait<T = Scalar> + HomogenousKernel + Default + Send + Sync,
    SourceToTargetData: SourceToTargetDataTrait + Send + Sync,
    <Scalar as RlstScalar>::Real: Default + Equivalence + Float,
    Self: SourceTranslation + ExchangeGhostData,
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
            // 3. Exchange packets (point to point)

            match self.communication_mode {
                CommunicationMode::P2P => self.v_list_p2p(),
                CommunicationMode::Subcomm => self.v_list_subcomm(),
            }

            // Update metadata
            self.update_v_list_metadata();

            //     // 4. Pass all root multipole data to root node so that final part of upward pass can occur on root node
        }

        // Gather root multipoles at nominated node
        self.gather_root_multipoles();

        // Now can proceed with remainder of the upward pass on chosen node, and some of the downward pass
        {

        }

        // Scatter root locals back to local trees
        self.scatter_root_locals();

        // Now remainder of downward pass can happen in parallel on each process

        Ok(())
    }

    fn kernel(&self) -> &Self::Kernel {
        &self.kernel
    }

    fn reset_metadata(&mut self) {
        // have to update
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

impl<Scalar, Kernel, SourceToTargetData> ExchangeGhostData
    for KiFmmMultiNode<Scalar, Kernel, SourceToTargetData>
where
    Scalar: RlstScalar + Default + Equivalence + Float,
    Kernel: KernelTrait<T = Scalar> + HomogenousKernel + Default + Send + Sync,
    SourceToTargetData: SourceToTargetDataTrait + Send + Sync,
    <Scalar as RlstScalar>::Real: Default + Equivalence + Float,
    Self: SourceTranslation,
{
    fn gather_root_multipoles(&mut self) {

    }

    fn scatter_root_locals(&mut self) {

    }

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

    fn u_list_p2p(&mut self) {
        let world_rank = self.rank;
        let size = self.communicator.size();

        // 1. Use query packet to figure out where to send each query physically
        let mut packets = vec![Vec::new(); size as usize];

        for &query in self.particle_query_packet.iter() {
            for (destination_rank, range) in self.all_ranges.chunks(2).enumerate() {
                if range[0] <= query.finest_first_child() && query.finest_first_child() <= range[1]
                {
                    packets[destination_rank].push(query)
                }
            }
        }

        let messages = packets
            .iter()
            .map(|p| if !p.is_empty() { 1 } else { 0 })
            .collect_vec();

        let packet_destinations = packets
            .iter()
            .enumerate()
            .filter_map(|(rank, packet)| {
                if packet.len() > 0 {
                    Some(rank as i32)
                } else {
                    None
                }
            })
            .collect_vec();

        let packets = packets
            .into_iter()
            .filter_map(|p| if p.len() > 0 { Some(p) } else { None })
            .collect_vec();

        let packet_sizes = packets.iter().map(|p| p.len() as i32).collect_vec();

        let mut total_messages = vec![0i32; size as usize];

        // 2. All to all to figure out with whom to communicate

        self.communicator
            .all_reduce_into(&messages, &mut total_messages, SystemOperation::sum());

        // 2. Need to now find packet sizes and sources for queries for particle data
        let recv_count = total_messages[world_rank as usize];
        let send_count = packets.len() as i32;
        let nreqs = recv_count + send_count;

        let mut received_packet_sizes = vec![0 as i32; recv_count as usize];
        let mut received_packet_sources = vec![0 as i32; recv_count as usize];

        mpi::request::multiple_scope(nreqs as usize, |scope, coll| {
            for (i, &destination_rank) in packet_destinations.iter().enumerate() {
                let tag = destination_rank;
                let sreq = self
                    .communicator
                    .process_at_rank(destination_rank)
                    .immediate_send_with_tag(scope, &packet_sizes[i], tag);
                coll.add(sreq);
            }

            for (i, size) in received_packet_sizes.iter_mut().enumerate() {
                let (msg, status) = loop {
                    // Spin for message availability. There is no guarantee that
                    // immediate sends, even to the same process, will be immediately
                    // visible to an immediate probe.
                    let preq = self
                        .communicator
                        .any_process()
                        .immediate_matched_probe_with_tag(world_rank);
                    if let Some(p) = preq {
                        break p;
                    }
                };

                let rreq = msg.immediate_matched_receive_into(scope, size);
                received_packet_sources[i] = status.source_rank();

                coll.add(rreq);
            }

            let mut complete = vec![];
            coll.wait_all(&mut complete);
        });

        // 3. Identify available U list data
        let mut recv_buffers = Vec::new();
        for packet_size in received_packet_sizes {
            recv_buffers.push(vec![
                MortonKey::<Scalar::Real>::default();
                packet_size as usize
            ])
        }

        // Barrier required to ensure that receive buffers are set up.
        self.communicator.barrier();

        mpi::request::multiple_scope(nreqs as usize, |scope, coll| {
            for (i, packet) in packets.iter().enumerate() {
                let sreq = self
                    .communicator
                    .process_at_rank(packet_destinations[i])
                    .immediate_send_with_tag(scope, &packet[..], packet_destinations[i]);
                coll.add(sreq);
            }

            for (i, buffer) in recv_buffers.iter_mut().enumerate() {
                let rreq = self
                    .communicator
                    .process_at_rank(received_packet_sources[i])
                    .immediate_receive_into(scope, &mut buffer[..]);

                coll.add(rreq);
            }

            let mut complete = vec![];
            coll.wait_all(&mut complete);
        });

        let mut available_received_packets = Vec::new();
        let mut available_received_packets_sizes = Vec::new();

        // Check for inclusion in local tree
        for recv_buffer in recv_buffers.iter() {
            let mut tmp = Vec::new();

            for key in recv_buffer.iter() {
                let mut index = 0;
                for (fmm_idx, tree) in self.tree.source_tree.trees.iter().enumerate() {
                    if tree.keys_set.contains(key) {
                        index = fmm_idx;
                        break;
                    }
                }

                if let Some(_coords) = self.tree.source_tree.trees[index].coordinates(key) {
                    tmp.push(*key)
                }
            }

            available_received_packets_sizes.push(tmp.len() as i32);
            available_received_packets.push(tmp);
        }

        // 3. Communicate ghost data
        let mut send_buffers_len = Vec::new();
        let mut send_buffers_index_pointers = Vec::new();

        for packet in available_received_packets.iter() {
            let mut tmp = Vec::new(); // index pointer

            let mut packet_size = 0;
            let mut index_pointer = 0;

            for key in packet.iter() {
                let mut index = 0;
                for (fmm_idx, tree) in self.tree.source_tree.trees.iter().enumerate() {
                    if tree.keys_set.contains(key) {
                        index = fmm_idx;
                        break;
                    }
                }

                let ncoords =
                    if let Some(coords) = self.tree.source_tree.trees[index].coordinates(key) {
                        coords.len()
                    } else {
                        0
                    };

                tmp.push((index_pointer, index_pointer + ncoords));
                packet_size += ncoords;
                index_pointer += ncoords
            }

            send_buffers_len.push(packet_size as i32);
            send_buffers_index_pointers.push(tmp);
        }

        // Allocate send buffers
        let mut send_buffers = Vec::new();
        for (packet, &packet_size) in available_received_packets
            .iter()
            .zip(send_buffers_len.iter())
        {
            let mut tmp = vec![Scalar::Real::default(); packet_size as usize];

            let mut index_pointer = 0;

            for key in packet.iter() {
                let mut index = 0;
                for (fmm_idx, tree) in self.tree.source_tree.trees.iter().enumerate() {
                    if tree.keys_set.contains(key) {
                        index = fmm_idx;
                        break;
                    }
                }

                if let Some(coords) = self.tree.source_tree.trees[index].coordinates(key) {
                    let ncoords = coords.len();
                    tmp[index_pointer..index_pointer + ncoords].copy_from_slice(coords);
                    index_pointer += ncoords
                } else {
                    index_pointer += 0
                }
            }

            send_buffers.push(tmp);
        }

        // Send back how many of requested U list are actually available, and what they are
        let mut available_requested_packets_sizes = vec![0 as i32; send_count as usize];
        mpi::request::multiple_scope(nreqs as usize, |scope, coll| {
            for (i, packet_size) in available_received_packets_sizes.iter().enumerate() {
                let destination_rank = received_packet_sources[i];
                let sreq = self
                    .communicator
                    .process_at_rank(destination_rank)
                    .immediate_send_with_tag(scope, packet_size, destination_rank);

                coll.add(sreq);
            }

            for (i, size) in available_requested_packets_sizes.iter_mut().enumerate() {
                let source_rank = packet_destinations[i];
                let rreq = self
                    .communicator
                    .process_at_rank(source_rank)
                    .immediate_receive_into(scope, size);
                coll.add(rreq);
            }

            let mut complete = vec![];
            coll.wait_all(&mut complete);
        });

        // Now send morton keys of query that actually exist
        let mut available_requested_packets = Vec::new();
        for (i, &size) in available_requested_packets_sizes.iter().enumerate() {
            available_requested_packets
                .push(vec![MortonKey::<Scalar::Real>::default(); size as usize])
        }

        // Barrier required to ensure that receive buffers are set up.
        self.communicator.barrier();

        mpi::request::multiple_scope(nreqs as usize, |scope, coll| {
            for (i, packet) in available_received_packets.iter().enumerate() {
                let destination_rank = received_packet_sources[i];
                let sreq = self
                    .communicator
                    .process_at_rank(destination_rank)
                    .immediate_send_with_tag(scope, &packet[..], destination_rank);
                coll.add(sreq);
            }

            for (i, buffer) in available_requested_packets.iter_mut().enumerate() {
                let source_rank = packet_destinations[i];
                let rreq = self
                    .communicator
                    .process_at_rank(source_rank)
                    .immediate_receive_into(scope, &mut buffer[..]);

                coll.add(rreq);
            }

            let mut complete = vec![];
            coll.wait_all(&mut complete);
        });

        // Now send packet sizes of queries that actually exist
        let mut available_requested_packet_sizes_particle = vec![0 as i32; send_count as usize];
        mpi::request::multiple_scope(nreqs as usize, |scope, coll| {
            for (i, packet_size) in send_buffers_len.iter().enumerate() {
                let destination_rank = received_packet_sources[i];
                let sreq = self
                    .communicator
                    .process_at_rank(destination_rank)
                    .immediate_send_with_tag(scope, packet_size, destination_rank);

                coll.add(sreq);
            }

            for (i, size) in available_requested_packet_sizes_particle
                .iter_mut()
                .enumerate()
            {
                let source_rank = packet_destinations[i];
                let rreq = self
                    .communicator
                    .process_at_rank(source_rank)
                    .immediate_receive_into(scope, size);
                coll.add(rreq);
            }

            let mut complete = vec![];
            coll.wait_all(&mut complete);
        });

        // Allocate receive buffers for particle data
        let mut recv_buffers = Vec::new();
        for &packet_size in available_requested_packet_sizes_particle.iter() {
            let tmp = vec![Scalar::Real::default(); packet_size as usize];
            recv_buffers.push(tmp)
        }

        // Barrier required to ensure that receive buffers are set up.
        self.communicator.barrier();

        // Communicate ghost coordinate data
        mpi::request::multiple_scope(nreqs as usize, |scope, coll| {
            for (i, packet) in send_buffers.iter().enumerate() {
                let destination_rank = received_packet_sources[i];
                let sreq = self
                    .communicator
                    .process_at_rank(destination_rank)
                    .immediate_send_with_tag(scope, &packet[..], destination_rank);
                coll.add(sreq);
            }

            for (i, buffer) in recv_buffers.iter_mut().enumerate() {
                let source_rank = packet_destinations[i];
                let rreq = self
                    .communicator
                    .process_at_rank(source_rank)
                    .immediate_receive_into(scope, &mut buffer[..]);

                coll.add(rreq);
            }

            let mut complete = vec![];
            coll.wait_all(&mut complete);
        });

        // Insert ghost particle data into local tree
        self.ghost_u_list_octants = available_requested_packets;
        self.ghost_u_list_data = recv_buffers;
    }

    fn v_list_p2p(&mut self) {
        let world_rank = self.rank;
        let size = self.communicator.size();

        // Using query packet figure out where to send each query physically, using the computed ranges_min.
        {
            // This can be multithreaded
            let mut packets = vec![Vec::new(); size as usize];

            for &query in self.multipole_query_packet.iter() {
                for (destination_rank, range) in self.all_ranges.chunks(2).enumerate() {
                    if range[0] <= query.finest_first_child()
                        && query.finest_first_child() <= range[1]
                    {
                        packets[destination_rank].push(query)
                    }
                }
            }

            let messages = packets
                .iter()
                .map(|p| if !p.is_empty() { 1 } else { 0 })
                .collect_vec();

            let packet_destinations = packets
                .iter()
                .enumerate()
                .filter_map(|(rank, packet)| {
                    if packet.len() > 0 {
                        Some(rank as i32)
                    } else {
                        None
                    }
                })
                .collect_vec();

            let packets = packets
                .into_iter()
                .filter_map(|p| if p.len() > 0 { Some(p) } else { None })
                .collect_vec();

            let packet_sizes = packets.iter().map(|p| p.len() as i32).collect_vec();

            let mut total_messages = vec![0i32; size as usize];

            // All to all to figure out with whom to communicate
            self.communicator.all_reduce_into(
                &messages,
                &mut total_messages,
                SystemOperation::sum(),
            );

            // Break into subcommunicators so multipoles can be broadcast
            let recv_count = total_messages[world_rank as usize];
            let send_count = packets.len() as i32;
            let nreqs = recv_count + send_count;

            let mut received_packet_sizes = vec![0 as i32; recv_count as usize];
            let mut received_packet_sources = vec![0 as i32; recv_count as usize];

            mpi::request::multiple_scope(nreqs as usize, |scope, coll| {
                for (i, &destination_rank) in packet_destinations.iter().enumerate() {
                    let tag = destination_rank;
                    let sreq = self
                        .communicator
                        .process_at_rank(destination_rank)
                        .immediate_send_with_tag(scope, &packet_sizes[i], tag);
                    coll.add(sreq);
                }

                for (i, size) in received_packet_sizes.iter_mut().enumerate() {
                    let (msg, status) = loop {
                        // Spin for message availability. There is no guarantee that
                        // immediate sends, even to the same process, will be immediately
                        // visible to an immediate probe.
                        let preq = self
                            .communicator
                            .any_process()
                            .immediate_matched_probe_with_tag(world_rank);
                        if let Some(p) = preq {
                            break p;
                        }
                    };

                    let rreq = msg.immediate_matched_receive_into(scope, size);
                    received_packet_sources[i] = status.source_rank();

                    coll.add(rreq);
                }

                let mut complete = vec![];
                coll.wait_all(&mut complete);
            });

            // Allocate receive buffers
            let mut recv_buffers = Vec::new();
            for packet_size in received_packet_sizes {
                recv_buffers.push(vec![
                    MortonKey::<Scalar::Real>::default();
                    packet_size as usize
                ])
            }

            // Barrier required to ensure that receive buffers are set up.
            self.communicator.barrier();

            mpi::request::multiple_scope(nreqs as usize, |scope, coll| {
                for (i, packet) in packets.iter().enumerate() {
                    let sreq = self
                        .communicator
                        .process_at_rank(packet_destinations[i])
                        .immediate_send_with_tag(scope, &packet[..], packet_destinations[i]);
                    coll.add(sreq);
                }

                for (i, buffer) in recv_buffers.iter_mut().enumerate() {
                    let rreq = self
                        .communicator
                        .process_at_rank(received_packet_sources[i])
                        .immediate_receive_into(scope, &mut buffer[..]);

                    coll.add(rreq);
                }

                let mut complete = vec![];
                coll.wait_all(&mut complete);
            });

            // Lookup locally available multipole data
            let mut available_received_packets = Vec::new();
            let mut available_received_packets_sizes = Vec::new();

            for recv_buffer in recv_buffers.iter() {
                // Check for inclusion in local tree
                let mut tmp = Vec::new();

                for key in recv_buffer.iter() {
                    if self.tree.source_tree.keys_set.contains(key) {
                        tmp.push(*key)
                    }
                }

                available_received_packets_sizes.push(tmp.len() as i32);
                available_received_packets.push(tmp);
            }

            // Allocate multipole buffer to be sent
            let mut send_buffers = Vec::new();
            for packet in available_received_packets.iter() {
                let mut multipole_idx = 0;
                let mut tmp =
                    vec![Scalar::default(); packet.len() * self.ncoeffs_equivalent_surface];

                for key in packet.iter() {
                    let mut index = 0;
                    for (fmm_idx, tree) in self.tree.source_tree.trees.iter().enumerate() {
                        if tree.keys_set.contains(key) {
                            index = fmm_idx;
                            break;
                        }
                    }

                    let multipole = self.multipole(index, key).unwrap();
                    tmp[multipole_idx * self.ncoeffs_equivalent_surface
                        ..(multipole_idx + 1) * self.ncoeffs_equivalent_surface]
                        .copy_from_slice(multipole);

                    multipole_idx += 1;
                }

                send_buffers.push(tmp);
            }

            // Send back how many are actually available, and expected message size
            let mut available_requested_packets_sizes = vec![0 as i32; send_count as usize];
            mpi::request::multiple_scope(nreqs as usize, |scope, coll| {
                for (i, packet_size) in available_received_packets_sizes.iter().enumerate() {
                    let destination_rank = received_packet_sources[i];
                    let sreq = self
                        .communicator
                        .process_at_rank(destination_rank)
                        .immediate_send_with_tag(scope, packet_size, destination_rank);

                    coll.add(sreq);
                }

                for (i, size) in available_requested_packets_sizes.iter_mut().enumerate() {
                    let source_rank = packet_destinations[i];
                    let rreq = self
                        .communicator
                        .process_at_rank(source_rank)
                        .immediate_receive_into(scope, size);
                    coll.add(rreq);
                }

                let mut complete = vec![];
                coll.wait_all(&mut complete);
            });

            // Now send morton keys of query that actually exist
            let mut available_requested_packets = Vec::new();
            for (i, &size) in available_requested_packets_sizes.iter().enumerate() {
                available_requested_packets
                    .push(vec![MortonKey::<Scalar::Real>::default(); size as usize])
            }

            // Barrier required to ensure that receive buffers are set up.
            self.communicator.barrier();

            mpi::request::multiple_scope(nreqs as usize, |scope, coll| {
                for (i, packet) in available_received_packets.iter().enumerate() {
                    let destination_rank = received_packet_sources[i];
                    let sreq = self
                        .communicator
                        .process_at_rank(destination_rank)
                        .immediate_send_with_tag(scope, &packet[..], destination_rank);
                    coll.add(sreq);
                }

                for (i, buffer) in available_requested_packets.iter_mut().enumerate() {
                    let source_rank = packet_destinations[i];
                    let rreq = self
                        .communicator
                        .process_at_rank(source_rank)
                        .immediate_receive_into(scope, &mut buffer[..]);

                    coll.add(rreq);
                }

                let mut complete = vec![];
                coll.wait_all(&mut complete);
            });

            // Allocate receive buffers for multipole data
            let mut recv_buffers = Vec::new();
            for &packet_size in available_requested_packets_sizes.iter() {
                let tmp = vec![
                    Scalar::default();
                    (packet_size as usize) * self.ncoeffs_equivalent_surface
                ];
                recv_buffers.push(tmp);
            }

            // Barrier required to ensure that receive buffers are set up.
            self.communicator.barrier();

            // Communicate ghost multipole data
            mpi::request::multiple_scope(nreqs as usize, |scope, coll| {
                for (i, packet) in send_buffers.iter().enumerate() {
                    let destination_rank = received_packet_sources[i];
                    let sreq = self
                        .communicator
                        .process_at_rank(destination_rank)
                        .immediate_send_with_tag(scope, &packet[..], destination_rank);
                    coll.add(sreq);
                }

                for (i, buffer) in recv_buffers.iter_mut().enumerate() {
                    let source_rank = packet_destinations[i];
                    let rreq = self
                        .communicator
                        .process_at_rank(source_rank)
                        .immediate_receive_into(scope, &mut buffer[..]);

                    coll.add(rreq);
                }

                let mut complete = vec![];
                coll.wait_all(&mut complete);
            });

            self.ghost_v_list_octants = available_received_packets;
            self.ghost_v_list_data = recv_buffers;
        }
    }

    fn u_list_subcomm(&mut self) {}

    fn v_list_subcomm(&mut self) {
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

    fn update_u_list_metadata(&mut self) {
        // potential index pointers need to be updated, and local trees need to be updated
        // corresponding tree index should be available in the request data to avoid a double loop

    }

    fn update_v_list_metadata(&mut self) {
        // Next step, generate metadata, will require a re-allocation for optimal performance of kernels for M2L in local trees
        // Need to be careful to insert siblings, even if they don't have any multipole data from ghosts so that kernels work.

        // Recreate tree index lookups
        // Copy local multipole data to new tree, with addition of ghost multipoles

        // Final step, setup new tree on nominated node for global FMM calculation, with all the leaf level
        // multipole data it needs to do both upward and doward passes
    }
}
