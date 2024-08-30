//! Multi Node FMM
//! Single Node FMM
use std::{collections::HashMap, time::Instant};

use green_kernels::traits::Kernel as KernelTrait;

use itertools::Itertools;
use mpi::{
    collective::SystemOperation, ffi::RSMPI_SUM, request::WaitGuard, topology::SimpleCommunicator, traits::{Communicator, Destination, Equivalence, Source}
};
use num::Float;
use rlst::RlstScalar;

use mpi::collective::CommunicatorCollectives;

use crate::{
    fmm::types::{FmmEvalType, KiFmm},
    traits::{
        field::SourceToTargetData as SourceToTargetDataTrait,
        fmm::{
            FmmOperatorData, HomogenousKernel, MultiNodeFmm, SourceToTargetTranslation,
            SourceTranslation, TargetTranslation,
        },
        tree::SingleNodeTreeTrait,
        types::{FmmError, FmmOperatorTime, FmmOperatorType},
    },
    tree::{helpers::all_to_allv_sparse, types::MortonKey},
    Fmm, MultiNodeFmmTree,
};

use super::types::KiFmmMultiNode;

impl<Scalar, Kernel, SourceToTargetData> MultiNodeFmm
    for KiFmmMultiNode<Scalar, Kernel, SourceToTargetData>
where
    Scalar: RlstScalar + Default + Equivalence + Float,
    Kernel: KernelTrait<T = Scalar> + HomogenousKernel + Default + Send + Sync,
    SourceToTargetData: SourceToTargetDataTrait + Send + Sync,
    <Scalar as RlstScalar>::Real: Default + Equivalence + Float,
    Self: SourceTranslation,
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
            //     // 3. Exchange packets (point to point)
            self.exchange_multipoles();

            //     // 4. Pass all root multipole data to root node so that final part of upward pass can occur on root node
        }

        // Now can proceed with remainder of the upward pass on chosen node, and some of the downward pass
        // {
        //     if self.communicator.rank() == 0 {
        //         // Global upward pass
        //         for level in (1..self.tree.source_tree.global_depth).rev() {}

        //         // Global downward pass
        //         for level in 2..=self.tree.target_tree.global_depth {
        //             if level > 2 {}
        //         }
        //     }

        //     // Exchange root multipole data back to required MPI processes
        // }

        // Now remainder of downward pass can happen in parallel on each process
        // {
        //     // local leaf level operations
        //     // fmm.p2p()?;
        //     // fmm.l2p()?;
        // }

        Ok(())
    }

    fn kernel(&self) -> &Self::Kernel {
        &self.kernel
    }

    fn exchange_multipoles(&mut self) {
        let rank = self.rank;
        let size = self.communicator.size();

        // 1. Gather ranges_min on all processes (should be defined also by interaction lists)
        let mut ranges_min = Vec::new();
        let mut ranges_max= Vec::new();
        for fmm_idx in 0..self.nsource_trees {
            ranges_min.push(self.tree.source_tree.trees[fmm_idx].owned_range().finest_first_child());
            ranges_max.push(self.tree.source_tree.trees[fmm_idx].owned_range().finest_last_child());
        }


        let range = [
            *ranges_min.iter().min().unwrap(),
            *ranges_max.iter().max().unwrap(),
        ];

        // let nranges = ranges_min.len() as i32;
        // let mut all_nranges = vec![0i32; size as usize];

        // self.communicator.all_gather_into(&nranges, &mut all_nranges);

        // println!("RANK {:?} {:?}", rank, all_nranges);

        // let recv_displacements = all_nranges
        //     .iter()
        //     .scan(0, |acc, &x| {
        //         let tmp = *acc;
        //         *acc += x;
        //         Some(tmp)
        //     })
        //     .collect_vec();


        let mut all_ranges = vec![MortonKey::<Scalar::Real>::default(); (size * 2) as usize];
        self.communicator.all_gather_into(&range, &mut all_ranges);

        // debugging code
        // if rank == 0 {
        //     for rank in 0..size {
        //         println!(
        //             "HERE {:?} {:?}",
        //             rank,
        //             &all_ranges[(rank as usize) * 2..(rank as usize + 1) * 2]
        //         );
        //     }
        // }

        // Using query packet figure out where to send each query physically, using the computed ranges_min.
        {
            let mut packets = vec![Vec::new(); size as usize];

            for &query in self.query_packet.iter() {
                for (destination_rank, range) in all_ranges.chunks(2).enumerate() {

                    if range[0] <= query.finest_first_child() && query.finest_first_child() <= range[1] {
                        packets[destination_rank].push(query)
                    }
                }
            }
            let messages = packets.iter().map(|p| if !p.is_empty() { 1} else { 0}).collect_vec();

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
                .filter_map(|p| {
                    if p.len() > 0 {
                        Some(p)
                    } else {
                        None
                    }
                }).collect_vec();

            let packet_sizes = packets.iter().map(|p| p.len() as i32 ).collect_vec();

            let mut total_messages = vec![0i32; size as usize];
            // All to all to figure out with whom to communicate
            self.communicator.all_reduce_into(&messages, &mut total_messages, SystemOperation::sum());

            let recv_count = total_messages[rank as usize];
            let send_count = packets.len() as i32;
            let nreqs = recv_count + send_count;

            // Can now set up point to point, or break up into subcommunicators
            let mut received_packet_sizes = vec![0 as i32; recv_count as usize];
            let mut received_packet_sources = vec![0 as i32; recv_count as usize];

            mpi::request::multiple_scope(nreqs as usize, |scope, coll| {
                for (i, &rank) in packet_destinations.iter().enumerate() {
                    let tag = rank;
                    let sreq = self.communicator.process_at_rank(rank).immediate_send_with_tag(
                        scope,
                        &packet_sizes[i],
                        tag,
                    );
                    coll.add(sreq);
                }

                for (i, size) in received_packet_sizes.iter_mut().enumerate() {
                    let (msg, status) = loop {
                        // Spin for message availability. There is no guarantee that
                        // immediate sends, even to the same process, will be immediately
                        // visible to an immediate probe.
                        let preq = self.communicator.any_process().immediate_matched_probe_with_tag(rank);
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


            if rank == 0 {
                println!("message sizes at rank {:?} {:?} {:?}", rank, received_packet_sizes, received_packet_sources);
            }

            if rank == 7 {
                println!("message sizes at rank {:?} {:?} {:?}", rank, received_packet_sizes, received_packet_sources);
            }


            // debugging code
            // if rank == 0 {
            //     println!("Queries for rank {:?}", rank);
            //     for (i, packet) in packets.iter().enumerate() {
            //         println!("RANK {:?} PACKETS {:?}", i, packet.len());
            //     }
            // }

            // if rank == 1 {
            //     println!("Queries for rank {:?}", rank);
            //     for (i, packet) in packets.iter().enumerate() {
            //         println!("RANK {:?} PACKETS {:?}", i, packet.len());
            //     }
            // }

            // Once you've found where to request packets from, you must check for existence, by sending packets point to point.


        }

        // 2. Receive packets from contributors
        // Need to have a second range for each multi node FMM (excluding the interaction list)
        // This defines the owned octants,
        {}

        // 3. Form packets for users
        {
            //     // Need to check all keys for which ranges_min they fall into
            //     let mut users = HashMap::new();

            //     for n in 0..size {
            //         if n != rank {
            //             users.insert(n, Vec::new());
            //         }
            //     }

            //     for (fmm_index, fmm) in self.fmms.iter().enumerate() {
            //         for &key in fmm.tree.source_tree.all_keys().unwrap() {
            //             for &(l, r) in all_domains.iter() {
            //                 if key >= l && key <= r {
            //                     users.get_mut(&l.rank).unwrap().push((fmm_index, key))
            //                 }
            //             }
            //         }
            //     }

            //     // Can form packets with user data
            //     for (user, required_keys) in users.iter() {

            //         // Need to have a better way of doing this, accounting for the potential variation by level
            //         let ncoeffs_equivalent_surface = self.fmms[0].ncoeffs_equivalent_surface[0];

            //         let mut packet = vec![Scalar::zero(); required_keys.len() * ncoeffs_equivalent_surface];
            //         let mut count = 0;

            //         for (fmm_index, key) in required_keys.iter() {
            //             // Shouldn't need double index, should be looking up multipole data by fmm_index in the first place.
            //             if let Some(multipole) = self.fmms[*fmm_index].multipole(key) {

            //                 packet[count * ncoeffs_equivalent_surface .. (count + 1) * ncoeffs_equivalent_surface].copy_from_slice(multipole);
            //                 count += 1;
            //             }
            //         }

            //         // Send packet to user process

            //         // 1 . Communicate packet size

            //         // 2. Communicate packet

            //         // 4. Communicate index pointers

            //     }
        }
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
