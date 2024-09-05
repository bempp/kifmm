//! Multi Node FMM
//! Single Node FMM
use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
    os::raw::c_void,
    time::Instant,
};

use green_kernels::traits::Kernel as KernelTrait;

use itertools::{izip, Itertools};
use mpi::{
    collective::SystemOperation,
    datatype::{Partition, PartitionMut},
    ffi::RSMPI_SUM,
    raw::{AsRaw, FromRaw},
    request::WaitGuard,
    topology::{Color, SimpleCommunicator},
    traits::{Communicator, Destination, Equivalence, Group, Partitioned, Root, Source},
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
    types::{IndexPointer, KiFmmMultiNode, Layout},
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
    fn gather_root_multipoles(&mut self) {
        let size = self.communicator.size();
        let rank = self.communicator.rank();

        let nroot_multipoles = self.tree.source_tree.trees.len() as i32;

        // Nominated rank chosen to run global upward pass
        let root_rank = 0;
        let root_process = self.communicator.process_at_rank(root_rank);

        if rank == root_rank {
            let mut all_nroot_multipoles = vec![0i32; size as usize];
            root_process.gather_into_root(&nroot_multipoles, &mut all_nroot_multipoles);

            // Allocate memory to store root multipoles

            // Allocate memory for global tree on which global FMM is being run
        } else {
            root_process.gather_into(&nroot_multipoles);
        }
    }

    fn scatter_root_locals(&mut self) {}

    fn set_layout(&mut self) {
        let size = self.communicator.size();

        // 1. Gather ranges_min on all processes (should be defined also by interaction lists)
        let mut ranges = Vec::new();
        for tree_idx in 0..self.nsource_trees {
            ranges.push(self.tree.source_tree.trees[tree_idx].owned_range().unwrap());
        }

        let nranges = ranges.len() as i32;
        let mut all_ranges_counts = vec![0i32; size as usize];
        self.communicator
            .all_gather_into(&nranges, &mut all_ranges_counts);

        let mut all_ranges_displacements = Vec::new();
        let mut displacement = 0;
        for &count in all_ranges_counts.iter() {
            all_ranges_displacements.push(displacement);
            displacement += count
        }

        let total_ranges = all_ranges_counts.iter().sum::<i32>();

        let mut raw = vec![MortonKey::<Scalar::Real>::default(); total_ranges as usize];
        let counts;
        let displacements;

        {
            let mut partition =
                PartitionMut::new(&mut raw, all_ranges_counts, &all_ranges_displacements[..]);
            self.communicator
                .all_gather_varcount_into(&ranges, &mut partition);
            counts = partition.counts().clone().to_vec();
            displacements = partition.displs().clone().to_vec();
        }

        let raw_set = raw.iter().cloned().collect();

        let mut ranks = Vec::new();

        for i in 0..raw.len() as i32 {
            let mut rank = 0;

            while rank < displacements.len() - 1 {
                let curr_displacement = displacements[rank + 1];

                if i < curr_displacement {
                    break;
                }

                rank += 1;
            }

            ranks.push(rank as i32);
        }

        let mut range_to_rank = HashMap::new();

        for (&range, &rank) in raw.iter().zip(ranks.iter()) {
            range_to_rank.insert(range, rank);
        }

        let layout = Layout {
            raw,
            raw_set,
            counts,
            displacements,
            ranks,
            range_to_rank,
        };

        self.layout = layout;
    }

    fn u_list_exchange(&mut self) {
        let world_rank = self.rank;
        let size = self.communicator.size();



    }

    fn v_list_exchange(&mut self) {
        // Two sets of communications, first for the multipoles that exist
        // secondly for the associated coefficient data

        // Begin by calculating receive counts, this requires an all to all over the neighbourhood
        // communicator only
        let neighbourhood_size = self.neighbourhood_communicator_v.size();

        let mut neighbourhood_send_counts = vec![0i32; neighbourhood_size as usize];

        for (global_rank, &send_count) in self.v_list_send_counts.iter().enumerate() {
            if let Some(local_rank) = self.neighbourhood_communicator_v.global_to_local_rank(global_rank as i32) {
                neighbourhood_send_counts[local_rank as usize] = send_count
            }
        }

        let mut neighbourhood_receive_counts = vec![0i32; neighbourhood_size as usize];
        self.neighbourhood_communicator_v.all_to_all_into(&neighbourhood_send_counts, &mut neighbourhood_receive_counts);

        // Now can create displacements
        let mut neighbourhood_send_displacements = Vec::new();
        let mut neighbourhood_receive_displacements = Vec::new();

        let mut send_counter = 0;
        let mut receive_counter = 0;

        for (send_count, receive_count) in neighbourhood_send_counts.iter().zip(neighbourhood_receive_counts.iter()) {
            neighbourhood_send_displacements.push(send_counter);
            neighbourhood_receive_displacements.push(receive_counter);
            send_counter +=  send_count;
            receive_counter += receive_count;
        }

        let total_receive_count = receive_counter;

        // Assign origin ranks to all received queries
        let mut receive_counter = 0;
        let mut received_queries_source_ranks = vec![0i32; total_receive_count as usize];
        for (i, &receive_count) in neighbourhood_receive_counts.iter().enumerate() {

            let curr_receive_counter = receive_counter;
            receive_counter +=  receive_count as usize;

            let rank = self.neighbourhood_communicator_v.neighbours[i];
            let tmp = vec![rank as i32; receive_count as usize];
            received_queries_source_ranks[curr_receive_counter..receive_counter].copy_from_slice(tmp.as_slice());
        }

        let mut received_queries = vec![0u64; total_receive_count as usize];
        // Create partition
        {
            let partition_send = Partition::new(&self.v_list_queries, neighbourhood_send_counts, neighbourhood_send_displacements);
            let mut partition_receive = PartitionMut::new(&mut received_queries, neighbourhood_receive_counts, &neighbourhood_receive_displacements[..]);
            self.neighbourhood_communicator_v.all_to_all_varcount_into(&partition_send, &mut partition_receive);
        }

        // Now have to check for locally contained keys
        let mut available_queries = Vec::new();
        let mut available_queries_source_ranks = Vec::new();
        for (&query, &source_rank) in received_queries.iter().zip(received_queries_source_ranks.iter()) {
            let key = MortonKey::from_morton(query, None);
            if self.tree.source_tree.keys_set.contains(&key) {
                available_queries.push(key.morton);
                available_queries_source_ranks.push(source_rank);
            }
        }

        let mut received_queries_send_counts_ = HashMap::new();
        for global_rank in available_queries_source_ranks.iter() {
            *received_queries_send_counts_.entry(*global_rank).or_insert(0) += 1;
        }

        let mut received_queries_send_counts = vec![0i32; neighbourhood_size as usize];

        for (&global_rank, &send_count) in received_queries_send_counts_.iter() {
            if let Some(local_rank) = self.neighbourhood_communicator_v.global_to_local_rank(global_rank) {
                received_queries_send_counts[local_rank as usize] = send_count;
            }
        }

        let mut received_queries_receive_counts = vec![0i32; neighbourhood_size as usize];
        self.neighbourhood_communicator_v.all_to_all_into(&received_queries_send_counts, &mut received_queries_receive_counts);

        // Now can create displacements
        let mut received_queries_send_displacements = Vec::new();
        let mut received_queries_receive_displacements = Vec::new();

        let mut send_counter = 0;
        let mut receive_counter = 0;

        for (send_count, receive_count) in received_queries_send_counts.iter().zip(received_queries_receive_counts.iter()) {
            received_queries_send_displacements.push(send_counter);
            received_queries_receive_displacements.push(receive_counter);
            send_counter +=  send_count;
            receive_counter += receive_count;
        }

        let total_receive_count = receive_counter;
        let total_send_count = send_counter;

        // Create partition and get back requests
        let mut requested_queries = vec![0u64; total_receive_count as usize];
        {
            let partition_send = Partition::new(&available_queries, received_queries_send_counts, received_queries_send_displacements);
            let mut partition_receive = PartitionMut::new(&mut requested_queries, received_queries_receive_counts, &received_queries_receive_displacements[..]);
            self.neighbourhood_communicator_v.all_to_all_varcount_into(&partition_send, &mut partition_receive);
        }


        // Allocate buffers to store requested multipole data
        let mut send_multipoles = vec![Scalar::default(); (total_send_count as usize) * self.ncoeffs_equivalent_surface];
        let mut multipole_idx = 0;
        for &query in available_queries.iter() {

            let key = MortonKey::from_morton(query, None);

            // Lookup corresponding source tree at this rank
            let mut tree_idx = 0;
            for (i, tree) in self.tree.source_tree.trees.iter().enumerate() {
                if tree.keys_set.contains(&key) {
                    tree_idx = 0;
                    break
                }
            }

            let multipole = self.multipole(tree_idx, &key).unwrap();

            send_multipoles[multipole_idx * self.ncoeffs_equivalent_surface..(multipole_idx + 1) * self.ncoeffs_equivalent_surface].copy_from_slice(
                multipole
            );

            multipole_idx += 1;
        }

        // Allocate buffers to store received multipole data
        let mut receive_multipoles = vec![Scalar::default(); (total_receive_count as usize) * self.ncoeffs_equivalent_surface];


        // Have to create multipole displacements and counts



        // if self.rank == 3 {
        //     println!("rank {:?} {:?}", self.rank, self.neighbourhood_communicator_v.neighbours);
        //     println!("rank {:?} {:?}", self.rank, &received_queries.len());
        //     println!("rank {:?} {:?}", self.rank, &requested_queries[0..3]);
        //     println!("rank {:?} {:?}", self.rank, &requested_queries.len());
        // }

        // if self.rank == 0 {
        //     println!("rank {:?} {:?}", self.rank, self.neighbourhood_communicator_v.neighbours);
        //     println!("rank {:?} {:?}", self.rank, &received_queries.len());
        //     println!("rank {:?} {:?}", self.rank, &received_queries[0..4]);
        // }



    }
}

