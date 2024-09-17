//! Multi Node FMM
//! Single Node FMM
use std::{
    collections::{HashMap, HashSet},
    time::Instant,
};

use green_kernels::traits::Kernel as KernelTrait;

use itertools::Itertools;
use mpi::{
    datatype::{Partition, PartitionMut},
    topology::SimpleCommunicator,
    traits::{Communicator, Equivalence, Partitioned, Root},
};

use num::Float;
use rlst::RlstScalar;

use mpi::collective::CommunicatorCollectives;

use crate::{
    fmm::types::KiFmm,
    traits::{
        field::{
            SourceToTargetData as SourceToTargetDataTrait, SourceToTargetTranslationMetadata,
            SourceToTargetTranslationMetadataGhostTrees,
        },
        fmm::{
            FmmGlobalFmmMetadata, FmmMetadata, FmmOperatorData, GhostExchange, HomogenousKernel,
            SourceToTargetTranslation, SourceTranslation, TargetTranslation,
        },
        tree::SingleNodeTreeTrait,
        types::{FmmError, FmmOperatorTime, FmmOperatorType},
    },
    tree::types::{GhostTreeU, GhostTreeV, MortonKey},
    Fmm, MultiNodeFmmTree, SingleNodeFmmTree,
};

#[cfg(feature = "mpi")]
use crate::{
    fmm::types::{KiFmmMultiNode, Layout},
    traits::fmm::MultiNodeFmm,
};

#[cfg(feature = "mpi")]
impl<Scalar, Kernel, SourceToTargetData, SourceToTargetDataSingleNode> MultiNodeFmm
    for KiFmmMultiNode<Scalar, Kernel, SourceToTargetData, SourceToTargetDataSingleNode>
where
    Scalar: RlstScalar + Default + Equivalence + Float,
    Kernel: KernelTrait<T = Scalar> + HomogenousKernel + Default + Send + Sync,
    SourceToTargetData: SourceToTargetDataTrait + Send + Sync,
    SourceToTargetDataSingleNode: SourceToTargetDataTrait + Send + Sync,
    <Scalar as RlstScalar>::Real: Default + Equivalence + Float,
    Self: SourceTranslation
        + GhostExchange
        + TargetTranslation
        + SourceToTargetTranslation
        + SourceToTargetTranslationMetadata,
    KiFmm<Scalar, Kernel, SourceToTargetDataSingleNode>: Fmm + SourceToTargetTranslationMetadata,
    GhostTreeV<Scalar, SourceToTargetData>:
        SourceToTargetTranslationMetadataGhostTrees<Scalar = Scalar>,
{
    type Scalar = Scalar;
    type Kernel = Kernel;
    type Tree = MultiNodeFmmTree<Scalar::Real, SimpleCommunicator>;

    fn dim(&self) -> usize {
        3
    }

    fn evaluate(&mut self, timed: bool) -> Result<(), FmmError> {
        // Run upward pass on local trees, up to local depth
        let total_depth = self.tree.source_tree.local_depth + self.tree.source_tree.global_depth;
        let global_depth = self.tree.source_tree.global_depth;

        {
            let s = Instant::now();
            self.p2m()?;
            self.times
                .push(FmmOperatorTime::from_instant(FmmOperatorType::P2M, s));

            for level in ((global_depth + 1)..=total_depth).rev() {
                let s = Instant::now();
                self.m2m(level).unwrap();
                self.times.push(FmmOperatorTime::from_instant(
                    FmmOperatorType::M2M(level),
                    s,
                ));
            }
        }

        // At this point the exchange needs to happen of multipole data
        self.v_list_exchange();

        // Can construct Metadata for ghost data at this point

        // Gather root multipoles at nominated node
        self.gather_global_fmm_at_root();

        // M2L displacements depend on existence, so must happen at runtime for Ghost tree and Global FMM Tree
        self.ghost_tree_v.displacements(
            &self.tree.target_tree.trees[..],
            total_depth,
            global_depth,
        );

        if self.rank == 0 {
            self.global_fmm.displacements(); // at root rank,
        }

        // Now can proceed with remainder of the upward pass on chosen node, and some of the downward pass
        if self.rank == 0 {
            self.global_fmm.upward_pass(timed)?; // Needs metadata
            self.global_fmm.downward_pass(timed)?; // avoid leaf level computations
        }

        // Scatter root locals back to local trees
        self.scatter_global_fmm_from_root();

        // Now remainder of downward pass can happen in parallel on each process, similar to how I've written the local upward passes
        // new kernels have to reflect ghost data, and potentially multiple local source trees
        // {
        //     let depth = self.tree.source_tree.local_depth + self.tree.source_tree.global_depth;
        //     for level in 2..=depth {
        //         if level > 2 {
        //             let s = Instant::now();
        //             self.l2l(level)?;
        //             self.times.push(FmmOperatorTime::from_instant(
        //                 FmmOperatorType::L2L(level),
        //                 s,
        //             ));
        //         }
        //         let s = Instant::now();
        //         self.m2l(level)?;
        //         self.times.push(FmmOperatorTime::from_instant(
        //             FmmOperatorType::M2L(level),
        //             s,
        //         ));
        //     }

        //     // Leaf level computation
        //     let s = Instant::now();
        //     self.p2p()?;
        //     self.times
        //         .push(FmmOperatorTime::from_instant(FmmOperatorType::P2P, s));
        //     let s = Instant::now();
        //     self.l2p()?;
        //     self.times
        //         .push(FmmOperatorTime::from_instant(FmmOperatorType::L2P, s));
        // }

        Ok(())
    }

    fn kernel(&self) -> &Self::Kernel {
        &self.kernel
    }

    fn check_surface_order(&self, _level: u64) -> usize {
        self.check_surface_order
    }

    fn equivalent_surface_order(&self, _level: u64) -> usize {
        self.equivalent_surface_order
    }

    fn ncoeffs_check_surface(&self, _level: u64) -> usize {
        self.ncoeffs_check_surface
    }

    fn ncoeffs_equivalent_surface(&self, _level: u64) -> usize {
        self.ncoeffs_equivalent_surface
    }

    fn multipole(
        &self,
        source_tree_idx: usize,
        key: &<<<Self::Tree as crate::traits::tree::MultiNodeFmmTreeTrait>::Tree as crate::traits::tree::MultiNodeTreeTrait>::Tree as crate::traits::tree::SingleNodeTreeTrait>::Node,
    ) -> Option<&[Self::Scalar]> {
        if source_tree_idx < self.tree.source_tree.n_trees {
            if let Some(&key_idx) = self.tree.source_tree.trees[source_tree_idx].level_index(key) {
                let multipole_ptr =
                    self.level_multipoles[source_tree_idx][key.level() as usize][key_idx];
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

    fn multipoles(&self, source_tree_idx: usize, level: u64) -> Option<&[Self::Scalar]> {
        if source_tree_idx < self.tree.source_tree.n_trees {
            let multipole_ptr = &self.level_multipoles[source_tree_idx][level as usize][0];
            let nsources = self.tree.source_tree.trees[source_tree_idx]
                .n_keys(level)
                .unwrap();
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

impl<Scalar, Kernel, SourceToTargetData, SourceToTargetDataSingleNode> GhostExchange
    for KiFmmMultiNode<Scalar, Kernel, SourceToTargetData, SourceToTargetDataSingleNode>
where
    Scalar: RlstScalar + Default + Equivalence + Float,
    Kernel: KernelTrait<T = Scalar> + HomogenousKernel + Default + Send + Sync,
    SourceToTargetData: SourceToTargetDataTrait + Send + Sync + Default,
    SourceToTargetDataSingleNode: SourceToTargetDataTrait + Send + Sync + Default,
    <Scalar as RlstScalar>::Real: Default + Equivalence + Float,
    Self: SourceTranslation
        + MultiNodeFmm<
            Scalar = Scalar,
            Tree = MultiNodeFmmTree<<Scalar as RlstScalar>::Real, SimpleCommunicator>,
        >,
    KiFmm<Scalar, Kernel, SourceToTargetDataSingleNode>: SourceToTargetTranslationMetadata
        + FmmMetadata<Scalar = Scalar, Charges = Scalar>
        + FmmOperatorData
        + Fmm<Scalar = Scalar, Tree = SingleNodeFmmTree<Scalar::Real>>,
{
    fn gather_global_fmm_at_root(&mut self) {
        let size = self.communicator.size();
        let rank = self.communicator.rank();

        let nroot_multipoles = self.tree.source_tree.trees.len() as i32;
        let nroot_locals = self.tree.target_tree.trees.len() as i32;

        // Nominated rank chosen to run global upward pass
        let root_rank = 0;
        let root_process = self.communicator.process_at_rank(root_rank);

        if rank == root_rank {
            // 0. Set metadata for result, stored in a new single node FMM
            let mut result = KiFmm::<Scalar, Kernel, SourceToTargetDataSingleNode>::default();
            result.equivalent_surface_order = vec![self.equivalent_surface_order];
            result.check_surface_order = vec![self.check_surface_order];
            result.ncoeffs_equivalent_surface = vec![self.ncoeffs_equivalent_surface];
            result.ncoeffs_check_surface = vec![self.ncoeffs_check_surface];
            result.kernel_eval_size = 1;

            // 1. Gather multipole data
            let mut global_multipoles_counts = vec![0i32; size as usize];
            root_process.gather_into_root(&nroot_multipoles, &mut global_multipoles_counts);

            // Calculate displacements and counts for associated morton keys
            let mut global_multipoles_displacements = Vec::new();
            let mut displacement = 0;
            for &count in global_multipoles_counts.iter() {
                global_multipoles_displacements.push(displacement);
                displacement += count
            }

            // Allocate memory for locally contained data
            let mut local_roots = Vec::new();
            let mut local_multipole = vec![
                Scalar::default();
                (nroot_multipoles as usize)
                    * self.ncoeffs_equivalent_surface
            ];

            for (tree_idx, source_tree) in self.tree.source_tree.trees.iter().enumerate() {
                let root_multipole = self.multipole(tree_idx, &source_tree.root).unwrap();
                local_multipole[tree_idx * self.ncoeffs_equivalent_surface
                    ..(tree_idx + 1) * self.ncoeffs_equivalent_surface]
                    .copy_from_slice(root_multipole);
                local_roots.push(source_tree.root)
            }

            // Calculate displacements and counts for multipole data
            let global_multipoles_bufs_counts = global_multipoles_counts
                .iter()
                .map(|c| c * self.ncoeffs_equivalent_surface as i32)
                .collect_vec();
            let mut global_multipoles_bufs_displacements = Vec::new();
            let mut displacement = 0;
            for &count in global_multipoles_bufs_counts.iter() {
                global_multipoles_bufs_displacements.push(displacement);
                displacement += count
            }

            // Allocate memory to store root multipoles
            let n = global_multipoles_bufs_counts.iter().sum::<i32>();
            let mut global_multipoles =
                vec![Scalar::default(); (n as usize) * self.ncoeffs_equivalent_surface];

            let mut partition = PartitionMut::new(
                &mut global_multipoles,
                &global_multipoles_bufs_counts[..],
                &global_multipoles_bufs_displacements[..],
            );
            root_process.gather_varcount_into_root(&local_multipole, &mut partition);

            let n = global_multipoles_counts.iter().sum::<i32>();
            let mut global_roots = vec![MortonKey::<Scalar::Real>::default(); n as usize];
            let mut partition = PartitionMut::new(
                &mut global_roots,
                &global_multipoles_counts[..],
                &global_multipoles_displacements[..],
            );

            root_process.gather_varcount_into_root(&local_roots, &mut partition);

            // Need to also insert sibling data if it's missing into multipole buffer so that upward pass
            // will run even if this doesn't exist remotely.
            // Also need to insert ancestors
            let mut global_keys_set: HashSet<_> = global_roots.iter().cloned().collect();
            let mut global_keys_to_index = HashMap::new();
            for (i, global_root) in global_roots.iter().enumerate() {
                global_keys_to_index.insert(global_root, i);
            }

            let mut global_leaves_set: HashSet<_> = global_roots.iter().cloned().collect();

            for (_i, global_root) in global_roots.iter().enumerate() {
                let siblings = global_root.siblings();
                let ancestors = global_root.ancestors(None);

                for &sibling in siblings.iter() {
                    if !global_keys_set.contains(&sibling) {
                        global_keys_set.insert(sibling); // add siblings to obtained roots
                    }
                    if !global_leaves_set.contains(&sibling) {
                        global_leaves_set.insert(sibling); // add siblings to obtained roots
                    }
                }

                for &ancestor in ancestors.iter() {
                    if !global_keys_set.contains(&ancestor) {
                        global_keys_set.insert(ancestor); // add siblings to obtained roots
                    }
                }
            }

            // Ensure that all siblings of ancestors are also included
            let global_keys_set: HashSet<_> = global_keys_set
                .iter()
                .flat_map(|key| {
                    if key.level() != 0 {
                        key.siblings()
                    } else {
                        vec![*key]
                    }
                })
                .collect();

            let mut global_keys = global_keys_set.iter().cloned().collect_vec();
            global_keys.sort(); // Store in Morton order

            let mut global_leaves = global_leaves_set.iter().cloned().collect_vec();
            global_leaves.sort(); // Store in Morton order

            // Global multipole data with missing siblings if they don't exist globally with zeros for coefficients
            let mut global_multipoles_with_ancestors =
                vec![Scalar::zero(); global_keys.len() * self.ncoeffs_equivalent_surface];

            for (new_idx, root) in global_keys.iter().enumerate() {
                if let Some(old_idx) = global_keys_to_index.get(root) {
                    let multipole = &global_multipoles[old_idx * self.ncoeffs_equivalent_surface
                        ..(old_idx + 1) * self.ncoeffs_equivalent_surface];
                    global_multipoles_with_ancestors[new_idx * self.ncoeffs_equivalent_surface
                        ..(new_idx + 1) * self.ncoeffs_equivalent_surface]
                        .copy_from_slice(multipole);
                }
            }

            // Insert multipoles and keys into result
            result.multipole_metadata(
                global_multipoles_with_ancestors,
                global_keys_set,
                global_keys,
                global_leaves_set,
                global_leaves,
                self.tree.source_tree.global_depth,
                &self.tree.domain,
            );

            // 2. Gather local data
            let mut global_locals_counts = vec![0i32; size as usize];
            root_process.gather_into_root(&nroot_locals, &mut global_locals_counts);

            // Calculate displacements and counts for associated morton keys
            let mut global_locals_displacements = Vec::new();
            let mut displacement = 0;
            for &count in global_locals_counts.iter() {
                global_locals_displacements.push(displacement);
                displacement += count
            }

            // Allocate memory for locally contained data
            let mut local_roots = Vec::new();

            for (_tree_idx, target_tree) in self.tree.target_tree.trees.iter().enumerate() {
                local_roots.push(target_tree.root)
            }

            let n = global_locals_counts.iter().sum::<i32>();
            let mut global_roots = vec![MortonKey::<Scalar::Real>::default(); n as usize];
            let mut partition = PartitionMut::new(
                &mut global_roots,
                &global_locals_counts[..],
                &global_locals_displacements[..],
            );

            root_process.gather_varcount_into_root(&local_roots, &mut partition);

            // Save somewhere the origin of all the local roots so I can broadcast back
            self.local_roots = global_roots.iter().cloned().collect();
            self.local_roots_counts = global_locals_counts;
            self.local_roots_displacements = global_locals_displacements;

            // Need to also insert sibling data if it's missing into multipole buffer so that upward pass
            // will run even if this doesn't exist remotely.
            // Also insert ancestor data
            let mut global_keys_set: HashSet<_> = global_roots.iter().cloned().collect();
            let mut global_leaves_set: HashSet<_> = global_roots.iter().cloned().collect();
            let mut global_keys_to_index = HashMap::new();
            for (i, global_root) in global_roots.iter().enumerate() {
                global_keys_to_index.insert(global_root, i);
            }

            for (_i, global_root) in global_roots.iter().enumerate() {
                let siblings = global_root.siblings();
                let ancestors = global_root.ancestors(None);

                for &sibling in siblings.iter() {
                    if !global_keys_set.contains(&sibling) {
                        global_keys_set.insert(sibling); // add siblings to obtained roots
                    }
                    if !global_leaves_set.contains(&sibling) {
                        global_leaves_set.insert(sibling); // add siblings to obtained roots
                    }
                }

                for &ancestor in ancestors.iter() {
                    if !global_keys_set.contains(&ancestor) {
                        global_keys_set.insert(ancestor); // add siblings to obtained roots
                    }
                }
            }

            let global_keys_set: HashSet<_> = global_keys_set
                .iter()
                .flat_map(|key| {
                    if key.level() != 0 {
                        key.siblings()
                    } else {
                        vec![*key]
                    }
                })
                .collect();

            let mut global_keys = global_keys_set.iter().cloned().collect_vec();
            global_keys.sort(); // Store in Morton order

            let mut global_leaves = global_leaves_set.iter().cloned().collect_vec();
            global_leaves.sort(); // Store in Morton order

            // Global locals data with missing siblings if they don't exist globally with zeros for coefficients
            let global_locals_with_ancestors =
                vec![Scalar::zero(); global_keys.len() * self.ncoeffs_equivalent_surface];

            // Insert locals and keys into result
            result.local_metadata(
                global_locals_with_ancestors,
                global_keys_set,
                global_keys,
                global_leaves_set,
                global_leaves,
                self.tree.target_tree.global_depth,
                &self.tree.domain,
            );

            // Set global fmm to result
            self.global_fmm = result;
        } else {
            // 1. Send multipoles and multipole buffers
            // Allocate buffers of send multipoles
            root_process.gather_into(&nroot_multipoles);

            let mut local_multipole = vec![
                Scalar::default();
                (nroot_multipoles as usize)
                    * self.ncoeffs_equivalent_surface
            ];
            let mut local_roots = Vec::new();

            for (tree_idx, source_tree) in self.tree.source_tree.trees.iter().enumerate() {
                let root_multipole = self.multipole(tree_idx, &source_tree.root).unwrap();
                local_multipole[tree_idx * self.ncoeffs_equivalent_surface
                    ..(tree_idx + 1) * self.ncoeffs_equivalent_surface]
                    .copy_from_slice(root_multipole);
                local_roots.push(source_tree.root)
            }

            root_process.gather_varcount_into(&local_multipole);
            root_process.gather_varcount_into(&local_roots);

            // 2. Send locals
            root_process.gather_into(&nroot_locals);
            let mut local_roots = Vec::new();

            for (_tree_idx, target_tree) in self.tree.target_tree.trees.iter().enumerate() {
                local_roots.push(target_tree.root)
            }

            root_process.gather_varcount_into(&local_roots);
        }
    }

    fn scatter_global_fmm_from_root(&mut self) {
        // Have to identify locations of each local first via a gather.
        // should really be in the 'gather ranges' part

        let rank = self.communicator.rank();

        let nroots = self.tree.target_tree.trees.len();
        let receive_buffer_size = nroots * self.ncoeffs_equivalent_surface;
        let mut receive_buffer = vec![Scalar::default(); receive_buffer_size];

        // Nominated rank chosen to run global upward pass
        let root_rank = 0;
        let root_process = self.communicator.process_at_rank(root_rank);

        if rank == root_rank {
            let send_buffer_size = self.local_roots.len() * self.ncoeffs_equivalent_surface;
            let mut send_buffer = vec![Scalar::default(); send_buffer_size];

            // Lookup local data to be sent back from global FMM
            let mut root_idx = 0;
            for root in self.local_roots.iter() {
                if let Some(local) = self.global_fmm.local(root) {
                    send_buffer[root_idx * self.ncoeffs_equivalent_surface
                        ..(root_idx + 1) * self.ncoeffs_equivalent_surface]
                        .copy_from_slice(local);
                    root_idx += 1;
                }
            }

            // Displace items to send back by ncoeffs
            let counts = self
                .local_roots_counts
                .iter()
                .map(|&c| c * (self.ncoeffs_equivalent_surface as i32))
                .collect_vec();
            let displacements = self
                .local_roots_displacements
                .iter()
                .map(|&d| d * (self.ncoeffs_equivalent_surface as i32))
                .collect_vec();

            let partition = Partition::new(&send_buffer, counts, &displacements[..]);

            root_process.scatter_varcount_into_root(&partition, &mut receive_buffer);
        } else {
            root_process.scatter_varcount_into(&mut receive_buffer);
        }
    }

    fn set_layout(&mut self) {
        let size = self.communicator.size();

        // 1. Gather ranges_min on all processes (should be defined also by interaction lists)
        let mut ranges = Vec::new();
        for tree_idx in 0..self.tree.source_tree.n_trees {
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
            counts = partition.counts().to_vec();
            displacements = partition.displs().to_vec();
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

        self.source_layout = layout;
    }

    fn u_list_exchange(&mut self) {
        // Similar to V list, begin by calculating receive counts
        let neighbourhood_size = self.neighbourhood_communicator_u.size();

        let mut neighbourhood_send_counts = vec![0i32; neighbourhood_size as usize];

        for (global_rank, &send_count) in self.u_list_send_counts.iter().enumerate() {
            if let Some(local_rank) = self
                .neighbourhood_communicator_u
                .global_to_local_rank(global_rank as i32)
            {
                neighbourhood_send_counts[local_rank as usize] = send_count
            }
        }

        let mut neighbourhood_receive_counts = vec![0i32; neighbourhood_size as usize];
        self.neighbourhood_communicator_u.all_to_all_into(
            &neighbourhood_send_counts,
            &mut neighbourhood_receive_counts,
        );

        // Now can create displacements
        let mut neighbourhood_send_displacements = Vec::new();
        let mut neighbourhood_receive_displacements = Vec::new();

        let mut send_counter = 0;
        let mut receive_counter = 0;

        for (send_count, receive_count) in neighbourhood_send_counts
            .iter()
            .zip(neighbourhood_receive_counts.iter())
        {
            neighbourhood_send_displacements.push(send_counter);
            neighbourhood_receive_displacements.push(receive_counter);
            send_counter += send_count;
            receive_counter += receive_count;
        }

        let total_receive_count = receive_counter;

        // Assign origin ranks to all received queries
        let mut receive_counter = 0;
        let mut received_queries_source_ranks = vec![0i32; total_receive_count as usize];
        for (i, &receive_count) in neighbourhood_receive_counts.iter().enumerate() {
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
                &self.u_list_queries,
                neighbourhood_send_counts,
                neighbourhood_send_displacements,
            );
            let mut partition_receive = PartitionMut::new(
                &mut received_queries,
                neighbourhood_receive_counts,
                &neighbourhood_receive_displacements[..],
            );
            self.neighbourhood_communicator_u
                .all_to_all_varcount_into(&partition_send, &mut partition_receive);
        }

        // Now have to check for locally contained keys from received queries
        let mut available_queries_source_ranks = Vec::new();
        let mut available_queries_sizes = Vec::new();
        let mut available_queries_bufs = Vec::new();
        for (&query, &source_rank) in received_queries
            .iter()
            .zip(received_queries_source_ranks.iter())
        {
            let key = MortonKey::from_morton(query, None);

            // Lookup corresponding source tree at this rank
            let mut tree_idx = 0;
            for (i, tree) in self.tree.source_tree.trees.iter().enumerate() {
                if tree.keys_set.contains(&key) {
                    tree_idx = i;
                    break;
                }
            }

            // Only send back if it contains coordinate data
            if let Some(coords) = self.tree.source_tree.trees[tree_idx].coordinates(&key) {
                available_queries_source_ranks.push(source_rank);
                available_queries_sizes.push(coords.len() as i32);
                available_queries_bufs.push(coords)
            }
        }

        // // Calculate send counts of available queries to send back
        let mut received_queries_sizes_send_counts_ = HashMap::new(); // particle data counts
        for (i, global_rank) in available_queries_source_ranks.iter().enumerate() {
            *received_queries_sizes_send_counts_
                .entry(*global_rank)
                .or_insert(0) += available_queries_sizes[i];
        }

        let mut received_queries_sizes_send_counts = vec![0i32; neighbourhood_size as usize];

        for (&global_rank, &send_count) in received_queries_sizes_send_counts_.iter() {
            if let Some(local_rank) = self
                .neighbourhood_communicator_v
                .global_to_local_rank(global_rank)
            {
                received_queries_sizes_send_counts[local_rank as usize] = send_count;
            }
        }

        // Calculate counts for requested queries associated coordinate data
        let mut requested_queries_sizes_receive_counts = vec![0i32; neighbourhood_size as usize];
        self.neighbourhood_communicator_u.all_to_all_into(
            &received_queries_sizes_send_counts,
            &mut requested_queries_sizes_receive_counts,
        );

        // Can now create displacements for coordinate data
        let mut received_queries_sizes_send_displacements = Vec::new();
        let mut requested_queries_sizes_receive_displacements = Vec::new();

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

        // Allocate buffers to store requested coordinate data to be send
        let mut send_coordinates = vec![Scalar::Real::default(); total_send_sizes_count as usize];
        let mut displacement = 0;
        for buf in available_queries_bufs {
            let new_displacement = displacement + buf.len();
            send_coordinates[displacement..new_displacement].copy_from_slice(buf);
            displacement = new_displacement;
        }

        // Allocate buffers to store requested coordinate data at this proc
        let mut ghost_coordinates =
            vec![Scalar::Real::default(); total_receive_sizes_count as usize];

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

        let depth = self.tree.source_tree.local_depth + self.tree.source_tree.global_depth;
        let domain = &self.tree.domain;
        self.ghost_tree_u = GhostTreeU::from_ghost_data(depth, domain, ghost_coordinates).unwrap();

        if self.rank == 3 {
            println!(
                "rank {:?} {:?}",
                self.rank, self.neighbourhood_communicator_u.neighbours
            );
            println!(
                "rank {:?} {:?}",
                self.rank,
                &requested_queries_sizes_receive_counts.len()
            );
            println!(
                "rank {:?} {:?}",
                self.rank, &requested_queries_sizes_receive_displacements
            );
            println!(
                "ghost tree {:?}",
                self.ghost_tree_u.leaves_to_coordinates.keys().len()
            );
            println!("ghost tree {:?}", self.ghost_tree_u.leaves.len());
        }
    }

    fn v_list_exchange(&mut self) {
        // Begin by calculating receive counts, this requires an all to all over the neighbourhood
        // communicator only
        let neighbourhood_size = self.neighbourhood_communicator_v.size();

        let mut neighbourhood_send_counts = vec![0i32; neighbourhood_size as usize];

        for (global_rank, &send_count) in self.v_list_send_counts.iter().enumerate() {
            if let Some(local_rank) = self
                .neighbourhood_communicator_v
                .global_to_local_rank(global_rank as i32)
            {
                neighbourhood_send_counts[local_rank as usize] = send_count
            }
        }

        let mut neighbourhood_receive_counts = vec![0i32; neighbourhood_size as usize];
        self.neighbourhood_communicator_v.all_to_all_into(
            &neighbourhood_send_counts,
            &mut neighbourhood_receive_counts,
        );

        // Now can create displacements
        let mut neighbourhood_send_displacements = Vec::new();
        let mut neighbourhood_receive_displacements = Vec::new();

        let mut send_counter = 0;
        let mut receive_counter = 0;

        for (send_count, receive_count) in neighbourhood_send_counts
            .iter()
            .zip(neighbourhood_receive_counts.iter())
        {
            neighbourhood_send_displacements.push(send_counter);
            neighbourhood_receive_displacements.push(receive_counter);
            send_counter += send_count;
            receive_counter += receive_count;
        }

        let total_receive_count = receive_counter;

        // Assign origin ranks to all received queries
        let mut receive_counter = 0;
        let mut received_queries_source_ranks = vec![0i32; total_receive_count as usize];
        for (i, &receive_count) in neighbourhood_receive_counts.iter().enumerate() {
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
                &self.v_list_queries,
                neighbourhood_send_counts,
                neighbourhood_send_displacements,
            );
            let mut partition_receive = PartitionMut::new(
                &mut received_queries,
                neighbourhood_receive_counts,
                &neighbourhood_receive_displacements[..],
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
            let key = MortonKey::from_morton(query, None);
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

        let mut received_queries_send_counts = vec![0i32; neighbourhood_size as usize];

        for (&global_rank, &send_count) in received_queries_send_counts_.iter() {
            if let Some(local_rank) = self
                .neighbourhood_communicator_v
                .global_to_local_rank(global_rank)
            {
                received_queries_send_counts[local_rank as usize] = send_count;
            }
        }

        // Calculate counts for requested queries
        let mut requested_queries_receive_counts = vec![0i32; neighbourhood_size as usize];
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

            send_multipoles_counts.push(send_count * self.ncoeffs_equivalent_surface as i32);
            receive_multipoles_counts.push(receive_count * self.ncoeffs_equivalent_surface as i32);
            send_multipoles_displacements
                .push(send_counter * self.ncoeffs_equivalent_surface as i32);
            receive_multipoles_displacements
                .push(send_counter * self.ncoeffs_equivalent_surface as i32);

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
            vec![Scalar::default(); (total_send_count as usize) * self.ncoeffs_equivalent_surface];
        let mut multipole_idx = 0;
        for &query in available_queries.iter() {
            let key = MortonKey::from_morton(query, None);

            // Lookup corresponding source tree at this rank
            let mut tree_idx = 0;
            for (_i, tree) in self.tree.source_tree.trees.iter().enumerate() {
                if tree.keys_set.contains(&key) {
                    tree_idx = 0;
                    break;
                }
            }

            let multipole = self.multipole(tree_idx, &key).unwrap();

            send_multipoles[multipole_idx * self.ncoeffs_equivalent_surface
                ..(multipole_idx + 1) * self.ncoeffs_equivalent_surface]
                .copy_from_slice(multipole);

            multipole_idx += 1;
        }

        // Allocate buffers to store requested multipole data
        let mut ghost_multipoles = vec![
            Scalar::default();
            (total_receive_count as usize)
                * self.ncoeffs_equivalent_surface
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

        let depth = self.tree.source_tree.global_depth + self.tree.source_tree.local_depth;

        let ghost_keys = requested_queries
            .iter()
            .map(|&k| MortonKey::from_morton(k, None))
            .collect_vec();

        self.ghost_tree_v = GhostTreeV::from_ghost_data(
            ghost_keys,
            ghost_multipoles,
            depth,
            self.ncoeffs_equivalent_surface,
        )
        .unwrap();

        // if self.rank == 3 {
        //     println!("rank {:?} {:?}", self.rank, self.neighbourhood_communicator_v.neighbours);
        //     println!("rank {:?} {:?}", self.rank, &self.ncoeffs_equivalent_surface);
        //     println!("rank {:?} {:?}", self.rank, &receive_multipoles.len());
        //     println!("rank {:?} {:?}", self.rank, &requested_queries.len());
        // }

        // if self.rank == 0 {
        //     println!("rank {:?} {:?}", self.rank, self.neighbourhood_communicator_v.neighbours);
        //     println!("rank {:?} {:?}", self.rank, &received_queries.len());
        //     println!("rank {:?} {:?}", self.rank, &received_queries[0..4]);
        // }
    }
}
