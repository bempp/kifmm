use std::collections::HashMap;
use std::collections::HashSet;

use green_kernels::traits::Kernel as KernelTrait;
use itertools::Itertools;
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
use crate::tree::types::MortonKey;
use crate::tree::SingleNodeTree;
use crate::SingleFmm;
use crate::SingleNodeFmmTree;

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

            let mut origin_rank = Vec::new();
            for (rank, &count) in global_locals_counts.iter().enumerate() {
                for _ in 0..(count as usize) {
                    origin_rank.push(rank as Rank)
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
        // // Have to identify locations of each local first via a gather.
        // // should really be in the 'gather ranges' part

        // let rank = self.communicator.rank();

        // let nroots = self.tree.target_tree.trees.len();
        // let receive_buffer_size = nroots * self.n_coeffs_equivalent_surface;
        // let mut receive_buffer = vec![Scalar::default(); receive_buffer_size];

        // // Nominated rank chosen to run global upward pass
        // let root_rank = 0;
        // let root_process = self.communicator.process_at_rank(root_rank);

        // if rank == root_rank {
        //     let send_buffer_size = self.local_roots.len() * self.n_coeffs_equivalent_surface;
        //     let mut send_buffer = vec![Scalar::default(); send_buffer_size];

        //     // Lookup local data to be sent back from global FMM
        //     let mut root_idx = 0;
        //     for root in self.local_roots.iter() {
        //         if let Some(local) = self.global_fmm.local(root) {
        //             send_buffer[root_idx * self.n_coeffs_equivalent_surface
        //                 ..(root_idx + 1) * self.n_coeffs_equivalent_surface]
        //                 .copy_from_slice(local);
        //             root_idx += 1;
        //         }
        //     }

        //     // Displace items to send back by ncoeffs
        //     let counts = self
        //         .local_roots_counts
        //         .iter()
        //         .map(|&c| c * (self.n_coeffs_equivalent_surfaceas i32))
        //         .collect_vec();

        //     let displacements = self
        //         .local_roots_displacements
        //         .iter()
        //         .map(|&d| d * (self.n_coeffs_equivalent_surfaceas i32))
        //         .collect_vec();

        //     let partition = Partition::new(&send_buffer, counts, &displacements[..]);

        //     root_process.scatter_varcount_into_root(&partition, &mut receive_buffer);
        // } else {
        //     root_process.scatter_varcount_into(&mut receive_buffer);
        // }
    }

    fn u_list_exchange(&mut self) {}

    fn v_list_exchange(&mut self) {}
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
