use std::collections::HashMap;

use green_kernels::traits::Kernel as KernelTrait;
use itertools::Itertools;
use mpi::collective::CommunicatorCollectives;
use mpi::datatype::Partition;
use mpi::datatype::PartitionMut;
use mpi::datatype::Partitioned;
use mpi::topology::Communicator;
use mpi::traits::Equivalence;
use mpi::traits::Root;
use num::Float;
use rlst::RlstScalar;

use crate::traits::field::SourceToTargetData as SourceToTargetDataTrait;
use crate::traits::fmm::HomogenousKernel;
use crate::traits::parallel::GhostExchange;
use crate::traits::tree::SingleTree;
use crate::tree::types::MortonKey;

use super::types::{KiFmmMulti, Layout};

impl<Scalar, Kernel, SourceToTargetData> GhostExchange
    for KiFmmMulti<Scalar, Kernel, SourceToTargetData>
where
    Scalar: RlstScalar + Default + Equivalence + Float,
    <Scalar as RlstScalar>::Real: RlstScalar + Default + Equivalence + Float,
    Kernel: KernelTrait<T = Scalar> + HomogenousKernel + Default + Send + Sync,
    SourceToTargetData: SourceToTargetDataTrait,
{
    fn gather_global_fmm_at_root(&mut self) {}

    fn scatter_global_fmm_from_root(&mut self) {
        // // Have to identify locations of each local first via a gather.
        // // should really be in the 'gather ranges' part

        // let rank = self.communicator.rank();

        // let nroots = self.tree.target_tree.trees.len();
        // let receive_buffer_size = nroots * self.ncoeffs_equivalent_surface;
        // let mut receive_buffer = vec![Scalar::default(); receive_buffer_size];

        // // Nominated rank chosen to run global upward pass
        // let root_rank = 0;
        // let root_process = self.communicator.process_at_rank(root_rank);

        // if rank == root_rank {
        //     let send_buffer_size = self.local_roots.len() * self.ncoeffs_equivalent_surface;
        //     let mut send_buffer = vec![Scalar::default(); send_buffer_size];

        //     // Lookup local data to be sent back from global FMM
        //     let mut root_idx = 0;
        //     for root in self.local_roots.iter() {
        //         if let Some(local) = self.global_fmm.local(root) {
        //             send_buffer[root_idx * self.ncoeffs_equivalent_surface
        //                 ..(root_idx + 1) * self.ncoeffs_equivalent_surface]
        //                 .copy_from_slice(local);
        //             root_idx += 1;
        //         }
        //     }

        //     // Displace items to send back by ncoeffs
        //     let counts = self
        //         .local_roots_counts
        //         .iter()
        //         .map(|&c| c * (self.ncoeffs_equivalent_surface as i32))
        //         .collect_vec();

        //     let displacements = self
        //         .local_roots_displacements
        //         .iter()
        //         .map(|&d| d * (self.ncoeffs_equivalent_surface as i32))
        //         .collect_vec();

        //     let partition = Partition::new(&send_buffer, counts, &displacements[..]);

        //     root_process.scatter_varcount_into_root(&partition, &mut receive_buffer);
        // } else {
        //     root_process.scatter_varcount_into(&mut receive_buffer);
        // }
    }

    fn set_source_layout(&mut self) {
        let size = self.communicator.size();

        // 1. Gather ranges on all processes, define by roots they own
        let mut ranges = Vec::new();
        for tree_idx in 0..self.tree.source_tree.n_trees {
            ranges.push(self.tree.source_tree.trees[tree_idx].root());
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

    fn u_list_exchange(&mut self) {}

    fn v_list_exchange(&mut self) {}
}
