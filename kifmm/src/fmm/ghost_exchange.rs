use std::collections::HashMap;

use green_kernels::traits::Kernel as KernelTrait;
use mpi::collective::CommunicatorCollectives;
use mpi::datatype::PartitionMut;
use mpi::datatype::Partitioned;
use mpi::topology::Communicator;
use mpi::traits::Equivalence;
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

    fn scatter_global_fmm_from_root(&mut self) {}

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
