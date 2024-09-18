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

    fn u_list_exchange(&mut self) {}

    fn v_list_exchange(&mut self) {}
}
