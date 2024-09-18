use std::collections::HashMap;

use green_kernels::traits::Kernel as KernelTrait;
use itertools::Itertools;
use mpi::collective::CommunicatorCollectives;
use mpi::datatype::Partition;
use mpi::datatype::PartitionMut;
use mpi::datatype::Partitioned;
use mpi::topology::Communicator;
use mpi::topology::SimpleCommunicator;
use mpi::traits::Equivalence;
use mpi::traits::Root;
use mpi::Count;
use num::Float;
use pulp::Scalar;
use rlst::RlstScalar;

use crate::traits::field::SourceToTargetData as SourceToTargetDataTrait;
use crate::traits::fmm::HomogenousKernel;
use crate::traits::fmm::MultiFmm;
use crate::traits::fmm::SourceToTargetTranslation;
use crate::traits::parallel::GhostExchange;
use crate::traits::tree::MultiFmmTree;
use crate::traits::tree::MultiTree;
use crate::traits::tree::SingleTree;
use crate::tree::types::MortonKey;
use crate::tree::MultiNodeTree;
use crate::tree::SingleNodeTree;
use crate::MultiNodeFmmTree;

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
{
    fn gather_global_fmm_at_root(&mut self) {
        let size = self.communicator.size();
        let rank = self.communicator.rank();

        // Nominated rank chosen to run the global upward pass
        let root_rank = 0;
        let root_process = self.communicator.process_at_rank(root_rank);

        if rank == root_rank {
            // 1. Gather multipole data from root processes on all ranks
            let n_root_multipoles = self.tree.source_tree().n_trees();
            let mut global_multipoles_counts = vec![0 as Count; size as usize];
            root_process.gather_into_root(&n_root_multipoles, &mut global_multipoles_counts);

            // 1.1 Calculate displacements and counts for associated morton keys
            let mut global_multipoles_displacements = Vec::new();
            let mut displacement = 0;
            for &count in global_multipoles_counts.iter() {
                global_multipoles_displacements.push(displacement);
                displacement += count;
            }

            // 1.1 Allocate emory for locally contained data
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

            // 1.1 Calculate displacements and counts for multipole data
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

            // 1.1. Allocate memory to store received multipoles
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
            root_process.gather_varcount_into(&multipole_roots)
        }
    }

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
