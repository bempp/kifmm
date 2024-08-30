//! Multi Node FMM
//! Single Node FMM
use std::{collections::HashMap, time::Instant};

use green_kernels::traits::Kernel as KernelTrait;

use mpi::{
    topology::SimpleCommunicator,
    traits::{Communicator, Equivalence},
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
            for level in (global_depth..=(local_depth + global_depth)).rev() {
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
            // self.exchange_multipoles();

            // 4. Pass all root multipole data to root node so that final part of upward pass can occur on root node
        }

        // Now can proceed with remainder of the upward pass on chosen node, and some of the downward pass
        {
            if self.communicator.rank() == 0 {
                // Global upward pass
                for level in (1..self.tree.source_tree.global_depth).rev() {}

                // Global downward pass
                for level in 2..=self.tree.target_tree.global_depth {
                    if level > 2 {}
                }
            }

            // Exchange root multipole data back to required MPI processes
        }

        // Now remainder of downward pass can happen in parallel on each process
        {
            // local leaf level operations
            // fmm.p2p()?;
            // fmm.l2p()?;
        }

        Ok(())
    }

    fn kernel(&self) -> &Self::Kernel {
        &self.kernel
    }

    fn exchange_multipoles(&mut self) {
        // let rank = self.rank;
        // let size = self.communicator.size();

        // // 1. Gather ranges on all processes (should be defined also by interaction lists)
        // let mut all_ranges = vec![0u64; (size as usize) * 2];

        // {
        //     // let ranges = self.ranges.iter().flat_map(|&(a, b)| vec![a, b]).collect_vec();
        //     let range = vec![self.ranges.first().unwrap().0, self.ranges.last().unwrap().1];

        //     let counts = vec![2 as i32 ; size as usize];
        //     let displs = counts
        //         .iter()
        //         .scan(0, |acc, &x| {
        //             let tmp = *acc;
        //             *acc += x;
        //             Some(tmp)
        //         })
        //         .collect_vec();

        //     let mut partition = PartitionMut::new(&mut all_ranges, counts, &displs[..]);
        //     self.communicator
        //         .all_gather_varcount_into(&range[..], &mut partition);
        // }

        // let all_domains = all_ranges
        //     .chunks_exact(2)
        //     .enumerate()
        //     .map(|(rank, domain)| {
        //         (
        //             MortonKey::<Scalar::Real>::from_morton(domain[0], Some(rank as i32)),
        //             MortonKey::<Scalar::Real>::from_morton(domain[1], Some(rank as i32)),
        //         )
        //     })
        //     .collect_vec();

        // // 2. Receive packets from contributors
        // // Need to have a second range for each multi node FMM (excluding the interaction list)
        // // This defines the owned octants,

        // // 3. Form packets for users
        // {
        //     // Need to check all keys for which ranges they fall into
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
        // }
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
        if fmm_idx < self.nfmms {
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
        if fmm_idx < self.nfmms {

            let multipole_ptr = &self.level_multipoles[fmm_idx][level as usize][0];
            let nsources = self.tree.source_tree.trees[fmm_idx].n_keys(level).unwrap();
            unsafe {Some(std::slice::from_raw_parts(multipole_ptr.raw, self.ncoeffs_equivalent_surface * nsources))}

        } else {
            None
        }
    }
}
