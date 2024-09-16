//! Even simpler than sample sort, statically define key distribution without sampling, allows simple pinning of MPI processes to CPUs
//!

use std::fmt::Debug;

use itertools::Itertools;
use mpi::{
    datatype::{Partition, PartitionMut},
    topology::SimpleCommunicator,
    traits::{Communicator, CommunicatorCollectives, Equivalence},
    Count,
};

/// Simple sort is a bucket style sort specialised for octrees. In this case, the number of 'splitters'
/// defines the leaves of an octree of depth log(n_spliters). This means that the number of octree leaves
/// must match the number of MPI processes.
///
/// This is suitable for approximately uniform distributions, with low-numbers of MPI processes. In which case
/// it assures a relatively uniform load balance, such that each MPI process is pinned to a given physical CPU.
pub fn simplesort<T>(
    arr: &mut Vec<T>,
    comm: &SimpleCommunicator,
    splitters: &[T],
) -> Result<(), std::io::Error>
where
    T: Equivalence + Ord + Default + Clone + Debug,
{
    if splitters.len() != (comm.size() - 1).try_into().unwrap() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "The number of splitters must be size-1, i.e. need to explicitly define split for all MPI processes",
        ));
    }

    let size = comm.size();

    // Perform local sort
    arr.sort();

    // Sort local data into buckets
    let nsplitters = size - 1;
    let mut splitter_index: i32 = 0;
    let mut l = 0;
    let mut count = 0;
    let mut splitter_indices = vec![(0i32, 0i32); nsplitters as usize];

    for (_i, item) in arr.iter().enumerate() {
        while splitter_index < nsplitters && item >= &splitters[splitter_index as usize] {
            if count > 0 {
                // Record the segment from l to l + count - 1
                splitter_indices[splitter_index as usize] = (l, l + count - 1);
                // Update l to the start of the next segment
                l += count;
            } else {
                // If count is 0, we need to move l forward by 1 to ensure correct indexing
                splitter_indices[splitter_index as usize] = (l, l);
                l += 1;
            }
            // Reset count and move to the next splitter
            count = 0;
            splitter_index += 1;
        }

        // If we haven't encountered a splitter that this item is larger than, increment count
        count += 1;
    }

    splitter_indices.push((l, l + count - 1));

    let counts_snd = splitter_indices
        .iter()
        .map(|(l, r)| if r - l > 0 { r - l + 1 } else { 0 })
        .collect_vec();

    let displs_snd = counts_snd
        .iter()
        .scan(0, |acc, &x| {
            let tmp = *acc;
            *acc += x;
            Some(tmp)
        })
        .collect_vec();

    let mut counts_recv = vec![0 as Count; size as usize];

    comm.all_to_all_into(&counts_snd, &mut counts_recv);

    let displs_recv = counts_recv
        .iter()
        .scan(0, |acc, &x| {
            let tmp = *acc;
            *acc += x;
            Some(tmp)
        })
        .collect_vec();

    let total = counts_recv.iter().sum::<Count>();

    let mut received = vec![T::default(); total as usize];
    let mut partition_received: PartitionMut<[T], Vec<i32>, &[i32]> =
        PartitionMut::new(&mut received[..], counts_recv, &displs_recv[..]);
    let partition_snd = Partition::new(&arr[..], counts_snd, &displs_snd[..]);

    comm.all_to_all_varcount_into(&partition_snd, &mut partition_received);
    received.sort();

    *arr = received;

    Ok(())
}
