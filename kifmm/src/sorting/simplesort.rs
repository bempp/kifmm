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
///
/// # Arguments
/// * `array`- Local part of distributed array to be sorted
/// * `communicator`- Reference to underlying MPI communicator
/// * `splitters` - The buckets used to define the sort, must match the number of MPI processes, are equivalent to the max value
///  of the first splitters.len() - 1 buckets
pub fn simplesort<T>(
    array: &mut Vec<T>,
    communicator: &SimpleCommunicator,
    splitters: &mut [T],
) -> Result<(), std::io::Error>
where
    T: Equivalence + Ord + Default + Clone + Debug,
{
    if splitters.len() != ((communicator.size() - 1) as usize) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "The number of splitters must be size-1, i.e. need to explicitly define split for all MPI processes",
        ));
    }

    // Splitters must also be ordered
    splitters.sort();

    let size = communicator.size();

    // Perform local sort
    array.sort();

    // Sort local data into buckets
    let mut ranks = Vec::new();
    for item in array.iter() {
        let mut rank_index = -1i32;
        for (i, splitter) in splitters.iter().enumerate() {
            if item < splitter {
                rank_index = i as i32;
                break;
            }
            if rank_index == -1i32 {
                rank_index = splitters.len() as i32
            }
        }

        ranks.push(rank_index);
    }

    let mut counts_snd = vec![0i32; size as usize];
    for &rank in ranks.iter() {
        counts_snd[rank as usize] += 1
    }

    let displs_snd = counts_snd
        .iter()
        .scan(0, |acc, &x| {
            let tmp = *acc;
            *acc += x;
            Some(tmp)
        })
        .collect_vec();

    let mut counts_recv = vec![0 as Count; size as usize];

    communicator.all_to_all_into(&counts_snd, &mut counts_recv);

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
    let partition_snd = Partition::new(&array[..], counts_snd, &displs_snd[..]);

    communicator.all_to_all_varcount_into(&partition_snd, &mut partition_received);
    received.sort();

    *array = received;

    Ok(())
}
