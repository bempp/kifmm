//! Implementation of Sample Sort algorithm
use std::fmt::Debug;

use itertools::Itertools;
use mpi::{
    datatype::{Partition, PartitionMut},
    topology::SimpleCommunicator,
    traits::{Communicator, CommunicatorCollectives, Equivalence},
    Count,
};
use rand::{thread_rng, Rng};

/// A sample sort implementation, the number of samples must be at most
/// the size of the local array, and greater than 0.
///
/// # Arguments
/// * `array`- Local part of distributed array to be sorted
/// * `communicator`- Reference to underlying MPI communicator
/// * `n_samples` - Number of local samples to take from each array
pub fn samplesort<T>(
    array: &mut Vec<T>,
    communicator: &SimpleCommunicator,
    n_samples: usize,
) -> Result<(), std::io::Error>
where
    T: Equivalence + Ord + Default + Clone + Debug,
{
    if n_samples > array.len() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "n_samples must be less than length of array",
        ));
    } else if n_samples == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "n_samples must be greater than 0",
        ));
    }

    let size = communicator.size();

    // Perform local sort
    array.sort();

    // 1. Collect k samples from each process onto all other processes
    let mut received_samples = vec![T::default(); n_samples * (size as usize)];
    let mut rng = thread_rng();
    let sample_idxs: Vec<usize> = (0..n_samples)
        .map(|_| rng.gen_range(0..array.len()))
        .collect();

    let local_samples = sample_idxs
        .into_iter()
        .map(|i| array[i].clone())
        .collect_vec();

    communicator.all_gather_into(&local_samples[..], &mut received_samples[..]);

    // Ignore first k samples to ensure size-1 splitters
    received_samples.sort();
    received_samples = received_samples[n_samples..].to_vec();

    // Every k'th sample defines a bucket
    let splitters = received_samples
        .into_iter()
        .step_by(n_samples)
        .collect_vec();

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
    let mut partition_received =
        PartitionMut::new(&mut received[..], counts_recv, &displs_recv[..]);
    let partition_snd = Partition::new(&array[..], counts_snd, &displs_snd[..]);

    communicator.all_to_all_varcount_into(&partition_snd, &mut partition_received);
    received.sort();
    *array = received;

    Ok(())
}
