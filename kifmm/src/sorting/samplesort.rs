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

/// A sample sort implementation, where 'k' is the number of samples to take from each sub-array.
/// Will be approximately load balanced for uniform distributions.
pub fn samplesort<T>(
    arr: &mut Vec<T>,
    comm: &SimpleCommunicator,
    k: usize,
) -> Result<(), std::io::Error>
where
    T: Equivalence + Ord + Default + Clone + Debug,
{
    if k > arr.len() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "k must be less than length of array",
        ));
    } else if k == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "k must be greater than 0",
        ));
    }

    let size = comm.size();

    // Perform local sort
    arr.sort();

    // 1. Collect k samples from each process onto all other processes
    let mut received_samples = vec![T::default(); k * (size as usize)];
    let mut rng = thread_rng();
    let sample_idxs: Vec<usize> = (0..k).map(|_| rng.gen_range(0..arr.len())).collect();

    let local_samples = sample_idxs
        .into_iter()
        .map(|i| arr[i].clone())
        .collect_vec();

    comm.all_gather_into(&local_samples[..], &mut received_samples[..]);

    // Ignore first k samples to ensure size-1 splitters
    received_samples.sort();
    received_samples = received_samples[k..].to_vec();

    // Every k'th sample defines a bucket
    let splitters = received_samples.into_iter().step_by(k).collect_vec();
    let nsplitters = size - 1;
    // let mut counts_snd = Vec::new();

    // 2. Sort local data into buckets
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
    let mut partition_received =
        PartitionMut::new(&mut received[..], counts_recv, &displs_recv[..]);
    let partition_snd = Partition::new(&arr[..], counts_snd, &displs_snd[..]);

    comm.all_to_all_varcount_into(&partition_snd, &mut partition_received);
    received.sort();
    *arr = received;

    Ok(())
}
