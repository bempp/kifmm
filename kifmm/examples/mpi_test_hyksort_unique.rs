//? mpirun -n {{NPROCESSES}} --features "mpi"
#[cfg(feature = "mpi")]
use kifmm::hyksort::hyksort;
#[cfg(feature = "mpi")]
use mpi::traits::{Communicator, Destination, Source};
#[cfg(feature = "mpi")]
use rand::Rng;
#[cfg(feature = "mpi")]
use std::collections::HashSet;

#[cfg(feature = "mpi")]
fn main() {
    // Setup MPI
    let universe = mpi::initialize().unwrap();
    let comm = universe.world();
    let k = 2;
    let size = comm.size();
    let rank = comm.rank();

    // Select unique random integers
    let mut rng = rand::thread_rng();
    let nsamples = 1000;
    let arr: Vec<i32> = (0..nsamples)
        .map(|_| rng.gen_range(rank * nsamples..rank * nsamples + nsamples))
        .collect();
    let arr_set: HashSet<i32> = arr.iter().cloned().collect();
    let mut arr: Vec<i32> = arr_set.into_iter().collect();

    // Sort
    let _ = hyksort(&mut arr, k, comm.duplicate());

    // Test that elements are globally sorted
    let min = *arr.iter().min().unwrap();
    let max = *arr.iter().max().unwrap();

    // Gather all bounds at root
    let next_rank = if rank + 1 < size { rank + 1 } else { 0 };
    let previous_rank = if rank > 0 { rank - 1 } else { size - 1 };

    let previous_process = comm.process_at_rank(previous_rank);
    let next_process = comm.process_at_rank(next_rank);

    // Send min to partner
    if rank > 0 {
        previous_process.send(&min);
    }

    let mut partner_min: i32 = 0;

    if rank < (size - 1) {
        next_process.receive_into(&mut partner_min);
    }

    if rank < size - 1 {
        assert!(max < partner_min)
    }

    // Test that each node's portion is locally sorted
    for i in 0..(arr.iter().len() - 1) {
        let a = arr[i];
        let b = arr[i + 1];
        assert!(a <= b);
    }

    if rank == 0 {
        println!("...test_hyksort_unique passed")
    }
}

#[cfg(not(feature = "mpi"))]
fn main() {}
