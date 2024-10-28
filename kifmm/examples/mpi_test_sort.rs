//? mpirun -n {{NPROCESSES}} --features "mpi"

#[cfg(feature = "mpi")]
#[allow(dead_code)]
mod mpi {

    use kifmm::sorting::{hyksort, samplesort, simplesort};
    use mpi::{
        topology::SimpleCommunicator,
        traits::{Communicator, Destination, Equivalence, Source},
    };
    use rand::Rng;

    /// Check if `n` is a power of two
    fn power_of_two(n: i32) -> bool {
        n != 0 && (n & (n - 1)) == 0
    }

    fn test_sort<T: Equivalence + Ord + Default + Clone + Copy>(
        sorted_arr: &[T],
        comm: &SimpleCommunicator,
        label: String,
    ) {
        let rank = comm.rank();
        let size = comm.size();

        // Test that there is no overlap between elements on each processor and that they are
        // globally sorted
        let min = *sorted_arr.iter().min().unwrap();
        let max = *sorted_arr.iter().max().unwrap();

        // Gather all bounds at root
        let next_rank = if rank + 1 < size { rank + 1 } else { 0 };
        let previous_rank = if rank > 0 { rank - 1 } else { size - 1 };

        let previous_process = comm.process_at_rank(previous_rank);
        let next_process = comm.process_at_rank(next_rank);

        // Send min to partner
        if rank > 0 {
            previous_process.send(&min);
        }

        let mut partner_min = T::default();

        if rank < (size - 1) {
            next_process.receive_into(&mut partner_min);
        }

        if rank < size - 1 {
            assert!(max < partner_min)
        }

        // Test that each node's portion is locally sorted
        for i in 0..(sorted_arr.iter().len() - 1) {
            let a = sorted_arr[i];
            let b = sorted_arr[i + 1];
            assert!(a <= b);
        }

        if rank == 0 {
            println!("...test_{} passed", label)
        }
    }

    pub fn main() {
        // Setup MPI
        let universe = mpi::initialize().unwrap();
        let comm = universe.world();

        // Test Hyksort
        {
            let subcomm_size = 2;
            // Only works if the communicator size is a power of two
            // Subcomm size must also be a power of two
            if power_of_two(comm.size()) && power_of_two(subcomm_size) {
                // Select random integers, with duplicates
                let mut rng = rand::thread_rng();
                let n = 1000;
                let mut arr: Vec<i32> = (0..n).map(|_| rng.gen_range(0..=10000)).collect();

                // Sort
                let _ = hyksort(&mut arr, subcomm_size, &comm);

                let label = "hyksort".to_string();
                test_sort(&arr, &comm, label);
            }
        }

        // Test Sample Sort
        {
            // Select random integers, with duplicates
            let mut rng = rand::thread_rng();
            let n = 1000;
            let mut arr: Vec<i32> = (0..n).map(|_| rng.gen_range(0..=10000)).collect();

            let number_of_samples = 100;

            // Sort
            let _ = samplesort(&mut arr, &comm, number_of_samples);

            let label = "samplesort".to_string();
            test_sort(&arr, &comm, label);
        }

        // Test Bucket Sort
        {
            // Select random integers, with duplicates
            let n_buckets = comm.size();
            let mut rng = rand::thread_rng();
            let n = 10000;
            let step_size = 100;
            let max = n_buckets * step_size;
            let mut arr: Vec<i32> = (0..n).map(|_| rng.gen_range(0..=max)).collect();

            // Number of splitters must match the number of MPI processes
            let mut splitters = Vec::new();

            for i in 1..=(n_buckets - 1) {
                splitters.push(i * step_size)
            }

            // Sort
            simplesort(&mut arr, &comm, &mut splitters).unwrap();

            let label = "simplesort".to_string();
            test_sort(&arr, &comm, label);
        }
    }
}

#[cfg(feature = "mpi")]
use mpi::main;

#[cfg(not(feature = "mpi"))]
fn main() {}
