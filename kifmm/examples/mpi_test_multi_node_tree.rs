//? mpirun -n {{NPROCESSES}} --features "mpi"
#![allow(unused_imports)]
use num::traits::Float;
use rand::distributions::uniform::SampleUniform;

use rlst::{RawAccess, RlstScalar};

use kifmm::{traits::tree::SingleNodeTreeTrait, tree::helpers::points_fixture};

#[cfg(feature = "mpi")]
use mpi::{environment::Universe, topology::SimpleCommunicator, traits::Equivalence, traits::*};

#[cfg(feature = "mpi")]
use kifmm::tree::types::{Domain, MortonKey, MultiNodeTreeNew};

// /// Test that the leaves on separate nodes do not overlap.
// #[cfg(feature = "mpi")]
// fn test_no_overlaps<T: RlstScalar + Equivalence + Float + Default>(
//     world: &SimpleCommunicator,
//     tree: &MultiNodeTree<T, SimpleCommunicator>,
// ) {
//     // Communicate bounds from each process
//     let max = tree.all_leaves_set().unwrap().iter().max().unwrap();
//     let min = tree.all_leaves_set().unwrap().iter().min().unwrap();

//     // Gather all bounds at root
//     let size = world.size();
//     let rank = world.rank();

//     let next_rank = if rank + 1 < size { rank + 1 } else { 0 };
//     let previous_rank = if rank > 0 { rank - 1 } else { size - 1 };

//     let previous_process = world.process_at_rank(previous_rank);
//     let next_process = world.process_at_rank(next_rank);

//     // Send min to partner
//     if rank > 0 {
//         previous_process.send(min);
//     }

//     let mut partner_min = MortonKey::default();

//     if rank < (size - 1) {
//         next_process.receive_into(&mut partner_min);
//     }

//     // Test that the partner's minimum node is greater than the process's maximum node
//     if rank < size - 1 {
//         assert!(max < &partner_min)
//     }
// }

// /// Test that the globally defined domain contains all the points at a given node.
// #[cfg(feature = "mpi")]
// fn test_global_bounds<T: RlstScalar + Equivalence + Float + SampleUniform>(
//     world: &SimpleCommunicator,
// ) {
//     let n_points = 10000;
//     let points = points_fixture::<T>(n_points, None, None, None);

//     let comm = world.duplicate();

//     let domain = Domain::<T>::from_global_points(points.data(), &comm);

//     // Test that all local points are contained within the global domain
//     for i in 0..n_points {
//         let x = points.data()[i];
//         let y = points.data()[i + n_points];
//         let z = points.data()[i + 2 * n_points];

//         assert!(domain.origin[0] <= x && x <= domain.origin[0] + domain.side_length[0]);
//         assert!(domain.origin[1] <= y && y <= domain.origin[1] + domain.side_length[1]);
//         assert!(domain.origin[2] <= z && z <= domain.origin[2] + domain.side_length[2]);
//     }
// }

// /// Test that all leaves are mapped
// #[cfg(feature = "mpi")]
// fn test_n_leaves<T: RlstScalar + Equivalence + Float + SampleUniform>(
//     world: &SimpleCommunicator,
//     tree: &MultiNodeTree<T, SimpleCommunicator>,
// ) {
//     let n_leaves = tree.n_leaves().unwrap();

//     let size = world.size() as usize;
//     let mut counts = vec![0usize; size];

//     world.all_gather_into(&n_leaves, &mut counts[..]);

//     let n_leaves_tot = counts.iter().sum::<usize>() as i32;
//     let expected = 8i32.pow(tree.depth() as u32);

//     if world.rank() == 0 {
//         assert_eq!(n_leaves_tot, expected)
//     }
// }

/// Test that all leaves are mapped
#[cfg(feature = "mpi")]
fn test_n_points<T: RlstScalar + Equivalence + Float + SampleUniform>(
    world: &SimpleCommunicator,
    tree: &MultiNodeTreeNew<T, SimpleCommunicator>,
    points_per_proc: usize,
) {
    use kifmm::tree::types::MultiNodeTreeNew;

    let mut n_points = 0;

    for t in tree.trees.iter() {
        n_points += t.n_coordinates_tot().unwrap();
    }

    let size = world.size() as usize;
    let mut counts = vec![0usize; size];

    world.all_gather_into(&n_points, &mut counts[..]);

    let n_points_tot = counts.iter().sum::<usize>() as usize;
    let expected = points_per_proc * size;

    if world.rank() == 0 {
        assert_eq!(n_points_tot, expected)
    }
}

#[cfg(feature = "mpi")]
fn main() {
    // Setup an MPI environment

    use kifmm::tree::types::MultiNodeTreeNew;

    let universe: Universe = mpi::initialize().unwrap();
    let world = universe.world();
    let comm = world.duplicate();

    // Setup tree parameters
    let prune_empty = false;
    let depth = 5;
    let n_points = 10000;
    let local_depth = 3;
    let global_depth = 1;

    // Generate some random test data local to each process
    let points = points_fixture::<f32>(n_points, None, None, None);

    // Create a uniform tree
    let uniform =
        MultiNodeTreeNew::new(points.data(), local_depth, global_depth, true, None, &world)
            .unwrap();

    // test_no_overlaps(&comm, &uniform);
    // if world.rank() == 0 {
    //     println!("\t ... test_no_overlaps passed on uniform tree");
    // }

    // test_global_bounds::<f32>(&comm);
    // if world.rank() == 0 {
    //     println!("\t ... test_global_bounds passed on uniform tree");
    // }

    // test_n_leaves(&comm, &uniform);
    // if world.rank() == 0 {
    //     println!("\t ... test_n_leaves passed on uniform tree");
    // }

    test_n_points(&comm, &uniform, n_points);
    if world.rank() == 0 {
        println!("\t ... test_n_points passed on uniform tree");
    }

    // let prune_empty = true;
    // let sparse = MultiNodeTree::new(points.data(), depth, prune_empty, None, &comm).unwrap();

    // test_no_overlaps(&comm, &sparse);
    // if world.rank() == 0 {
    //     println!("\t ... test_no_overlaps passed on sparse tree");
    // }

    // test_global_bounds::<f32>(&comm);
    // if world.rank() == 0 {
    //     println!("\t ... test_global_bounds passed on sparse tree");
    // }

    // test_n_points(&comm, &sparse, n_points);
    // if world.rank() == 0 {
    //     println!("\t ... test_n_points passed on sparse tree");
    // }
}

#[cfg(not(feature = "mpi"))]
fn main() {}
