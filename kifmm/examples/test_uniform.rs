// // ? mpirun -n {{NPROCESSES}} --features "mpi"
#![allow(unused_imports)]
use num::traits::Float;
use rand::distributions::uniform::SampleUniform;

use rlst::{RawAccess, RlstScalar};

use kifmm::{traits::tree::Tree, tree::helpers::points_fixture};

#[cfg(feature = "mpi")]
use mpi::{environment::Universe, topology::UserCommunicator, traits::*};

#[cfg(feature = "mpi")]
use kifmm::tree::types::{Domain, MortonKey, MultiNodeTree};

/// Test that the leaves on separate nodes do not overlap.
#[cfg(feature = "mpi")]
fn test_no_overlaps<T: Float + Default + RlstScalar<Real = T>>(
    world: &UserCommunicator,
    tree: &MultiNodeTree<T>,
) {
    // Communicate bounds from each process
    let max = tree.all_leaves_set().unwrap().iter().max().unwrap();
    let min = tree.all_leaves_set().unwrap().iter().min().unwrap();

    // Gather all bounds at root
    let size = world.size();
    let rank = world.rank();

    let next_rank = if rank + 1 < size { rank + 1 } else { 0 };
    let previous_rank = if rank > 0 { rank - 1 } else { size - 1 };

    let previous_process = world.process_at_rank(previous_rank);
    let next_process = world.process_at_rank(next_rank);

    // Send min to partner
    if rank > 0 {
        previous_process.send(min);
    }

    let mut partner_min = MortonKey::default();

    if rank < (size - 1) {
        next_process.receive_into(&mut partner_min);
    }

    // Test that the partner's minimum node is greater than the process's maximum node
    if rank < size - 1 {
        assert!(max < &partner_min)
    }
}

/// Test that the globally defined domain contains all the points at a given node.
#[cfg(feature = "mpi")]
fn test_global_bounds<T: RlstScalar + Float + Default + Equivalence + SampleUniform>(
    world: &UserCommunicator,
) {
    let npoints = 10000;
    let points = points_fixture::<T>(npoints, None, None, None);

    let comm = world.duplicate();

    let domain = Domain::from_global_points(points.data(), &comm);

    // Test that all local points are contained within the global domain
    for i in 0..npoints {
        let x = points.data()[i];
        let y = points.data()[i + npoints];
        let z = points.data()[i + 2 * npoints];

        assert!(domain.origin[0] <= x && x <= domain.origin[0] + domain.diameter[0]);
        assert!(domain.origin[1] <= y && y <= domain.origin[1] + domain.diameter[1]);
        assert!(domain.origin[2] <= z && z <= domain.origin[2] + domain.diameter[2]);
    }
}

/// Test that all leaves are mapped
#[cfg(feature = "mpi")]
fn test_nleaves<T: RlstScalar<Real = T> + Float + Default + Equivalence + SampleUniform>(
    world: &UserCommunicator,
    tree: &MultiNodeTree<T>,
) {
    let nleaves = tree.nleaves().unwrap();

    let size = world.size() as usize;
    let mut counts = vec![0usize; size];

    world.all_gather_into(&nleaves, &mut counts[..]);

    let nleaves_tot = counts.iter().sum::<usize>() as i32;
    let expected = 8i32.pow(tree.depth() as u32);

    if world.rank() == 0 {
        assert_eq!(nleaves_tot, expected)
    }
}

#[cfg(feature = "mpi")]
fn main() {
    // Setup an MPI environment

    let universe: Universe = mpi::initialize().unwrap();
    let world = universe.world();
    let comm = world.duplicate();

    // Setup tree parameters
    let sparse = false;
    let hyksort_subcomm_size = 2;
    let depth = 3;
    let n_points = 10000;

    // Generate some random test data local to each process
    let points = points_fixture::<f32>(n_points, None, None, None);

    // Create a uniform tree
    let tree = MultiNodeTree::new(
        points.data(),
        depth,
        sparse,
        None,
        &comm,
        hyksort_subcomm_size,
    )
    .unwrap();

    test_no_overlaps(&comm, &tree);
    if world.rank() == 0 {
        println!("\t ... test_no_overlaps passed on uniform tree");
    }

    test_global_bounds::<f32>(&comm);
    if world.rank() == 0 {
        println!("\t ... test_global_bounds passed on uniform tree");
    }

    test_nleaves(&comm, &tree);
    if world.rank() == 0 {
        println!("\t ... test_nleaves passed on uniform tree");
    }
}

#[cfg(not(feature = "mpi"))]
fn main() {}
