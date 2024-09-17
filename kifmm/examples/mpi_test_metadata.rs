//? mpirun -n {{NPROCESSES}} --features "mpi"
#![allow(unused_imports)]
use num::traits::Float;
use rand::distributions::uniform::SampleUniform;

use rlst::{RawAccess, RlstScalar};

use kifmm::{traits::tree::SingleTree, tree::helpers::points_fixture};

#[cfg(feature = "mpi")]
use mpi::{environment::Universe, topology::SimpleCommunicator, traits::Equivalence, traits::*};

#[cfg(feature = "mpi")]
use kifmm::tree::types::{Domain, MortonKey, MultiNodeTree};

#[cfg(feature = "mpi")]
fn main() {
    // Setup an MPI environment

    use green_kernels::laplace_3d::Laplace3dKernel;
    use kifmm::{fmm::types::MultiNodeBuilder, tree::types::SortKind, FftFieldTranslation};

    let universe: Universe = mpi::initialize().unwrap();
    let world = universe.world();
    let comm = world.duplicate();

    // Setup tree parameters
    let prune_empty = false;
    let n_points = 10000;
    let local_depth = 3;
    let global_depth = 1;
    let sort_kind = SortKind::Samplesort { k: 100 };

    // FMM parameters
    let expansion_order = 4;
    let kernel = Laplace3dKernel::<f32>::new();
    let source_to_target = FftFieldTranslation::<f32>::new(None);

    // Generate some random test data local to each process
    let points = points_fixture::<f32>(n_points, None, None, None);

    let mut fmm = MultiNodeBuilder::new()
        .tree(
            &comm,
            points.data(),
            points.data(),
            local_depth,
            global_depth,
            prune_empty,
            sort_kind,
        )
        .unwrap()
        .parameters(expansion_order, kernel, source_to_target)
        .unwrap()
        .build();
}

#[cfg(not(feature = "mpi"))]
fn main() {}
