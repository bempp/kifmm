#[cfg(feature = "mpi")]
fn main() {
    // use green_kernels::laplace_3d::Laplace3dKernel;
    // use kifmm::{
    //     tree::{helpers::points_fixture, types::SortKind},
    //     FftFieldTranslation,
    // };

    // use rayon::ThreadPoolBuilder;

    // use mpi::traits::*;
    // use rlst::RawAccess;

    // let (universe, _threading) = mpi::initialize_with_threading(mpi::Threading::Single).unwrap();
    // // let universe = mpi::initialize().unwrap();
    // let world = universe.world();
    // let comm = world.duplicate();

    // // Tree parameters
    // let prune_empty = true;
    // let n_points = 10000;
    // let local_depth = 2;
    // let global_depth = 2;
    // let sort_kind = SortKind::Samplesort { k: 100 };

    // // Fmm Parameters
    // let expansion_order = 4;
    // let kernel = Laplace3dKernel::<f32>::new();
    // let source_to_target = FftFieldTranslation::<f32>::new(None);

    // // Generate some random test data local to each process
    // let points = points_fixture::<f32>(n_points, None, None, None);

    // ThreadPoolBuilder::new()
    //     .num_threads(1)
    //     .build_global()
    //     .unwrap();

    // let fmm = MultiNodeBuilder::new()
    //     .tree(
    //         &comm,
    //         points.data(),
    //         points.data(),
    //         local_depth,
    //         global_depth,
    //         prune_empty,
    //         sort_kind,
    //     )
    //     .unwrap()
    //     .parameters(expansion_order, kernel, source_to_target)
    //     .unwrap()
    //     .build()
    //     .unwrap();

    // Test that displacements are calculated correctly for local FMMs

    // Test that displacements are calculated correctly for Ghost FMMs

    // Test that displacements are calculated correctly for global FMM
}

#[cfg(not(feature = "mpi"))]
fn main() {}
