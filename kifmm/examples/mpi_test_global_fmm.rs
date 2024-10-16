#[cfg(feature = "mpi")]
fn main() {
    use green_kernels::laplace_3d::Laplace3dKernel;
    use kifmm::{
        fmm::types::MultiNodeBuilder,
        traits::fmm::EvaluateMulti,
        tree::{helpers::points_fixture, types::SortKind},
        // BlasFieldTranslationSaRcmp,
        FftFieldTranslation,
    };

    use rayon::ThreadPoolBuilder;

    use mpi::traits::*;
    use rlst::RawAccess;

    let (universe, _threading) = mpi::initialize_with_threading(mpi::Threading::Single).unwrap();
    // let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let comm = world.duplicate();

    // Tree parameters
    let prune_empty = true;
    let n_points = 10000;
    let local_depth = 2;
    let global_depth = 2;
    let sort_kind = SortKind::Samplesort { k: 100 };

    // Fmm Parameters
    let expansion_order = 4;
    let kernel = Laplace3dKernel::<f32>::new();
    let source_to_target = FftFieldTranslation::<f32>::new(None);
    // let source_to_target =
    //     BlasFieldTranslationSaRcmp::<f32>::new(None, None, kifmm::FmmSvdMode::Deterministic);

    // Generate some random test data local to each process
    let points = points_fixture::<f32>(n_points, None, None, None);

    ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()
        .unwrap();

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
        .build()
        .unwrap();

    fmm.evaluate(false).unwrap();

    // Test that the gathered global FMM contains all the expected multipole data required for
    // partial upward pass.
}

#[cfg(not(feature = "mpi"))]
fn main() {}
