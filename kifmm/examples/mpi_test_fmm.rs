#[cfg(feature = "mpi")]
fn main() {
    use green_kernels::laplace_3d::Laplace3dKernel;
    use kifmm::{
        fmm::types::MultiNodeBuilder,
        traits::fmm::EvaluateMulti,
        tree::{helpers::points_fixture, types::SortKind},
        FftFieldTranslation,
    };

    use mpi::traits::*;
    use rlst::RawAccess;

    let (universe, _threading) = mpi::initialize_with_threading(mpi::Threading::Funneled).unwrap();
    let world = universe.world();
    let comm = world.duplicate();

    // Tree parameters
    let prune_empty = true;
    let n_points = 10000;
    let local_depth = 1;
    let global_depth = 3;
    let sort_kind = SortKind::Samplesort { k: 100 };

    // Fmm Parameters
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
        .build()
        .unwrap();

    fmm.evaluate(false).unwrap();
}

#[cfg(not(feature = "mpi"))]
fn main() {}
