#[cfg(feature = "mpi")]
fn main() {
    use green_kernels::{laplace_3d::Laplace3dKernel, traits::Kernel, types::GreenKernelEvalType};
    use kifmm::{
        fmm::types::MultiNodeBuilder,
        traits::{
            fmm::{DataAccessMulti, EvaluateMulti},
            tree::{MultiFmmTree, MultiTree},
        },
        tree::{helpers::points_fixture, types::SortKind},
        BlasFieldTranslationSaRcmp, Evaluate, FftFieldTranslation, SingleNodeBuilder,
    };

    use rayon::ThreadPoolBuilder;

    use mpi::{datatype::PartitionMut, traits::*};
    use rlst::RawAccess;

    let (universe, _threading) = mpi::initialize_with_threading(mpi::Threading::Single).unwrap();
    let world = universe.world();
    let comm = world.duplicate();

    // Tree parameters
    let prune_empty = true;
    let n_points = 10000;
    let local_depth = 3;
    let global_depth = 2;
    let sort_kind = SortKind::Samplesort { n_samples: 100 };

    // Fmm Parameters
    let expansion_order = 4;
    let kernel = Laplace3dKernel::<f32>::new();

    ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()
        .unwrap();

    let source_to_target = FftFieldTranslation::<f32>::new(None);

    // Generate some random test data local to each process
    let points = points_fixture::<f32>(n_points, None, None, Some(world.rank() as u64));

    let mut multi_fmm = MultiNodeBuilder::new(false)
        .tree(
            &comm,
            points.data(),
            points.data(),
            local_depth.clone(),
            global_depth.clone(),
            prune_empty,
            sort_kind.clone(),
        )
        .unwrap()
        .parameters(expansion_order.clone(), kernel.clone(), source_to_target)
        .unwrap()
        .build()
        .unwrap();

    multi_fmm.evaluate().unwrap();
}

#[cfg(not(feature = "mpi"))]
fn main() {}
