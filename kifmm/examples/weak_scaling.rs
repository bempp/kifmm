#[cfg(feature = "mpi")]
fn main() {
    use green_kernels::{laplace_3d::Laplace3dKernel, traits::Kernel, types::GreenKernelEvalType};
    use kifmm::{
        fmm::types::MultiNodeBuilder,
        traits::{
            fmm::{DataAccessMulti, EvaluateMulti},
            general::multi_node::GhostExchange,
            tree::{MultiFmmTree, MultiTree},
        },
        tree::{
            helpers::points_fixture,
            types::{Domain, MortonKey, SortKind},
        },
        Evaluate, FftFieldTranslation, SingleNodeBuilder,
    };

    use rayon::ThreadPoolBuilder;

    use mpi::{datatype::PartitionMut, traits::*};
    use rlst::RawAccess;

    let (universe, _threading) = mpi::initialize_with_threading(mpi::Threading::Single).unwrap();
    let world = universe.world();
    let comm = world.duplicate();

    // Tree parameters
    let prune_empty = false;
    let n_points = 10000;
    let local_depth = 3;
    let global_depth = 2;
    let sort_kind = SortKind::Samplesort { k: 100 };

    // Fmm Parameters
    let expansion_order = 4;
    let kernel = Laplace3dKernel::<f32>::new();
    let source_to_target = FftFieldTranslation::<f32>::new(None);

    // Generate some random test data local to each process
    let points = points_fixture::<f32>(n_points, None, None, Some(world.rank() as u64));

    ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()
        .unwrap();

    let mut fmm = MultiNodeBuilder::new()
        .tree(
            &comm,
            points.data(),
            points.data(),
            local_depth.clone(),
            global_depth.clone(),
            prune_empty,
            sort_kind,
        )
        .unwrap()
        .parameters(expansion_order.clone(), kernel, source_to_target)
        .unwrap()
        .build()
        .unwrap();

    fmm.evaluate(true);

    println!("");
}

#[cfg(not(feature = "mpi"))]
fn main() {}
