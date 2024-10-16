#[cfg(feature = "mpi")]
fn main() {
    use green_kernels::laplace_3d::Laplace3dKernel;
    use kifmm::{
        fmm::types::MultiNodeBuilder,
        traits::tree::{MultiFmmTree, SingleFmmTree, SingleTree},
        tree::{
            helpers::points_fixture,
            types::{MortonKey, SortKind},
        },
        FftFieldTranslation,
    };

    use rayon::ThreadPoolBuilder;

    use mpi::{collective::SystemOperation, traits::*};
    use rlst::RawAccess;

    let (universe, _threading) = mpi::initialize_with_threading(mpi::Threading::Single).unwrap();
    // let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let comm = world.duplicate();
    let rank = comm.rank();

    // Tree parameters
    let prune_empty = true;
    let n_points = 10000;
    let local_depth = 3;
    let global_depth = 1;
    let sort_kind = SortKind::Samplesort { k: 1000 };

    // Fmm Parameters
    let expansion_order = 4;
    let kernel = Laplace3dKernel::<f32>::new();
    let source_to_target = FftFieldTranslation::<f32>::new(None);

    // Generate some random test data local to each process
    let points = points_fixture::<f32>(n_points, None, None, None);

    ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()
        .unwrap();

    // Queries are set as a part of the build
    let fmm = MultiNodeBuilder::new()
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

    // Test neighbourhood communicator setup

    // Expect these to match the global communicator
    assert_eq!(
        fmm.tree.u_list_query.send_counts.len(),
        fmm.tree.source_tree().comm.size() as usize
    );
    assert_eq!(
        fmm.tree.u_list_query.receive_counts.len(),
        fmm.tree.source_tree().comm.size() as usize
    );
    assert_eq!(
        fmm.tree.v_list_query.send_counts.len(),
        fmm.tree.source_tree().comm.size() as usize
    );
    assert_eq!(
        fmm.tree.v_list_query.receive_counts.len(),
        fmm.tree.source_tree().comm.size() as usize
    );

    // Test that the received leaves in the U ghost tree are a subset of the requested leaves, and that they
    // are associated with charge/coordinate data
    for leaf in fmm.ghost_fmm_u.tree.source_tree().leaves.iter() {
        assert!(fmm.tree.u_list_query.queries.contains(&leaf.morton));
        assert!(fmm
            .ghost_fmm_u
            .tree
            .source_tree()
            .coordinates(leaf)
            .is_some());
    }
}

#[cfg(not(feature = "mpi"))]
fn main() {}
