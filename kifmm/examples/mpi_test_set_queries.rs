#[cfg(feature = "mpi")]
fn main() {
    use green_kernels::laplace_3d::Laplace3dKernel;
    use itertools::Itertools;
    use kifmm::{
        fmm::types::MultiNodeBuilder,
        tree::{
            helpers::points_fixture,
            types::{MortonKey, SortKind},
        },
        // BlasFieldTranslationSaRcmp,
        FftFieldTranslation,
    };

    use rayon::ThreadPoolBuilder;

    use mpi::{collective::SystemOperation, traits::*};
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

    // Test U list queries

    // Test that requests are really not contained globally, but in global tree.
    for &query in fmm.tree.u_list_query.queries.iter() {
        let key = MortonKey::from_morton(query);
        let ancestors = key.ancestors();

        // Test that the containing rank is unique
        let intersection = ancestors
            .intersection(&fmm.tree.source_layout.raw_set)
            .collect_vec();
        assert_eq!(intersection.len(), 1);

        // Test that the containing rank is not this one
        let &rank = fmm
            .tree
            .source_layout
            .rank_from_key(intersection[0])
            .unwrap();
        assert_ne!(rank, fmm.rank);
    }

    // Test that send and receive counts are properly matched up
    let total_send_count = fmm.tree.u_list_query.send_counts.iter().sum::<i32>();
    let mut total_receive_count = 0;

    for rank in 0..comm.size() {
        fmm.communicator.all_reduce_into(
            &fmm.tree.u_list_query.receive_counts[rank as usize],
            &mut total_receive_count,
            SystemOperation::sum(),
        );
        if rank == fmm.rank {
            assert_eq!(total_receive_count, total_send_count);
        }
    }

    // Test that the queries are sorted by destination rank
    let mut found = Vec::new();
    for query in fmm.tree.u_list_query.queries.iter() {
        found.push(
            *fmm.tree
                .source_layout
                .rank_from_key(&MortonKey::from_morton(*query))
                .unwrap(),
        );
    }

    let mut min = found[0];
    for &item in found.iter() {
        assert!(item >= min);
        min = item
    }

    // Test V list queries

    // Test that requests are really not contained globally, but in global tree.
    for &query in fmm.tree.v_list_query.queries.iter() {
        let key = MortonKey::from_morton(query);
        let ancestors = key.ancestors();

        // Test that the containing rank is unique
        let intersection = ancestors
            .intersection(&fmm.tree.source_layout.raw_set)
            .collect_vec();
        assert_eq!(intersection.len(), 1);

        // Test that the containing rank is not this one
        let &rank = fmm
            .tree
            .source_layout
            .rank_from_key(intersection[0])
            .unwrap();
        assert_ne!(rank, fmm.rank);
    }
}

#[cfg(not(feature = "mpi"))]
fn main() {}
