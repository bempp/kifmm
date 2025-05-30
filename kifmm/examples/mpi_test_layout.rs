//? mpirun -n {{NPROCESSES}} --features "mpi"

#[cfg(feature = "mpi")]
fn main() {
    use green_kernels::laplace_3d::Laplace3dKernel;
    use itertools::Itertools;
    use kifmm::{
        fmm::types::MultiNodeBuilder,
        traits::fmm::DataAccessMulti,
        tree::{
            helpers::points_fixture,
            types::{MortonKey, SortKind},
        },
        FftFieldTranslation,
    };

    use rayon::ThreadPoolBuilder;

    use mpi::{
        collective::SystemOperation,
        datatype::PartitionMut,
        traits::{Communicator, CommunicatorCollectives},
    };
    use rlst::RawAccess;

    let (universe, _threading) = mpi::initialize_with_threading(mpi::Threading::Single).unwrap();
    let world = universe.world();
    let comm = world.duplicate();

    if world.size() >= 2 {
        // Tree parameters
        let prune_empty = true;
        let n_points = 10000;
        let local_depth = 2;
        let global_depth = 2;
        let sort_kind = SortKind::Samplesort { n_samples: 100 };

        // Fmm Parameters
        let expansion_order = [4];
        let kernel = Laplace3dKernel::<f32>::new();
        let source_to_target = FftFieldTranslation::<f32>::new(None);

        // Generate some random test data local to each process
        let points = points_fixture::<f32>(n_points, None, None, None);
        let charges = vec![1f32; n_points];

        ThreadPoolBuilder::new()
            .num_threads(1)
            .build_global()
            .unwrap();

        // Queries are set as a part of the build
        let fmm = MultiNodeBuilder::new(false)
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
            .parameters(
                &charges,
                &expansion_order,
                kernel,
                green_kernels::types::GreenKernelEvalType::Value,
                source_to_target,
            )
            .unwrap()
            .build()
            .unwrap();

        // Gather layout manually for testing
        let local_domain = &fmm.tree.source_tree.roots;
        let n_local_domain = local_domain.len() as i32;

        let mut n_global_domain = vec![0i32; comm.size() as usize];

        comm.all_gather_into(&n_local_domain, &mut n_global_domain);

        // Now communicate the actual roots
        let n_global_roots = n_global_domain.iter().sum::<i32>();
        let mut global_roots = vec![MortonKey::<f32>::default(); n_global_roots as usize];

        let counts = n_global_domain;

        let mut displacement = 0;
        let mut displacements = Vec::new();
        for count in counts.iter() {
            displacements.push(displacement);
            displacement += *count;
        }

        let mut partition = PartitionMut::new(&mut global_roots, &counts[..], &displacements[..]);

        comm.all_gather_varcount_into(&local_domain[..], &mut partition);

        let mut ranks = Vec::new();
        for (i, &count) in counts.iter().enumerate() {
            ranks.append(&mut vec![i as i32; count as usize])
        }

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
            assert_ne!(rank, fmm.rank());

            // Calculate rank from layout directly, and test that it is
            // as expected
            let rank_idx = global_roots
                .iter()
                .position(|&x| x.morton == intersection[0].morton)
                .unwrap();
            let expected_rank = ranks[rank_idx];
            assert_eq!(expected_rank, rank);
        }

        // Test that send and receive counts are properly matched up
        let total_send_count = fmm.tree.u_list_query.send_counts.iter().sum::<i32>();
        let mut total_receive_count = 0;

        for rank in 0..comm.size() {
            fmm.communicator().all_reduce_into(
                &fmm.tree.u_list_query.receive_counts[rank as usize],
                &mut total_receive_count,
                SystemOperation::sum(),
            );
            if rank == fmm.rank() {
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

        if comm.rank() == 0 {
            println!("...test_layout_u_list_queries passed")
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
            assert_ne!(rank, fmm.rank());

            // Calculate rank from layout directly, and test that it is
            // as expected
            let rank_idx = global_roots
                .iter()
                .position(|&x| x.morton == intersection[0].morton)
                .unwrap();
            let expected_rank = ranks[rank_idx];
            assert_eq!(expected_rank, rank);
        }

        // Test that send and receive counts are properly matched up
        let total_send_count = fmm.tree.v_list_query.send_counts.iter().sum::<i32>();
        let mut total_receive_count = 0;

        for rank in 0..comm.size() {
            fmm.communicator().all_reduce_into(
                &fmm.tree.v_list_query.receive_counts[rank as usize],
                &mut total_receive_count,
                SystemOperation::sum(),
            );
            if rank == fmm.rank() {
                assert_eq!(total_receive_count, total_send_count);
            }
        }

        // Test that the queries are sorted by destination rank
        let mut found = Vec::new();
        for query in fmm.tree.v_list_query.queries.iter() {
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

        if comm.rank() == 0 {
            println!("...test_layout_v_list_queries passed")
        }
    }
}

#[cfg(not(feature = "mpi"))]
fn main() {}
