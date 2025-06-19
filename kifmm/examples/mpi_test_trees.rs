//? mpirun -n {{NPROCESSES}} --features "mpi"

#[cfg(feature = "mpi")]
fn main() {
    use green_kernels::{laplace_3d::Laplace3dKernel, types::GreenKernelEvalType};
    use kifmm::{
        fmm::types::MultiNodeBuilder,
        traits::{
            fmm::{DataAccess, DataAccessMulti},
            general::multi_node::GhostExchange,
            tree::{MultiFmmTree, MultiTree},
        },
        tree::{
            helpers::points_fixture,
            types::{MortonKey, SortKind},
        },
        FftFieldTranslation, SingleNodeBuilder,
    };

    use rayon::ThreadPoolBuilder;

    use mpi::{
        datatype::PartitionMut,
        traits::{Communicator, Root},
    };
    use rlst::RawAccess;

    let (universe, _threading) = mpi::initialize_with_threading(mpi::Threading::Single).unwrap();
    let world = universe.world();
    let comm = world.duplicate();

    // Tree parameters
    let prune_empty = false;
    let n_points = 100000;
    let local_depth = 3;
    let global_depth = 2;
    let sort_kind = SortKind::Samplesort { n_samples: 100 };

    // Fmm Parameters
    let expansion_order = 4;
    let kernel = Laplace3dKernel::<f32>::new();
    let source_to_target = FftFieldTranslation::<f32>::new(None);

    // Generate some random test data local to each process
    let points = points_fixture::<f32>(n_points, None, None, Some(world.rank() as u64));
    let charges = vec![1f32; n_points];

    ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()
        .unwrap();

    let mut fmm = MultiNodeBuilder::new(false)
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
            &[expansion_order],
            kernel,
            GreenKernelEvalType::Value,
            source_to_target,
        )
        .unwrap()
        .build()
        .unwrap();

    fmm.gather_global_fmm_at_root();

    // Gather all coordinates for the test
    let root_process = comm.process_at_rank(0);
    let n_coords = fmm.tree().source_tree().coordinates.len() as i32;
    let mut all_coordinates = vec![0f32; 3 * n_points * world.size() as usize];

    if world.rank() == 0 {
        let mut coordinates_counts = vec![0i32; comm.size() as usize];
        root_process.gather_into_root(&n_coords, &mut coordinates_counts);

        let mut coordinates_displacements = Vec::new();
        let mut counter = 0;
        for &count in coordinates_counts.iter() {
            coordinates_displacements.push(counter);
            counter += count;
        }

        let local_coords = fmm.tree().source_tree().all_coordinates().unwrap();

        let mut partition = PartitionMut::new(
            &mut all_coordinates,
            coordinates_counts,
            coordinates_displacements,
        );

        root_process.gather_varcount_into_root(local_coords, &mut partition);
    } else {
        root_process.gather_into(&n_coords);

        let local_coords = fmm.tree().source_tree().all_coordinates().unwrap();
        root_process.gather_varcount_into(local_coords);
    }

    // Gather all keys
    let mut keys_counts = vec![0i32; world.size() as usize];
    if world.rank() == 0 {
        let n_keys = fmm.tree.source_tree.keys.len() as i32;

        root_process.gather_into_root(&n_keys, &mut keys_counts);
    } else {
        let n_keys = fmm.tree.source_tree.keys.len() as i32;
        root_process.gather_into(&n_keys);
    }

    let mut all_keys = vec![MortonKey::<f32>::default(); keys_counts.iter().sum::<i32>() as usize];

    if world.rank() == 0 {
        let mut keys_displacements = Vec::new();
        let mut displacement = 0;
        for count in keys_counts.iter() {
            keys_displacements.push(displacement);
            displacement += count;
        }

        let mut partition = PartitionMut::new(&mut all_keys, keys_counts, keys_displacements);

        let keys = &fmm.tree.source_tree.keys.keys;
        root_process.gather_varcount_into_root(&keys[..], &mut partition);
    } else {
        let keys = &fmm.tree.source_tree.keys.keys;
        root_process.gather_varcount_into(&keys[..]);
    }

    // Gather all leaves
    let mut leaves_counts = vec![0i32; world.size() as usize];
    if world.rank() == 0 {
        let n_leaves = fmm.tree.source_tree.leaves.len() as i32;

        root_process.gather_into_root(&n_leaves, &mut leaves_counts);
    } else {
        let n_leaves = fmm.tree.source_tree.leaves.len() as i32;
        root_process.gather_into(&n_leaves);
    }

    let mut all_leaves =
        vec![MortonKey::<f32>::default(); leaves_counts.iter().sum::<i32>() as usize];

    if world.rank() == 0 {
        let mut leaves_displacements = Vec::new();
        let mut displacement = 0;
        for count in leaves_counts.iter() {
            leaves_displacements.push(displacement);
            displacement += count;
        }

        let mut partition = PartitionMut::new(&mut all_leaves, leaves_counts, leaves_displacements);

        let leaves = &fmm.tree.source_tree.leaves.keys;
        root_process.gather_varcount_into_root(&leaves[..], &mut partition);
    } else {
        let leaves = &fmm.tree.source_tree.leaves.keys;
        root_process.gather_varcount_into(&leaves[..]);
    }

    if world.rank() == 0 {
        let single_fmm = SingleNodeBuilder::new(false)
            .tree(
                &all_coordinates,
                &all_coordinates,
                None,
                Some(local_depth + global_depth),
                prune_empty,
            )
            .unwrap()
            .parameters(
                &vec![1f32; all_coordinates.len() / 3],
                &vec![expansion_order; (local_depth + global_depth + 1) as usize],
                Laplace3dKernel::new(),
                GreenKernelEvalType::Value,
                FftFieldTranslation::new(None),
            )
            .unwrap()
            .build()
            .unwrap();

        // Test that all keys are contained
        for key in all_keys.iter() {
            assert!(single_fmm.tree().source_tree.keys_set.contains(key))
        }
        println!("...test_keys passed");

        // Test that all leaves span the same domain as if constructed locally
        // assert_eq!(single_fmm.tree().source_tree.leaves.len(), all_leaves.len());

        for leaf in all_leaves.iter() {
            assert!(single_fmm.tree().source_tree.leaves_set.contains(leaf))
        }

        println!("...test_leaves passed");
    }
}

#[cfg(not(feature = "mpi"))]
fn main() {}
