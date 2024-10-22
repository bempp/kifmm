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

    // Test FFT field translation
    {
        let source_to_target = FftFieldTranslation::<f32>::new(None);

        // Generate some random test data local to each process
        let points = points_fixture::<f32>(n_points, None, None, Some(world.rank() as u64));

        let mut multi_fmm = MultiNodeBuilder::new()
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

        multi_fmm.evaluate(false).unwrap();

        // Gather all coordinates for the test
        let root_process = comm.process_at_rank(0);
        let n_coords = multi_fmm.tree().source_tree().coordinates.len() as i32;
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

            let local_coords = multi_fmm.tree().source_tree().all_coordinates().unwrap();

            let mut partition = PartitionMut::new(
                &mut all_coordinates,
                coordinates_counts,
                coordinates_displacements,
            );

            root_process.gather_varcount_into_root(local_coords, &mut partition);
        } else {
            root_process.gather_into(&n_coords);

            let local_coords = multi_fmm.tree().source_tree().all_coordinates().unwrap();
            root_process.gather_varcount_into(local_coords);
        }

        if world.rank() == 0 {
            let mut single_fmm = SingleNodeBuilder::new()
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
            single_fmm.evaluate(false).unwrap();
            let mut expected = vec![0f32; &multi_fmm.tree.target_tree.coordinates.len() / 3];

            multi_fmm.kernel.evaluate_st(
                GreenKernelEvalType::Value,
                &all_coordinates,
                &multi_fmm.tree.target_tree.coordinates,
                &vec![1f32; n_points * world.size() as usize],
                &mut expected,
            );

            let distributed = &multi_fmm.potentials;

            distributed
                .iter()
                .zip(expected.iter())
                .for_each(|(f, e)| assert!(((f - e).abs() / e.abs()) < 1e-3));

            println!("...test_fmm M2L=FFT passed")

            // let leaf = &single_fmm.tree.target_tree.leaves[0];
            // let found_single = single_fmm.potential(&leaf);
            // let found = fmm.potential(&leaf);

            // println!(
            //     "{:?} expected: {:?} \n found: {:?} \n found single {:?}",
            //     world.rank(),
            //     leaf,
            //     found,
            //     found_single
            // );
        }
    }

    // Test BLAS field translation
    {
        // let source_to_target = FftFieldTranslation::<f32>::new(None);
        let source_to_target =
            BlasFieldTranslationSaRcmp::<f32>::new(None, None, kifmm::FmmSvdMode::Deterministic);

        // Generate some random test data local to each process
        let points = points_fixture::<f32>(n_points, None, None, Some(world.rank() as u64));

        let mut multi_fmm = MultiNodeBuilder::new()
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

        multi_fmm.evaluate(false).unwrap();

        // Gather all coordinates for the test
        let root_process = comm.process_at_rank(0);
        let n_coords = multi_fmm.tree().source_tree().coordinates.len() as i32;
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

            let local_coords = multi_fmm.tree().source_tree().all_coordinates().unwrap();

            let mut partition = PartitionMut::new(
                &mut all_coordinates,
                coordinates_counts,
                coordinates_displacements,
            );

            root_process.gather_varcount_into_root(local_coords, &mut partition);
        } else {
            root_process.gather_into(&n_coords);

            let local_coords = multi_fmm.tree().source_tree().all_coordinates().unwrap();
            root_process.gather_varcount_into(local_coords);
        }

        if world.rank() == 0 {
            let mut single_fmm = SingleNodeBuilder::new()
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
            single_fmm.evaluate(false).unwrap();
            let mut expected = vec![0f32; &multi_fmm.tree.target_tree.coordinates.len() / 3];

            multi_fmm.kernel.evaluate_st(
                GreenKernelEvalType::Value,
                &all_coordinates,
                &multi_fmm.tree.target_tree.coordinates,
                &vec![1f32; n_points * world.size() as usize],
                &mut expected,
            );

            let distributed = &multi_fmm.potentials;

            distributed
                .iter()
                .zip(expected.iter())
                .for_each(|(f, e)| assert!(((f - e).abs() / e.abs()) < 1e-3));

            // let leaf = &single_fmm.tree.target_tree.leaves[0];
            // let found_single = single_fmm.potential(&leaf);
            // let found = fmm.potential(&leaf);

            // println!(
            //     "{:?} expected: {:?} \n found: {:?} \n found single {:?}",
            //     world.rank(),
            //     leaf,
            //     found,
            //     found_single
            // );
            println!("...test_fmm M2L=BLAS passed")
        }
    }
}

#[cfg(not(feature = "mpi"))]
fn main() {}
