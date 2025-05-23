//? mpirun -n {{NPROCESSES}} --features "mpi"

#[cfg(feature = "mpi")]
fn main() {
    use green_kernels::{laplace_3d::Laplace3dKernel, traits::Kernel, types::GreenKernelEvalType};
    use itertools::izip;
    use kifmm::{
        fmm::types::MultiNodeBuilder,
        traits::{
            fmm::{DataAccessMulti, EvaluateMulti},
            tree::{MultiFmmTree, MultiTree},
        },
        tree::{helpers::points_fixture, types::SortKind},
        FftFieldTranslation,
    };

    use mpi::{
        datatype::PartitionMut,
        traits::{Communicator, Root},
    };
    use rlst::RawAccess;

    fn test_fmm(
        name: String,
        expansion_order: &[usize],
        world: &mpi::topology::SimpleCommunicator,
    ) {
        let comm = world.duplicate();

        // Tree parameters
        let prune_empty = true;
        let n_points = 10000;
        let local_depth = 3;
        let global_depth = 2;
        let sort_kind = SortKind::Samplesort { n_samples: 100 };

        // Fmm Parameters

        let kernel = Laplace3dKernel::<f32>::new();
        // let source_to_target =
        //     BlasFieldTranslationSaRcmp::<f32>::new(None, None, kifmm::FmmSvdMode::Deterministic);
        let source_to_target = FftFieldTranslation::<f32>::new(None);

        // Generate some random test data local to each process
        let points = points_fixture::<f32>(n_points, None, None, None);
        let charges = vec![1f32; n_points];

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
                &expansion_order,
                kernel,
                GreenKernelEvalType::Value,
                source_to_target,
            )
            .unwrap()
            .build()
            .unwrap();

        // Perform upward and downward passes
        fmm.evaluate().unwrap();

        // Gather all coordinates for the test
        let root_process = fmm.communicator().process_at_rank(0);
        let n_coords = fmm.tree().source_tree().coordinates.len() as i32;
        let mut all_coordinates = vec![0f32; 3 * n_points * world.size() as usize];

        if world.rank() == 0 {
            let mut coordinates_counts = vec![0i32; fmm.communicator().size() as usize];
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

        // Perform upward pass on global fmm
        if world.rank() == 0 {
            let targets = fmm.tree().target_tree().all_coordinates().unwrap();
            let mut expected = vec![0f32; targets.len() / 3];

            fmm.kernel().evaluate_st(
                GreenKernelEvalType::Value,
                &all_coordinates,
                &fmm.tree().target_tree().all_coordinates().unwrap(),
                &vec![1f32; n_points * (world.size() as usize)],
                &mut expected,
            );

            let found = fmm.potentials().unwrap();

            let mut num = 0.0;
            let mut den = 0.0;
            for (expected, found) in izip!(&expected, found) {
                num += (expected - found).abs().powf(2.0);
                den += expected.abs().powf(2.0);
            }

            let l2_error = (num / den).powf(0.5);

            println!(
                "Global Upward Pass rank {:?} l2 {:?} \n expected {:?} found {:?}",
                world.rank(),
                l2_error,
                &expected[0..5],
                &found[0..5]
            );

            println!("...test_fmm_{} passed", name);
        }
    }

    let (universe, _threading) = mpi::initialize_with_threading(mpi::Threading::Funneled).unwrap();
    let world = universe.world();

    // let expansion_order = [4];
    // test_fmm(
    //     "single expansion order all levels".to_string(),
    //     &expansion_order,
    //     &world,
    // );

    let expansion_order = [4, 4, 5, 4, 5, 4];
    test_fmm(
        "variable expansion order per level".to_string(),
        &expansion_order,
        &world,
    );
}

#[cfg(not(feature = "mpi"))]
fn main() {}
