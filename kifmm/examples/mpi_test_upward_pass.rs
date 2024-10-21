#[cfg(feature = "mpi")]
fn main() {
    use green_kernels::{laplace_3d::Laplace3dKernel, traits::Kernel, types::GreenKernelEvalType};
    use kifmm::traits::general::multi_node::GhostExchange;
    use kifmm::{
        fmm::types::MultiNodeBuilder,
        traits::{
            fmm::{DataAccess, DataAccessMulti, EvaluateMulti},
            tree::{FmmTreeNode, MultiFmmTree, MultiTree, SingleFmmTree, SingleTree},
        },
        tree::{constants::ALPHA_INNER, helpers::points_fixture, types::SortKind},
        Evaluate,
        // FftFieldTranslation,
    };
    use kifmm::{BlasFieldTranslationSaRcmp, FmmSvdMode};
    use mpi::{datatype::PartitionMut, traits::*};
    use rlst::{RawAccess, RlstScalar};

    let (universe, _threading) = mpi::initialize_with_threading(mpi::Threading::Funneled).unwrap();
    let world = universe.world();
    let comm = world.duplicate();

    // Tree parameters
    let prune_empty = true;
    let n_points = 10000;
    let local_depth = 3;
    let global_depth = 1;
    let sort_kind = SortKind::Samplesort { k: 100 };

    // Fmm Parameters
    let expansion_order = 4;
    let kernel = Laplace3dKernel::<f32>::new();
    // let source_to_target = FftFieldTranslation::<f32>::new(None);
    let source_to_target =
        BlasFieldTranslationSaRcmp::<f32>::new(None, None, FmmSvdMode::Deterministic);

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

    // Perform partial upward pass on each rank
    fmm.evaluate_leaf_sources(false).unwrap();
    fmm.evaluate_upward_pass(false).unwrap();

    // Test at roots of local trees for result of partial upward passes
    let roots = fmm.tree().source_tree().roots();

    let distant_point = vec![100000., 0., 0.];
    let mut expected = vec![0.];
    let mut found = vec![0.];

    for root in roots.iter() {
        let multipole = fmm.multipole(root).unwrap();

        let upward_equivalent_surface = root.surface_grid(
            fmm.equivalent_surface_order,
            fmm.tree().domain(),
            ALPHA_INNER as f32,
        );

        fmm.kernel().evaluate_st(
            GreenKernelEvalType::Value,
            &upward_equivalent_surface,
            &distant_point,
            multipole,
            &mut found,
        );
    }

    let coords = fmm.tree().source_tree().all_coordinates().unwrap();
    let charges = vec![1f32; coords.len() / 3];

    fmm.kernel().evaluate_st(
        GreenKernelEvalType::Value,
        &coords,
        &distant_point,
        &charges,
        &mut expected,
    );

    let abs_error = RlstScalar::abs(expected[0] - found[0]);
    let rel_error = abs_error / expected[0];

    if world.rank() == 0 {
        println!(
            "Local Upward Pass rank {:?} abs {:?} rel {:?} \n expected {:?} found {:?}",
            world.rank(),
            abs_error,
            rel_error,
            expected,
            found
        );
    }

    let threshold = 1e-3;
    assert!(rel_error <= threshold);

    // Gather all coordinates for the test
    let root_process = fmm.communicator.process_at_rank(0);
    let n_coords = fmm.tree().source_tree().coordinates.len() as i32;
    let mut all_coordinates = vec![0f32; 3 * n_points * world.size() as usize];

    if world.rank() == 0 {
        let mut coordinates_counts = vec![0i32; fmm.communicator.size() as usize];
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

    // Gather global FMM
    fmm.gather_global_fmm_at_root();

    // Perform upward pass on global fmm
    if world.rank() == 0 {
        fmm.global_fmm.evaluate_upward_pass(false).unwrap();

        // Test that all multipoles are the same
        // test root multipole
        let root = fmm.global_fmm.tree().source_tree().root();
        let multipole = fmm.global_fmm.multipole(&root).unwrap();

        let distant_point = vec![100000., 0., 0.];
        let mut expected = vec![0.];
        let mut found = vec![0.];

        let upward_equivalent_surface = root.surface_grid(
            fmm.equivalent_surface_order,
            fmm.tree().domain(),
            ALPHA_INNER as f32,
        );

        fmm.kernel().evaluate_st(
            GreenKernelEvalType::Value,
            &upward_equivalent_surface,
            &distant_point,
            multipole,
            &mut found,
        );

        fmm.kernel().evaluate_st(
            GreenKernelEvalType::Value,
            &all_coordinates,
            &distant_point,
            &vec![1.0; n_points * world.size() as usize],
            &mut expected,
        );

        let abs_error = RlstScalar::abs(expected[0] - found[0]);
        let rel_error = abs_error / expected[0];

        if world.rank() == 0 {
            println!(
                "Global Upward Pass rank {:?} abs {:?} rel {:?} \n expected {:?} found {:?}",
                world.rank(),
                abs_error,
                rel_error,
                expected,
                found
            );
        }

        let threshold = 1e-3;
        assert!(rel_error <= threshold);
    }
}

#[cfg(not(feature = "mpi"))]
fn main() {}
