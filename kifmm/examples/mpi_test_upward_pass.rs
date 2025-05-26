//? mpirun -n {{NPROCESSES}} --features "mpi"

#[cfg(feature = "mpi")]
fn main() {
    use green_kernels::helmholtz_3d::Helmholtz3dKernel;
    use green_kernels::{laplace_3d::Laplace3dKernel, traits::Kernel, types::GreenKernelEvalType};
    use itertools::izip;
    use kifmm::traits::general::multi_node::GhostExchange;
    use kifmm::{
        fmm::types::MultiNodeBuilder,
        traits::{
            fmm::{DataAccess, DataAccessMulti, EvaluateMulti},
            tree::{FmmTreeNode, MultiFmmTree, MultiTree, SingleFmmTree, SingleTree},
        },
        tree::{constants::ALPHA_INNER, helpers::points_fixture, types::SortKind},
        Evaluate, FftFieldTranslation,
    };
    use kifmm::{BlasFieldTranslationSaRcmp, FmmSvdMode, MultiNodeFmmTree};
    use mpi::collective::SystemOperation;
    use mpi::topology::SimpleCommunicator;
    use mpi::traits::{CommunicatorCollectives, Equivalence};
    use mpi::{
        datatype::PartitionMut,
        traits::{Communicator, Root},
    };
    use mpi_sys::RSMPI_SUM;
    use num_complex::Complex32;
    use num::{Float, One};
    use rlst::{c32, RawAccess, RlstScalar};

    fn test_multi_node_helmholtz_upward_pass_helper<T>(
        name: String,
        mut fmm: Box<
            dyn EvaluateMulti<
                Scalar = T,
                Kernel = Helmholtz3dKernel<T>,
                Tree = MultiNodeFmmTree<T::Real, SimpleCommunicator>,
            >,
        >,
        eval_type: GreenKernelEvalType,
        threshold: T::Real,
    ) where
        T: RlstScalar<Complex = T> + Equivalence,
        <T as RlstScalar>::Real: Equivalence + Float,
    {
        // Test the global part of the upward pass
        let alpha_inner = T::from(ALPHA_INNER).unwrap().re();

        // Perform partial upward pass on each rank
        fmm.evaluate_leaf_sources().unwrap();
        fmm.evaluate_upward_pass().unwrap();

        // Test at roots of local trees for result of partial upward passes
        let roots = fmm.tree().source_tree().roots();

        let distant_point = vec![T::real(10000.),  T::real(0.0),  T::real(0.0)];
        let mut expected = vec![T::zero()];
        let mut found = vec![T::zero()];

        for root in roots.iter() {
            let multipole = fmm.multipole(root).unwrap();

            let upward_equivalent_surface = root.surface_grid(
                fmm.equivalent_surface_order(fmm.tree().source_tree().global_depth()),
                fmm.tree().domain(),
                alpha_inner,
            );

            fmm.kernel().evaluate_st(
                eval_type,
                &upward_equivalent_surface,
                &distant_point,
                multipole,
                &mut found,
            );
        }

        let coords = fmm.tree().source_tree().all_coordinates().unwrap();
        let charges = vec![T::one(); coords.len() / 3];

        fmm.kernel().evaluate_st(
            eval_type,
            coords,
            &distant_point,
            &charges,
            &mut expected,
        );

        let mut num = T::real(0.0);
        let mut den = T::real(0.0);

        for (&expected, &found) in izip!(&expected, &found) {
            // squared error in complex difference
            let diff_re = expected.re() - found.re();
            let diff_im = expected.im() - found.im();
            num += RlstScalar::powf(diff_re, T::real(2.0)) + RlstScalar::powf(diff_im, T::real(2.0));

            // squared magnitude of expected
            den += RlstScalar::powf(expected.re(), T::real(2.0)) + RlstScalar::powf(expected.im(), T::real(2.0));
        }


        // now take square root
        let l2_error = if den != T::real(0.0) {
            RlstScalar::sqrt(num) / RlstScalar::sqrt(den)
        } else {
            T::real(0.0) // or handle division-by-zero error
        };

        // if fmm.rank() == 0 {
        // }

            println!(
                "Local Upward Pass rank {:?} l2 err {:?} \n expected {:?} found {:?}",
                fmm.rank(),
                l2_error,
                expected,
                found
            );
        assert!(l2_error <= threshold);
        println!("...test_upward_pass_global_tree {} passed", name);


    }


    fn test_multi_node_laplace_upward_pass_helper<T: RlstScalar<Real = T> + Float + Default + Equivalence>(
        name: String,
        mut fmm: Box<
            dyn EvaluateMulti<
                Scalar = T,
                Kernel = Laplace3dKernel<T>,
                Tree = MultiNodeFmmTree<T, SimpleCommunicator>,
            >,
        >,
        eval_type: GreenKernelEvalType,
        threshold: T,
    ) where
        <T as RlstScalar>::Real: Equivalence,
    {

        // Test the global part of the upward pass
        let alpha_inner = T::from(ALPHA_INNER).unwrap();

        // Perform partial upward pass on each rank
        fmm.evaluate_leaf_sources().unwrap();
        fmm.evaluate_upward_pass().unwrap();

        // Test at roots of local trees for result of partial upward passes
        let roots = fmm.tree().source_tree().roots();

        let distant_point = vec![T::from(10000.).unwrap(), T::zero(), T::zero()];
        let mut expected = vec![T::zero()];
        let mut found = vec![T::zero()];

        for root in roots.iter() {
            let multipole = fmm.multipole(root).unwrap();

            let upward_equivalent_surface = root.surface_grid(
                fmm.equivalent_surface_order(fmm.tree().source_tree().global_depth()),
                fmm.tree().domain(),
                alpha_inner,
            );

            fmm.kernel().evaluate_st(
                eval_type,
                &upward_equivalent_surface,
                &distant_point,
                multipole,
                &mut found,
            );
        }

        let coords = fmm.tree().source_tree().all_coordinates().unwrap();
        let charges = vec![T::one(); coords.len() / 3];

        fmm.kernel().evaluate_st(
            eval_type,
            coords,
            &distant_point,
            &charges,
            &mut expected,
        );

        let mut num = T::real(0.0);
        let mut den = T::real(0.0);
        for (&expected, &found) in izip!(&expected, &found) {
            num += RlstScalar::powf(RlstScalar::abs(expected - found), T::real(2.0));
            den += RlstScalar::powf(RlstScalar::abs(expected), T::real(2.0));
        }

        let l2_error = RlstScalar::powf(num / den, T::real(0.5));

        if fmm.rank() == 0 {
            println!(
                "Local Upward Pass rank {:?} l2 err {:?} \n expected {:?} found {:?}",
                fmm.rank(),
                l2_error,
                expected,
                found
            );
        }

        assert!(l2_error <= threshold);
        println!("...test_upward_pass_global_tree {} passed", name);

        // // Gather all coordinates for the test
        // let root_process = fmm.communicator().process_at_rank(0);
        // // let n_coords = fmm.tree().source_tree().coordinates.len() as i32;

        // // Communicate total number of coordinates
        // let mut n_coords_tot = 0;
        // let n_coords_r = fmm.tree().source_tree().n_coordinates_tot().unwrap_or(0) as i32;
        // fmm.communicator()
        //     .all_reduce_into(&n_coords_r, &mut n_coords_tot, SystemOperation::sum());
        // let mut all_coordinates =
        //     vec![<T as RlstScalar>::Real::from(0.0).unwrap(); (n_coords_tot as usize)];

        // if fmm.rank() == 0 {
        //     let mut coordinates_counts = vec![0i32; fmm.communicator().size() as usize];
        //     root_process.gather_into_root(&n_coords_r, &mut coordinates_counts);

        //     let mut coordinates_displacements = Vec::new();
        //     let mut counter = 0;
        //     for &count in coordinates_counts.iter() {
        //         coordinates_displacements.push(counter);
        //         counter += count;
        //     }

        //     let local_coords = fmm.tree().source_tree().all_coordinates().unwrap();

        //     let mut partition = PartitionMut::new(
        //         &mut all_coordinates,
        //         coordinates_counts,
        //         coordinates_displacements,
        //     );

        //     root_process.gather_varcount_into_root(local_coords, &mut partition);
        // } else {
        //     root_process.gather_into(&n_coords_r);

        //     let local_coords = fmm.tree().source_tree().all_coordinates().unwrap();
        //     root_process.gather_varcount_into(local_coords);
        // }

        // Gather global FMM
        // fmm.gather_global_fmm_at_root();

        // Perform upward pass on global fmm
        // if fmm.rank() == 0 {
        //     fmm.global_fmm.evaluate_upward_pass().unwrap();

        //     // Test that all multipoles are the same
        //     // test root multipole
        //     let root = fmm.global_fmm.tree().source_tree().root();
        //     let multipole = fmm.global_fmm.multipole(&root).unwrap();

        //     let distant_point = vec![100000., 0., 0.];
        //     let mut expected = vec![0.];
        //     let mut found = vec![0.];

        //     let upward_equivalent_surface = root.surface_grid(
        //         fmm.equivalent_surface_order(0),
        //         fmm.tree().domain(),
        //         ALPHA_INNER as f32,
        //     );

        //     fmm.kernel().evaluate_st(
        //         GreenKernelEvalType::Value,
        //         &upward_equivalent_surface,
        //         &distant_point,
        //         multipole,
        //         &mut found,
        //     );

        //     fmm.kernel().evaluate_st(
        //         GreenKernelEvalType::Value,
        //         &all_coordinates,
        //         &distant_point,
        //         &vec![1.0; n_points * world.size() as usize],
        //         &mut expected,
        //     );

        //     let abs_error = RlstScalar::abs(expected[0] - found[0]);
        //     let rel_error = abs_error / expected[0];

        //     println!(
        //         "Global Upward Pass rank {:?} abs {:?} rel {:?} \n expected {:?} found {:?}",
        //         world.rank(),
        //         abs_error,
        //         rel_error,
        //         expected,
        //         found
        //     );

        //     let threshold = 1e-3;
        //     assert!(rel_error <= threshold);

    }

    // // Test Laplace FMM
    // // N.B global tree refined to depth 3 to ensure that the global upward pass is also being run
    // {
    //     let (universe, _threading) =
    //         mpi::initialize_with_threading(mpi::Threading::Single).unwrap();
    //     let world = universe.world();
    //     let comm = world.duplicate();

    //     let n_points = 10000;
    //     let charges = vec![1f32; n_points];
    //     let eval_type = GreenKernelEvalType::Value;
    //     let source_to_target = FftFieldTranslation::new(None);
    //     let sources = points_fixture(n_points, None, None, None);
    //     let local_depth = 3;
    //     let global_depth = 3;
    //     let prune_empty = true;

    //     // Single expansion order
    //     {
    //         let expansion_order = [5];

    //         let fmm = MultiNodeBuilder::new(false)
    //             .tree(
    //                 &comm.duplicate(),
    //                 sources.data(),
    //                 sources.data(),
    //                 local_depth,
    //                 global_depth,
    //                 prune_empty,
    //                 SortKind::Samplesort { n_samples: 10 },
    //             )
    //             .unwrap()
    //             .parameters(
    //                 &charges,
    //                 &expansion_order,
    //                 Laplace3dKernel::new(),
    //                 eval_type,
    //                 source_to_target.clone(),
    //             )
    //             .unwrap()
    //             .build()
    //             .unwrap();

    //         test_multi_node_laplace_upward_pass_helper(
    //             "fixed_expansion_order".to_string(),
    //             Box::new(fmm),
    //             eval_type,
    //             1e-4,
    //         );
    //     }

    //     // Test case with multiple expansion orders which vary by level
    //     {
    //         let expansion_order = [4, 4, 5, 4, 5, 4, 5];
    //         assert!(expansion_order.len() == (global_depth + local_depth + 1).try_into().unwrap());

    //         let fmm = MultiNodeBuilder::new(false)
    //             .tree(
    //                 &comm.duplicate(),
    //                 sources.data(),
    //                 sources.data(),
    //                 local_depth,
    //                 global_depth,
    //                 prune_empty,
    //                 SortKind::Samplesort { n_samples: 10 },
    //             )
    //             .unwrap()
    //             .parameters(
    //                 &charges,
    //                 &expansion_order,
    //                 Laplace3dKernel::new(),
    //                 eval_type,
    //                 source_to_target.clone(),
    //             )
    //             .unwrap()
    //             .build()
    //             .unwrap();

    //         test_multi_node_laplace_upward_pass_helper(
    //             "fixed_expansion_order".to_string(),
    //             Box::new(fmm),
    //             eval_type,
    //             1e-4,
    //         );
    //     }
    // }


    // Test Helmholtz FMM
    {

        let (universe, _threading) =
            mpi::initialize_with_threading(mpi::Threading::Single).unwrap();
        let world = universe.world();
        let comm = world.duplicate();

        let n_points = 10000;
        let charges = vec![Complex32::one(); n_points];
        let eval_type = GreenKernelEvalType::Value;
        let source_to_target = FftFieldTranslation::new(None);
        let sources = points_fixture(n_points, None, None, None);
        let local_depth = 3;
        let global_depth = 3;
        let prune_empty = true;

        // Single expansion order
        {
            let expansion_order = [5];

            let fmm = MultiNodeBuilder::new(false)
                .tree(
                    &comm.duplicate(),
                    sources.data(),
                    sources.data(),
                    local_depth,
                    global_depth,
                    prune_empty,
                    SortKind::Samplesort { n_samples: 10 },
                )
                .unwrap()
                .parameters(
                    &charges,
                    &expansion_order,
                    Helmholtz3dKernel::new(1.0),
                    eval_type,
                    source_to_target.clone(),
                )
                .unwrap()
                .build()
                .unwrap();

            test_multi_node_helmholtz_upward_pass_helper(
                "fixed_expansion_order".to_string(),
                Box::new(fmm),
                eval_type,
                1e-3,
            );
        }



    }
}

#[cfg(not(feature = "mpi"))]
fn main() {}
