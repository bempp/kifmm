//? mpirun -n {{NPROCESSES}} --features "mpi"

#[cfg(feature = "mpi")]
fn main() {
    use green_kernels::helmholtz_3d::Helmholtz3dKernel;
    use green_kernels::{laplace_3d::Laplace3dKernel, traits::Kernel, types::GreenKernelEvalType};
    use itertools::izip;
    use kifmm::MultiNodeFmmTree;
    use kifmm::{
        fmm::types::MultiNodeBuilder,
        traits::{
            fmm::EvaluateMulti,
            tree::{FmmTreeNode, MultiFmmTree, MultiTree},
        },
        tree::{constants::ALPHA_INNER, helpers::points_fixture, types::SortKind},
        FftFieldTranslation,
    };
    use mpi::topology::SimpleCommunicator;
    use mpi::traits::{Communicator, Equivalence};
    use num::{Float, One};
    use num_complex::Complex32;
    use rlst::{RawAccess, RlstScalar};

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

        let distant_point = vec![T::real(10000.), T::real(0.0), T::real(0.0)];
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

        fmm.kernel()
            .evaluate_st(eval_type, coords, &distant_point, &charges, &mut expected);

        let mut num = T::real(0.0);
        let mut den = T::real(0.0);

        for (&expected, &found) in izip!(&expected, &found) {
            // squared error in complex difference
            let diff_re = expected.re() - found.re();
            let diff_im = expected.im() - found.im();
            num +=
                RlstScalar::powf(diff_re, T::real(2.0)) + RlstScalar::powf(diff_im, T::real(2.0));

            // squared magnitude of expected
            den += RlstScalar::powf(expected.re(), T::real(2.0))
                + RlstScalar::powf(expected.im(), T::real(2.0));
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

    fn test_multi_node_laplace_upward_pass_helper<
        T: RlstScalar<Real = T> + Float + Default + Equivalence,
    >(
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

        fmm.kernel()
            .evaluate_st(eval_type, coords, &distant_point, &charges, &mut expected);

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
    }

    let (universe, _threading) = mpi::initialize_with_threading(mpi::Threading::Single).unwrap();
    let world = universe.world();

    // Test Laplace FMM
    // N.B global tree refined to depth 3 to ensure that the global upward pass is also being run
    {
        let comm = world.duplicate();

        let n_points = 10000;
        let charges = vec![1f32; n_points];
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
                    Laplace3dKernel::new(),
                    eval_type,
                    source_to_target.clone(),
                )
                .unwrap()
                .build()
                .unwrap();

            test_multi_node_laplace_upward_pass_helper(
                "fixed_expansion_order".to_string(),
                Box::new(fmm),
                eval_type,
                1e-4,
            );
        }

        // Test case with multiple expansion orders which vary by level
        {
            let expansion_order = [4, 4, 5, 4, 5, 4, 5];
            assert!(expansion_order.len() == (global_depth + local_depth + 1).try_into().unwrap());

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
                    Laplace3dKernel::new(),
                    eval_type,
                    source_to_target.clone(),
                )
                .unwrap()
                .build()
                .unwrap();

            test_multi_node_laplace_upward_pass_helper(
                "fixed_expansion_order".to_string(),
                Box::new(fmm),
                eval_type,
                1e-4,
            );
        }
    }

    // Test Helmholtz FMM
    {
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
