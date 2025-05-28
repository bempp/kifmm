//? mpirun -n {{NPROCESSES}} --features "mpi"

#[cfg(feature = "mpi")]
fn main() {
    use green_kernels::{
        helmholtz_3d::Helmholtz3dKernel, laplace_3d::Laplace3dKernel, traits::Kernel,
        types::GreenKernelEvalType,
    };
    use itertools::{izip, Itertools};
    use kifmm::{
        fmm::types::MultiNodeBuilder,
        linalg::rsvd::MatrixRsvd,
        traits::{
            fmm::EvaluateMulti,
            general::single_node::Epsilon,
            tree::{MultiFmmTree, MultiTree},
        },
        tree::{helpers::points_fixture, types::SortKind},
        BlasFieldTranslationIa, BlasFieldTranslationSaRcmp, FftFieldTranslation, MultiNodeFmmTree,
    };

    use mpi::{
        datatype::PartitionMut,
        topology::SimpleCommunicator,
        traits::{Communicator, Equivalence, Root},
    };
    use num::{Float, One};
    use rlst::{c32, RawAccess, RlstScalar};

    fn test_multi_node_helmholtz_fmm_helper<
        T: RlstScalar<Complex = T> + Epsilon + MatrixRsvd + Default + Equivalence,
    >(
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
        <T as RlstScalar>::Real: Equivalence + Float + Epsilon,
    {
        // Run the FMM
        fmm.evaluate().unwrap();

        // TODO add test for matrix input
        let _eval_size = match eval_type {
            GreenKernelEvalType::Value => 1,
            GreenKernelEvalType::ValueDeriv => 4,
        };

        let charges_rank = fmm.charges().unwrap();
        let source_coordinates_rank = fmm.tree().source_tree().all_coordinates().unwrap();
        let n_sources_rank = charges_rank.len() as i32;

        // Gather all coordinates and charges for the test
        let root_process = fmm.communicator().process_at_rank(0);

        if fmm.communicator().rank() == 0 {
            // Communicate counts
            let mut sources_counts = vec![0i32; fmm.communicator().size() as usize];
            root_process.gather_into_root(&n_sources_rank, &mut sources_counts);
            let coordinates_counts = sources_counts.iter().map(|c| c * 3).collect_vec();

            let mut sources_displacements = Vec::new();
            let mut counter = 0;
            for &count in sources_counts.iter() {
                sources_displacements.push(counter);
                counter += count;
            }

            let mut coordinates_displacements = Vec::new();
            let mut counter = 0;
            for &count in coordinates_counts.iter() {
                coordinates_displacements.push(counter);
                counter += count;
            }

            let n_sources = sources_counts.iter().sum::<i32>();

            let mut all_coordinates = vec![T::Real::default(); 3 * n_sources as usize];
            let mut all_charges = vec![T::default(); n_sources as usize];

            // Communicate charges
            let mut partition =
                PartitionMut::new(&mut all_charges, sources_counts, sources_displacements);

            root_process.gather_varcount_into_root(charges_rank, &mut partition);

            // Communicate coordinates
            let mut partition = PartitionMut::new(
                &mut all_coordinates,
                coordinates_counts,
                coordinates_displacements,
            );

            root_process.gather_varcount_into_root(source_coordinates_rank, &mut partition);

            let target_coordinates_rank = fmm.tree().target_tree().all_coordinates().unwrap();
            let n_targets = target_coordinates_rank.len() / 3;
            let mut expected = vec![T::default(); n_targets];

            fmm.kernel().evaluate_st(
                GreenKernelEvalType::Value,
                &all_coordinates,
                target_coordinates_rank,
                &all_charges,
                &mut expected,
            );

            // Test metadata

            let found = fmm.potentials().unwrap();

            let mut num = T::real(0.0);
            let mut den = T::real(0.0);

            for (expected, &found) in izip!(expected, found) {
                // squared error in complex difference
                let diff_re = expected.re() - found.re();
                let diff_im = expected.im() - found.im();
                num += RlstScalar::powf(diff_re, T::real(2.0))
                    + RlstScalar::powf(diff_im, T::real(2.0));

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

            assert!(l2_error <= threshold);
            println!("...test_helmholtz_fmm_{} passed", name);
        } else {
            root_process.gather_into(&n_sources_rank);

            // Communicate charges
            root_process.gather_varcount_into(charges_rank);

            // Communicate coordinates
            root_process.gather_varcount_into(source_coordinates_rank);
        }
    }

    fn test_multi_node_laplace_fmm_helper<T: RlstScalar<Real = T> + Float + Default + Equivalence>(
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
        // Run the FMM
        fmm.evaluate().unwrap();

        // TODO add test for matrix input
        let _eval_size = match eval_type {
            GreenKernelEvalType::Value => 1,
            GreenKernelEvalType::ValueDeriv => 4,
        };

        let charges_rank = fmm.charges().unwrap();
        let source_coordinates_rank = fmm.tree().source_tree().all_coordinates().unwrap();
        let n_sources_rank = charges_rank.len() as i32;

        // Gather all coordinates and charges for the test
        let root_process = fmm.communicator().process_at_rank(0);

        if fmm.communicator().rank() == 0 {
            // Communicate counts
            let mut sources_counts = vec![0i32; fmm.communicator().size() as usize];
            root_process.gather_into_root(&n_sources_rank, &mut sources_counts);
            let coordinates_counts = sources_counts.iter().map(|c| c * 3).collect_vec();

            let mut sources_displacements = Vec::new();
            let mut counter = 0;
            for &count in sources_counts.iter() {
                sources_displacements.push(counter);
                counter += count;
            }

            let mut coordinates_displacements = Vec::new();
            let mut counter = 0;
            for &count in coordinates_counts.iter() {
                coordinates_displacements.push(counter);
                counter += count;
            }

            let n_sources = sources_counts.iter().sum::<i32>();

            let mut all_coordinates = vec![T::Real::default(); 3 * n_sources as usize];
            let mut all_charges = vec![T::default(); n_sources as usize];

            // Communicate charges
            let mut partition =
                PartitionMut::new(&mut all_charges, sources_counts, sources_displacements);

            root_process.gather_varcount_into_root(charges_rank, &mut partition);

            // Communicate coordinates
            let mut partition = PartitionMut::new(
                &mut all_coordinates,
                coordinates_counts,
                coordinates_displacements,
            );

            root_process.gather_varcount_into_root(source_coordinates_rank, &mut partition);

            let target_coordinates_rank = fmm.tree().target_tree().all_coordinates().unwrap();
            let n_targets = target_coordinates_rank.len() / 3;
            let mut expected = vec![T::default(); n_targets];

            fmm.kernel().evaluate_st(
                GreenKernelEvalType::Value,
                &all_coordinates,
                target_coordinates_rank,
                &all_charges,
                &mut expected,
            );

            let found = fmm.potentials().unwrap();

            let mut num = T::real(0.0);
            let mut den = T::real(0.0);
            for (&expected, &found) in izip!(&expected, found) {
                num += RlstScalar::powf(RlstScalar::abs(expected - found), T::real(2.0));
                den += RlstScalar::powf(RlstScalar::abs(expected), T::real(2.0));
            }

            let l2_error = RlstScalar::powf(num / den, T::real(0.5));

            assert!(l2_error <= threshold);
            println!("...test_laplace_fmm_{} passed", name);
        } else {
            root_process.gather_into(&n_sources_rank);

            // Communicate charges
            root_process.gather_varcount_into(charges_rank);

            // Communicate coordinates
            root_process.gather_varcount_into(source_coordinates_rank);
        }
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
        let sources = points_fixture(n_points, None, None, None);
        let local_depth = 3;
        let global_depth = 3;
        let prune_empty = true;

        // FFT Field translation
        let source_to_target = FftFieldTranslation::new(None);
        // Test case with a single expansion order applied at all levels
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

            test_multi_node_laplace_fmm_helper(
                "laplace_fixed_expansion_order_fft_m2l".to_string(),
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

            test_multi_node_laplace_fmm_helper(
                "laplace_variable_expansion_order_fft_m2l".to_string(),
                Box::new(fmm),
                eval_type,
                1e-4,
            );
        }

        // BLAS field translation
        let source_to_target =
            BlasFieldTranslationSaRcmp::new(None, None, kifmm::FmmSvdMode::Deterministic);
        // Test case with a single expansion order applied at all levels
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

            test_multi_node_laplace_fmm_helper(
                "laplace_fixed_expansion_order_blas_m2l".to_string(),
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

            test_multi_node_laplace_fmm_helper(
                "laplace_variable_expansion_order_blas_m2l".to_string(),
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
        let charges = vec![c32::one(); n_points];
        let eval_type = GreenKernelEvalType::Value;
        let sources = points_fixture(n_points, None, None, None);
        let local_depth = 3;
        let global_depth = 3;
        let prune_empty = true;

        let wavenumber = 2.0;

        // Test FFT field translation at low frequencies
        let source_to_target = FftFieldTranslation::new(None);
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
                    SortKind::Samplesort { n_samples: 1000 },
                )
                .unwrap()
                .parameters(
                    &charges,
                    &expansion_order,
                    Helmholtz3dKernel::new(wavenumber),
                    eval_type,
                    source_to_target.clone(),
                )
                .unwrap()
                .build()
                .unwrap();

            test_multi_node_helmholtz_fmm_helper(
                "low_frequency_fixed_expansion_order_fft_m2l".to_string(),
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
                    SortKind::Samplesort { n_samples: 1000 },
                )
                .unwrap()
                .parameters(
                    &charges,
                    &expansion_order,
                    Helmholtz3dKernel::new(wavenumber),
                    eval_type,
                    source_to_target.clone(),
                )
                .unwrap()
                .build()
                .unwrap();

            test_multi_node_helmholtz_fmm_helper(
                "low_frequency_variable_expansion_order_fft_m2l".to_string(),
                Box::new(fmm),
                eval_type,
                1e-4,
            );
        }

        // Test BLAS field translation at low frequencies
        let source_to_target =
            BlasFieldTranslationIa::new(None, None, kifmm::FmmSvdMode::Deterministic);
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
                    SortKind::Samplesort { n_samples: 1000 },
                )
                .unwrap()
                .parameters(
                    &charges,
                    &expansion_order,
                    Helmholtz3dKernel::new(wavenumber),
                    eval_type,
                    source_to_target.clone(),
                )
                .unwrap()
                .build()
                .unwrap();

            test_multi_node_helmholtz_fmm_helper(
                "low_frequency_fixed_expansion_order_blas_m2l".to_string(),
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
                    SortKind::Samplesort { n_samples: 1000 },
                )
                .unwrap()
                .parameters(
                    &charges,
                    &expansion_order,
                    Helmholtz3dKernel::new(wavenumber),
                    eval_type,
                    source_to_target.clone(),
                )
                .unwrap()
                .build()
                .unwrap();

            test_multi_node_helmholtz_fmm_helper(
                "low_frequency_variable_expansion_order_blas_m2l".to_string(),
                Box::new(fmm),
                eval_type,
                1e-4,
            );
        }
    }
}

#[cfg(not(feature = "mpi"))]
fn main() {}
