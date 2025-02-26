//? mpirun -n {{NPROCESSES}} --features "mpi"

#[cfg(feature = "mpi")]
fn main() {
    // use green_kernels::{laplace_3d::Laplace3dKernel, traits::Kernel, types::GreenKernelEvalType};
    // use kifmm::{
    //     fmm::types::MultiNodeBuilder,
    //     traits::{
    //         fmm::{DataAccessMulti, EvaluateMulti},
    //         tree::{MultiFmmTree, MultiTree},
    //     },
    //     tree::{helpers::points_fixture, types::SortKind},
    //     ChargeHandler, Evaluate, FftFieldTranslation, SingleNodeBuilder,
    // };

    // use rand::{rngs::StdRng, Rng, SeedableRng};
    // use rayon::ThreadPoolBuilder;

    // use mpi::{
    //     datatype::PartitionMut,
    //     traits::{Communicator, Root},
    // };
    // use rlst::{rlst_dynamic_array1, RawAccess, RawAccessMut};

    // let (universe, _threading) = mpi::initialize_with_threading(mpi::Threading::Funneled).unwrap();
    // let world = universe.world();
    // let comm = world.duplicate();

    // // Tree parameters
    // let prune_empty = true;
    // let n_points = 10000;
    // let local_depth = 1;
    // let global_depth = 2;
    // let sort_kind = SortKind::Samplesort { n_samples: 100 };

    // // Fmm Parameters
    // let expansion_order = 4;
    // let kernel = Laplace3dKernel::<f32>::new();

    // ThreadPoolBuilder::new()
    //     .num_threads(2)
    //     .build_global()
    //     .unwrap();

    // // Test FFT field translation
    // {
    //     let source_to_target = FftFieldTranslation::<f32>::new(None);

    //     // Generate some random test data local to each process
    //     let points = points_fixture::<f32>(n_points, None, None, Some(world.rank() as u64));
    //     let mut rng = StdRng::seed_from_u64(comm.rank() as u64);
    //     let mut charges = rlst_dynamic_array1!(f32, [n_points]);
    //     charges.data_mut().iter_mut().for_each(|c| *c = rng.gen());

    //     let mut multi_fmm = MultiNodeBuilder::new(false)
    //         .tree(
    //             &comm,
    //             points.data(),
    //             points.data(),
    //             local_depth,
    //             global_depth,
    //             prune_empty,
    //             sort_kind.clone(),
    //         )
    //         .unwrap()
    //         .parameters(
    //             charges.data(),
    //             expansion_order,
    //             kernel.clone(),
    //             GreenKernelEvalType::Value,
    //             source_to_target,
    //         )
    //         .unwrap()
    //         .build()
    //         .unwrap();

    //     multi_fmm.evaluate().unwrap();

    //     // Change the charges associated with the FMM object
    //     let new_charges =
    //         vec![multi_fmm.rank() as f32; multi_fmm.tree().source_tree().global_indices.len()];
    //     multi_fmm.attach_charges_ordered(&new_charges).unwrap();
    //     multi_fmm.evaluate().unwrap();

    //     // Gather all coordinates and charges for the test
    //     let root_process = comm.process_at_rank(0);
    //     let n_coords = multi_fmm.tree().source_tree().coordinates.len() as i32;
    //     let n_charges = multi_fmm.tree().source_tree().global_indices.len() as i32;
    //     let mut all_charges = vec![0f32; n_points * world.size() as usize];
    //     let mut all_coordinates = vec![0f32; 3 * n_points * world.size() as usize];

    //     if world.rank() == 0 {
    //         let mut charges_counts = vec![0i32; comm.size() as usize];
    //         let mut coordinates_counts = vec![0i32; comm.size() as usize];
    //         root_process.gather_into_root(&n_coords, &mut coordinates_counts);
    //         root_process.gather_into_root(&n_charges, &mut charges_counts);

    //         let mut coordinates_displacements = Vec::new();
    //         let mut counter = 0;
    //         for &count in coordinates_counts.iter() {
    //             coordinates_displacements.push(counter);
    //             counter += count;
    //         }

    //         let mut charges_displacements = Vec::new();
    //         let mut counter = 0;
    //         for &count in charges_counts.iter() {
    //             charges_displacements.push(counter);
    //             counter += count;
    //         }

    //         let local_coords = multi_fmm.tree().source_tree().all_coordinates().unwrap();
    //         let local_charges = multi_fmm.charges().unwrap();

    //         let mut partition = PartitionMut::new(
    //             &mut all_coordinates,
    //             coordinates_counts,
    //             coordinates_displacements,
    //         );

    //         root_process.gather_varcount_into_root(local_coords, &mut partition);

    //         let mut partition =
    //             PartitionMut::new(&mut all_charges, charges_counts, charges_displacements);

    //         root_process.gather_varcount_into_root(local_charges, &mut partition);
    //     } else {
    //         root_process.gather_into(&n_coords);
    //         root_process.gather_into(&n_charges);

    //         let local_coords = multi_fmm.tree().source_tree().all_coordinates().unwrap();
    //         root_process.gather_varcount_into(local_coords);

    //         let local_charges = multi_fmm.charges().unwrap();
    //         root_process.gather_varcount_into(local_charges);
    //     }

    //     if world.rank() == 0 {
    //         let mut single_fmm = SingleNodeBuilder::new(false)
    //             .tree(
    //                 &all_coordinates,
    //                 &all_coordinates,
    //                 None,
    //                 Some(local_depth + global_depth),
    //                 prune_empty,
    //             )
    //             .unwrap()
    //             .parameters(
    //                 &all_charges,
    //                 &vec![expansion_order; (local_depth + global_depth + 1) as usize],
    //                 Laplace3dKernel::new(),
    //                 GreenKernelEvalType::Value,
    //                 FftFieldTranslation::new(None),
    //             )
    //             .unwrap()
    //             .build()
    //             .unwrap();
    //         single_fmm.evaluate().unwrap();
    //         let mut expected = vec![0f32; &multi_fmm.tree.target_tree.coordinates.len() / 3];

    //         multi_fmm.kernel().evaluate_st(
    //             GreenKernelEvalType::Value,
    //             &all_coordinates,
    //             &multi_fmm.tree.target_tree.coordinates,
    //             &all_charges,
    //             &mut expected,
    //         );

    //         let distributed = multi_fmm.potentials().unwrap();

    //         // println!(
    //         //     "{:?} expected: {:?} \n found: {:?}",
    //         //     world.rank(),
    //         //     &distributed[0..100],
    //         //     &expected[0..100]
    //         // );

    //         // let mut err = 0f32;

    //         // for (l, r) in izip!(distributed, expected) {
    //         //     println!("l {:?}, r {:?}", l, r);
    //         //     err += (l-r).abs()/r.abs();
    //         // }

    //         // println!("ERROR {:?}", err/(multi_fmm.tree.target_tree.n_coordinates_tot().unwrap() as f32));

    //         if multi_fmm.communicator().size() > 1 {
    //             distributed
    //                 .iter()
    //                 .zip(expected.iter())
    //                 .for_each(|(f, e)| assert!(((f - e).abs() / e.abs()) < 1e-2));
    //         }

    //         println!("...test_attach_charges_ordered passed");
    //     }
    // }
}

#[cfg(not(feature = "mpi"))]
fn main() {}
