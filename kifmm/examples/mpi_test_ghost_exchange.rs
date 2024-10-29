#[cfg(feature = "mpi")]
fn approx_equal(a: f32, b: f32, epsilon: f32) -> bool {
    (a - b).abs() < epsilon
}

#[cfg(feature = "mpi")]
fn main() {
    use green_kernels::{laplace_3d::Laplace3dKernel, types::GreenKernelEvalType};
    use itertools::Itertools;
    use kifmm::{
        fmm::types::MultiNodeBuilder,
        traits::{
            fmm::{DataAccess, DataAccessMulti, EvaluateMulti},
            tree::{MultiFmmTree, MultiTree, SingleFmmTree, SingleTree},
        },
        tree::{helpers::points_fixture, types::SortKind},
        Evaluate, FftFieldTranslation, SingleNodeBuilder,
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
    let rank = comm.rank();

    // Tree parameters
    let prune_empty = false;
    let n_points = 10000;
    let local_depth = 3;
    let global_depth = 2;
    let sort_kind = SortKind::Samplesort { n_samples: 1000 };

    // Fmm Parameters
    let expansion_order = 4;
    let kernel = Laplace3dKernel::<f32>::new();
    let source_to_target = FftFieldTranslation::<f32>::new(None);

    // Generate some random test data local to each process
    let points = points_fixture::<f32>(n_points, None, None, Some(rank as u64));

    ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()
        .unwrap();

    // Queries are set as a part of the build
    let mut multi_fmm = MultiNodeBuilder::new(false)
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

    multi_fmm.evaluate().unwrap();

    // Test neighbourhood communicator setup
    // Expect these to match the global communicator
    assert_eq!(
        multi_fmm.tree.u_list_query.send_counts.len(),
        multi_fmm.tree.source_tree().communicator.size() as usize
    );
    assert_eq!(
        multi_fmm.tree.u_list_query.receive_counts.len(),
        multi_fmm.tree.source_tree().communicator.size() as usize
    );
    assert_eq!(
        multi_fmm.tree.v_list_query.send_counts.len(),
        multi_fmm.tree.source_tree().communicator.size() as usize
    );
    assert_eq!(
        multi_fmm.tree.v_list_query.receive_counts.len(),
        multi_fmm.tree.source_tree().communicator.size() as usize
    );

    // Test that the interaction lists remain the same with respect to a single node FMM
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
        let mut single_fmm = SingleNodeBuilder::new(false)
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

        single_fmm.evaluate().unwrap();

        // Test that V list exchange is correct at a given level
        let level = 3;
        for key in multi_fmm.tree.target_tree.keys(level).unwrap() {
            let mut interaction_list = key
                .parent()
                .neighbors()
                .iter()
                .flat_map(|pn| pn.children())
                .filter(|pnc| {
                    !key.is_adjacent(pnc) && single_fmm.tree().source_tree.keys_set.contains(pnc)
                })
                .collect_vec();

            // check against interaction list found for global fmm
            let mut distributed_interaction_list = key
                .parent()
                .neighbors()
                .iter()
                .flat_map(|pn| pn.children())
                .filter(|pnc| {
                    !key.is_adjacent(pnc) && multi_fmm.tree.source_tree.keys_set.contains(pnc)
                })
                .collect_vec();

            let mut ghost_interaction_list = key
                .parent()
                .neighbors()
                .iter()
                .flat_map(|pn| pn.children())
                .filter(|pnc| {
                    !key.is_adjacent(pnc)
                        && multi_fmm
                            .ghost_fmm_v
                            .tree()
                            .source_tree
                            .keys_set
                            .contains(pnc)
                })
                .collect_vec();

            distributed_interaction_list.append(&mut ghost_interaction_list);

            interaction_list.sort();
            distributed_interaction_list.sort();

            // Test that the whole of the interaction list is captured either locally, or in the data received from ghosts
            for s in interaction_list.iter() {
                assert!(distributed_interaction_list.contains(s))
            }

            // Test that the data remains the same
            for s in interaction_list.iter() {
                let m1 = single_fmm.multipole(s).unwrap();

                if let Some(m2) = multi_fmm.multipole(s) {
                    m1.iter()
                        .zip(m2.iter())
                        .for_each(|(f, e)| assert!(((f - e).abs() / e.abs()) < 1e-3));
                }
                if let Some(m2) = multi_fmm.ghost_fmm_v.multipole(s) {
                    m1.iter()
                        .zip(m2.iter())
                        .for_each(|(f, e)| assert!(((f - e).abs() / e.abs()) < 1e-3));
                }
            }
        }

        let root = multi_fmm.global_fmm.tree().source_tree().root();
        let distributed_root_multipole = multi_fmm.global_fmm.multipole(&root).unwrap();
        let single_root_multipole = &single_fmm.multipole(&root).unwrap();

        let threshold = 1e-2;
        let error = distributed_root_multipole
            .iter()
            .zip(single_root_multipole.iter())
            .map(|(l, r)| (l - r).abs())
            .collect_vec()
            .iter()
            .sum::<f32>();
        assert!(error <= threshold);

        println!("...test_v_list_exchange passed");

        // Test that U list exchange is correct
        for leaf in multi_fmm.tree.target_tree.leaves.iter() {
            let interaction_list = leaf
                .neighbors()
                .iter()
                .cloned()
                .filter(|n| {
                    single_fmm.tree().source_tree.keys_set.contains(n)
                        && single_fmm.tree().source_tree.coordinates(n).is_some()
                })
                .collect_vec();

            // check against interaction list found for global fmm
            let mut distributed_interaction_list = leaf
                .neighbors()
                .iter()
                .cloned()
                .filter(|n| {
                    multi_fmm.tree.source_tree.leaves_set.contains(n)
                        && multi_fmm.tree.source_tree.coordinates(n).is_some()
                })
                .collect_vec();

            let mut ghost_interaction_list = leaf
                .neighbors()
                .iter()
                .cloned()
                .filter(|n| {
                    multi_fmm
                        .ghost_fmm_u
                        .tree()
                        .source_tree
                        .leaves_set
                        .contains(n)
                        && multi_fmm
                            .ghost_fmm_u
                            .tree()
                            .source_tree
                            .coordinates(n)
                            .is_some()
                })
                .collect_vec();

            distributed_interaction_list.append(&mut ghost_interaction_list);

            // println!("SAME: {:?}={:?}", interaction_list.len(), distributed_interaction_list.len());

            let i1 = interaction_list.iter().map(|s| s.morton).collect_vec();
            let i2 = distributed_interaction_list
                .iter()
                .map(|s| s.morton)
                .collect_vec();

            // Test that the whole of the interaction list is captured either locally, or in the data received from ghosts
            for s in interaction_list.iter() {
                assert!(
                    distributed_interaction_list.contains(s),
                    "Test failed: element {:?} is missing in distributed_interaction_list. at leaf: {:?} \n: {:?} = {:?} \n is it contained locally? {:?} or in ghost tree? {:?}",
                    s, leaf.morton, i1, i2, multi_fmm.tree.source_tree.keys_set.contains(s), multi_fmm.ghost_fmm_u.tree().source_tree.keys_set.contains(s)
                );
            }

            for s in interaction_list.iter() {
                if let Some(c1) = single_fmm.tree().source_tree.coordinates(s) {
                    // Look for coordinates

                    if multi_fmm.tree.source_tree.keys_set.contains(s) {
                        let c2 = multi_fmm.tree.source_tree.coordinates(s).unwrap();

                        c1.iter()
                            .zip(c2.iter())
                            .for_each(|(&c1, &c2)| assert!(approx_equal(c1, c2, 1e-6)));
                        assert!(c1.len() == c2.len());
                    } else {
                        let c2 = multi_fmm
                            .ghost_fmm_u
                            .tree()
                            .source_tree
                            .coordinates(s)
                            .unwrap();
                        c1.iter()
                            .zip(c2.iter())
                            .for_each(|(&c1, &c2)| assert!(approx_equal(c1, c2, 1e-6)));
                        assert!(c1.len() == c2.len());
                    }
                }
            }

            // Test that the target coordinates are the same
            for s in multi_fmm.tree.target_tree.leaves.iter() {
                if let Some(c1) = multi_fmm.tree.target_tree.coordinates(s) {
                    let c2 = single_fmm.tree().target_tree.coordinates(s).unwrap();
                    assert_eq!(c1.len(), c2.len());
                }
            }
        }

        println!("...test_u_list_exchange passed");
    }
}

#[cfg(not(feature = "mpi"))]
fn main() {}
