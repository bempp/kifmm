fn approx_equal(a: f32, b: f32, epsilon: f32) -> bool {
    (a - b).abs() < epsilon
}

fn find_subvector(haystack: &[f32], needle: &[f32], epsilon: f32) -> Option<usize> {
    // Ensure the needle is smaller or equal in length to the haystack
    if needle.len() > haystack.len() {
        return None;
    }

    for i in 0..=haystack.len() - needle.len() {
        if haystack[i..i + needle.len()]
            .iter()
            .zip(needle.iter())
            .all(|(&h, &n)| approx_equal(h, n, epsilon))
        {
            return Some(i); // Return the starting index
        }
    }
    None // Return None if the subvector is not found
}

#[cfg(feature = "mpi")]
fn main() {
    use green_kernels::{laplace_3d::Laplace3dKernel, types::GreenKernelEvalType};
    use itertools::{izip, Itertools};
    use kifmm::{
        fmm::types::MultiNodeBuilder,
        traits::{
            fmm::{DataAccessMulti, EvaluateMulti},
            general::multi_node::GhostExchange,
            tree::{MultiFmmTree, MultiTree, SingleFmmTree, SingleTree},
        },
        tree::{
            helpers::points_fixture,
            types::{MortonKey, SortKind},
        },
        DataAccess, Evaluate, FftFieldTranslation, SingleNodeBuilder,
    };

    use rayon::ThreadPoolBuilder;

    use mpi::{collective::SystemOperation, datatype::PartitionMut, traits::*};
    use rlst::RawAccess;

    let (universe, _threading) = mpi::initialize_with_threading(mpi::Threading::Single).unwrap();
    // let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let comm = world.duplicate();
    let rank = comm.rank();

    // Tree parameters
    let prune_empty = false;
    let n_points = 10000;
    let local_depth = 3;
    let global_depth = 2;
    let sort_kind = SortKind::Samplesort { k: 1000 };

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

    fmm.evaluate(false).unwrap();

    // Test neighbourhood communicator setup

    // Expect these to match the global communicator
    assert_eq!(
        fmm.tree.u_list_query.send_counts.len(),
        fmm.tree.source_tree().comm.size() as usize
    );
    assert_eq!(
        fmm.tree.u_list_query.receive_counts.len(),
        fmm.tree.source_tree().comm.size() as usize
    );
    assert_eq!(
        fmm.tree.v_list_query.send_counts.len(),
        fmm.tree.source_tree().comm.size() as usize
    );
    assert_eq!(
        fmm.tree.v_list_query.receive_counts.len(),
        fmm.tree.source_tree().comm.size() as usize
    );

    // Test that the interaction lists remain the same with respect to a single node FMM
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

    // Test that

    // let needle = vec![0.18512774, 0.15292609, 0.2774359];
    // let epsilon = 1e-6; // Tolerance for approximate equality

    // match find_subvector(&fmm.tree().source_tree().all_coordinates().unwrap(), &needle, epsilon) {
    //     Some(index) => println!("Subvector found starting at index {} at rank {:?}", index, rank),
    //     None => println!("Subvector not found in rank {:?}", rank),
    // }

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

        // Test that V list exchange is correct
        // pick a box at a level
        let level = 3;

        for key in fmm.tree.target_tree.keys(level).unwrap() {
            let mut interaction_list = key
                .parent()
                .neighbors()
                .iter()
                .flat_map(|pn| pn.children())
                .filter(|pnc| {
                    !key.is_adjacent(pnc) && single_fmm.tree.source_tree.keys_set.contains(pnc)
                })
                .collect_vec();

            // check against interaction list found for global fmm
            let mut distributed_interaction_list = key
                .parent()
                .neighbors()
                .iter()
                .flat_map(|pn| pn.children())
                .filter(|pnc| !key.is_adjacent(pnc) && fmm.tree.source_tree.keys_set.contains(pnc))
                .collect_vec();

            let mut ghost_interaction_list = key
                .parent()
                .neighbors()
                .iter()
                .flat_map(|pn| pn.children())
                .filter(|pnc| {
                    !key.is_adjacent(pnc) && fmm.ghost_fmm_v.tree.source_tree.keys_set.contains(pnc)
                })
                .collect_vec();

            distributed_interaction_list.append(&mut ghost_interaction_list);

            interaction_list.sort();
            distributed_interaction_list.sort();

            // Test that the whole of the interaction list is captured either locally, or in the data received from ghosts
            for s in interaction_list.iter() {
                assert!(distributed_interaction_list.contains(s))
            }

            for s in interaction_list.iter() {
                let m1 = single_fmm.multipole(s).unwrap();

                if let Some(m2) = fmm.multipole(s) {
                    println!("same? {:?}={:?}", &m1[0..5], &m2[0..5])
                }

                if let Some(m2) = fmm.ghost_fmm_v.multipole(s) {
                    println!("same? {:?}={:?}", &m1[0..5], &m2[0..5])
                }
            }
        }

        let root = fmm.global_fmm.tree().source_tree().root();
        let distributed_root_multipole = fmm.global_fmm.multipole(&root).unwrap();
        let single_root_multipole = &single_fmm.multipole(&root).unwrap();

        let threshold = 1e-3;
        let error = distributed_root_multipole
            .iter()
            .zip(single_root_multipole.iter())
            .map(|(l, r)| (l - r).abs())
            .collect_vec()
            .iter()
            .sum::<f32>();
        // println!("error {:?}", error)
        assert!(error <= threshold);

        // // Test that U list exchange is correct
        // for leaf in fmm.tree.target_tree.leaves.iter() {
        //     let mut interaction_list = leaf
        //         .neighbors()
        //         .iter()
        //         .cloned()
        //         .filter(|n| {
        //             single_fmm.tree.source_tree.keys_set.contains(n)
        //                 && single_fmm.tree.source_tree.coordinates(n).is_some()
        //         })
        //         .collect_vec();

        //     // check against interaction list found for global fmm
        //     let mut distributed_interaction_list = leaf
        //         .neighbors()
        //         .iter()
        //         .cloned()
        //         .filter(|n| {
        //             fmm.tree.source_tree.leaves_set.contains(n)
        //                 && fmm.tree.source_tree.coordinates(n).is_some()
        //         })
        //         .collect_vec();

        //     let mut ghost_interaction_list = leaf
        //         .neighbors()
        //         .iter()
        //         .cloned()
        //         .filter(|n| {
        //             fmm.ghost_fmm_u.tree.source_tree.leaves_set.contains(n)
        //                 && fmm.ghost_fmm_u.tree.source_tree.coordinates(n).is_some()
        //         })
        //         .collect_vec();

        //     distributed_interaction_list.append(&mut ghost_interaction_list);

        //     // println!("SAME: {:?}={:?}", interaction_list.len(), distributed_interaction_list.len());

        //     let i1 = interaction_list.iter().map(|s| s.morton).collect_vec();
        //     let i2 = distributed_interaction_list
        //         .iter()
        //         .map(|s| s.morton)
        //         .collect_vec();

        //     // Test that the whole of the interaction list is captured either locally, or in the data received from ghosts
        //     for s in interaction_list.iter() {
        //         assert!(
        //             distributed_interaction_list.contains(s),
        //             "Test failed: element {:?} is missing in distributed_interaction_list. at leaf: {:?} \n: {:?} = {:?} \n is it contained locally? {:?} or in ghost tree? {:?}",
        //             s, leaf.morton, i1, i2, fmm.tree.source_tree.keys_set.contains(s), fmm.ghost_fmm_u.tree.source_tree.keys_set.contains(&s)
        //         );

        //         // println!("coordinates {:?}", fmm.tree.source_tree.coordinates);

        //         // let key = MortonKey::from_morton(4611686018427387909);

        //         // println!("CHECKING U GHOST TREE {:?}", fmm.ghost_fmm_u.tree.source_tree.leaves_to_coordinates.keys().contains(&key));
        //         // assert!(
        //         //     distributed_interaction_list.contains(s),
        //         //     "Test failed: expected coordinates {:?}
        //         //     \n found coordinates in local tree {:?} in ghost tree? {:?}
        //         //     \n found key in local keys {:?} in ghost keys {:?},
        //         //     \n found key in local leaves {:?} in ghost leaves {:?}",
        //         //     single_fmm.tree.source_tree.coordinates(&key),
        //         //     fmm.tree.source_tree.coordinates(&key),
        //         //     fmm.ghost_fmm_u.tree.source_tree.coordinates(&key),
        //         //     fmm.tree.source_tree.keys_set.contains(&key),
        //         //     fmm.ghost_fmm_u.tree.source_tree.keys_set.contains(&key),
        //         //     fmm.tree.source_tree.leaves_set.contains(&key),
        //         //     fmm.ghost_fmm_u.tree.source_tree.leaves_set.contains(&key),
        //         // );
        //     }

        //     for s in interaction_list.iter() {
        //         if let Some(c1) = single_fmm.tree.source_tree.coordinates(s) {
        //             // Look for coordinates

        //             if fmm.tree.source_tree.keys_set.contains(s) {
        //                 let c2 = fmm.tree.source_tree.coordinates(s).unwrap();

        //                 c1.iter()
        //                     .zip(c2.iter())
        //                     .for_each(|(&c1, &c2)| assert!(approx_equal(c1, c2, 1e-6)));
        //                 assert!(c1.len() == c2.len());
        //             } else {
        //                 let c2 = fmm.ghost_fmm_u.tree.source_tree.coordinates(s).unwrap();
        //                 c1.iter()
        //                     .zip(c2.iter())
        //                     .for_each(|(&c1, &c2)| assert!(approx_equal(c1, c2, 1e-6)));
        //                 assert!(c1.len() == c2.len());
        //             }
        //         }
        //     }

        //     // Test that the target coordinates are the same
        //     for s in fmm.tree.target_tree.leaves.iter() {
        //         if let Some(c1) = fmm.tree.target_tree.coordinates(s) {
        //             let c2 = single_fmm.tree.target_tree.coordinates(s).unwrap();
        //             assert_eq!(c1.len(), c2.len());
        //         }
        //     }

        //     //

        //     // Test that the interaction list coordinates are the same?

        //     // Need to test that the fetched coordinates are the same, as the interaction lists and the target coordinates
        //     // are the same
        // }
    }
}

#[cfg(not(feature = "mpi"))]
fn main() {}
