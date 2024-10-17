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
    let points = points_fixture::<f32>(n_points, None, None, None);

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

    // fmm.v_list_exchange();
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

        // pick a box at a level
        let level = 2;

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

            // for s in interaction_list.iter() {
            //     let m1 = single_fmm.multipole(s).unwrap();

            //     // if let Some(m2) = fmm.multipole(s) {
            //     //     println!("same? {:?}={:?}", &m1[0..5], &m2[0..5])
            //     // }

            //     if let Some(m2) = fmm.ghost_fmm_v.multipole(s) {
            //         println!("same? {:?}={:?}", &m1[0..5], &m2[0..5])
            //     }
            // }
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
    }
}

#[cfg(not(feature = "mpi"))]
fn main() {}
