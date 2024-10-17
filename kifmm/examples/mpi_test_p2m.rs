#[cfg(feature = "mpi")]
fn main() {
    use std::collections::HashMap;

    use green_kernels::{laplace_3d::Laplace3dKernel, types::GreenKernelEvalType};
    use itertools::{izip, Itertools};
    use kifmm::{
        fmm::types::MultiNodeBuilder,
        traits::{
            field::SourceTranslation,
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
    let world = universe.world();
    let comm = world.duplicate();

    // Tree parameters
    let prune_empty = true;
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
    let multi_fmm = MultiNodeBuilder::new()
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

    multi_fmm.p2m().unwrap();

    // Test that the interaction lists remain the same with respect to a single node FMM
    // Gather all data for the test

    let root_process = comm.process_at_rank(0);
    let n_coords = multi_fmm.tree().source_tree().coordinates.len() as i32;
    let mut all_coordinates = vec![0f32; 3 * n_points * world.size() as usize];

    let mut global_leaves_counts = vec![0i32; world.size() as usize];

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

        // Gather leaf counts
        let n_leaves = multi_fmm.tree.source_tree.leaves.len() as i32;
        root_process.gather_into_root(&n_leaves, &mut global_leaves_counts);
    } else {
        root_process.gather_into(&n_coords);

        let local_coords = multi_fmm.tree().source_tree().all_coordinates().unwrap();
        root_process.gather_varcount_into(local_coords);

        // gather leaf counts
        let n_leaves = multi_fmm.tree.source_tree.leaves.len() as i32;
        root_process.gather_into(&n_leaves);
    }

    let mut global_leaves =
        vec![MortonKey::<f32>::default(); global_leaves_counts.iter().sum::<i32>() as usize];
    let mut global_multipoles = vec![
        0f32;
        multi_fmm.n_coeffs_equivalent_surface
            * global_leaves_counts.iter().sum::<i32>() as usize
    ];

    if world.rank() == 0 {
        let mut global_leaves_displacements = Vec::new();
        let mut displacement = 0;
        for count in global_leaves_counts.iter() {
            global_leaves_displacements.push(displacement);
            displacement += count;
        }

        let mut global_multipoles_counts = Vec::new();
        let mut global_multipoles_displacements = Vec::new();
        let mut displacement = 0;
        for &count in global_leaves_counts.iter() {
            global_multipoles_counts.push(count * multi_fmm.n_coeffs_equivalent_surface as i32);
            global_multipoles_displacements.push(displacement);
            displacement += multi_fmm.n_coeffs_equivalent_surface as i32 * count;
        }

        let mut partition = PartitionMut::new(
            &mut global_leaves,
            global_leaves_counts,
            global_leaves_displacements,
        );

        let leaves = &multi_fmm.tree.source_tree.leaves;
        root_process.gather_varcount_into_root(&leaves[..], &mut partition);

        let mut multipoles = vec![0f32; multi_fmm.n_coeffs_equivalent_surface * leaves.len()];
        for (i, leaf) in leaves.iter().enumerate() {
            let l = i * multi_fmm.n_coeffs_equivalent_surface;
            let r = l + multi_fmm.n_coeffs_equivalent_surface;
            multipoles[l..r].copy_from_slice(multi_fmm.multipole(leaf).unwrap());
        }

        let mut partition = PartitionMut::new(
            &mut global_multipoles,
            global_multipoles_counts,
            global_multipoles_displacements,
        );

        root_process.gather_varcount_into_root(&multipoles, &mut partition);
    } else {
        let leaves = &multi_fmm.tree.source_tree.leaves;

        let mut multipoles = vec![0f32; multi_fmm.n_coeffs_equivalent_surface * leaves.len()];
        for (i, leaf) in leaves.iter().enumerate() {
            let l = i * multi_fmm.n_coeffs_equivalent_surface;
            let r = l + multi_fmm.n_coeffs_equivalent_surface;
            multipoles[l..r].copy_from_slice(multi_fmm.multipole(leaf).unwrap());
        }

        root_process.gather_varcount_into(&leaves[..]);
        root_process.gather_varcount_into(&multipoles);
    }

    if world.rank() == 0 {
        let single_fmm = SingleNodeBuilder::new()
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

        single_fmm.p2m().unwrap();

        let leaf_level = local_depth + global_depth;
        let leaves = &single_fmm.tree.source_tree.leaves;

        let mut re_ordered_global_leaves = global_leaves.iter().cloned().collect_vec();
        re_ordered_global_leaves.sort();

        let mut old_key_to_index = HashMap::new();
        for (old_idx, leaf) in global_leaves.iter().enumerate() {
            old_key_to_index.insert(leaf, old_idx);
        }

        let mut new_key_to_index = HashMap::new();
        for (new_idx, leaf) in re_ordered_global_leaves.iter().enumerate() {
            new_key_to_index.insert(leaf, new_idx);
        }

        let mut re_ordered_global_multipoles = vec![0f32; global_multipoles.len()];

        for (leaf, new_idx) in new_key_to_index.iter() {
            let old_idx = old_key_to_index.get(leaf).unwrap();
            let l = old_idx * multi_fmm.n_coeffs_equivalent_surface;
            let r = l + multi_fmm.n_coeffs_equivalent_surface;

            let new_l = new_idx * multi_fmm.n_coeffs_equivalent_surface;
            let new_r = new_l + multi_fmm.n_coeffs_equivalent_surface;

            re_ordered_global_multipoles[new_l..new_r].copy_from_slice(&global_multipoles[l..r]);
        }

        global_leaves.sort();

        let error = &single_fmm
            .multipoles(leaf_level)
            .unwrap()
            .iter()
            .zip(re_ordered_global_multipoles.iter())
            .map(|(l, r)| (l - r).abs())
            .collect_vec();

        let threshold = 1e-5;
        assert_eq!(leaves.len(), global_leaves.len());
        assert!(error.iter().sum::<f32>() <= threshold);
    }
}

#[cfg(not(feature = "mpi"))]
fn main() {}
