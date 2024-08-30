//? mpirun -n {{NPROCESSES}} --features "mpi"

use green_kernels::traits::Kernel;
use kifmm::{
    fmm::types::KiFmmMultiNode,
    traits::{
        field::SourceToTargetData,
        fmm::SourceTranslation,
        tree::{FmmTreeNode, SingleNodeTreeTrait},
    },
    tree::constants::ALPHA_INNER,
};
use pulp::Scalar;

fn main() {
    use green_kernels::laplace_3d::Laplace3dKernel;
    use kifmm::{
        fmm::types::{FftFieldTranslationMultiNode, MultiNodeBuilder},
        traits::fmm::MultiNodeFmm,
        tree::{helpers::points_fixture, types::MultiNodeTreeNew},
        FftFieldTranslation,
    };
    use mpi::{environment::Universe, traits::Communicator};
    use rlst::RawAccess;

    let universe: Universe = mpi::initialize().unwrap();
    let world = universe.world();
    let comm = world.duplicate();

    // Setup tree parameters
    let prune_empty = false;
    let n_points = 10000;
    let local_depth = 4;
    let global_depth = 1;

    let expansion_order = 6;

    // Generate some random test data local to each process
    let points = points_fixture::<f32>(n_points, None, None, None);

    // Attach some charges
    let charges = vec![1f32; n_points];

    // Create a distributed FMM object
    let mut fmm = MultiNodeBuilder::new()
        .tree(
            points.data(),
            points.data(),
            local_depth,
            global_depth,
            prune_empty,
            &world,
        )
        .unwrap()
        .parameters(
            expansion_order,
            Laplace3dKernel::<f32>::new(),
            FftFieldTranslationMultiNode::<f32>::new(None),
        )
        .unwrap()
        .build()
        .unwrap();

    // Test local upward pass
    {
        // Upward pass algorithm
        let depth = local_depth + global_depth;

        fmm.p2m().unwrap();

        for level in ((global_depth + 1)..=depth).rev() {
            fmm.m2m(level).unwrap();
        }

        // Test at local roots
        for fmm_idx in 0..fmm.nfmms {

            // Test root multipole
            {
                let root = fmm.tree.source_tree.trees[fmm_idx].root;
                let upward_equivalent_surface = root.surface_grid(
                    fmm.equivalent_surface_order,
                    &fmm.tree.domain,
                    ALPHA_INNER as f32,
                );
                let root_multipole = fmm.multipole(fmm_idx, &root).unwrap();

                let all_coordinates = fmm.tree.source_tree.trees[fmm_idx].all_coordinates().unwrap();

                let test_point = vec![1000f32, 0f32, 0f32];

                let mut expected = vec![0f32];
                let mut found = vec![0f32];

                let charges = vec![1f32; all_coordinates.len() / 3];

                fmm.kernel.evaluate_st(green_kernels::types::EvalType::Value, all_coordinates, &test_point, &charges, &mut expected);
                fmm.kernel.evaluate_st(green_kernels::types::EvalType::Value, &upward_equivalent_surface, &test_point, &root_multipole, &mut found);

                let abs_err  = (expected[0] - found[0]).abs();
                let rel_err = abs_err / expected[0];
                assert!(rel_err <= 1e-5);
            }

            if fmm.rank == 0 {
                println!("...test_root_multipole: PASSSED")
            }

            // Test at random leaf
            {
                let mut leaf_idx = 0;
                loop {
                    let leaf = fmm.tree.source_tree.trees[fmm_idx].leaves[leaf_idx];

                    if let Some(_coords) = fmm.tree.source_tree.trees[fmm_idx].coordinates(&leaf) {
                        break;
                    }
                    leaf_idx += 1;
                }

                let leaf = fmm.tree.source_tree.trees[fmm_idx].leaves[leaf_idx];
                let leaf_coords = fmm.tree.source_tree.trees[fmm_idx].coordinates(&leaf).unwrap();

                let leaf_multipole = fmm.multipole(fmm_idx, &leaf).unwrap();

                let upward_equivalent_surface = leaf.surface_grid(
                    fmm.equivalent_surface_order,
                    &fmm.tree.domain,
                    ALPHA_INNER as f32,
                );

                let test_point = vec![1000f32, 0f32, 0f32];

                let mut expected = vec![0f32];
                let mut found = vec![0f32];

                let charges = vec![1f32; leaf_coords.len() / 3];

                fmm.kernel.evaluate_st(green_kernels::types::EvalType::Value, leaf_coords, &test_point, &charges, &mut expected);
                fmm.kernel.evaluate_st(green_kernels::types::EvalType::Value, &upward_equivalent_surface, &test_point, &leaf_multipole, &mut found);

                let abs_err  = (expected[0] - found[0]).abs();
                let rel_err = abs_err / expected[0];
                assert!(rel_err <= 1e-5);
            }
            if fmm.rank == 0 {
                println!("...test_leaf_multipoles: PASSSED")
            }
        }
    }
}
