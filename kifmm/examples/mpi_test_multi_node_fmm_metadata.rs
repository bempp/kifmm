//? mpirun -n {{NPROCESSES}} --features "mpi"

use green_kernels::traits::Kernel;
use kifmm::{
    fmm::types::KiFmmMultiNode,
    traits::{field::SourceToTargetData, tree::SingleNodeTreeTrait},
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

    /// Test that the level index pointers are created correctly
    {
        /// 1. Test that the outer loop is over all local roots
        let nfmms = fmm.nfmms;
        assert_eq!(fmm.level_index_pointer_multipoles.len(), nfmms);
        assert_eq!(fmm.level_index_pointer_locals.len(), nfmms);

        /// 2. Test that each item is of size equal to to the number of global levels
        let depth = local_depth + global_depth;
        assert_eq!(
            fmm.level_index_pointer_multipoles[0].len(),
            (depth + 1) as usize
        );

        /// 3. Test that only levels associated with the local tree are populated
        for level in 0..global_depth {
            assert!(fmm.level_index_pointer_multipoles[0][level as usize].is_empty());
            assert!(fmm.level_index_pointer_locals[0][level as usize].is_empty());
        }

        for level in global_depth..depth {
            assert!(!fmm.level_index_pointer_multipoles[0][level as usize].is_empty());
            assert!(!fmm.level_index_pointer_locals[0][level as usize].is_empty());
        }

        if fmm.communicator.rank() == 0 {
            println!("...test_level_index_pointer_multinode: PASSED")
        }
    }

    /// Test that level expansion pointers are created correctly
    {
        // 1. Test that the outer loop is over all local roots
        let nfmms = fmm.nfmms;
        assert_eq!(fmm.level_locals.len(), nfmms);
        assert_eq!(fmm.level_multipoles.len(), nfmms);

        /// 2. Test that each item is of size equal to to the number of global levels
        let depth = local_depth + global_depth;
        assert_eq!(fmm.level_locals[0].len(), (depth + 1) as usize);
        assert_eq!(fmm.level_multipoles[0].len(), (depth + 1) as usize);

        // 3. Test that only levels associated with the local tree are populated.
        for level in 0..global_depth {
            assert!(fmm.level_locals[0][level as usize].is_empty());
            assert!(fmm.level_multipoles[0][level as usize].is_empty());
        }

        for level in global_depth..depth {
            assert!(!fmm.level_locals[0][level as usize].is_empty());
            assert!(!fmm.level_multipoles[0][level as usize].is_empty());
        }

        if fmm.communicator.rank() == 0 {
            println!("...test_level_expansion_pointers_multinode: PASSED")
        }
    }

    /// Test that leaf expansions have leaf expansion pointers have been properly set
    {
        let nfmms = fmm.nfmms;

        // 1. Test that the outer loop is over all local roots
        let nfmms = fmm.nfmms;
        assert_eq!(fmm.leaf_locals.len(), nfmms);
        assert_eq!(fmm.leaf_multipoles.len(), nfmms);

        // 2. Test that the number of pointers is equal to the number of leaves
        let nleaves = fmm.tree.source_tree.trees[0].n_leaves().unwrap();
        assert_eq!(nleaves, fmm.leaf_multipoles[0].len());
        let nleaves = fmm.tree.target_tree.trees[0].n_leaves().unwrap();
        assert_eq!(nleaves, fmm.leaf_locals[0].len());

        if fmm.communicator.rank() == 0 {
            println!("...test_leaf_expansion_pointers_multinode: PASSED")
        }
    }
}
