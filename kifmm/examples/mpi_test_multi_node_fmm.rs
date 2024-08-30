//? mpirun -n {{NPROCESSES}} --features "mpi"

#[cfg(feature = "mpi")]
fn main() {
    // Setup an MPI environment

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
    let local_depth = 3;
    let global_depth = 1;

    let expansion_order = 6;

    // Generate some random test data local to each process
    let points = points_fixture::<f32>(n_points, None, None, None);

    let charges = vec![1f32; n_points];

    // Create a uniform tree
    let mut fmm = MultiNodeBuilder::new()
        .tree(
            points.data(),
            points.data(),
            local_depth,
            global_depth,
            prune_empty,
            &world,
            kifmm::tree::multi_node::SortKind::Simplesort,
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

    fmm.evaluate(true);

    // println!(
    //     "RANK {:?} {:?}",
    //     fmm.communicator.rank(),
    //     fmm.tree.source_tree.trees.len()
    // );
    // if fmm.communicator.rank() == 0 {
    // for source_tree in fmm.tree.source_tree.trees.iter() {
    //     println!("RANK {:?} {:?} {:?} {:?}", fmm.communicator.rank(), source_tree.leaves.iter().min().unwrap(), source_tree.leaves.iter().max().unwrap(), source_tree.leaves.len());
    // }
    // }

    // Charge setting has to be after construction in multi-node
    // fmm.set_charges();

    // Evaluate distributed FMM
    // fmm.evaluate();
}
