//? mpirun -n {{NPROCESSES}} --features "mpi"

#[cfg(feature = "mpi")]
fn main() {
    // Setup an MPI environment

    use green_kernels::laplace_3d::Laplace3dKernel;
    use kifmm::{
        fmm::types::{FftFieldTranslationMultiNode, MultiNodeBuilder},
        traits::fmm::{GhostExchange, MultiNodeFmm},
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

    // let charges = vec![1f32; n_points];

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

    // Test U list exchange, run as a part of pre-computation.

    if fmm.communicator.rank() == 0 {
        println!("{:?}", fmm.ghost_u_list_octants.len());
        println!("{:?}", fmm.ghost_u_list_octants[1].len());

        for octant in fmm.ghost_u_list_octants[1].iter() {
            println!("requesting local tree ID {:?}", octant.rank)
        }
    }

    // Test V list exchange
    // fmm.v_list_p2p();
}
