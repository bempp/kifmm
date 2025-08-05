//! Create a multi-node tree, distributed with MPI

fn main() {
    use kifmm::tree::{helpers::points_fixture, MultiNodeTree, SortKind};
    use rlst::RawAccess;

    let universe = mpi::initialize().unwrap();
    let comm = universe.world();

    // Generate some random points at each process, will be sorted into Morton order
    let n_points_per_process = 100000;
    let points_per_process = points_fixture::<f32>(n_points_per_process, None, None, None);

    // Tree parameters
    let n_samples = 1000; // Sampling parameter for sampling sort
    let sort_kind = SortKind::Samplesort { n_samples }; // We also implement Hyksort and Bucket sort

    // We perform a split of the tree into 'local' and 'global' trees, where the local trees have corresponding local roots split across
    // MPI processes, and the global tree, corresponding to the upper levels shared by all local trees as ancestors, is stored on a nominated
    // node.
    let local_depth = 3;
    let global_depth = 2;
    let domain = None; // Specify, or alternatively construct from data
    let prune_empty = true; // Optionally prune empty leaves and their branches

    // Call the tree constructor
    let _tree = MultiNodeTree::new(
        &comm,
        points_per_process.data(),
        local_depth,
        global_depth,
        domain,
        sort_kind,
        prune_empty,
    )
    .unwrap();
}
