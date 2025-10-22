//! Run a parametrised  distributed FMM with FFT based M2L in f32
use clap::Parser;
use green_kernels::laplace_3d::Laplace3dKernel;
use kifmm::{
    traits::{
        tree::MultiFmmTree,
        types::{CommunicationType, FmmOperatorType, MPICollectiveType, MetadataType},
    },
    tree::{
        helpers::{points_fixture, points_fixture_sphere},
        types::SortKind,
    },
    DataAccessMulti, EvaluateMulti, FftFieldTranslation, MultiNodeBuilder,
};
use mpi::traits::*;
use rayon::ThreadPoolBuilder;
use rlst::RawAccess;
use std::{collections::HashMap, time::Instant};

/// Struct for parsing command-line arguments
#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = String::from("x"))]
    id: String,

    #[arg(long, default_value_t = 3)]
    expansion_order: usize,

    /// Whether to prune empty nodes
    #[arg(long, default_value_t = false)]
    prune_empty: bool,

    /// Number of points per MPI process
    #[arg(long, default_value_t = 10000)]
    n_points: usize,

    /// Local depth
    #[arg(long, default_value_t = 1)]
    local_depth: u64,

    /// Global depth
    #[arg(long, default_value_t = 1)]
    global_depth: u64,

    /// FFT Hadamard Block Size
    #[arg(long, default_value_t = 16)]
    block_size: usize,

    /// Number of threads per MPI process
    #[arg(long, default_value_t = 1)]
    n_threads: usize,

    /// Number of samples in parallel sample sort, must be
    /// less than the number of samples and greater than 0
    #[arg(long, default_value_t = 10)]
    n_samples: usize,

    /// Particle distribution (0 - uniform, 1 - sphere)
    #[arg(long, default_value_t = 0)]
    distribution: u64,
}

fn main() {
    let (universe, _threading) =
        mpi::initialize_with_threading(mpi::Threading::Serialized).unwrap();
    let world = universe.world();
    let comm = world.duplicate();

    // Tree parameters
    let args = Args::parse();
    let expansion_order = args.expansion_order;
    let prune_empty = args.prune_empty;
    let n_points = args.n_points;
    let local_depth = args.local_depth;
    let global_depth = args.global_depth;
    let block_size = args.block_size;
    let n_threads = args.n_threads;
    let n_samples = args.n_samples;
    let distribution = args.distribution;
    let id = args.id;

    assert!(n_samples > 0 && n_samples < n_points);

    let sort_kind = SortKind::Samplesort { n_samples };

    // Fmm Parameters
    let kernel = Laplace3dKernel::<f32>::new();

    ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build_global()
        .unwrap();

    let source_to_target = FftFieldTranslation::<f32>::new(Some(block_size));

    // Generate some random test data local to each process

    let points;
    if distribution == 0 {
        points = points_fixture::<f32>(n_points, None, None, Some(world.rank() as u64));
    } else if distribution == 1 {
        points = points_fixture_sphere::<f32>(n_points, Some(world.rank() as u64))
    } else {
        panic!("Unknown distribution")
    }

    let charges = vec![1f32; n_points];

    let mut multi_fmm = MultiNodeBuilder::new(true)
        .tree(
            &comm,
            points.data(),
            points.data(),
            local_depth,
            global_depth,
            prune_empty,
            sort_kind.clone(),
        )
        .unwrap()
        .parameters(
            &charges,
            &[expansion_order],
            kernel.clone(),
            green_kernels::types::GreenKernelEvalType::Value,
            source_to_target,
        )
        .unwrap()
        .build()
        .unwrap();

    let start = Instant::now();
    multi_fmm.evaluate().unwrap();
    let runtime = start.elapsed().as_millis();

    let mut mean_roots_per_rank_source_tree = 0.;
    if multi_fmm.rank() == 0 {
        let mut m: HashMap<i32, f64> = HashMap::new();
        for x in multi_fmm.tree().source_tree().all_roots_ranks.iter() {
            *m.entry(*x).or_default() += 1.0;
        }

        let n_ranks = m.len();
        let mut tmp = 0.0;
        for (_rank, n_keys) in m.iter() {
            tmp += n_keys;
        }

        mean_roots_per_rank_source_tree = tmp / (n_ranks as f64);
    }

    let mut mean_roots_per_rank_target_tree = 0.;
    if multi_fmm.rank() == 0 {
        let mut m: HashMap<i32, f64> = HashMap::new();
        for x in multi_fmm.tree().target_tree().all_roots_ranks.iter() {
            *m.entry(*x).or_default() += 1.0;
        }

        let n_ranks = m.len();
        let mut tmp = 0.0;
        for (_rank, n_keys) in m.iter() {
            tmp += n_keys;
        }

        mean_roots_per_rank_target_tree = tmp / (n_ranks as f64);
    }

    // Destructure operator times
    let mut operator_times = HashMap::new();

    // Destructure FMM MPI times
    let mut mpi_times = HashMap::new();

    for (&op_type, op_time) in multi_fmm.mpi_times.iter() {
        match op_type {
            MPICollectiveType::AlltoAll => {
                mpi_times.insert("all_to_all", op_time.time);
            }
            MPICollectiveType::AlltoAllV => {
                mpi_times.insert("all_to_all_v", op_time.time);
            }

            MPICollectiveType::NeighbourAlltoAll => {
                mpi_times.insert("neighbour_all_to_all", op_time.time);
            }
            MPICollectiveType::NeighbourAlltoAllv => {
                mpi_times.insert("neighbour_all_to_all_v", op_time.time);
            }

            MPICollectiveType::NeighbourAlltoAllvRuntime => {
                mpi_times.insert("neighbour_all_to_all_v_runtime", op_time.time);
            }

            MPICollectiveType::Gather => {
                mpi_times.insert("gather", op_time.time);
            }
            MPICollectiveType::Scatter => {
                mpi_times.insert("scatter", op_time.time);
            }

            MPICollectiveType::GatherV => {
                mpi_times.insert("gather_v", op_time.time);
            }
            MPICollectiveType::ScatterV => {
                mpi_times.insert("scatter_v", op_time.time);
            }

            MPICollectiveType::GatherVRuntime => {
                mpi_times.insert("gather_v_runtime", op_time.time);
            }
            MPICollectiveType::ScatterVRuntime => {
                mpi_times.insert("scatter_v_runtime", op_time.time);
            }

            MPICollectiveType::AllGather => {
                mpi_times.insert("all_gather", op_time.time);
            }
            MPICollectiveType::AllGatherV => {
                mpi_times.insert("all_gather_v", op_time.time);
            }

            MPICollectiveType::DistGraphCreate => {
                mpi_times.insert("dist_graph_create", op_time.time);
            }

            MPICollectiveType::Sort => {
                mpi_times.insert("sort", op_time.time);
            }
        }
    }

    for (&op_type, op_time) in multi_fmm.tree.mpi_times.iter() {
        match op_type {
            MPICollectiveType::AlltoAll => {
                mpi_times.insert("tree_all_to_all", op_time.time);
            }
            MPICollectiveType::AlltoAllV => {
                mpi_times.insert("tree_all_to_all_v", op_time.time);
            }

            MPICollectiveType::NeighbourAlltoAll => {
                mpi_times.insert("tree_neighbour_all_to_all", op_time.time);
            }
            MPICollectiveType::NeighbourAlltoAllv => {
                mpi_times.insert("tree_neighbour_all_to_all_v", op_time.time);
            }
            MPICollectiveType::NeighbourAlltoAllvRuntime => {
                mpi_times.insert("tree_neighbour_all_to_all_v_runtime", op_time.time);
            }

            MPICollectiveType::GatherVRuntime => {
                mpi_times.insert("tree_gather_v_runtime", op_time.time);
            }
            MPICollectiveType::ScatterVRuntime => {
                mpi_times.insert("tree_scatter_v_runtime", op_time.time);
            }

            MPICollectiveType::Gather => {
                mpi_times.insert("tree_gather", op_time.time);
            }
            MPICollectiveType::Scatter => {
                mpi_times.insert("tree_scatter", op_time.time);
            }

            MPICollectiveType::GatherV => {
                mpi_times.insert("tree_gather_v", op_time.time);
            }
            MPICollectiveType::ScatterV => {
                mpi_times.insert("tree_scatter_v", op_time.time);
            }

            MPICollectiveType::AllGather => {
                mpi_times.insert("tree_all_gather", op_time.time);
            }
            MPICollectiveType::AllGatherV => {
                mpi_times.insert("tree_all_gather_v", op_time.time);
            }

            MPICollectiveType::DistGraphCreate => {
                mpi_times.insert("tree_dist_graph_create", op_time.time);
            }

            MPICollectiveType::Sort => {
                mpi_times.insert("tree_sort", op_time.time);
            }
        }
    }

    for (&op_type, op_time) in multi_fmm.operator_times.iter() {
        match op_type {
            FmmOperatorType::P2M => {
                operator_times.insert("p2m", op_time.time);
            }
            FmmOperatorType::P2P => {
                operator_times.insert("p2p", op_time.time);
            }
            FmmOperatorType::L2P => {
                operator_times.insert("l2p", op_time.time);
            }
            FmmOperatorType::M2L(_) => {
                if let Some(existing) = operator_times.get_mut("m2l") {
                    *existing += op_time.time;
                } else {
                    operator_times.insert("m2l", op_time.time);
                }
            }
            FmmOperatorType::M2M(_) => {
                if let Some(existing) = operator_times.get_mut("m2m") {
                    *existing += op_time.time;
                } else {
                    operator_times.insert("m2m", op_time.time);
                }
            }
            FmmOperatorType::L2L(_) => {
                if let Some(existing) = operator_times.get_mut("l2l") {
                    *existing += op_time.time;
                } else {
                    operator_times.insert("l2l", op_time.time);
                }
            }
        }
    }

    // Destructure communication times
    let mut communication_times = HashMap::new();
    for (&comm_type, comm_time) in multi_fmm.communication_times.iter() {
        match comm_type {
            CommunicationType::SourceTree => {
                communication_times.insert("source_tree", comm_time.time);
            }
            CommunicationType::TargetTree => {
                communication_times.insert("target_tree", comm_time.time);
            }
            CommunicationType::SourceDomain => {
                communication_times.insert("source_domain", comm_time.time);
            }
            CommunicationType::TargetDomain => {
                communication_times.insert("target_domain", comm_time.time);
            }
            CommunicationType::Layout => {
                communication_times.insert("layout", comm_time.time);
            }
            CommunicationType::GhostExchangeV => {
                communication_times.insert("ghost_exchange_v", comm_time.time);
            }
            CommunicationType::GhostExchangeVRuntime => {
                communication_times.insert("ghost_exchange_v_runtime", comm_time.time);
            }
            CommunicationType::GhostExchangeU => {
                communication_times.insert("ghost_exchange_u", comm_time.time);
            }
            CommunicationType::GatherGlobalFmm => {
                communication_times.insert("gather_global_fmm", comm_time.time);
            }
            CommunicationType::ScatterGlobalFmm => {
                communication_times.insert("scatter_global_fmm", comm_time.time);
            }
        }
    }

    // Destructure metadata times
    let mut metadata_times = HashMap::new();

    for (&metadata_type, metadata_time) in multi_fmm.metadata_times.iter() {
        match metadata_type {
            MetadataType::SourceToTargetData => {
                metadata_times.insert("source_to_target_data", metadata_time.time);
            }
            MetadataType::SourceData => {
                metadata_times.insert("source_data", metadata_time.time);
            }
            MetadataType::TargetData => {
                metadata_times.insert("target_data", metadata_time.time);
            }
            MetadataType::GlobalFmm => {
                metadata_times.insert("global_fmm", metadata_time.time);
            }
            MetadataType::GhostFmmV => {
                metadata_times.insert("ghost_fmm_v", metadata_time.time);
            }
            MetadataType::GhostFmmU => {
                metadata_times.insert("ghost_fmm_u", metadata_time.time);
            }

            MetadataType::DisplacementMap => {
                metadata_times.insert("displacement_map", metadata_time.time);
            }

            MetadataType::MetadataCreation => {
                metadata_times.insert("metadata_creation", metadata_time.time);
            }
        }
    }

    println!(
        "{:?},{:?},{:?},{:?},{:?},{:?},{:?},{:?},\
         {:?},{:?},{:?},{:?},{:?},{:?},{:?},{:?},{:?},{:?}, \
         {:?},{:?},{:?},{:?},{:?},{:?},{:?},{:?}, \
         {:?},{:?},{:?},{:?},{:?},{:?},{:?},{:?},{:?}, \
         {:?},{:?},{:?},{:?},{:?},{:?},{:?},{:?},{:?},{:?},{:?},{:?},{:?},{:?},{:?}\
         {:?},{:?},{:?},{:?},{:?},{:?},{:?},{:?},{:?},{:?},{:?},{:?},{:?},{:?},{:?}",
        id,
        multi_fmm.rank(),
        runtime,
        operator_times.get("p2m").unwrap_or(&0),
        operator_times.get("m2m").unwrap_or(&0),
        operator_times.get("l2l").unwrap_or(&0),
        operator_times.get("m2l").unwrap_or(&0),
        operator_times.get("p2p").unwrap_or(&0),
        communication_times.get("source_tree").unwrap_or(&0),
        communication_times.get("target_tree").unwrap_or(&0),
        communication_times.get("source_domain").unwrap_or(&0),
        communication_times.get("target_domain").unwrap_or(&0),
        communication_times.get("layout").unwrap_or(&0),
        communication_times.get("ghost_exchange_v").unwrap_or(&0),
        communication_times
            .get("ghost_exchange_v_runtime")
            .unwrap_or(&0),
        communication_times.get("ghost_exchange_u").unwrap_or(&0),
        communication_times.get("gather_global_fmm").unwrap_or(&0),
        communication_times.get("scatter_global_fmm").unwrap_or(&0),
        metadata_times.get("source_to_target_data").unwrap_or(&0),
        metadata_times.get("source_data").unwrap_or(&0),
        metadata_times.get("target_data").unwrap_or(&0),
        metadata_times.get("global_fmm").unwrap_or(&0),
        metadata_times.get("ghost_fmm_v").unwrap_or(&0),
        metadata_times.get("ghost_fmm_u").unwrap_or(&0),
        metadata_times.get("displacement_map").unwrap_or(&0),
        metadata_times.get("metadata_creation").unwrap_or(&0),
        args.expansion_order,
        args.n_points,
        args.local_depth,
        args.global_depth,
        args.block_size,
        args.n_threads,
        args.n_samples,
        mean_roots_per_rank_source_tree,
        mean_roots_per_rank_target_tree,
        mpi_times.get("all_to_all").unwrap_or(&0),
        mpi_times.get("all_to_all_v").unwrap_or(&0),
        mpi_times.get("neighbour_all_to_all").unwrap_or(&0),
        mpi_times.get("neighbour_all_to_all_v").unwrap_or(&0),
        mpi_times
            .get("neighbour_all_to_all_v_runtime")
            .unwrap_or(&0),
        mpi_times.get("gather").unwrap_or(&0),
        mpi_times.get("scatter").unwrap_or(&0),
        mpi_times.get("gather_v").unwrap_or(&0),
        mpi_times.get("scatter_v").unwrap_or(&0),
        mpi_times.get("gather_v_runtime").unwrap_or(&0),
        mpi_times.get("scatter_v_runtime").unwrap_or(&0),
        mpi_times.get("all_gather").unwrap_or(&0),
        mpi_times.get("all_gather_v").unwrap_or(&0),
        mpi_times.get("dist_graph_create").unwrap_or(&0),
        mpi_times.get("sort").unwrap_or(&0),
        mpi_times.get("tree_all_to_all").unwrap_or(&0),
        mpi_times.get("tree_all_to_all_v").unwrap_or(&0),
        mpi_times.get("tree_neighbour_all_to_all").unwrap_or(&0),
        mpi_times.get("tree_neighbour_all_to_all_v").unwrap_or(&0),
        mpi_times
            .get("tree_neighbour_all_to_all_v_runtime")
            .unwrap_or(&0),
        mpi_times.get("tree_gather").unwrap_or(&0),
        mpi_times.get("tree_scatter").unwrap_or(&0),
        mpi_times.get("tree_gather_v").unwrap_or(&0),
        mpi_times.get("tree_scatter_v").unwrap_or(&0),
        mpi_times.get("tree_gather_v_runtime").unwrap_or(&0),
        mpi_times.get("tree_scatter_v_runtime").unwrap_or(&0),
        mpi_times.get("tree_all_gather").unwrap_or(&0),
        mpi_times.get("tree_all_gather_v").unwrap_or(&0),
        mpi_times.get("tree_dist_graph_create").unwrap_or(&0),
        mpi_times.get("tree_sort").unwrap_or(&0),
    );
}
