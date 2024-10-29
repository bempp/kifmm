//! Run a parametrised  distributed FMM with BLAS based M2L in f32
use clap::Parser;
use green_kernels::laplace_3d::Laplace3dKernel;
use kifmm::{
    traits::types::{CommunicationType, FmmOperatorType, MetadataType},
    tree::{helpers::points_fixture, types::SortKind},
    BlasFieldTranslationSaRcmp, EvaluateMulti, FmmSvdMode, MultiNodeBuilder,
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

    /// Singular Value Cutoff
    #[arg(long, default_value_t = 1e-7)]
    threshold: f32,

    /// Number of threads per MPI process
    #[arg(long, default_value_t = 1)]
    n_threads: usize,

    /// Number of samples in parallel sample sort, must be
    /// less than the number of samples and greater than 0
    #[arg(long, default_value_t = 10)]
    n_samples: usize,
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
    let threshold = args.threshold;
    let n_threads = args.n_threads;
    let n_samples = args.n_samples;
    let id = args.id;

    assert!(n_samples > 0 && n_samples < n_points);

    let sort_kind = SortKind::Samplesort { n_samples };

    // Fmm Parameters
    let kernel = Laplace3dKernel::<f32>::new();

    ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build_global()
        .unwrap();

    let source_to_target =
        BlasFieldTranslationSaRcmp::<f32>::new(Some(threshold), None, FmmSvdMode::Deterministic);

    // Generate some random test data local to each process
    let points = points_fixture::<f32>(n_points, None, None, Some(world.rank() as u64));

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
        .parameters(expansion_order, kernel.clone(), source_to_target)
        .unwrap()
        .build()
        .unwrap();

    let start = Instant::now();
    multi_fmm.evaluate().unwrap();
    let runtime = start.elapsed().as_millis();

    // Destructure operator times
    let mut operator_times = HashMap::new();

    for time in multi_fmm.operator_times.iter() {
        match &time.operator {
            FmmOperatorType::P2M => {
                operator_times.insert("p2m", time.time);
            }
            FmmOperatorType::P2P => {
                operator_times.insert("p2p", time.time);
            }
            FmmOperatorType::L2P => {
                operator_times.insert("l2p", time.time);
            }
            FmmOperatorType::M2L(_) => {
                if let Some(existing) = operator_times.get_mut("m2l") {
                    *existing += time.time;
                } else {
                    operator_times.insert("m2l", time.time);
                }
            }
            FmmOperatorType::M2M(_) => {
                if let Some(existing) = operator_times.get_mut("m2m") {
                    *existing += time.time;
                } else {
                    operator_times.insert("m2m", time.time);
                }
            }
            FmmOperatorType::L2L(_) => {
                if let Some(existing) = operator_times.get_mut("l2l") {
                    *existing += time.time;
                } else {
                    operator_times.insert("l2l", time.time);
                }
            }
        }
    }

    // Destructure communication times
    let mut communication_times = HashMap::new();
    for time in multi_fmm.communication_times.iter() {
        match &time.operator {
            CommunicationType::SourceTree => {
                communication_times.insert("source_tree", time.time);
            }
            CommunicationType::TargetTree => {
                communication_times.insert("target_tree", time.time);
            }
            CommunicationType::SourceDomain => {
                communication_times.insert("source_domain", time.time);
            }
            CommunicationType::TargetDomain => {
                communication_times.insert("target_domain", time.time);
            }
            CommunicationType::Layout => {
                communication_times.insert("layout", time.time);
            }
            CommunicationType::GhostExchangeV => {
                communication_times.insert("ghost_exchange_v", time.time);
            }
            CommunicationType::GhostExchangeVRuntime => {
                communication_times.insert("ghost_exchange_v_runtime", time.time);
            }
            CommunicationType::GhostExchangeU => {
                communication_times.insert("ghost_exchange_u", time.time);
            }
            CommunicationType::GatherGlobalFmm => {
                communication_times.insert("gather_global_fmm", time.time);
            }
            CommunicationType::ScatterGlobalFmm => {
                communication_times.insert("scatter_global_fmm", time.time);
            }
        }
    }

    // Destructure metadata times
    let mut metadata_times = HashMap::new();

    for time in multi_fmm.metadata_times.iter() {
        match &time.operator {
            MetadataType::SourceToTargetData => {
                metadata_times.insert("source_to_target_data", time.time);
            }
            MetadataType::SourceData => {
                metadata_times.insert("source_data", time.time);
            }
            MetadataType::TargetData => {
                metadata_times.insert("target_data", time.time);
            }
            MetadataType::GlobalFmm => {
                metadata_times.insert("global_fmm", time.time);
            }
            MetadataType::GhostFmmV => {
                metadata_times.insert("ghost_fmm_v", time.time);
            }
            MetadataType::GhostFmmU => {
                metadata_times.insert("ghost_fmm_u", time.time);
            }
        }
    }

    println!(
        "{:?},{:?},{:?},{:?},{:?},{:?},{:?},{:?}, \
         {:?},{:?},{:?},{:?},{:?},{:?},{:?},{:?},{:?},{:?}, \
         {:?},{:?},{:?},{:?},{:?},{:?},\
         {:?},{:?},{:?},{:?},{:?},{:?},{:?}",
        id,
        multi_fmm.rank,
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
        args.expansion_order,
        args.n_points,
        args.local_depth,
        args.global_depth,
        args.threshold,
        args.n_threads,
        args.n_samples
    );
}
