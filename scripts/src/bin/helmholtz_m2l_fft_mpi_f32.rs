//! Run a parametrised  distributed FMM with FFT based M2L in f32
use clap::Parser;
use green_kernels::{helmholtz_3d::Helmholtz3dKernel, traits::Kernel, types::GreenKernelEvalType};
use itertools::{izip, Itertools};
use kifmm::{
    traits::{
        tree::{MultiFmmTree, MultiTree},
        types::{CommunicationType, FmmOperatorType, MetadataType},
    },
    tree::{helpers::points_fixture, types::SortKind},
    DataAccessMulti, EvaluateMulti, FftFieldTranslation, MultiNodeBuilder,
};
use mpi::traits::*;
use num::{complex::ComplexFloat, One};
use rayon::ThreadPoolBuilder;
use rlst::{c32, rlst_dynamic_array2, RawAccess, RawAccessMut, RlstScalar};
use std::{collections::HashMap, time::Instant};

/// Struct for parsing command-line arguments
#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = String::from("x"))]
    id: String,

    #[arg(long, default_value_t = 3)]
    leaf_expansion_order: usize,

    #[arg(long, default_value_t = 1.0)]
    expansion_order_multiplier: f64,

    #[arg(long, default_value_t = 10.0)]
    wavenumber: f64,

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
}

fn main() {
    let (universe, _threading) =
        mpi::initialize_with_threading(mpi::Threading::Serialized).unwrap();
    let world = universe.world();
    let comm = world.duplicate();

    // Tree parameters
    let args = Args::parse();
    let leaf_expansion_order = args.leaf_expansion_order;
    let prune_empty = args.prune_empty;
    let n_points = args.n_points;
    let local_depth = args.local_depth;
    let global_depth = args.global_depth;
    let block_size = args.block_size;
    let n_threads = args.n_threads;
    let n_samples = args.n_samples;
    let id = args.id;
    let wavenumber = args.wavenumber;
    let expansion_order_multiplier = args.expansion_order_multiplier;

    assert!(n_samples > 0 && n_samples < n_points);

    let sort_kind = SortKind::Samplesort { n_samples };

    // Fmm Parameters
    let expansion_order = std::iter::successors(Some(leaf_expansion_order), move |&prev| {
        let result = (prev as f64 * expansion_order_multiplier).ceil() as usize;
        Some(result)
    })
    .take((global_depth + local_depth + 1) as usize)
    .collect_vec()
    .into_iter()
    .rev()
    .collect_vec();

    let kernel = Helmholtz3dKernel::new(wavenumber as f32);

    ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build_global()
        .unwrap();

    let source_to_target = FftFieldTranslation::new(Some(block_size));

    // Generate some random test data local to each process
    let points = points_fixture::<f32>(n_points, None, None, Some(world.rank() as u64));
    let tmp = vec![c32::one(); n_points];
    let mut charges = rlst_dynamic_array2!(c32, [n_points, 1]);
    charges.data_mut().copy_from_slice(&tmp);

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
            charges.data(),
            &expansion_order,
            kernel.clone(),
            green_kernels::types::GreenKernelEvalType::Value,
            source_to_target,
        )
        .unwrap()
        .build()
        .unwrap();

    // Evaluate FMM
    let start = Instant::now();
    multi_fmm.evaluate().unwrap();
    let runtime = start.elapsed().as_millis();

    // Run convergence test on root node
    let mut l2_error = 0.0;
    // Test on root rank only, regenerate test data to avoid communication of it
    if multi_fmm.rank() == 0 {

        let size = multi_fmm.communicator().size();
        let mut all_coordinates =  vec![0f32; 3 * n_points * (size as usize)];

        let mut offset = 0;
        for seed in 0..size {
            let block = points_fixture::<f32>(n_points, None, None, Some(seed as u64));
            let new_offset = offset + n_points * 3;
            all_coordinates[offset..new_offset].copy_from_slice(block.data());
            offset = new_offset;
        }

        let all_charges = vec![c32::one(); n_points * size as usize];

        let targets_rank = multi_fmm.tree().target_tree().all_coordinates().unwrap();
        let n_targets = targets_rank.len() / 3;
        let mut expected = vec![c32::default(); n_targets];

        let take = 10;
        multi_fmm.kernel().evaluate_mt(
            GreenKernelEvalType::Value,
            &all_coordinates,
            &targets_rank[0..take*3],
            &all_charges,
            &mut expected[0..take],
        );

        // Calculate L2 error
        let found = multi_fmm.potentials().unwrap();

        let mut num = 0.0f32;
        let mut den = 0.0f32;

        for (expected, &found) in izip!(expected, found).take(take) {
            // squared error in complex difference
            let diff_re = expected.re() - found.re();
            let diff_im = expected.im() - found.im();
            num += RlstScalar::powf(diff_re, 2.0f32) + RlstScalar::powf(diff_im, 2.0f32);

            // squared magnitude of expected
            den += RlstScalar::powf(expected.re(), 2.0f32) + RlstScalar::powf(expected.im(), 2.0f32);
        }

        // now take square root
        l2_error = if den != 0.0f32 {
            RlstScalar::sqrt(num) / RlstScalar::sqrt(den)
        } else {
            0.0 // or handle division-by-zero error
        };
    }

    // Destructure operator times
    let mut operator_times = HashMap::new();

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
         {:?},{:?},{:?},{:?},{:?},{:?}, {:?}, {:?}, \
         {:?},{:?},{:?},{:?},{:?},{:?},{:?}, \
         {:?}, {:?}",
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
        args.leaf_expansion_order,
        args.n_points,
        args.local_depth,
        args.global_depth,
        args.block_size,
        args.n_threads,
        args.n_samples,
        args.expansion_order_multiplier,
        l2_error
    );
}
