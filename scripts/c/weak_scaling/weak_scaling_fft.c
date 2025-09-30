#include "helpers.h"

#include <mpi.h>
#ifdef CRAYPAT
#include <pat_api.h>
#endif

#ifdef FMM_USE_DOUBLE
typedef double real_t;
#define FMM_ALLOCATOR laplace_fft_f64_mpi_alloc
#else
typedef float real_t;
#define FMM_ALLOCATOR laplace_fft_f32_mpi_alloc
#endif

static void fill_random(real_t *arr, size_t n) {
    for (size_t i = 0; i < n; i++) {
        arr[i] = (real_t)drand48();
    }
}
int main(int argc, char **argv) {

    Config cfg;
    set_defaults(&cfg);

    // First check for config
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--config") == 0 && i+1 < argc) {
            load_config(argv[i+1], &cfg);
            break;
        }
    }

    // CLI override
    parse_cli(argc, argv, &cfg);

    // Initialise points/charges
    real_t *sources = (real_t *)malloc(3 * cfg.n_sources * sizeof(real_t));
    real_t *targets = (real_t *)malloc(3 * cfg.n_targets * sizeof(real_t));
    real_t *charges = (real_t *)malloc(cfg.n_sources * sizeof(real_t));

    fill_random(sources, 3 * cfg.n_sources);
    fill_random(targets, 3 * cfg.n_targets);
    fill_random(charges, cfg.n_sources);

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm comm;
    MPI_Comm_dup(MPI_COMM_WORLD, &comm);
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // disable for tree and operator construction
    #ifdef CRAYPAT
    PAT_record(PAT_STATE_OFF);
    #endif

    int block_size = 128;

    struct FmmEvaluatorMPI *evaluator = FMM_ALLOCATOR(
        cfg.timed, cfg.expansion_order, cfg.nexpansion_order, cfg.eval_type,
        (const void *)sources, cfg.n_sources * 3, (const void *)targets,
        cfg.n_targets * 3, (const void *)charges, cfg.n_sources, cfg.prune_empty, cfg.local_depth,
        cfg.global_depth, block_size, cfg.sort_kind, cfg.n_samples, (void *)(uintptr_t)comm); // cast to uintptr_t required on cray

    // enable for FMM computation
    #ifdef CRAYPAT
    PAT_record(PAT_STATE_ON);
    #endif
    double t_start = MPI_Wtime();
    evaluate_mpi(evaluator);
    double t_end = MPI_Wtime();

    #ifdef CRAYPAT
    PAT_record(PAT_STATE_OFF);
    #endif

    // Collect timings
    uint64_t runtime = (uint64_t)((t_end - t_start) * 1000.0); // milliseconds
    FmmOperatorTimes *op_times = operator_times_mpi(evaluator);
    CommunicationTimes *comm_times = communication_times_mpi(evaluator);
    MetadataTimes *meta_times = metadata_times_mpi(evaluator);

    uint64_t p2m, m2m, l2l, m2l, l2p, p2p;
    collect_operator_times(op_times, &p2m, &m2m, &l2l, &m2l, &l2p, &p2p);

    uint64_t src_tree, tgt_tree, src_dom, tgt_dom, layout;
    uint64_t ghost_v, ghost_v_rt, ghost_u, gather_fmm, scatter_fmm;
    collect_communication_times(comm_times,
        &src_tree, &tgt_tree, &src_dom, &tgt_dom, &layout,
        &ghost_v, &ghost_v_rt, &ghost_u, &gather_fmm, &scatter_fmm);

    uint64_t src_to_tgt, src_data, tgt_data, glob_fmm;
    uint64_t ghost_fmm_v, ghost_fmm_u, disp_map, meta_creation;
    collect_metadata_times(meta_times,
        &src_to_tgt, &src_data, &tgt_data, &glob_fmm,
        &ghost_fmm_v, &ghost_fmm_u, &disp_map, &meta_creation);

    char *row = build_row("0", rank, runtime, p2m, m2m, l2l, m2l, p2p,
        src_tree, tgt_tree, src_dom, tgt_dom, layout, ghost_v, ghost_v_rt,
        ghost_u, gather_fmm, scatter_fmm, src_to_tgt, src_data, tgt_data,
        glob_fmm, ghost_fmm_v, ghost_fmm_u,
        *cfg.expansion_order, cfg.n_sources,
        cfg.local_depth, cfg.global_depth,
        block_size,
        cfg.nthreads, cfg.n_samples);

    // Print to stdout
    printf("%s\n", row);
    free(row);
    free(sources);
    free(targets);
    free(charges);
    free_fmm_evaluator_mpi(evaluator);
    free(cfg.expansion_order);

    MPI_Finalize();
    return 0;
}