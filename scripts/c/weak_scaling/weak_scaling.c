#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include <mpi.h>
#ifdef CRAYPAT
#include <pat_api.h>
#endif

#include "kifmm_rs.h"

typedef struct {
    // Tree parameters
    bool prune_empty;
    uint64_t local_depth;
    uint64_t global_depth;
    uint64_t sort_kind;
    uint64_t n_samples;

    // Runtime parameters
    bool eval_type;
    bool timed;
    int n_sources;
    int n_targets;

    // FMM Parameters
    uintptr_t *expansion_order;
    uintptr_t nexpansion_order;
} Config;

void set_defaults(Config *cfg) {
    // --- Tree parameters ---
    cfg->prune_empty   = true;
    cfg->local_depth   = 1;
    cfg->global_depth  = 1;
    cfg->sort_kind     = 1;
    cfg->n_samples     = 100;

    // --- Runtime parameters ---
    cfg->eval_type     = true;
    cfg->timed         = true;
    cfg->n_sources     = 100000;
    cfg->n_targets     = 100000;

    // --- FMM parameters ---
    cfg->nexpansion_order = cfg->local_depth + cfg->global_depth;
    cfg->expansion_order  = malloc(cfg->nexpansion_order * sizeof(uintptr_t));
    if (!cfg->expansion_order) {
        perror("malloc expansion_order");
        exit(EXIT_FAILURE);
    }
    for (uintptr_t i = 0; i < cfg->nexpansion_order; i++) {
        cfg->expansion_order[i] = 6; // default order per level
    }
}

void resize_expansion_order(Config *cfg) {
    free(cfg->expansion_order);
    cfg->nexpansion_order = cfg->local_depth + cfg->global_depth;
    cfg->expansion_order  = malloc(cfg->nexpansion_order * sizeof(uintptr_t));
    if (!cfg->expansion_order) {
        perror("malloc expansion_order");
        exit(EXIT_FAILURE);
    }
    for (uintptr_t i = 0; i < cfg->nexpansion_order; i++) {
        cfg->expansion_order[i] = 6; // refill defaults
    }
}


void load_config(const char *filename, Config *cfg) {
    FILE *f = fopen(filename, "r");
    if (!f) { perror("open config"); return; }

    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#' || line[0] == '\n') continue;

        char key[128], val[128];
        if (sscanf(line, "%127[^=]=%127s", key, val) == 2) {
            if (strcmp(key, "prune_empty") == 0) {
                cfg->prune_empty = (strcmp(val, "true") == 0);
            } else if (strcmp(key, "local_depth") == 0) {
                cfg->local_depth = strtoull(val, NULL, 10);
                resize_expansion_order(cfg);
            } else if (strcmp(key, "global_depth") == 0) {
                cfg->global_depth = strtoull(val, NULL, 10);
                resize_expansion_order(cfg);
            } else if (strcmp(key, "sort_kind") == 0) {
                cfg->sort_kind = strtoull(val, NULL, 10);
            } else if (strcmp(key, "n_samples") == 0) {
                cfg->n_samples = strtoull(val, NULL, 10);
            } else if (strcmp(key, "eval_type") == 0) {
                cfg->eval_type = (strcmp(val, "true") == 0);
            } else if (strcmp(key, "timed") == 0) {
                cfg->timed = (strcmp(val, "true") == 0);
            } else if (strcmp(key, "n_sources") == 0) {
                cfg->n_sources = atoi(val);
            } else if (strcmp(key, "n_targets") == 0) {
                cfg->n_targets = atoi(val);
            } else if (strcmp(key, "expansion_order") == 0) {
                if (strchr(val, ',')) {
                    char *token = strtok(val, ",");
                    size_t count = 0;
                    while (token && count < cfg->nexpansion_order) {
                        cfg->expansion_order[count++] = strtoull(token, NULL, 10);
                        token = strtok(NULL, ",");
                    }
                } else {
                    uintptr_t order = strtoull(val, NULL, 10);
                    for (uintptr_t i = 0; i < cfg->nexpansion_order; i++) {
                        cfg->expansion_order[i] = order;
                    }
                }
            }
        }
    }
    fclose(f);
}

void parse_cli(int argc, char **argv, Config *cfg) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--prune_empty") == 0 && i+1 < argc) {
            cfg->prune_empty = (strcmp(argv[++i], "true") == 0);
        } else if (strcmp(argv[i], "--local_depth") == 0 && i+1 < argc) {
            cfg->local_depth = strtoull(argv[++i], NULL, 10);
            resize_expansion_order(cfg);
        } else if (strcmp(argv[i], "--global_depth") == 0 && i+1 < argc) {
            cfg->global_depth = strtoull(argv[++i], NULL, 10);
            resize_expansion_order(cfg);
        } else if (strcmp(argv[i], "--sort_kind") == 0 && i+1 < argc) {
            cfg->sort_kind = strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--n_samples") == 0 && i+1 < argc) {
            cfg->n_samples = strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--eval_type") == 0 && i+1 < argc) {
            cfg->eval_type = (strcmp(argv[++i], "true") == 0);
        } else if (strcmp(argv[i], "--timed") == 0 && i+1 < argc) {
            cfg->timed = (strcmp(argv[++i], "true") == 0);
        } else if (strcmp(argv[i], "--n_sources") == 0 && i+1 < argc) {
            cfg->n_sources = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--n_targets") == 0 && i+1 < argc) {
            cfg->n_targets = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--expansion_order") == 0 && i+1 < argc) {
            char *val = argv[++i];
            if (strchr(val, ',')) {
                // comma-separated list
                char *token = strtok(val, ",");
                size_t count = 0;
                while (token && count < cfg->nexpansion_order) {
                    cfg->expansion_order[count++] = strtoull(token, NULL, 10);
                    token = strtok(NULL, ",");
                }
            } else {
                uintptr_t order = strtoull(val, NULL, 10);
                for (uintptr_t j = 0; j < cfg->nexpansion_order; j++) {
                    cfg->expansion_order[j] = order;
                }
            }
        }
    }
}


// Points per process
const int n_sources = 100000;
const int n_targets = 100000;

double drand() { return (double)rand() / (double)RAND_MAX; }
float frand() { return (float)rand() / (float)RAND_MAX; }

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
    double *sources = (double *)malloc(3 * n_sources * sizeof(double));
    double *targets = (double *)malloc(3 * n_targets * sizeof(double));
    double *charges = (double *)malloc(n_sources * sizeof(double));

    for (int i = 0; i < 3 * cfg.n_sources; ++i) {
        sources[i] = drand();
    }

    for (int i = 0; i < 3 * cfg.n_targets; ++i) {
        targets[i] = drand();
    }

    for (int i = 0; i < cfg.n_sources; ++i) {
        charges[i] = drand();
    }

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

    struct FmmEvaluatorMPI *evaluator = laplace_fft_f64_mpi_alloc(
        cfg.timed, cfg.expansion_order, cfg.nexpansion_order, cfg.eval_type,
        (const void *)sources, cfg.n_sources * 3, (const void *)targets,
        cfg.n_targets * 3, (const void *)charges, cfg.n_sources, cfg.prune_empty, cfg.local_depth,
        cfg.global_depth, block_size, cfg.sort_kind, cfg.n_samples, (void *)(uintptr_t)comm); // cast to uintptr_t required on cray

    // enable for FMM computation
    #ifdef CRAYPAT
    PAT_record(PAT_STATE_ON);
    #endif

    evaluate_mpi(evaluator);

    #ifdef CRAYPAT
    PAT_record(PAT_STATE_OFF);
    #endif

    FmmOperatorTimes *times = operator_times_mpi(evaluator);

    free(sources);
    free(targets);
    free(charges);
    free_fmm_evaluator_mpi(evaluator);
    free(cfg.expansion_order);

    MPI_Finalize();
    return 0;
}