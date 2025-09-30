#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <inttypes.h>  // for PRIu64

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
    int nthreads;

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
    cfg->nexpansion_order = 1; // default: single order
    cfg->expansion_order  = malloc(cfg->nexpansion_order * sizeof(uintptr_t));
    if (!cfg->expansion_order) {
        perror("malloc expansion_order");
        exit(EXIT_FAILURE);
    }
    cfg->expansion_order[0] = 6; // default order
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

        char key[128], val[256];
        if (sscanf(line, "%127[^=]=%255s", key, val) == 2) {
            if (strcmp(key, "prune_empty") == 0) {
                cfg->prune_empty = (strcmp(val, "true") == 0);
            } else if (strcmp(key, "local_depth") == 0) {
                cfg->local_depth = strtoull(val, NULL, 10);
            } else if (strcmp(key, "nthreads") == 0) {
                cfg->nthreads = strtoull(val, NULL, 1);
            } else if (strcmp(key, "global_depth") == 0) {
                cfg->global_depth = strtoull(val, NULL, 10);
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
            } else if (strcmp(key, "nexpansion_order") == 0) {
                cfg->nexpansion_order = strtoull(val, NULL, 10);
                free(cfg->expansion_order);
                cfg->expansion_order = malloc(cfg->nexpansion_order * sizeof(uintptr_t));
                if (!cfg->expansion_order) {
                    perror("malloc expansion_order");
                    exit(EXIT_FAILURE);
                }
                for (uintptr_t i = 0; i < cfg->nexpansion_order; i++) {
                    cfg->expansion_order[i] = 6; // initialize default
                }
            } else if (strcmp(key, "expansion_order") == 0) {
                if (strchr(val, ',')) {
                    char tmp[256];
                    strncpy(tmp, val, sizeof(tmp)-1);
                    tmp[sizeof(tmp)-1] = '\0';
                    char *token = strtok(tmp, ",");
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
        } else if (strcmp(argv[i], "--global_depth") == 0 && i+1 < argc) {
            cfg->global_depth = strtoull(argv[++i], NULL, 10);
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
        } else if (strcmp(argv[i], "--nexpansion_order") == 0 && i+1 < argc) {
            cfg->nexpansion_order = strtoull(argv[++i], NULL, 10);
            free(cfg->expansion_order);
            cfg->expansion_order = malloc(cfg->nexpansion_order * sizeof(uintptr_t));
            if (!cfg->expansion_order) {
                perror("malloc expansion_order");
                exit(EXIT_FAILURE);
            }
            for (uintptr_t j = 0; j < cfg->nexpansion_order; j++) {
                cfg->expansion_order[j] = 6;
            }
        } else if (strcmp(argv[i], "--expansion_order") == 0 && i+1 < argc) {
            char *val = argv[++i];
            if (strchr(val, ',')) {
                char tmp[256];
                strncpy(tmp, val, sizeof(tmp)-1);
                tmp[sizeof(tmp)-1] = '\0';
                char *token = strtok(tmp, ",");
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


double drand() { return (double)rand() / (double)RAND_MAX; }
float frand() { return (float)rand() / (float)RAND_MAX; }


void collect_operator_times(const FmmOperatorTimes *op_times,
                            uint64_t *p2m, uint64_t *m2m,
                            uint64_t *l2l, uint64_t *m2l,
                            uint64_t *l2p, uint64_t *p2p) {
    *p2m = *m2m = *l2l = *m2l = *l2p = *p2p = 0;

    for (uintptr_t i = 0; i < op_times->length; i++) {
        const FmmOperatorEntry *entry = &op_times->times[i];
        switch (entry->op_type.tag) {
            case FmmOperatorType_P2M:
                *p2m = entry->time.time;
                break;
            case FmmOperatorType_P2P:
                *p2p = entry->time.time;
                break;
            case FmmOperatorType_L2P:
                *l2p = entry->time.time;
                break;
            case FmmOperatorType_M2M:
                *m2m += entry->time.time;
                break;
            case FmmOperatorType_M2L:
                *m2l += entry->time.time;
                break;
            case FmmOperatorType_L2L:
                *l2l += entry->time.time;
                break;
        }
    }
}


void collect_metadata_times(const MetadataTimes *meta_times,
                            uint64_t *source_to_target_data,
                            uint64_t *source_data,
                            uint64_t *target_data,
                            uint64_t *global_fmm,
                            uint64_t *ghost_fmm_v,
                            uint64_t *ghost_fmm_u,
                            uint64_t *displacement_map,
                            uint64_t *metadata_creation) {
    *source_to_target_data = *source_data = *target_data = 0;
    *global_fmm = *ghost_fmm_v = *ghost_fmm_u = 0;
    *displacement_map = *metadata_creation = 0;

    for (uintptr_t i = 0; i < meta_times->length; i++) {
        const MetadataEntry *entry = &meta_times->times[i];
        switch (entry->metadata_type) {
            case MetadataType_SourceToTargetData:
                *source_to_target_data = entry->time.time;
                break;
            case MetadataType_SourceData:
                *source_data = entry->time.time;
                break;
            case MetadataType_TargetData:
                *target_data = entry->time.time;
                break;
            case MetadataType_GlobalFmm:
                *global_fmm = entry->time.time;
                break;
            case MetadataType_GhostFmmV:
                *ghost_fmm_v = entry->time.time;
                break;
            case MetadataType_GhostFmmU:
                *ghost_fmm_u = entry->time.time;
                break;
            case MetadataType_DisplacementMap:
                *displacement_map = entry->time.time;
                break;
            case MetadataType_MetadataCreation:
                *metadata_creation = entry->time.time;
                break;
        }
    }
}


void collect_communication_times(const CommunicationTimes *comm_times,
                                 uint64_t *source_tree,
                                 uint64_t *target_tree,
                                 uint64_t *source_domain,
                                 uint64_t *target_domain,
                                 uint64_t *layout,
                                 uint64_t *ghost_exchange_v,
                                 uint64_t *ghost_exchange_v_runtime,
                                 uint64_t *ghost_exchange_u,
                                 uint64_t *gather_global_fmm,
                                 uint64_t *scatter_global_fmm) {
    *source_tree = *target_tree = *source_domain = *target_domain = 0;
    *layout = *ghost_exchange_v = *ghost_exchange_v_runtime = 0;
    *ghost_exchange_u = *gather_global_fmm = *scatter_global_fmm = 0;

    for (uintptr_t i = 0; i < comm_times->length; i++) {
        const CommunicationEntry *entry = &comm_times->times[i];
        switch (entry->comm_type) {
            case CommunicationType_SourceTree:
                *source_tree = entry->time.time;
                break;
            case CommunicationType_TargetTree:
                *target_tree = entry->time.time;
                break;
            case CommunicationType_SourceDomain:
                *source_domain = entry->time.time;
                break;
            case CommunicationType_TargetDomain:
                *target_domain = entry->time.time;
                break;
            case CommunicationType_Layout:
                *layout = entry->time.time;
                break;
            case CommunicationType_GhostExchangeV:
                *ghost_exchange_v = entry->time.time;
                break;
            case CommunicationType_GhostExchangeVRuntime:
                *ghost_exchange_v_runtime = entry->time.time;
                break;
            case CommunicationType_GhostExchangeU:
                *ghost_exchange_u = entry->time.time;
                break;
            case CommunicationType_GatherGlobalFmm:
                *gather_global_fmm = entry->time.time;
                break;
            case CommunicationType_ScatterGlobalFmm:
                *scatter_global_fmm = entry->time.time;
                break;
        }
    }
}

char *build_row(const char *id,
                int rank,
                uint64_t runtime,
                // operator times
                uint64_t p2m, uint64_t m2m, uint64_t l2l,
                uint64_t m2l, uint64_t p2p,
                // communication times
                uint64_t source_tree, uint64_t target_tree,
                uint64_t source_domain, uint64_t target_domain,
                uint64_t layout,
                uint64_t ghost_exchange_v, uint64_t ghost_exchange_v_runtime,
                uint64_t ghost_exchange_u,
                uint64_t gather_global_fmm, uint64_t scatter_global_fmm,
                // metadata times
                uint64_t source_to_target_data, uint64_t source_data,
                uint64_t target_data, uint64_t global_fmm,
                uint64_t ghost_fmm_v, uint64_t ghost_fmm_u,
                // parameters
                size_t expansion_order, size_t n_points,
                uint64_t local_depth, uint64_t global_depth,
                size_t block_size, size_t n_threads, size_t n_samples)
{
    size_t bufsize = 2048;
    char *buf = malloc(bufsize);
    if (!buf) return NULL;

    snprintf(buf, bufsize,
        "%s,%d,%" PRIu64 ","
        "%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ","
        "%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ","
        "%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ","
        "%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ","
        "%zu,%zu,%" PRIu64 ",%" PRIu64 ",%zu,%zu,%zu",
        id, rank, runtime,
        p2m, m2m, l2l, m2l, p2p,
        source_tree, target_tree, source_domain, target_domain, layout,
        ghost_exchange_v, ghost_exchange_v_runtime, ghost_exchange_u,
        gather_global_fmm, scatter_global_fmm,
        source_to_target_data, source_data, target_data, global_fmm,
        ghost_fmm_v, ghost_fmm_u,
        expansion_order, n_points, local_depth, global_depth,
        block_size, n_threads, n_samples);

    return buf; // caller must free()
}


