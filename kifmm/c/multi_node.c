#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "kifmm_rs.h"

// Points per process
const int n_sources = 100000;
const int n_targets = 100000;

double drand() { return (double)rand() / RAND_MAX; }

int main(int argc, char **argv) {

  double *sources = (double *)malloc(3 * n_sources * sizeof(double));
  double *targets = (double *)malloc(3 * n_targets * sizeof(double));
  double *charges = (double *)malloc(n_sources * sizeof(double));

  for (int i = 0; i < 3 * n_sources; ++i) {
    sources[i] = drand();
  }

  for (int i = 0; i < 3 * n_targets; ++i) {
    targets[i] = drand();
  }

  for (int i = 0; i < n_sources; ++i) {
    charges[i] = drand();
  }

  bool prune_empty = true;
  uint64_t local_depth = 1;
  uint64_t global_depth = 1;
  bool eval_type = true; // evaluate potentials
  bool timed = true;

  uintptr_t expansion_order[] = {6};
  uintptr_t nexpansion_order = 1;
  uintptr_t block_size = 32;

  uint64_t sort_kind = 1;    // sample sort
  uintptr_t n_samples = 100; // number of samples in sample sort

  // MPI
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm comm;
  MPI_Comm_dup(MPI_COMM_WORLD, &comm);
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  struct FmmEvaluatorMPI *evaluator = laplace_fft_f64_mpi_alloc(
      timed, expansion_order, nexpansion_order, eval_type,
      (const void *)sources, n_sources * 3, (const void *)targets,
      n_targets * 3, (const void *)charges, n_sources, prune_empty, local_depth,
      global_depth, block_size, sort_kind, n_samples, (void *)comm);

  evaluate_mpi(evaluator);
  FmmOperatorTimes *times = operator_times_mpi(evaluator);

  if (times->length > 0) {

    MortonKeys *leaves = leaves_target_tree_mpi(evaluator);

    printf("\n");
    printf("Number of leaf keys (n): %zu at rank = %u/%u\n", leaves->len, rank,
           size - 1);
    for (uintptr_t i = 0; i < 5; ++i) {
      printf("Element %zu: %llu\n", i, leaves->data[i]);
    }

    Potentials *potentials = leaf_potentials_mpi(evaluator, leaves->data[0]);

    Coordinates *coordinates =
        coordinates_source_tree_mpi(evaluator, leaves->data[0]);

    printf("\n");
    printf("Number of coordinates at rank = %zu/%d: %zu\n",
        (size_t)coordinates->len, rank, size - 1);
    const double *coords = (const double *)coordinates->data;
    for (uintptr_t i = 0; i < 5; ++i) {
      printf("Element %zu: [%f, %f, %f]\n", i, coords[i * 3], coords[i * 3 + 1],
             coords[i * 3 + 2]);
    }

    printf("\n");
    printf("Number of potentials: %zu\n", potentials->n);
    Potential *pot = &potentials->data[0];
    const double *data = (const double *)pot->data;
    for (uintptr_t i = 0; i < 5; ++i) {
      printf("Element %zu: %f\n", i, data[i]);
    }

    printf("\n");
    printf("Time Operators, rank = %u/%u \n", rank, size - 1);
    for (uintptr_t i = 0; i < times->length; i++) {

      FmmOperatorEntry op_entry = times->times[i];

      printf("Time: %llu ms, Operator: ", op_entry.time.time);

      switch (op_entry.op_type.tag) {
      case FmmOperatorType_P2M:
        printf("P2M\n");
        break;
      case FmmOperatorType_M2M:
        printf("M2M (%llu)\n", op_entry.op_type.m2m);
        break;
      case FmmOperatorType_M2L:
        printf("M2L (%llu)\n", op_entry.op_type.m2l);
        break;
      case FmmOperatorType_L2L:
        printf("L2L (%llu)\n", op_entry.op_type.l2l);
        break;
      case FmmOperatorType_L2P:
        printf("L2P\n");
        break;
      case FmmOperatorType_P2P:
        printf("P2P\n");
        break;
      default:
        printf("Unknown operator\n");
        break;
      }
    }

  } else {
    printf("FMM not timed");
  }

  // Cleanup
  free(sources);
  free(targets);
  free(charges);
  free_fmm_evaluator_mpi(evaluator);

  MPI_Finalize();

  return 0;
}