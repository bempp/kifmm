#include "kifmm_rs.h"
#include <stdio.h>
#include <stdlib.h>

const int n_sources = 100000;
const int n_targets = 100000;

double drand() { return (double)rand() / RAND_MAX; }

int main() {
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
  uint64_t n_crit = 150;
  uint64_t depth = 0;
  double singular_value_threshold = 0.001;
  bool eval_type = true; // evaluate potentials
  bool timed = true;

  uintptr_t expansion_order[] = {6};
  uintptr_t nexpansion_order = 1;
  uintptr_t block_size = 32;

  // Instantiate a Laplace evaluator
  struct FmmEvaluator *evaluator = laplace_fft_f64_alloc(
      timed,
      expansion_order, nexpansion_order, eval_type, (const void *)sources,
      n_sources * 3, (const void *)targets, n_targets * 3, (const void *)charges,
      n_sources, prune_empty, n_crit, depth, block_size);

   evaluate(evaluator);
   FmmOperatorTimes *times =operator_times(evaluator);

   if (times->length > 0) {
    MortonKeys *leaves = leaves_target_tree(evaluator);

    printf("\n");
    printf("Number of leaf keys (n): %zu\n", leaves->len);
    for (uintptr_t i = 0; i < 5; ++i) {
      printf("Element %zu: %llu\n", i, leaves->data[i]);
    }

    Potentials *potentials = leaf_potentials(evaluator, leaves->data[123]);

    Coordinates *coordinates =
        coordinates_source_tree(evaluator, leaves->data[123]);

    printf("\n");
    printf("Number of coordinates: %zu\n", coordinates->len);
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
    printf("Time Operators\n");
    for (uintptr_t i = 0; i < times->length; i++) {
      FmmOperatorTime op_time = times->times[i];
      printf("Time: %llu ms, Operator: ", op_time.time);

      switch (op_time.operator_.tag) {
      case FmmOperatorType_P2M:
        printf("P2M\n");
        break;
      case FmmOperatorType_M2M:
        printf("M2M (%llu)\n", op_time.operator_.m2m);
        break;
      case FmmOperatorType_M2L:
        printf("M2L (%llu)\n", op_time.operator_.m2l);
        break;
      case FmmOperatorType_L2L:
        printf("L2L (%llu)\n", op_time.operator_.l2l);
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

    // Cleanup
    free_morton_keys(leaves);
    free(potentials);
  } else {
    printf("FMM not timed \n");
  }

  // Cleanup
  free(sources);
  free(targets);
  free(charges);
  free_fmm_evaluator(evaluator);

  return 0;
}
