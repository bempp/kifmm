#include "kifmm_rs.h"
#include <stdio.h>
#include <stdlib.h>

const int NSOURCES = 10000;
const int NTARGETS = 10000;

double drand() { return (double)rand() / RAND_MAX; }

int main() {
  double *sources = (double *)malloc(3 * NSOURCES * sizeof(double));
  double *targets = (double *)malloc(3 * NTARGETS * sizeof(double));
  double *charges = (double *)malloc(NSOURCES * sizeof(double));

  for (int i = 0; i < 3 * NSOURCES; ++i) {
    sources[i] = drand();
  }

  for (int i = 0; i < 3 * NTARGETS; ++i) {
    targets[i] = drand();
  }

  for (int i = 0; i < NSOURCES; ++i) {
    charges[i] = drand();
  }

  bool prune_empty = true;
  uint64_t n_crit = 150;
  uint64_t depth = 0;
  double singular_value_threshold = 0.001;

  uintptr_t expansion_order[] = {6};
  uintptr_t nexpansion_order = 1;
  uintptr_t block_size = 32;

  // Instantiate a Laplace evaluator
  struct FmmEvaluator *evaluator = laplace_fft_f64_alloc(
      expansion_order, nexpansion_order, (const void *)sources, NSOURCES * 3,
      (const void *)targets, NTARGETS * 3, (const void *)charges, NSOURCES,
      prune_empty, n_crit, depth, block_size);

  bool timed = true;
  evaluate(evaluator, timed);

  // Potentials pot = potentials(evaluator);

  // const double *array = (const double *)pot.data;

  // // Print the array
  // for (uintptr_t i = 0; i < 5; ++i) {
  //   printf("%lf ", array[i]);
  // }

  MortonKeys *leaves = leaves_target_tree(evaluator);

    printf("Number of leaf keys (n): %zu\n", leaves->len);
  for (uintptr_t i = 0; i < 5; ++i) {
      printf("Element %zu: %llu\n", i, leaves->data[i]);
  }

  Coordinates *coordinates = coordinates_source_tree(evaluator, leaves->data[123]);
  printf("Number of coordinates: %zu\n", coordinates->len);

  const double *coords = (const double *)coordinates->data;
  for (uintptr_t i=0; i<5; ++i) {
      printf("Element %zu: [%f, %f, %f]\n", i, coords[i*3], coords[i*3 + 1], coords[i*3+2]);
  }
  // const uint64_t *array = (const uint64_t*)leaves.data;

  // Cleanup
  free(sources);
  free(targets);
  free(charges);
  free_fmm_evaluator(evaluator);

  return 0;
}
