#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "kifmm_rs.h"

// Points per process
const int n_sources = 100000;
const int n_targets = 100000;

double drand() { return (double)rand() / RAND_MAX; }

int main () {

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




    return 0;
}