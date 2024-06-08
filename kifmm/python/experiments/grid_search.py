import itertools
from time import time
import csv
import dask
from dask.distributed import Client, progress, LocalCluster
from dask.diagnostics import ProgressBar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kifmm_py import KiFmm

# @dask.delayed
def setup_fmm_blas(sources, targets, charges, kernel_eval_type, kernel, field_translation, surface_diff, svd_threshold, depth, expansion_order):
    tmp = [expansion_order] * (depth + 1)

    start_time = time()
    fmm = KiFmm(
        tmp,
        sources,
        targets,
        charges,
        kernel_eval_type,
        kernel,
        field_translation,
        prune_empty=True,
        timed=True,
        svd_threshold=svd_threshold,
        surface_diff=surface_diff,
        depth=depth
    )
    setup_time = time() - start_time
    return fmm, setup_time


@dask.delayed
def setup_fmm_fft(sources, targets, charges, kernel_eval_type, kernel, field_translation, depth, expansion_order):
    tmp = [expansion_order] * (depth + 1)

    start_time = time()
    fmm = KiFmm(
        tmp,
        sources,
        targets,
        charges,
        kernel_eval_type,
        kernel,
        field_translation,
        prune_empty=True,
        timed=True,
        depth=depth
    )
    setup_time = time() - start_time
    return fmm, setup_time


def main():
    dim = 3
    np.random.seed(0)

    cluster = LocalCluster()
    client = Client(cluster)
    print(client)

    # Single Precision BLAS
    dtype = np.float32
    ctype = np.complex64

    surface_diff_vec = [0, 1, 2]
    svd_threshold_vec = [None, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 2e-2]
    depth_vec = [4, 5]
    expansion_order_vec = [3, 4, 5, 6]

    # Set FMM Parameters
    n_vec = 1
    n_crit = None
    depth = 3
    n_sources = 1000000
    n_targets = 1000000
    kernel = "laplace"
    field_translation = "blas"
    kernel_eval_type = (
        "eval"
    )
    parameters = list(itertools.product(surface_diff_vec, svd_threshold_vec, depth_vec, expansion_order_vec))

    sources = np.reshape(
        np.random.rand(n_sources * dim), (n_sources, dim)
    ).astype(dtype)
    targets = np.reshape(
        np.random.rand(n_targets * dim), (n_targets, dim)
    ).astype(dtype)
    charges = np.reshape(
        np.random.rand(n_sources * n_vec), (n_sources, n_vec)
    ).astype(dtype)

    soucres = client.scatter


    rel_error = []
    times = []
    setup = []
    fmms = []

    for (i, (surface_diff, svd_threshold, depth, expansion_order)) in enumerate(parameters):
        fmm = setup_fmm_blas(
            sources,
            targets,
            charges,
            kernel_eval_type,
            kernel,
            field_translation,
            surface_diff,
            svd_threshold,
            depth,
            expansion_order
        )
        fmms.append(
            client.submit(
            setup_fmm_blas,
            sources,
            targets,
            charges,
            kernel_eval_type,
            kernel,
            field_translation,
            surface_diff,
            svd_threshold,
            depth,
            expansion_order
            ))

    # with ProgressBar():
        # fmms = dask.compute(fmms)

if __name__ == "__main__":
    main()
