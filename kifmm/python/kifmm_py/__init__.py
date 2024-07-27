"""
Python interface for KiFMM-rs
"""
from enum import Enum

from ._kifmm_rs import lib, ffi

import numpy as np


class EvalMode(Enum):
    Value=1
    ValueDeriv=4

CONSTRUCTORS = {
    np.dtypes.Float32DType: {
        "laplace": {"blas_svd": lib.laplace_blas_svd_f32_alloc, "blas_rsvd": lib.laplace_blas_rsvd_f32_alloc, "fft": lib.laplace_fft_f32_alloc},
        "helmholtz": {
            "blas_svd": lib.helmholtz_blas_svd_f32_alloc,
            "fft": lib.helmholtz_fft_f32_alloc,
        },
    },
    np.dtypes.Float64DType: {
        "laplace": {"blas_svd": lib.laplace_blas_svd_f64_alloc, "blas_rsvd": lib.laplace_blas_rsvd_f64_alloc, "fft": lib.laplace_fft_f64_alloc},
        "helmholtz": {
            "blas_svd": lib.helmholtz_blas_svd_f64_alloc,
            "fft": lib.helmholtz_fft_f64_alloc,
        },
    },
}

class RandomSvdSettings:
    def __init__(self, mode, n_iter, n_components, n_oversamples, random_state):

        if n_iter is None:
            self.n_iter = 1
        else:
            self.n_iter = n_iter

        if n_components is None:
            raise TypeError("n_components must be specified")
        else:
            self.n_components = n_components

        if n_oversamples is None:
            self.n_oversamples = 10
        else:
            self.n_oversamples = n_oversamples

        if random_state is None:
            self.random_state = 1
        else:
            self.random_state = random_state


class BlasFieldTranslation:
    def __init__(self, kernel, svd_threshold, random=False, n_iter=None, n_components=None, n_oversamples=None, random_state=None):
        self.kernel = kernel
        self.svd_threshold = svd_threshold

        if isinstance(self.kernel, HelmholtzKernel):
            if random:
                raise TypeError("Randomised compression unimplemented for this kernel")

        elif isinstance(self.kernel, LaplaceKernel):
            if random:
                self.rsvd = RandomSvdSettings(n_iter, n_components, n_oversamples, random_state)

        else:
            raise TypeError("Unsupported Kernel")


class FftFieldTranslation:
    def __init__(self, kernel, block_size=None):
        self.kernel = kernel
        self.block_size = block_size

        if isinstance(self.kernel, HelmholtzKernel) or isinstance(self.kernel, LaplaceKernel):
            pass
        else:
            raise TypeError("Unsupported Kernel")

class LaplaceKernel:
    def __init__(self, dtype, eval_mode):
        self.dtype = dtype
        self.eval_mode


class HelmholtzKernel:
    def __init__(self, dtype, wavenumber, eval_mode):
        self.dtype = dtype
        self.wavenumber = wavenumber
        self.eval_mode


class SingleNodeTree:
    def __init__(self, sources, targets, charges, n_crit=None, depth=None, prune_empty=False):
        """ Constructor for Single Node Trees
        Args:
            sources (np.ndarray): Source coordinates, real data expected in C order with shape '[n_points, dim]'
            targets (np.ndarray): Target coordinates, real data expected in C order with shape '[n_points, dim]'
            charges (np.ndarray): Charge data, real or complex (dependent on kernel) of shape dimensions '[n_charges, n_vecs]' where each of 'n_vecs' is associated with 'n_charges'. 'n_vecs' > 1 only supported with BLAS field translations.
        """
        # Check for valid n_crit
        try:
            if (n_crit is not None and depth is None) or (n_crit is None and depth is not None):
        except:
            raise TypeError(f"Either depth or n_crit can be set")

        try:
            assert isinstance(prune_empty, bool)

        self.sources = sources
        self.targets = targets
        self.charges = charges
        self.n_crit = n_crit
        self.depth = depth
        self.prune_empty = prune_empty


class KiFmm:
    """
    Wraps around the low level Rust interface.
    """

    def __init__(
        self,
        expansion_order,
        tree,
        field_translation,
        timed=False,
    ):
        """Constructor for Single Node FMMss.

        Args:
            expansion_order (list[int], int): The expansion order of the FMM, if specifying a depth expansion order must be specified for each tree level in a list
            tree (obj): Currently only 'SingleNodeTree' supported
            field_translation (obj): Either 'FftFieldTranslation' or 'BlasFieldTranslation'.
            timed (bool): Optionally return operator runtimes.
        """
        pass

    @property
    def target_global_indices(self):
        pass

    @property
    def source_global_indices(self):
        pass

    @property
    def target_leaves(self):
        pass

    @property
    def source_leaves(self):
        pass

    @property
    def all_potentials(self):
        pass

    def potential(self, key):
        pass

    def evaluate(self):
        pass

    def clear(self):
        pass

    def evaluate_kernel_st(self):
        pass

    def evaluate_kernel_mt(self):
        pass

    def plot(self):
        pass
