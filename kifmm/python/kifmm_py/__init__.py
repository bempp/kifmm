"""
Python interface for KiFMM-rs
"""
from enum import Enum

from ._kifmm_rs import lib, ffi

import numpy as np


KERNELS = ["laplace", "helmholtz"]
KERNEL_EVAL_TYPES = {"eval": 0, "eval_deriv": 1}
KERNEL_EVAL_SIZE = {"eval": 1, "eval_deriv": 4}
FIELD_TRANSLATION_TYPES = ["fft", "blas_svd", "blas_rsvd"]
KERNEL_DTYPE = {
    "laplace": [np.dtypes.Float32DType, np.dtypes.Float64DType],
    "helmholtz": [np.dtypes.Complex64DType, np.dtypes.Complex128DType],
}


class EvalMode(Enum):
    Value=1
    ValueDeriv=2


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


class KiFmm:
    """
    Wraps around the low level Rust interface.
    """

    def __init__(
        self,
        expansion_order,
        sources,
        targets,
        charges,
        kernel_eval_type,
        kernel,
        field_translation,
        n_crit=None,
        depth=None,
        prune_empty=True,
        timed=False,
        svd_threshold=None,
        wavenumber=None,
        surface_diff=None,
        block_size=None,
        rsvd=None
    ):
        """Constructor for Single Node FMMss.

        Args:
            expansion_order (list[int], int): The expansion order of the FMM, if specifying a depth expansion order must be specified for each tree level in a list
            sources (np.ndarray): Source coordinates, real data expected in C order with shape '[n_points, dim]'
            targets (np.ndarray): Target coordinates, real data expected in C order with shape '[n_points, dim]'
            charges (np.ndarray): Charge data, real or complex (dependent on kernel) of shape dimensions '[n_charges, n_vecs]' where each of 'n_vecs' is associated with 'n_charges'. 'n_vecs' > 1 only supported with BLAS field translations.
            kernel_eval_type (str):  Either 'eval_deriv' - to evaluate potentials and gradients, or 'eval' to evaluate potentials alone
            kernel (str): Either 'laplace' or 'helmholtz' supported.
            field_translation (str): Either 'fft' or 'blas'.
            n_crit (int, optional):  Maximum number of particles per leaf box, must be less than number of particles in domain. Must specify either n_crit or depth.
            depth (int, optional):  Maximum depth of octree, must match the number of dimension of the specified expansion orders. Must specify either n_crit or depth.
            prune_empty (bool, optional): Optionally drop empty leaf boxes for performance in FMM.
            timed (bool, optional): Whether or not to store operator runtimes in the 'times' attribute. Defaults to False.
            svd_threshold (float, optional): Defines a threshold for the SVD compression of M2L operators when using BLAS field translations, derived from machine precision if not set.
            Defaults to None for FFT field translations.
            wavenumber (float, optional): Must specify a wavenumber for Helmholtz kernels. Defaults to None for Laplace kernels.
            surface_diff (int, optional): Calculated as check_expansion_order-equivalent_expansion_order, used to provide more stability when using BLAS based field translations.
            Defaults to 0 if not specified for BLAS based field translations.
            block_size (int, optional): Maximum block size used in FFT based M2L translations, if not specified set to 128.
            rsvd (tuple(int), optional): If using BLAS based field translations, can optionally use randomised SVD to accelerate precomputations.
        """

        # Check valid tree spec
        try:
            assert (n_crit is None and depth is not None) or (
                n_crit is not None and depth is None
            )
        except:
            raise TypeError("Either of 'n_crit' or 'depth' must be supplied")

        # Check valid expansion order
        try:
            assert isinstance(expansion_order, list) or isinstance(expansion_order, int)
        except:
            raise TypeError(f"Expansion orders of type {type(expansion_order)}")

        try:
            if isinstance(expansion_order, list):
                for e in expansion_order:
                    assert e >= 2
            else:
                assert expansion_order >= 2
        except:
            raise TypeError(f"Expansion orders must be >= 2")

        try:
            if isinstance(expansion_order, list):
                assert (len(expansion_order) == depth + 1) and (
                    n_crit is None and depth is not None
                )
        except:
            raise TypeError(
                "Dimension of expansion order must match tree depth if using variable expansion order"
            )

        # Check that inputs are numpy arrays
        try:
            assert (
                isinstance(sources, np.ndarray)
                and isinstance(targets, np.ndarray)
                and isinstance(charges, np.ndarray)
            )
        except:
            raise TypeError(
                f"sources of type {type(sources)}, targets of type {type(targets)}, charges of type {type(charges)}"
            )

        try:
            assert (
                np.isreal(sources[0][0])
                and np.isreal(targets[0][0])
                and sources.shape[1] == 3
                and targets.shape[1] == 3
            )
        except:
            raise TypeError("sources and targets must be reals, and of shape [3, N]")

        try:
            assert type(sources[0].dtype) == type(targets[0].dtype)
        except:
            raise TypeError(
                f"sources of type {type(sources[0].dtype)}, targets of type {type(targets[0].dtype)} do not match."
            )

        # Check for valid n_crit
        try:
            if n_crit is not None:
                assert n_crit < sources.shape[0] and n_crit < targets.shape[0]
        except:
            raise TypeError(f"ncrit={ncrit} is too large for specified sources/targets")

        try:
            if n_crit is not None:
                assert isinstance(expansion_order, int)
        except:
            raise TypeError(
                f"Only a single expansion order must be specified if constructing a tree with 'n_crit'"
            )

        # Check for valid tree
        try:
            assert isinstance(prune_empty, bool)

        except:
            raise TypeError(f"'prune_empty' must be a boolean")

        # Check for valid kernel
        try:
            assert kernel in KERNELS
        except:
            raise TypeError(
                f"kernel '{kernel}' invalid choice, must be one of {KERNELS}"
            )

        # CHeck for valid kernel
        try:
            assert kernel_eval_type in KERNEL_EVAL_TYPES.keys()
        except:
            raise TypeError(
                f"kernel eval type '{kernel_eval_type}' invalid choice, must be one of {list(KERNEL_EVAL_TYPES.keys())}"
            )

        expected_dtypes = KERNEL_DTYPE[kernel]
        try:
            assert type(charges[0].dtype) in expected_dtypes
        except:
            raise TypeError(
                f"charges of the wrong type '{type(charges[0].dtype)}' for this kernel"
            )

        # Check that field translation is valid
        try:
            assert field_translation in FIELD_TRANSLATION_TYPES
        except:
            raise TypeError(
                f"field translation '{field_translation}' is not valid, expect one of {FIELD_TRANSLATION_TYPES}"
            )

        # Check for valid block size
        try:
            if block_size is not None:
                assert isinstance(block_size, int)
        except:
            raise TypeError(f"block sizes of type '{type(block_size)}', expected int")

        try:
            assert isinstance(timed, bool)
        except:
            raise TypeError(f"Must specify 'True' or 'False' for timed")

        self.dtype = type(sources[0].dtype)

        if isinstance(expansion_order, list):
            self.expansion_order = expansion_order
        else:
            self.expansion_order = [expansion_order]

        if field_translation == "blas_rsvd":
            self.rsvd = rsvd.settings

        if kernel == "helmholtz":
            try:
                assert wavenumber is not None
            except:
                raise TypeError("wavenumber must be set for Helmholtz kernels")

        self.sources = sources
        self.targets = targets
        self.charges = charges
        self.n_crit = n_crit
        self.depth = depth
        self.prune_empty = prune_empty
        self.kernel_eval_type = KERNEL_EVAL_TYPES[kernel_eval_type]
        self.kernel_eval_size = KERNEL_EVAL_SIZE[kernel_eval_type]
        self.kernel = kernel
        self.field_translation = field_translation
        self.constructor = CONSTRUCTORS[self.dtype][self.kernel][self.field_translation]
        self.svd_threshold = svd_threshold
        self.surface_diff = surface_diff
        self.block_size = block_size
        self.wavenumber = wavenumber

        if self.kernel == "laplace":
            if self.field_translation == "fft":
                pass
        #         self.fmm = self.constructor(
        #             self.expansion_order,
        #             self.sources,
        #             self.targets,
        #             self.charges,
        #             self.prune_empty,
        #             self.kernel_eval_type,
        #             self.n_crit,
        #             self.depth,
        #             self.block_size,
        #         )

            elif self.field_translation == "blas_svd":
                pass
        #         self.fmm = self.constructor(
        #             self.expansion_order,
        #             self.sources,
        #             self.targets,
        #             self.charges,
        #             self.prune_empty,
        #             self.kernel_eval_type,
        #             self.svd_threshold,
        #             self.n_crit,
        #             self.depth,
        #             self.surface_diff,
        #             self.rsvd
        #         )

            elif self.field_translation == "blas_rsvd":
                pass

            else:
                pass

        elif self.kernel == "helmholtz":
            if self.field_translation == "fft":
                pass
        #         self.fmm = self.constructor(
        #             self.expansion_order,
        #             self.sources,
        #             self.targets,
        #             self.charges,
        #             self.prune_empty,
        #             self.kernel_eval_type,
        #             self.wavenumber,
        #             self.n_crit,
        #             self.depth,
        #             self.block_size,
        #         )

            elif self.field_translation == "blas_svd":
                pass
        #         self.fmm = self.constructor(
        #             self.expansion_order,
        #             self.sources,
        #             self.targets,
        #             self.charges,
        #             self.prune_empty,
        #             self.kernel_eval_type,
        #             self.wavenumber,
        #             self.svd_threshold,
        #             self.n_crit,
        #             self.depth,
        #             self.surface_diff,
        #         )
            else:
                pass

        # self.source_keys = self.fmm.source_keys
        # self.source_keys_set = set(self.source_keys)
        # self.target_keys = self.fmm.target_keys
        # self.target_keys_set = set(self.target_keys)
        # self.source_leaves = self.fmm.source_leaves
        # self.source_leaves_set = set(self.source_leaves)
        # self.target_leaves = self.fmm.target_leaves
        # self.target_leaves_set = set(self.target_leaves)
        # self.source_tree_depth = self.fmm.source_tree_depth
        # self.target_tree_depth = self.fmm.target_tree_depth
        # self.source_global_indices = self.fmm.source_global_indices
        # self.target_global_indices = self.fmm.target_global_indices
        # self.timed = timed
        # self.times = None