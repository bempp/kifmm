"""
Python interface for KiFMM-rs
"""

from enum import Enum

from ._kifmm_rs import lib, ffi

import numpy as np


class EvalType(Enum):
    Value = 1
    ValueDeriv = 4


class RandomSvdSettings:
    def __init__(self, n_components, n_oversamples):

        if n_components is None:
            raise TypeError("n_components must be specified")
        else:
            self.n_components = n_components

        if n_oversamples is None:
            self.n_oversamples = 10
        else:
            self.n_oversamples = n_oversamples

        self.n_components_c = ffi.cast("uintptr_t", self.n_components)
        self.n_oversamples_c = ffi.cast("uintptr_t", self.n_oversamples)


class BlasFieldTranslation:
    def __init__(
        self,
        kernel,
        svd_threshold,
        surface_diff=0,
        random=False,
        n_components=None,
        n_oversamples=None,
    ):
        self.kernel = kernel
        self.svd_threshold = svd_threshold
        self.surface_diff = surface_diff
        self.random = random

        if isinstance(self.kernel, HelmholtzKernel):
            if self.random:
                raise TypeError("Randomised compression unimplemented for this kernel")

        elif isinstance(self.kernel, LaplaceKernel):
            if self.random:
                self.rsvd = RandomSvdSettings(
                    n_components,
                    n_oversamples,
                )

        else:
            raise TypeError("Unsupported Kernel")

        if self.kernel.dtype == np.float32:
            self.svd_threshold_c = ffi.cast("float", self.svd_threshold)
        elif self.kernel_dtype == np.float64:
            self.svd_threshold_c = ffi.cast("double", self.svd_threshold)
        else:
            raise TypeError("Unsupported dtype for svd_threshold")

        try:
            assert surface_diff >= 0
        except:
            raise TypeError("surface_diff must be positive or 0")

        self.surface_diff_c = ffi.cast("uintptr_t", self.surface_diff)


class FftFieldTranslation:
    def __init__(self, kernel, block_size=None):
        self.kernel = kernel
        self.block_size = block_size
        self.block_size_c = ffi.cast("uintptr_t", self.block_size)

        if isinstance(self.kernel, HelmholtzKernel) or isinstance(
            self.kernel, LaplaceKernel
        ):
            pass
        else:
            raise TypeError("Unsupported Kernel")


class LaplaceKernel:
    def __init__(self, dtype, eval_type):
        self.dtype = dtype
        self.eval_type = eval_type

        if self.eval_type == EvalType.Value:
            self.eval_type_c = ffi.cast("bool", True)
        elif self.eval_type == EvalType.Value:
            self.eval_type_c = ffi.cast("bool", False)
        else:
            raise TypeError("Unrecognised eval_type")


class HelmholtzKernel:
    def __init__(self, dtype, wavenumber, eval_type):
        self.dtype = dtype
        self.wavenumber = wavenumber
        self.eval_type = eval_type

        try:
            assert type(self.wavenumber) == self.dtype
        except:
            raise TypeError("Invalid wavenumber type")

        if self.eval_type == EvalType.Value:
            self.eval_type_c = ffi.cast("bool", True)
        elif self.eval_type == EvalType.Value:
            self.eval_type_c = ffi.cast("bool", False)
        else:
            raise TypeError("Unrecognised eval_type")

        if self.dtype == np.float32:
            self.wavenumber_c = ffi.cast("float", self.wavenumber)
        elif self.dtype == np.float64:
            self.wavenumber_c = ffi.cast("double", self.wavenumber)
        else:
            raise TypeError("Invalid dtype")


class SingleNodeTree:
    def __init__(
        self, sources, targets, charges, n_crit=None, depth=None, prune_empty=False
    ):
        """Constructor for Single Node Trees
        Args:
            sources (np.ndarray): Source coordinates, real data expected in C order with shape '[n_points, dim]'
            targets (np.ndarray): Target coordinates, real data expected in C order with shape '[n_points, dim]'
            charges (np.ndarray): Charge data, real or complex (dependent on kernel) of shape dimensions '[n_charges, n_vecs]'
            where each of 'n_vecs' is associated with 'n_charges'. 'n_vecs' > 1 only supported with BLAS field translations.
        """

        dim = 3
        try:
            assert len(sources) % dim == 0
            assert len(targets) % dim == 0
            assert len(sources) % len(charges) == 0
        except:
            raise TypeError("Incorrect dimension for sources, targets or charges")

        try:
            valid = (n_crit is None and depth is not None) or (
                n_crit is not None and depth is None
            )
            assert valid

            if n_crit is not None:
                assert (n_crit < len(sources)) and (n_crit < len(targets))

        except:
            raise TypeError("Only one of depth or n_crit must be set")

        try:
            assert isinstance(prune_empty, bool)
        except:
            raise TypeError("prune_empty must be set to a boolean")

        self.sources = sources
        self.targets = targets
        self.charges = charges
        self.n_crit = n_crit
        self.depth = depth
        self.prune_empty = prune_empty

        self.nsources_c = ffi.cast("uintptr_t", np.uint64(len(sources)))
        self.sources_c = ffi.cast("void * ", self.sources.ctypes.data)
        self.ntargets_c = ffi.cast("uintptr_t", np.uint64(len(targets)))
        self.targets_c = ffi.cast("void * ", self.targets.ctypes.data)
        self.ncharges_c = ffi.cast("uintptr_t", np.uint64(len(charges)))
        self.charges_c = ffi.cast("void * ", self.charges.ctypes.data)
        self.prune_empty_c = ffi.cast("bool", prune_empty)

        if self.n_crit is not None:
            self.n_crit_c = ffi.cast("uint64_t", n_crit)
        else:
            self.n_crit_c = ffi.cast("uint64_t", 0)

        if self.depth is not None:
            self.depth_c = ffi.cast("uint64_t", depth)
        else:
            self.depth_c = ffi.cast("uint64_t", 0)


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
        self.expansion_order = expansion_order
        self.tree = tree
        self.field_translation = field_translation
        self.timed = timed

        try:
            if self.tree.depth is None:
                assert len(expansion_order) == 1
            else:
                assert len(expansion_order) == self.tree.depth + 1
        except:
            raise TypeError(
                "expansion_order length must be depth + 1 if specified, otherwise of length 1 if n_crit specified"
            )

        self.expansion_order_c = ffi.cast("uintptr_t *", self.expansion_order)
        self.nexpansion_order_c = ffi.cast("uintptr_t", len(self.expansion_order))

        # Build FMM runtime object
        self._construct()

    def _construct(self):
        if isinstance(self.field_translation, FftFieldTranslation):
            if isinstance(self.field_translation.kernel, LaplaceKernel):
                if self.field_translation.kernel.dtype == np.float32:
                    self.fmm = lib.laplace_fft_f32_alloc(
                        self.expansion_order_c,
                        self.nexpansion_order_c,
                        self.kernel.eval_type_c,
                        self.tree.sources_c,
                        self.tree.nsources_c,
                        self.tree.targets_c,
                        self.tree.ntargets_c,
                        self.tree.charges_c,
                        self.tree.ncharges_c,
                        self.tree.prune_empty_c,
                        self.tree.n_crit_c,
                        self.tree.depth_c,
                        self.field_translation.block_size_c,
                    )

                elif self.field_translation.kernel.dtype == np.float64:
                    self.fmm = lib.laplace_fft_f64_alloc(
                        self.expansion_order_c,
                        self.nexpansion_order_c,
                        self.kernel.eval_type_c,
                        self.tree.sources_c,
                        self.tree.nsources_c,
                        self.tree.targets_c,
                        self.tree.ntargets_c,
                        self.tree.charges_c,
                        self.tree.ncharges_c,
                        self.tree.prune_empty_c,
                        self.tree.n_crit_c,
                        self.tree.depth_c,
                        self.field_translation.block_size_c,
                    )

                else:
                    raise TypeError("Unsupported datatype")

            elif isinstance(self.field_translation.kernel, HelmholtzKernel):
                if self.field_translation.kernel.dtype == np.float32:
                    self.fmm = lib.helmholtz_fft_f32_alloc(
                        self.expansion_order_c,
                        self.nexpansion_order_c,
                        self.kernel.eval_type_c,
                        self.kernel.wavenumber_c,
                        self.tree.sources_c,
                        self.tree.nsources_c,
                        self.tree.targets_c,
                        self.tree.ntargets_c,
                        self.tree.charges_c,
                        self.tree.ncharges_c,
                        self.tree.prune_empty_c,
                        self.tree.n_crit_c,
                        self.tree.depth_c,
                        self.field_translation.block_size_c,
                    )

                elif self.field_translation.kernel.dtype == np.float64:
                    self.fmm = lib.helmholtz_fft_f64_alloc(
                        self.expansion_order_c,
                        self.nexpansion_order_c,
                        self.kernel.eval_type_c,
                        self.kernel.wavenumber_c,
                        self.tree.sources_c,
                        self.tree.nsources_c,
                        self.tree.targets_c,
                        self.tree.ntargets_c,
                        self.tree.charges_c,
                        self.tree.ncharges_c,
                        self.tree.prune_empty_c,
                        self.tree.n_crit_c,
                        self.tree.depth_c,
                        self.field_translation.block_size_c,
                    )

                else:
                    raise TypeError("Unsupported datatype")

            else:
                raise TypeError("Unsupported kernel")

        elif isinstance(self.field_translation, BlasFieldTranslation):

            if isinstance(self.field_translation.kernel, LaplaceKernel):

                if self.field_translation.random:
                    if self.field_translation.kernel.dtype == np.float32:
                        self.fmm = lib.laplace_blas_rsvd_f32_alloc(
                            self.expansion_order_c,
                            self.nexpansion_order_c,
                            self.kernel_eval_type_c,
                            self.tree.sources_c,
                            self.tree.nsources_c,
                            self.tree.targets_c,
                            self.tree.ntargets_c,
                            self.tree.charges_c,
                            self.tree.ncharges_c,
                            self.tree.prune_empty_c,
                            self.tree.n_crit_c,
                            self.tree.depth_c,
                            self.field_translation.svd_threshold_c,
                            self.field_translation.surface_diff_c,
                            self.field_translation.rsvd.n_components_c,
                            self.field_translation.rsvd.n_oversamples_c,
                        )

                    elif self.field_translation.kernel.dtype == np.float64:
                        self.fmm = lib.laplace_blas_rsvd_f64_alloc(
                            self.expansion_order_c,
                            self.nexpansion_order_c,
                            self.kernel_eval_type_c,
                            self.tree.sources_c,
                            self.tree.nsources_c,
                            self.tree.targets_c,
                            self.tree.ntargets_c,
                            self.tree.charges_c,
                            self.tree.ncharges_c,
                            self.tree.prune_empty_c,
                            self.tree.n_crit_c,
                            self.tree.depth_c,
                            self.field_translation.svd_threshold_c,
                            self.field_translation.surface_diff_c,
                            self.field_translation.rsvd.n_components_c,
                            self.field_translation.rsvd.n_oversamples_c,
                        )

                    else:
                        raise TypeError("Unsupported datatype")

                else:
                    if self.field_translation.kernel.dtype == np.float32:
                        self.fmm = lib.laplace_blas_svd_f32_alloc(
                            self.expansion_order_c,
                            self.nexpansion_order_c,
                            self.kernel_eval_type_c,
                            self.tree.sources_c,
                            self.tree.nsources_c,
                            self.tree.targets_c,
                            self.tree.ntargets_c,
                            self.tree.charges_c,
                            self.tree.ncharges_c,
                            self.tree.prune_empty_c,
                            self.tree.n_crit_c,
                            self.tree.depth_c,
                            self.field_translation.svd_threshold_c,
                            self.field_translation.surface_diff_c,
                        )

                    elif self.field_translation.kernel.dtype == np.float64:
                        self.fmm = lib.laplace_blas_svd_f64_alloc(
                            self.expansion_order_c,
                            self.nexpansion_order_c,
                            self.kernel_eval_type_c,
                            self.tree.sources_c,
                            self.tree.nsources_c,
                            self.tree.targets_c,
                            self.tree.ntargets_c,
                            self.tree.charges_c,
                            self.tree.ncharges_c,
                            self.tree.prune_empty_c,
                            self.tree.n_crit_c,
                            self.tree.depth_c,
                            self.field_translation.svd_threshold_c,
                            self.field_translation.surface_diff_c,
                        )

                    else:
                        raise TypeError("Unsupported datatype")

            elif isinstance(self.field_translation.kernel, HelmholtzKernel):

                if self.field_translation.random:
                    raise TypeError(
                        "Randomised SVD not yet supported for Helmholtz Kernel"
                    )
                else:
                    if self.field_translation.kernel.dtype == np.float32:
                        self.fmm = lib.helmholtz_blas_svd_f32_alloc(
                            self.expansion_order_c,
                            self.nexpansion_order_c,
                            self.kernel_eval_type_c,
                            self.kernel.wavenumber_c,
                            self.tree.sources_c,
                            self.tree.nsources_c,
                            self.tree.targets_c,
                            self.tree.ntargets_c,
                            self.tree.charges_c,
                            self.tree.ncharges_c,
                            self.tree.prune_empty_c,
                            self.tree.n_crit_c,
                            self.tree.depth_c,
                            self.field_translation.svd_threshold_c,
                            self.field_translation.surface_diff_c,
                        )

                    elif self.field_translation.kernel.dtype == np.float64:
                        self.fmm = lib.helmholtz_blas_svd_f64_alloc(
                            self.expansion_order_c,
                            self.nexpansion_order_c,
                            self.kernel_eval_type_c,
                            self.kernel.wavenumber_c,
                            self.tree.sources_c,
                            self.tree.nsources_c,
                            self.tree.targets_c,
                            self.tree.ntargets_c,
                            self.tree.charges_c,
                            self.tree.ncharges_c,
                            self.tree.prune_empty_c,
                            self.tree.n_crit_c,
                            self.tree.depth_c,
                            self.field_translation.svd_threshold_c,
                            self.field_translation.surface_diff_c,
                        )

                    else:
                        raise TypeError("Unsupported datatype")

            else:
                raise TypeError("Unsupported kernel")

        else:
            raise TypeError(f"Unsupported field translation {self.field_translation}")

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
