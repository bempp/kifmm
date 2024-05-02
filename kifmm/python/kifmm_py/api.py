"""
Shallow Python API for KiFMM
"""

import re

from kifmm_py import kifmm_rust
import numpy as np


KERNELS = ["laplace", "helmholtz"]
KERNEL_EVAL_TYPES = {"eval": 0, "eval_deriv": 1}
FIELD_TRANSLATION_TYPES = ["fft", "blas"]
KERNEL_DTYPE = {
    "laplace": [np.dtypes.Float32DType, np.dtypes.Float64DType],
    "helmholtz": [np.dtypes.Complex64DType, np.dtypes.Complex128DType],
}

CONSTRUCTORS = {
    np.dtypes.Float32DType: {
        "laplace": {"blas": kifmm_rust.LaplaceBlas32, "fft": kifmm_rust.LaplaceFft32},
        "helmholtz": {
            "blas": kifmm_rust.HelmholtzBlas32,
            "fft": kifmm_rust.HelmholtzFft32,
        },
    },
    np.dtypes.Float64DType: {
        "laplace": {"blas": kifmm_rust.LaplaceBlas64, "fft": kifmm_rust.LaplaceFft64},
        "helmholtz": {
            "blas": kifmm_rust.HelmholtzBlas64,
            "fft": kifmm_rust.HelmholtzFft64,
        },
    },
}


class KiFmm:
    """
    Build and interact with Single Node Kernel Independent FMMs
    """

    def __init__(
        self,
        expansion_order,
        sources,
        targets,
        charges,
        n_crit,
        sparse,
        kernel_eval_type,
        kernel,
        field_translation,
        svd_threshold=None,
        wavenumber=None,
    ):
        # Check valid expansion order
        try:
            assert expansion_order >= 2 and isinstance(expansion_order, int)
        except:
            raise TypeError(f"Invalid expansion order {expansion_order}")

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
            raise TypeError("sources and targets must be reals, and of shape [N, 3]")

        try:
            assert type(sources[0].dtype) == type(targets[0].dtype)
        except:
            raise TypeError(
                f"sources of type {type(sources[0].dtype)}, targets of type {type(targets[0].dtype)} do not match."
            )

        # Check that arrays are in Fortran order
        try:
            assert np.isfortran(sources) and np.isfortran(targets)
        except:
            raise TypeError(f"sources, targets expected in Fortran order")

        # Check for valid n_crit
        try:
            assert n_crit < sources.shape[0] and n_crit < targets.shape[0]
        except:
            raise TypeError(f"ncrit={ncrit} is too large for these sources/targets")

        # Check for valid tree
        try:
            assert isinstance(sparse, bool)

        except:
            raise TypeError(f"'sparse' must be a boolean")

        # Check for valid kernel
        try:
            assert kernel in KERNELS
        except:
            raise TypeError(
                f"kernel '{kernel}' invalid choice, must be one of {KERNELS}"
            )

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

        self.dtype = type(sources[0].dtype)
        self.expansion_order = expansion_order
        self.sources = sources
        self.targets = targets
        self.charges = charges
        self.n_crit = n_crit
        self.sparse = sparse
        self.kernel_eval_type = KERNEL_EVAL_TYPES[kernel_eval_type]
        self.kernel = kernel
        self.field_translation = field_translation
        self.constructor = CONSTRUCTORS[self.dtype][self.kernel][self.field_translation]

        if self.field_translation == "blas":
            try:
                assert svd_threshold is not None
            except:
                raise TypeError(
                    "svd threshold must be set for BLAS based field translations"
                )

        self.svd_threshold = svd_threshold

        if self.kernel == "helmholtz":
            try:
                assert wavenumber is not None
            except:
                raise TypeError("wavenumber must be set for Helmholtz kernels")

        self.wavenumber = wavenumber

        if kernel == "laplace":
            if self.field_translation == "fft":
                self.fmm = self.constructor(
                    self.expansion_order,
                    self.sources,
                    self.targets,
                    self.charges,
                    self.n_crit,
                    self.sparse,
                    self.kernel_eval_type,
                )

            elif self.field_translation == "blas":
                self.fmm = self.constructor(
                    self.expansion_order,
                    self.sources,
                    self.targets,
                    self.charges,
                    self.n_crit,
                    self.sparse,
                    self.kernel_eval_type,
                    self.svd_threshold,
                )

        elif kernel == "helmholtz":
            if self.field_translation == "fft":
                self.fmm = self.constructor(
                    self.expansion_order,
                    self.sources,
                    self.targets,
                    self.charges,
                    self.n_crit,
                    self.sparse,
                    self.kernel_eval_type,
                    self.wavenumber,
                )

            elif self.field_translation == "blas":
                self.fmm = self.constructor(
                    self.expansion_order,
                    self.sources,
                    self.targets,
                    self.charges,
                    self.n_crit,
                    self.sparse,
                    self.kernel_eval_type,
                    self.wavenumber,
                    self.svd_threshold,
                )

    def evaluate(self):
        self.fmm.evaluate()

    def clear(self):
        pass

    def evaluate_kernel_st(self):
        pass

    def evaluate_kernel_mt(self):
        pass

    def __repr__(self):
        _type = match = re.search(r"'builtins\.([^']+)'", str(self.constructor)).group(
            1
        )
        res = f"type={_type}, expansion_order={self.expansion_order}, eval_type={self.kernel_eval_type}"

        if self.field_translation == "blas":
            res += f" svd_threshold={self.svd_threshold}"

        if self.kernel == "helmholtz":
            res += f" wavenumber={self.wavenumber}"

        return res
