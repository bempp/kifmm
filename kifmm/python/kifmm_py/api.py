"""
Python API
"""

import re

from kifmm_py import kifmm_rust
import numpy as np


KERNELS = ["laplace", "helmholtz"]
KERNEL_EVAL_TYPES = {"eval": 0, "eval_deriv": 1}
KERNEL_EVAL_SIZE = {"eval": 1, "eval_deriv": 4}
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
    Wraps around the low level Rust interface.
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
        """Constructor for Single Node FMMss.

        Args:
            expansion_order (int): The expansion order of the FMM
            sources (np.ndarray): Source coordinates, real data expected in column major order such that the shape is `[n_coords, dim]`
            targets (np.ndarray): Target coordinates, real data expected in column major order such that the shape is `[n_coords, dim]`
            charges (np.ndarray): Charge data, real or complex (dependent on kernel) of shape dimensions `[n_charges, n_vecs]` where each of `n_vecs` is associated with `n_charges`. `n_vecs` > 1 only supported with BLAS field translations.
            n_crit (int):  Maximum number of particles per leaf box, must be less than number of particles in domain.
            sparse (bool): Optionally drop empty leaf boxes for performance in FMM.
            kernel_eval_type (str):  Either `eval_deriv` - to evaluate potentials and gradients, or `eval` to evaluate potentials alone
            kernel (str): Either 'laplace' or 'helmholtz' supported.
            field_translation (str): Either 'fft' or 'blas'
            svd_threshold (float, optional): Must specify a threshold defining the SVD compression of M2L operators when using BLAS field translations. Defaults to None for FFT field translations.
            wavenumber (float, optional): Must specify a wavenumber for Helmholtz kernels. Defaults to None for Laplace kernels.
        """
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
        self.kernel_eval_size = KERNEL_EVAL_SIZE[kernel_eval_type]
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

        self.source_keys = self.fmm.source_keys
        self.source_keys_set = set(self.source_keys)
        self.target_keys = self.fmm.target_keys
        self.target_keys_set = set(self.target_keys)
        self.source_leaves = self.fmm.source_leaves
        self.source_leaves_set = set(self.source_leaves)
        self.target_leaves = self.fmm.target_leaves
        self.target_leaves_set = set(self.target_leaves)
        self.source_tree_depth = self.fmm.source_tree_depth
        self.target_tree_depth = self.fmm.target_tree_depth
        self.source_global_indices = self.fmm.source_global_indices
        self.target_global_indices = self.fmm.target_global_indices

    def evaluate(self):
        """Run the FMM."""
        self.fmm.evaluate()

    def clear(self, charges):
        """Clear currently assigned charges, assign new charges

        Args:
            charges (np.ndarray): Charge data, real or complex (dependent on kernel) of shape dimensions `[n_charges, n_vecs]` where each of `n_vecs` is associated with `n_charges`. `n_vecs` > 1 only supported with BLAS field translations.
        """
        try:
            assert isinstance(charges, np.ndarray)
        except:
            raise TypeError(f"charges of type {type(charges)}")

        expected_dtypes = KERNEL_DTYPE[kernel]
        try:
            assert type(charges[0].dtype) in expected_dtypes
        except:
            raise TypeError(
                f"charges of the wrong type '{type(charges[0].dtype)}' for this kernel"
            )

        self.fmm.clear(charges)

    def source_key_to_anchor(self, key):
        """Convert a Morton key to its respective anchor representation for source tree keys.

        Args:
            key (int): Morton Key

        Returns:
            np.array([int; 4]): Anchor of origin of each box, as well as octree level in the form [anchor[0], anchor[1], anchor[1], level]
        """
        try:
            assert key in self.source_keys_set
        except:
            raise ValueError(f"key {key} isn't in source keys")

        return self.fmm.source_key_to_anchor(key)

    def target_key_to_anchor(self, key):
        """Convert a Morton key to its respective anchor representation for target tree keys.

        Args:
            key (int): Morton Key

        Returns:
            np.array([int; 4]): Anchor of origin of each box, as well as octree level in the form [anchor[0], anchor[1], anchor[1], level]
        """
        try:
            assert key in self.target_keys_set
        except:
            raise ValueError(f"key {key} isn't in target keys")
        return self.fmm.target_key_to_anchor(key)

    def potentials(self, leaf):
        """Lookup potential data associated with a leaf, specified by its Morton key.

        Args:
            leaf (int): Morton key

        Returns:
            np.ndarray: Potential data, returned as a list where the length corresponds to the number of evaluations/charge vectors,
            and is stored in an order defined by 'global_indices'
        """
        return self.fmm.potentials(leaf)

    def source_coordinates(self, leaf):
        """Lookup coordinate data associated with a leaf in the source tree

        Args:
            leaf (int): Morton key.

        Returns:
            np.ndarray: Coordinates in Fortran order associated with this leaf.
        """
        try:
            assert leaf in self.source_leaves_set
        except:
            raise ValueError(f"key {key} isn't in source leaves")
        return self.fmm.source_coordinates(leaf)

    def target_coordinates(self, leaf):
        """Lookup coordinate data associated with a leaf in the target tree

        Args:
            leaf (int): Morton key.

        Returns:
            np.ndarray: Coordinates in Fortran order associated with this leaf.
        """
        try:
            assert leaf in self.target_leaves_set
        except:
            raise ValueError(f"key {key} isn't in target leaves")
        return self.fmm.target_coordinates(leaf)

    def evaluate_kernel(self, sources, targets, charges):
        """Evaluate the kernel function associated with this FMM, evaluation mode set by FMM.

        Args:
            sources (np.ndarray): Source coordinates, real data expected in column major order such that the shape is `[n_coords, dim]`
            targets (np.ndarray): Target coordinates, real data expected in column major order such that the shape is `[n_coords, dim]`
            charges (np.ndarray): Charge data, real or complex (dependent on kernel) of shape dimensions `[n_charges, n_vecs]` where each of `n_vecs` is associated with `n_charges`. `n_vecs` > 1 only supported with BLAS field translations.

        Returns:
            np.ndarray: Potentials/potential gradients associated with target coordinates.
        """
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

        expected_dtypes = KERNEL_DTYPE[self.kernel]
        try:
            assert type(charges[0].dtype) in expected_dtypes
        except:
            raise TypeError(
                f"charges of the wrong type '{type(charges[0].dtype)}' for this kernel"
            )

        return self.fmm.evaluate_kernel_st(sources, targets, charges)

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