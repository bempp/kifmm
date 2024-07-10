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

class RandomSvdSettings:
    def __init__(self, n_iter, n_components, n_oversamples, random_state):
        self.n_iter = n_iter
        self.n_components = n_components
        self.n_oversamples = n_oversamples
        self.random_state = random_state

    @property
    def settings(self):
        return (self.n_iter, self.n_components, self.n_oversamples, self.random_state)

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

        if field_translation == "blas":
            if rsvd == None:
                self.rsvd = None
            else:
                self.rsvd = rsvd.settings

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
                    self.prune_empty,
                    self.kernel_eval_type,
                    self.n_crit,
                    self.depth,
                    self.block_size,
                )

            elif self.field_translation == "blas":
                self.fmm = self.constructor(
                    self.expansion_order,
                    self.sources,
                    self.targets,
                    self.charges,
                    self.prune_empty,
                    self.kernel_eval_type,
                    self.svd_threshold,
                    self.n_crit,
                    self.depth,
                    self.surface_diff,
                    self.rsvd
                )

        elif kernel == "helmholtz":
            if self.field_translation == "fft":
                self.fmm = self.constructor(
                    self.expansion_order,
                    self.sources,
                    self.targets,
                    self.charges,
                    self.prune_empty,
                    self.kernel_eval_type,
                    self.wavenumber,
                    self.n_crit,
                    self.depth,
                    self.block_size,
                )

            elif self.field_translation == "blas":
                self.fmm = self.constructor(
                    self.expansion_order,
                    self.sources,
                    self.targets,
                    self.charges,
                    self.prune_empty,
                    self.kernel_eval_type,
                    self.wavenumber,
                    self.svd_threshold,
                    self.n_crit,
                    self.depth,
                    self.surface_diff,
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
        self.timed = timed
        self.times = None

    def evaluate(self):
        """Run the FMM."""
        self.times = self.fmm.evaluate(self.timed)

    def clear(self, charges):
        """Clear currently assigned charges, assign new charges

        Args:
            charges (np.ndarray): Charge data, real or complex (dependent on kernel) of shape dimensions '[n_charges, n_vecs]' where each of 'n_vecs' is associated with 'n_charges'. 'n_vecs' > 1 only supported with BLAS field translations.
        """
        try:
            assert isinstance(charges, np.ndarray)
        except:
            raise TypeError(f"charges of type {type(charges)}")

        expected_dtypes = KERNEL_DTYPE[self.kernel]
        try:
            assert type(charges[0].dtype) in expected_dtypes
        except:
            raise TypeError(
                f"charges of the wrong type '{type(charges[0].dtype)}' for this kernel"
            )

        self.fmm.clear(charges)

    @property
    def cutoff_ranks(self):
        try:
            assert self.field_translation == "blas"
        except:
            raise TypeError(
                "Cutoff ranks only available for FMMs with BLAS based field translations"
            )

        return self.fmm.cutoff_ranks

    @property
    def directional_cutoff_ranks(self):
        try:
            assert self.field_translation == "blas" and self.kernel == "laplace"
        except:
            raise TypeError(
                "Directional cutoff ranks only available for FMMs with BLAS based field translations and Laplace kernels"
            )

        return self.fmm.directional_cutoff_ranks

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
            and is stored in an order defined by 'global_indices'. Potential is of the shape '[kernel_eval_size, n_targets]',
            kernel_eval_size \in [1, 4] depending on whether the user is evaluating potentials or gradients.
        """
        return self.fmm.potentials(leaf)

    def all_potentials(self):
        """Lookup all potential data associated with FMM.

        Returns:
            np.ndarray: Potential data, returned as a list where the length corresponds to the number of evaluations/charge vectors,
            and is stored in an order defined by 'global_indices'. Each potential is of the shape '[kernel_eval_size, n_targets]',
            kernel_eval_size \in [1, 4] depending on whether the user is evaluating potentials or gradients.
        """
        return self.fmm.all_potentials()

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
            sources (np.ndarray): Source coordinates, real data expected in C order with shape is '[n_coords, dim]'
            targets (np.ndarray): Target coordinates, real data expected in C order with shape is '[n_coords, dim]'
            charges (np.ndarray): Charge data, real or complex (dependent on kernel) of shape dimensions '[n_charges, n_vecs]' where each of 'n_vecs' is associated with 'n_charges'. 'n_vecs' > 1 only supported with BLAS field translations.

        Returns:
            np.ndarray: Potentials/potential gradients associated with target coordinates. Potential is of the shape '[kernel_eval_size, n_targets]',
            kernel_eval_size \in [1, 4] depending on whether the user is evaluating potentials or gradients.
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
        except Exception as e:
            print(e)
            raise TypeError("sources and targets must be reals, and of shape [3, N]")

        try:
            assert type(sources[0].dtype) == type(targets[0].dtype)
        except:
            raise TypeError(
                f"sources of type {type(sources[0].dtype)}, targets of type {type(targets[0].dtype)} do not match."
            )

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
