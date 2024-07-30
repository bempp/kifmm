"""
Python interface for KiFMM-rs
"""

import ctypes
from enum import Enum

import numpy as np
from stl import mesh

from ._kifmm_rs import lib, ffi


def read_stl_triangle_mesh_vertices(filepath, dtype=np.float32):
    """Read STL into Fortran ordered NumPy array"""
    m = mesh.Mesh.from_file(filepath).vectors

    faces = m.reshape(-1, 3)
    faces = np.arange(faces.shape[0]).reshape(-1, 3)  # Assuming each face is a triangle

    x = m[:, :, 0].flatten()
    y = m[:, :, 1].flatten()
    z = m[:, :, 2].flatten()

    # Return as a single Fortran order array
    n = len(x)
    result = np.zeros((n, 3)).astype(dtype)
    result[:, 0] = x
    result[:, 1] = y
    result[:, 2] = z
    return (result, faces)


class Times:
    """
    Helper class to destructure operator runtimes.
    """

    def __init__(self, times_p):
        self.times_p = times_p
        self._times = dict()
        if self.times_p.length > 0:
            for i in range(0, self.times_p.length):
                operator_time = self.times_p.times[i]
                tag = operator_time.operator_.tag
                time = operator_time.time

                if tag == lib.FmmOperatorType_P2M:
                    operator_name = "P2M"
                elif tag == lib.FmmOperatorType_M2M:
                    level = operator_time.operator_.m2m
                    operator_name = f"M2M({level})"
                elif tag == lib.FmmOperatorType_M2L:
                    level = operator_time.operator_.m2l
                    operator_name = f"M2L({level})"
                elif tag == lib.FmmOperatorType_L2L:
                    level = operator_time.operator_.l2l
                    operator_name = f"L2L({level})"
                elif tag == lib.FmmOperatorType_L2P:
                    operator_name = "L2P"
                elif tag == lib.FmmOperatorType_P2P:
                    operator_name = "P2P"

                self._times[operator_name] = time

    def __repr__(self):
        return str(self._times)


class EvalType(Enum):
    """
    Kernel evaluation type
    """

    # Evaluate kernel values
    Value = 1

    # Evaluate values and derivatives
    ValueDeriv = 4


class RandomSvdSettings:
    def __init__(self, n_components, n_oversamples):
        """Set parameters for randomised SVD.

        Args:
            n_components (int): Approximate rank of matrix to be compressed
            n_oversamples (int): Number of oversamples used to sample subspace
        """
        if n_components is None:
            raise TypeError("n_components must be specified")
        else:
            self.n_components = n_components

        if n_oversamples is None:
            self.n_oversamples = 10
        else:
            self.n_oversamples = n_oversamples


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
        """Set parameters for SVD compressed BLAS accelerated M2L field translation.

        Args:
            kernel (Kernel): Associated kernel
            svd_threshold (float): Singular value cutoff used in compression
            surface_diff (int, optional): Difference in expansion order used to construct
            check and equivalent surfaces, surface_diff = check_surface_order-equivalent_surface_order.
            Defaults to 0.
            random (bool, optional): Whether or not to use random SVD. Defaults to False.
            n_components (_type_, optional): Parameter in random SVD. Defaults to None.
            n_oversamples (_type_, optional): Parameter in random SVD. Defaults to None.
        """
        self.kernel = kernel
        self.svd_threshold = self.kernel.dtype(svd_threshold)
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

        try:
            assert surface_diff >= 0
        except:
            raise TypeError("surface_diff must be positive or 0")


class FftFieldTranslation:
    def __init__(self, kernel, block_size=32):
        """Set parameters for FFT accelerated M2L field translation.

        Args:
            kernel (Kernel): Associated kernel
            block_size (int, optional): Block size determines cache usage by the implementation. Defaults to 32.
        """

        try:
            assert isinstance(block_size, int)
        except:
            raise TypeError("block_size must be an integer")

        if isinstance(kernel, HelmholtzKernel) or isinstance(kernel, LaplaceKernel):
            pass
        else:
            raise TypeError("Unsupported Kernel")

        self.block_size = block_size
        self.kernel = kernel


class Kernel:
    """Marker class for Kernels"""

    pass


class LaplaceKernel(Kernel):
    def __init__(self, dtype, eval_type):
        """Laplace Kernel

        Args:
            dtype (float): Associated datatype
            eval_type (EvalType): Value or ValueDeriv
        """
        self.dtype = dtype

        if eval_type == EvalType.Value:
            self.eval_type = True
        elif eval_type == EvalType.ValueDeriv:
            self.eval_type = False
        else:
            raise TypeError("Unrecognised eval_type")


class HelmholtzKernel(Kernel):
    def __init__(self, dtype, wavenumber, eval_type):
        """Helmholtz Kernel

        Args:
            dtype (float): Associated datatype
            wavenumber (float): Associated wavenumber
            eval_type (EvalType): Value or ValueDeriv
        """

        try:
            assert type(wavenumber) == self.dtype
        except:
            raise TypeError("Invalid wavenumber type")

        if eval_type == EvalType.Value:
            self.eval_type = True
        elif eval_type == EvalType.ValueDeriv:
            self.eval_type = False
        else:
            raise TypeError("Unrecognised eval_type")

        self.dtype = dtype
        self.wavenumber = wavenumber


class SingleNodeTree:
    def __init__(
        self, sources, targets, charges, n_crit=None, depth=None, prune_empty=False
    ):
        """Constructor for SingleNodeTrees.

        Args:
            sources (np.array): 1D buffer of source points, expected in Fortran order [x_1, x_2, ..., x_N, y_1, y_2, .., y_N, z_1, z_2, z_N]
            targets (np.array): 1D buffer of target points, expected in Fortran order [x_1, x_2, ..., x_N, y_1, y_2, .., y_N, z_1, z_2, z_N]
            charges (np.array): 1D buffer of charge associated with source points.
            n_crit (int, optional): Must set n_crit or depth, n_crit is the max number of particles in a given leaf box. Defaults to None.
            depth (int, optional): Must set n_crit or depth, depth specifies the max depth of the tree. Defaults to None.
            prune_empty (bool, optional): Whether or not to discard empty leaf boxes, and their siblings and ancestors. Defaults to False.

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

        if n_crit is not None:
            self.n_crit = n_crit
        else:
            self.n_crit = 0
        if depth is not None:
            self.depth = depth
        else:
            self.depth = 0

        self.sources = sources
        self.targets = targets
        self.charges = charges
        self.prune_empty = prune_empty
        self.nsources = len(sources)
        self.ntargets = len(targets)
        self.ncharges = len(charges)
        self.sources_c = ffi.cast("void* ", self.sources.ctypes.data)
        self.targets_c = ffi.cast("void* ", self.targets.ctypes.data)
        self.charges_c = ffi.cast("void* ", self.charges.ctypes.data)

    def new_charges(self, new_charges):

        try:
            assert len(new_charges) % self.nsources == 0
        except:
            TypeError("Incompatible number of new_charges for sources")

        self.ncharges = len(new_charges)
        self.charges = new_charges
        self.charges_c = ffi.cast("void* ", new_charges.ctypes.data)


class KiFmm:
    """Runtime FMM object"""

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
        self._expansion_order = expansion_order
        self._tree = tree
        self._field_translation = field_translation
        self._timed = timed
        dim = 3

        try:
            if self._tree.ncharges // (self._tree.nsources // dim) > 1:
                assert isinstance(self._field_translation, BlasFieldTranslation)
        except:
            raise TypeError(
                "Multiple charge vectors only supported with BlasFieldTranslation"
            )

        try:
            if self._tree.depth is None:
                assert len(expansion_order) == 1
            else:
                assert len(expansion_order) == (self._tree.depth + 1)
        except:
            raise TypeError(
                "expansion_order length must be depth + 1 if specified, otherwise of length 1 if n_crit specified"
            )

        self._nexpansion_order = len(self._expansion_order)
        self._expansion_order_c = ffi.cast(
            "uintptr_t*", self._expansion_order.ctypes.data
        )
        self._evaluated = False
        self.all_potentials = []

        # Build FMM runtime object
        self._construct()
        self._leaves_target_tree()
        self._leaves_source_tree()
        self._target_global_indices()

    def _construct(self):
        """Construct runtime FMM object"""
        if isinstance(self._field_translation, FftFieldTranslation):
            if isinstance(self._field_translation.kernel, LaplaceKernel):
                if self._field_translation.kernel.dtype == np.float32:
                    self._fmm = lib.laplace_fft_f32_alloc(
                        self._expansion_order_c,
                        self._nexpansion_order,
                        self._field_translation.kernel.eval_type,
                        self._tree.sources_c,
                        self._tree.nsources,
                        self._tree.targets_c,
                        self._tree.ntargets,
                        self._tree.charges_c,
                        self._tree.ncharges,
                        self._tree.prune_empty,
                        self._tree.n_crit,
                        self._tree.depth,
                        self._field_translation.block_size,
                    )
                    self.potential_dtype = np.float32

                elif self._field_translation.kernel.dtype == np.float64:
                    self._fmm = lib.laplace_fft_f64_alloc(
                        self._expansion_order_c,
                        self._nexpansion_order,
                        self._field_translation.kernel.eval_type,
                        self._tree.sources_c,
                        self._tree.nsources,
                        self._tree.targets_c,
                        self._tree.ntargets,
                        self._tree.charges_c,
                        self._tree.ncharges,
                        self._tree.prune_empty,
                        self._tree.n_crit,
                        self._tree.depth,
                        self._field_translation.block_size,
                    )
                    self.potential_dtype = np.float64

                else:
                    raise TypeError("Unsupported datatype")

            elif isinstance(self._field_translation.kernel, HelmholtzKernel):
                if self._field_translation.kernel.dtype == np.float32:
                    self._fmm = lib.helmholtz_fft_f32_alloc(
                        self._expansion_order_c,
                        self._nexpansion_order,
                        self._field_translation.kernel.eval_type,
                        self.kernel.wavenumber,
                        self._tree.sources_c,
                        self._tree.nsources,
                        self._tree.targets_c,
                        self._tree.ntargets,
                        self._tree.charges_c,
                        self._tree.ncharges,
                        self._tree.prune_empty,
                        self._tree.n_crit,
                        self._tree.depth,
                        self._field_translation.block_size,
                    )
                    self.potential_dtype = np.complex64

                elif self._field_translation.kernel.dtype == np.float64:
                    self._fmm = lib.helmholtz_fft_f64_alloc(
                        self._expansion_order_c,
                        self._nexpansion_order,
                        self._field_translation.kernel.eval_type,
                        self.kernel.wavenumber,
                        self._tree.sources_c,
                        self._tree.nsources,
                        self._tree.targets_c,
                        self._tree.ntargets,
                        self._tree.charges_c,
                        self._tree.ncharges,
                        self._tree.prune_empty,
                        self._tree.n_crit,
                        self._tree.depth,
                        self._field_translation.block_size,
                    )
                    self.potential_dtype = np.complex128

                else:
                    raise TypeError("Unsupported datatype")

            else:
                raise TypeError("Unsupported kernel")

        elif isinstance(self._field_translation, BlasFieldTranslation):

            if isinstance(self._field_translation.kernel, LaplaceKernel):

                if self._field_translation.random:
                    if self._field_translation.kernel.dtype == np.float32:
                        self._fmm = lib.laplace_blas_rsvd_f32_alloc(
                            self._expansion_order_c,
                            self._nexpansion_order,
                            self._field_translation.kernel.eval_type,
                            self._tree.sources_c,
                            self._tree.nsources,
                            self._tree.targets_c,
                            self._tree.ntargets,
                            self._tree.charges_c,
                            self._tree.ncharges,
                            self._tree.prune_empty,
                            self._tree.n_crit,
                            self._tree.depth,
                            self._field_translation.svd_threshold,
                            self._field_translation.surface_diff,
                            self._field_translation.rsvd.n_components,
                            self._field_translation.rsvd.n_oversamples,
                        )
                        self.potential_dtype = np.float32

                    elif self._field_translation.kernel.dtype == np.float64:

                        self._fmm = lib.laplace_blas_rsvd_f64_alloc(
                            self._expansion_order_c,
                            self._nexpansion_order,
                            self._field_translation.kernel.eval_type,
                            self._tree.sources_c,
                            self._tree.nsources,
                            self._tree.targets_c,
                            self._tree.ntargets,
                            self._tree.charges_c,
                            self._tree.ncharges,
                            self._tree.prune_empty,
                            self._tree.n_crit,
                            self._tree.depth,
                            self._field_translation.svd_threshold,
                            self._field_translation.surface_diff,
                            self._field_translation.rsvd.n_components,
                            self._field_translation.rsvd.n_oversamples,
                        )

                        self.potential_dtype = np.float64

                    else:
                        raise TypeError("Unsupported datatype")

                else:
                    if self._field_translation.kernel.dtype == np.float32:
                        self._fmm = lib.laplace_blas_svd_f32_alloc(
                            self._expansion_order_c,
                            self._nexpansion_order,
                            self._field_translation.kernel.eval_type,
                            self._tree.sources_c,
                            self._tree.nsources,
                            self._tree.targets_c,
                            self._tree.ntargets,
                            self._tree.charges_c,
                            self._tree.ncharges,
                            self._tree.prune_empty,
                            self._tree.n_crit,
                            self._tree.depth,
                            self._field_translation.svd_threshold,
                            self._field_translation.surface_diff,
                        )
                        self.potential_dtype = np.float32

                    elif self._field_translation.kernel.dtype == np.float64:
                        self._fmm = lib.laplace_blas_svd_f64_alloc(
                            self._expansion_order_c,
                            self._nexpansion_order,
                            self._field_translation.kernel.eval_type,
                            self._tree.sources_c,
                            self._tree.nsources,
                            self._tree.targets_c,
                            self._tree.ntargets,
                            self._tree.charges_c,
                            self._tree.ncharges,
                            self._tree.prune_empty,
                            self._tree.n_crit,
                            self._tree.depth,
                            self._field_translation.svd_threshold,
                            self._field_translation.surface_diff,
                        )
                        self.potential_dtype = np.float64

                    else:
                        raise TypeError("Unsupported datatype")

            elif isinstance(self._field_translation.kernel, HelmholtzKernel):

                if self._field_translation.random:
                    raise TypeError(
                        "Randomised SVD not yet supported for Helmholtz Kernel"
                    )
                else:
                    if self._field_translation.kernel.dtype == np.float32:
                        self._fmm = lib.helmholtz_blas_svd_f32_alloc(
                            self._expansion_order_c,
                            self._nexpansion_order,
                            self._field_translation.kernel.eval_type,
                            self.kernel.wavenumber,
                            self._tree.sources_c,
                            self._tree.nsources,
                            self._tree.targets_c,
                            self._tree.ntargets,
                            self._tree.charges_c,
                            self._tree.ncharges,
                            self._tree.prune_empty,
                            self._tree.n_crit,
                            self._tree.depth,
                            self._field_translation.svd_threshold,
                            self._field_translation.surface_diff,
                        )
                        self.potential_dtype = np.complex64

                    elif self._field_translation.kernel.dtype == np.float64:
                        self._fmm = lib.helmholtz_blas_svd_f64_alloc(
                            self._expansion_order_c,
                            self._nexpansion_order,
                            self._field_translation.kernel.eval_type,
                            self.kernel.wavenumber,
                            self._tree.sources_c,
                            self._tree.nsources,
                            self._tree.targets_c,
                            self._tree.ntargets,
                            self._tree.charges_c,
                            self._tree.ncharges,
                            self._tree.prune_empty,
                            self._tree.n_crit,
                            self._tree.depth,
                            self._field_translation.svd_threshold,
                            self._field_translation.surface_diff,
                        )
                        self.potential_dtype = np.complex128

                    else:
                        raise TypeError("Unsupported datatype")

            else:
                raise TypeError("Unsupported kernel")

        else:
            raise TypeError(f"Unsupported field translation {self._field_translation}")

    def _target_global_indices(self):
        global_indices_p = lib.global_indices_target_tree(self._fmm)
        ptr = ffi.cast("uintptr_t*", global_indices_p.data)
        self.target_global_indices = np.frombuffer(
            ffi.buffer(ptr, global_indices_p.len * ffi.sizeof("uintptr_t")),
            dtype=np.uint64,
        )

    def _leaves_target_tree(self):
        morton_keys_p = lib.leaves_target_tree(self._fmm)
        ptr = ffi.cast("uint64_t*", morton_keys_p.data)
        self.leaves_target_tree = np.frombuffer(
            ffi.buffer(ptr, morton_keys_p.len * ffi.sizeof("uint64_t")), dtype=np.uint64
        )

    def _leaves_source_tree(self):
        morton_keys_p = lib.leaves_source_tree(self._fmm)
        ptr = ffi.cast("uint64_t*", morton_keys_p.data)
        self.leaves_source_tree = np.frombuffer(
            ffi.buffer(ptr, morton_keys_p.len * ffi.sizeof("uint64_t")), dtype=np.uint64
        )

    def _all_potentials(self):
        dim = 3
        if self._evaluated:
            self._all_potentials_p = lib.all_potentials(self._fmm)

            all_potentials = KiFmm._cast_to_numpy_array(
                self._all_potentials_p.data,
                self._all_potentials_p.len,
                self.potential_dtype,
                ffi,
            )

            n_evals = len(all_potentials) // (self._tree.ntargets // dim)
            self.all_potentials = np.reshape(
                all_potentials, (n_evals, (self._tree.ntargets // dim))
            )

    def unsort_all_potentials(self):
        """Un-permute the evaluated potentials from Morton order to the input order"""
        if self._evaluated:
            self.all_potentials_r = np.zeros_like(self.all_potentials)

            for i in range(0, self.all_potentials.shape[0]):
                for j, k in enumerate(self.target_global_indices):
                    self.all_potentials_r[i][k] = self.all_potentials[i][j]

    @staticmethod
    def _cast_to_numpy_array(ptr, length, dtype, ffi):
        if dtype == np.float32:
            ctype_buffer = ffi.cast("float*", ptr)
            return np.frombuffer(
                ffi.buffer(ctype_buffer, length * ffi.sizeof("float")), dtype=np.float32
            )
        elif dtype == np.float64:
            ctype_buffer = ffi.cast("double*", ptr)
            return np.frombuffer(
                ffi.buffer(ctype_buffer, length * ffi.sizeof("double")),
                dtype=np.float64,
            )
        elif dtype == np.complex64:
            ctype_buffer = ffi.cast("float*", ptr)
            return np.frombuffer(
                ffi.buffer(ctype_buffer, length * ffi.sizeof("float")), dtype=np.float32
            ).view(np.complex64)
        elif dtype == np.complex128:
            ctype_buffer = ffi.cast("double*", ptr)
            return np.frombuffer(
                ffi.buffer(ctype_buffer, length * ffi.sizeof("double")),
                dtype=np.float64,
            ).view(np.complex128)
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

    def leaf_potentials(self, leaf):
        """Lookup potentials associated with each leaf

        Args:
            leaf (int): Leaf key

        Returns:
            np.array(self.potential_dtype): Potential data associated with this leaf
        """
        try:
            assert isinstance(leaf, int) or (isinstance(leaf, np.uint64))
        except:
            raise TypeError("leaf must be of type int")

        if self._evaluated:
            result = []

            potentials = lib.leaf_potentials(self._fmm, leaf)
            n = potentials.n
            for i in range(0, n):
                potential_p = potentials.data[i].data
                potential_len = potentials.data[i].len
                result.append(
                    KiFmm._cast_to_numpy_array(
                        potential_p, potential_len, self.potential_dtype, ffi
                    )
                )

            return np.array(result)

    def evaluate(self):
        """Evaluate FMM"""
        self.times = Times(lib.evaluate(self._fmm, self._timed))
        self._evaluated = True
        self._all_potentials()

    def clear(self, charges):
        """Clear charge data, and add new charge data

        Args:
            charges (np.array): New charge data
        """
        self._tree.new_charges(charges)
        lib.clear(self._fmm, self._tree.charges_c, self._tree.ncharges)

    def evaluate_kernel(self, eval_type, sources, targets, charges, result):
        """Evaluate associated kernel function

        Args:
            eval_type (EvalType): Value or ValueDeriv
            sources (np.array): 1D buffer of source points, expected in Fortran order [x_1, x_2, ..., x_N, y_1, y_2, .., y_N, z_1, z_2, z_N]
            targets (np.array): 1D buffer of target points, expected in Fortran order [x_1, x_2, ..., x_N, y_1, y_2, .., y_N, z_1, z_2, z_N]
            charges (np.array): 1D buffer of charge associated with source points.
            result (np.array): 1D buffer of result data, associated with target points.
        """

        dim = 3

        if eval_type == EvalType.Value:
            eval_type = True
        elif eval_type == EvalType.ValueDeriv:
            eval_type = False
        else:
            raise TypeError("Unrecognised eval_type")

        try:
            assert len(charges) == len(sources) // dim
        except:
            raise TypeError("Number of charges must match number of sources")

        try:
            assert result.dtype == sources.dtype
            assert result.dtype == targets.dtype
            assert result.dtype == charges.dtype
        except:
            raise TypeError("dtype of result must match source/charge/target data")

        try:
            if eval_type:
                assert len(result) == len(targets) // dim
            else:
                assert len(result) // 4 == len(targets) // dim

        except:
            raise TypeError("result vector must match number of targets")

        nsources = len(sources)
        ntargets = len(targets)
        ncharges = len(charges)
        nresult = len(result)
        sources_c = ffi.cast("void* ", sources.ctypes.data)
        targets_c = ffi.cast("void* ", targets.ctypes.data)
        charges_c = ffi.cast("void* ", charges.ctypes.data)
        result_c = ffi.cast("void* ", result.ctypes.data)

        lib.evaluate_kernel_st(
            self._fmm,
            eval_type,
            sources_c,
            nsources,
            targets_c,
            ntargets,
            charges_c,
            ncharges,
            result_c,
            nresult,
        )

    def coordinates_source_tree(self, leaf):
        try:
            assert isinstance(leaf, int) or (isinstance(leaf, np.uint64))
        except:
            raise TypeError("leaf must be of type int")

        coords = None
        if self._evaluated:
            coords_p = lib.coordinates_source_tree(self._fmm, leaf)
            coords = KiFmm._cast_to_numpy_array(
                coord_p.data, coords_p.len, self._field_translation.kernel.dtype, ffi
            )

        return coords

    def coordinates_target_tree(self, leaf):
        try:
            assert isinstance(leaf, int) or (isinstance(leaf, np.uint64))
        except:
            raise TypeError("leaf must be of type int")

        coords = None
        if self._evaluated:
            coords_p = lib.coordinates_target_tree(self._fmm, leaf)
            coords = KiFmm._cast_to_numpy_array(
                coords_p.data, coords_p.len, self._field_translation.kernel.dtype, ffi
            )

        return coords
