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


class OperatorTimes:
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


class MetadataTimes:
    """
    Helper class to destructure metadata runtimes.
    """

    def __init__(self, times_p):
        self.times_p = times_p
        self._times = dict()
        if self.times_p.length > 0:
            for i in range(0, self.times_p.length):
                metadata_time = self.times_p.times[i]
                tag = metadata_time.operator_
                time = metadata_time.time

                if tag == 0:
                    operator_name = "source_to_target_data"
                elif tag == 1:
                    operator_name = "source_data"
                elif tag == 2:
                    operator_name = "target_data"
                elif tag == 3:
                    operator_name = "global_fmm"
                elif tag == 4:
                    operator_name = "ghost_fmm_v"
                elif tag == 5:
                    operator_name = "ghost_fmm_u"

                self._times[operator_name] = time

    def __repr__(self):
        return str(self._times)


class CommunicationTimes:
    """
    Helper class to destructure communication runtimes.
    """

    def __init__(self, times_p):
        self.times_p = times_p
        self._times = dict()
        if self.times_p.length > 0:
            for i in range(0, self.times_p.length):
                communication_time = self.times_p.times[i]
                tag = communication_time.operator_
                time = communication_time.time

                if tag == 0:
                    operator_name = "source_tree"
                elif tag == 1:
                    operator_name = "target_tree"
                elif tag == 2:
                    operator_name = "target_domain"
                elif tag == 3:
                    operator_name = "source_domain"
                elif tag == 4:
                    operator_name = "layout"
                elif tag == 5:
                    operator_name = "ghost_exchange_v"
                elif tag == 6:
                    operator_name = "ghost_exchange_v_runtime"
                elif tag == 7:
                    operator_name = "ghost_exchange_u"
                elif tag == 8:
                    operator_name = "gather_global_fmm"
                elif tag == 9:
                    operator_name = "scatter_global_fmm"

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
    def __init__(self, n_components=0, n_oversamples=10):
        """Set Randomised SVD parameters

        Args:
            n_components (int, optional): Estimated rank of
            orthogonal subspace used to approximate kernel.
            Defaults to being determined from
            associated Kernel if set to 0.
            n_oversamples (int, optional): Number of oversamples
            used to approximate orthogonal subspace used to approximate
            kernel. Defaults to 10.
        """
        try:
            assert isinstance(n_components, int)
        except:
            raise TypeError("Expected int n_components")

        try:
            assert isinstance(n_oversamples, int)
        except:
            raise TypeError("n_oversamples must be an integer")

        try:
            assert n_oversamples >= 0
        except:
            raise TypeError("n_oversamples must be >= 0")

        self.n_components = n_components
        self.n_oversamples = n_oversamples


class BlasFieldTranslation:
    def __init__(
        self,
        kernel,
        svd_threshold,
        n_components=0,
        n_oversamples=10,
        surface_diff=0,
        random=False,
    ):
        """Set parameters for SVD compressed BLAS accelerated M2L field translation.

        Args:
            kernel (Kernel): Associated kernel
            svd_threshold (float): Singular value cutoff used in compression
            n_components (int, optional): Parameter in random SVD.
            n_oversamples (int, optional): Parameter in random SVD.
            surface_diff (int, optional): Difference in expansion order used to construct
            check and equivalent surfaces, surface_diff = check_surface_order-equivalent_surface_order.
            Defaults to 0.
            random (bool, optional): Whether or not to use random SVD. Defaults to False.
        """
        self.kernel = kernel
        self.svd_threshold = self.kernel.dtype(svd_threshold)
        self.surface_diff = surface_diff
        self.random = random

        if isinstance(self.kernel, HelmholtzKernel) or isinstance(
            self.kernel, LaplaceKernel
        ):
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

        self.eval_type_r = eval_type


class HelmholtzKernel(Kernel):
    def __init__(self, dtype, wavenumber, eval_type):
        """Helmholtz Kernel

        Args:
            dtype (float): Associated datatype
            wavenumber (float): Associated wavenumber
            eval_type (EvalType): Value or ValueDeriv
        """

        self.dtype = dtype

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

        self.eval_type_r = eval_type
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
            assert len(charges) % (len(sources) // dim) == 0
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
        self.n_sources = len(sources)
        self.n_targets = len(targets)
        self.n_charges = len(charges)
        self.sources_c = ffi.cast("void* ", self.sources.ctypes.data)
        self.targets_c = ffi.cast("void* ", self.targets.ctypes.data)
        self.charges_c = ffi.cast("void* ", self.charges.ctypes.data)

    def new_charges(self, new_charges):

        try:
            assert len(new_charges) % self.n_sources == 0
        except:
            TypeError("Incompatible number of new_charges for sources")

        self.n_charges = len(new_charges)
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

        self._evaluated = False
        self._expansion_order = expansion_order
        self._n_coeffs = 6 * (expansion_order - 1) ** 2 + 2
        self._tree = tree
        self._n_evals = self._tree.n_charges // (self._tree.n_sources // 3)
        self._field_translation = field_translation
        self._timed = timed
        dim = 3

        try:
            if self._n_evals > 1:
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

        self._n_expansion_order = len(self._expansion_order)
        self._expansion_order_c = ffi.cast(
            "uintptr_t*", self._expansion_order.ctypes.data
        )
        self.all_potentials = []
        self._morton_keys_refs = set()

        # Build FMM runtime object
        self._construct()
        self._keys_target_tree()
        self._keys_source_tree()
        self._leaves_target_tree()
        self._leaves_source_tree()
        self._target_global_indices()
        self._target_tree_depth()
        self._source_tree_depth()

    def _construct(self):
        """Construct runtime FMM object"""
        if isinstance(self._field_translation, FftFieldTranslation):
            if isinstance(self._field_translation.kernel, LaplaceKernel):
                if self._field_translation.kernel.dtype == np.float32:
                    self._fmm = lib.laplace_fft_f32_alloc(
                        self._timed,
                        self._expansion_order_c,
                        self._n_expansion_order,
                        self._field_translation.kernel.eval_type,
                        self._tree.sources_c,
                        self._tree.n_sources,
                        self._tree.targets_c,
                        self._tree.n_targets,
                        self._tree.charges_c,
                        self._tree.n_charges,
                        self._tree.prune_empty,
                        self._tree.n_crit,
                        self._tree.depth,
                        self._field_translation.block_size,
                    )
                    self.potential_dtype = np.float32

                elif self._field_translation.kernel.dtype == np.float64:
                    self._fmm = lib.laplace_fft_f64_alloc(
                        self._timed,
                        self._expansion_order_c,
                        self._n_expansion_order,
                        self._field_translation.kernel.eval_type,
                        self._tree.sources_c,
                        self._tree.n_sources,
                        self._tree.targets_c,
                        self._tree.n_targets,
                        self._tree.charges_c,
                        self._tree.n_charges,
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
                        self._timed,
                        self._expansion_order_c,
                        self._n_expansion_order,
                        self._field_translation.kernel.eval_type,
                        self._field_translation.kernel.wavenumber,
                        self._tree.sources_c,
                        self._tree.n_sources,
                        self._tree.targets_c,
                        self._tree.n_targets,
                        self._tree.charges_c,
                        self._tree.n_charges,
                        self._tree.prune_empty,
                        self._tree.n_crit,
                        self._tree.depth,
                        self._field_translation.block_size,
                    )
                    self.potential_dtype = np.complex64

                elif self._field_translation.kernel.dtype == np.float64:
                    self._fmm = lib.helmholtz_fft_f64_alloc(
                        self._timed,
                        self._expansion_order_c,
                        self._n_expansion_order,
                        self._field_translation.kernel.eval_type,
                        self._field_translation.kernel.wavenumber,
                        self._tree.sources_c,
                        self._tree.n_sources,
                        self._tree.targets_c,
                        self._tree.n_targets,
                        self._tree.charges_c,
                        self._tree.n_charges,
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
                            self._timed,
                            self._expansion_order_c,
                            self._n_expansion_order,
                            self._field_translation.kernel.eval_type,
                            self._tree.sources_c,
                            self._tree.n_sources,
                            self._tree.targets_c,
                            self._tree.n_targets,
                            self._tree.charges_c,
                            self._tree.n_charges,
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
                            self._timed,
                            self._expansion_order_c,
                            self._n_expansion_order,
                            self._field_translation.kernel.eval_type,
                            self._tree.sources_c,
                            self._tree.n_sources,
                            self._tree.targets_c,
                            self._tree.n_targets,
                            self._tree.charges_c,
                            self._tree.n_charges,
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
                            self._timed,
                            self._expansion_order_c,
                            self._n_expansion_order,
                            self._field_translation.kernel.eval_type,
                            self._tree.sources_c,
                            self._tree.n_sources,
                            self._tree.targets_c,
                            self._tree.n_targets,
                            self._tree.charges_c,
                            self._tree.n_charges,
                            self._tree.prune_empty,
                            self._tree.n_crit,
                            self._tree.depth,
                            self._field_translation.svd_threshold,
                            self._field_translation.surface_diff,
                        )
                        self.potential_dtype = np.float32

                    elif self._field_translation.kernel.dtype == np.float64:
                        self._fmm = lib.laplace_blas_svd_f64_alloc(
                            self._timed,
                            self._expansion_order_c,
                            self._n_expansion_order,
                            self._field_translation.kernel.eval_type,
                            self._tree.sources_c,
                            self._tree.n_sources,
                            self._tree.targets_c,
                            self._tree.n_targets,
                            self._tree.charges_c,
                            self._tree.n_charges,
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
                    if self._field_translation.kernel.dtype == np.float32:
                        self._fmm = lib.helmholtz_blas_rsvd_f32_alloc(
                            self._timed,
                            self._expansion_order_c,
                            self._n_expansion_order,
                            self._field_translation.kernel.eval_type,
                            self._field_translation.kernel.wavenumber,
                            self._tree.sources_c,
                            self._tree.n_sources,
                            self._tree.targets_c,
                            self._tree.n_targets,
                            self._tree.charges_c,
                            self._tree.n_charges,
                            self._tree.prune_empty,
                            self._tree.n_crit,
                            self._tree.depth,
                            self._field_translation.svd_threshold,
                            self._field_translation.surface_diff,
                            self._field_translation.rsvd.n_components,
                            self._field_translation.rsvd.n_oversamples,
                        )
                        self.potential_dtype = np.complex64

                    elif self._field_translation.kernel.dtype == np.float64:
                        self._fmm = lib.helmholtz_blas_rsvd_f64_alloc(
                            self._timed,
                            self._expansion_order_c,
                            self._n_expansion_order,
                            self._field_translation.kernel.eval_type,
                            self._field_translation.kernel.wavenumber,
                            self._tree.sources_c,
                            self._tree.n_sources,
                            self._tree.targets_c,
                            self._tree.n_targets,
                            self._tree.charges_c,
                            self._tree.n_charges,
                            self._tree.prune_empty,
                            self._tree.n_crit,
                            self._tree.depth,
                            self._field_translation.svd_threshold,
                            self._field_translation.surface_diff,
                            self._field_translation.rsvd.n_components,
                            self._field_translation.rsvd.n_oversamples,
                        )
                        self.potential_dtype = np.complex128

                else:
                    if self._field_translation.kernel.dtype == np.float32:
                        self._fmm = lib.helmholtz_blas_svd_f32_alloc(
                            self._timed,
                            self._expansion_order_c,
                            self._n_expansion_order,
                            self._field_translation.kernel.eval_type,
                            self._field_translation.kernel.wavenumber,
                            self._tree.sources_c,
                            self._tree.n_sources,
                            self._tree.targets_c,
                            self._tree.n_targets,
                            self._tree.charges_c,
                            self._tree.n_charges,
                            self._tree.prune_empty,
                            self._tree.n_crit,
                            self._tree.depth,
                            self._field_translation.svd_threshold,
                            self._field_translation.surface_diff,
                        )
                        self.potential_dtype = np.complex64

                    elif self._field_translation.kernel.dtype == np.float64:
                        self._fmm = lib.helmholtz_blas_svd_f64_alloc(
                            self._timed,
                            self._expansion_order_c,
                            self._n_expansion_order,
                            self._field_translation.kernel.eval_type,
                            self._field_translation.kernel.wavenumber,
                            self._tree.sources_c,
                            self._tree.n_sources,
                            self._tree.targets_c,
                            self._tree.n_targets,
                            self._tree.charges_c,
                            self._tree.n_charges,
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
        tmp = np.frombuffer(
            ffi.buffer(ptr, global_indices_p.len * ffi.sizeof("uintptr_t")),
            dtype=np.uint64,
        )

        self.target_global_indices = np.zeros_like(tmp)
        for i, j in enumerate(tmp):
            self.target_global_indices[j] = i

    def _source_tree_depth(self):
        self.source_tree_depth = lib.source_tree_depth(self._fmm)

    def _target_tree_depth(self):
        self.target_tree_depth = lib.target_tree_depth(self._fmm)

    def _leaves_target_tree(self):
        morton_keys_p = lib.leaves_target_tree(self._fmm)
        ptr = ffi.cast("uint64_t*", morton_keys_p.data)
        self.leaves_target_tree = np.frombuffer(
            ffi.buffer(ptr, morton_keys_p.len * ffi.sizeof("uint64_t")), dtype=np.uint64
        )

        self._morton_keys_refs.add(morton_keys_p)

    def _leaves_source_tree(self):
        morton_keys_p = lib.leaves_source_tree(self._fmm)
        ptr = ffi.cast("uint64_t*", morton_keys_p.data)
        self.leaves_source_tree = np.frombuffer(
            ffi.buffer(ptr, morton_keys_p.len * ffi.sizeof("uint64_t")), dtype=np.uint64
        )
        self._morton_keys_refs.add(morton_keys_p)

    def _keys_target_tree(self):
        morton_keys_p = lib.keys_target_tree(self._fmm)
        ptr = ffi.cast("uint64_t*", morton_keys_p.data)
        self.keys_target_tree = np.frombuffer(
            ffi.buffer(ptr, morton_keys_p.len * ffi.sizeof("uint64_t")), dtype=np.uint64
        )

        self._morton_keys_refs.add(morton_keys_p)

    def _keys_source_tree(self):
        morton_keys_p = lib.keys_source_tree(self._fmm)
        ptr = ffi.cast("uint64_t*", morton_keys_p.data)
        self.keys_source_tree = np.frombuffer(
            ffi.buffer(ptr, morton_keys_p.len * ffi.sizeof("uint64_t")), dtype=np.uint64
        )
        self._morton_keys_refs.add(morton_keys_p)

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

            # if self.
            self.all_potentials = np.reshape(
                all_potentials,
                (
                    self._n_evals,
                    self._tree.n_targets // dim,
                    self._field_translation.kernel.eval_type_r.value,
                ),
            )

            self.all_potentials_u = np.zeros_like(self.all_potentials)

            for i in range(0, self.all_potentials.shape[0]):
                self.all_potentials_u[i] = self.all_potentials[i][
                    self.target_global_indices
                ]

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

    def level(self, level, key):
        """Lookup key level"""
        try:
            assert isinstance(key, int) or (isinstance(key, np.uint64))
        except:
            raise TypeError("key must be of type int")

        return lib.level(key)

    def surface(self, alpha, expansion_order, key):
        """Calculate surface

        Args:
            alpha (float): scaling parameter
            expansion_order (int): expansion order
            key (int): key
        """
        try:
            assert isinstance(key, int) or (isinstance(key, np.uint64))
        except:
            raise TypeError("key must be of type int")

        coords_p = lib.surface(alpha, expansion_order, key)

        coords = KiFmm._cast_to_numpy_array(
            coord_p.data, coords_p.len, self._field_translation.kernel.dtype, ffi
        )

        return coords

    def local(self, key):
        """Lookup local associated with each key

        Args:
            key (int): key

        Returns:
            np.array(self.potential_dtype): Local data associated with this key
        """
        try:
            assert isinstance(key, int) or (isinstance(key, np.uint64))
        except:
            raise TypeError("key must be of type int")

        if len(self._expansion_order) > 1:
            idx = lib.level(key)
        else:
            idx = 0

        if self._evaluated:
            result = []

            local = lib.multipole(self._fmm, key)
            local_p = local.data
            local_len = local.len
            tmp = KiFmm._cast_to_numpy_array(
                local_p, local_len, self.potential_dtype, ffi
            )

            tmp = tmp.reshape(self._n_evals, self._n_coeffs[idx])
            return tmp

    def multipole(self, key):
        """Lookup multipole associated with each key

        Args:
            key (int): key

        Returns:
            np.array(self.potential_dtype): Multipole data associated with this key
        """
        try:
            assert isinstance(key, int) or (isinstance(key, np.uint64))
        except:
            raise TypeError("key must be of type int")

        if len(self._expansion_order) > 1:
            idx = lib.level(key)
        else:
            idx = 0

        if self._evaluated:
            result = []

            multipole = lib.multipole(self._fmm, key)
            multipole_p = multipole.data
            multipole_len = multipole.len
            tmp = KiFmm._cast_to_numpy_array(
                multipole_p, multipole_len, self.potential_dtype, ffi
            )

            tmp = tmp.reshape(self._n_evals, self._n_coeffs[idx])
            return tmp

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
        lib.evaluate(self._fmm)
        self._evaluated = True
        self._all_potentials()

    def operator_times(self):
        """Get operator runtimes"""
        if self._timed:
            return OperatorTimes(lib.operator_times(self._fmm))

    def metadata_times(self):
        """Get metadata runtimes"""
        if self._timed:
            return MetadataTimes(lib.metadata_times(self._fmm))

    def communication_times(self):
        """Get communication runtimes"""
        if self._timed:
            return CommunicationTimes(lib.communication_times(self._fmm))

    def clear(self):
        """Clear buffers to re-initialise FMM"""
        lib.clear(self._fmm)

    def attach_charges_unordered(self, charges):
        """Attach charges in initial point ordering, before Morton sort.

        Args:
            charges (np.array): New charge data
        """
        self._tree.new_charges(charges)
        lib.attach_charges_unordered(
            self._fmm, self._tree.charges_c, self._tree.n_charges
        )

    def attach_charges_ordered(self, charges):
        """Attach charges in final point ordering, after Morton sort.

        Args:
            charges (np.array): New charge data
        """
        self._tree.new_charges(charges)
        lib.attach_charges_ordered(
            self._fmm, self._tree.charges_c, self._tree.n_charges
        )

    def evaluate_kernel_st(self, eval_type, sources, targets, charges, result):
        """Evaluate associated kernel function in single threaded mode

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

        n_sources = len(sources)
        n_targets = len(targets)
        n_charges = len(charges)
        n_result = len(result)
        sources_c = ffi.cast("void* ", sources.ctypes.data)
        targets_c = ffi.cast("void* ", targets.ctypes.data)
        charges_c = ffi.cast("void* ", charges.ctypes.data)
        result_c = ffi.cast("void* ", result.ctypes.data)

        lib.evaluate_kernel_st(
            self._fmm,
            eval_type,
            sources_c,
            n_sources,
            targets_c,
            n_targets,
            charges_c,
            n_charges,
            result_c,
            n_result,
        )

    def evaluate_kernel_mt(self, eval_type, sources, targets, charges, result):
        """Evaluate associated kernel function in multi threaded mode

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

        n_sources = len(sources)
        n_targets = len(targets)
        n_charges = len(charges)
        n_result = len(result)
        sources_c = ffi.cast("void* ", sources.ctypes.data)
        targets_c = ffi.cast("void* ", targets.ctypes.data)
        charges_c = ffi.cast("void* ", charges.ctypes.data)
        result_c = ffi.cast("void* ", result.ctypes.data)

        lib.evaluate_kernel_mt(
            self._fmm,
            eval_type,
            sources_c,
            n_sources,
            targets_c,
            n_targets,
            charges_c,
            n_charges,
            result_c,
            n_result,
        )

    def coordinates_source_tree(self, leaf):
        """Lookup coordinates associated with a leaf in the source tree

        Args:
            leaf (int): Leaf Morton key

        Returns:
            np.array: Coordinates in Fortran order [x_1, x_2, ..., x_N, y_1, y_2, .., y_N, z_1, z_2, z_N]
        """
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
        """Lookup coordinates associated with a leaf in the target tree

        Args:
            leaf (int): Leaf Morton key

        Returns:
            np.array: Coordinates in Fortran order [x_1, x_2, ..., x_N, y_1, y_2, .., y_N, z_1, z_2, z_N]
        """
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

    def __del__(self):
        if self._evaluated:
            lib.free_fmm_evaluator(self._fmm)
            for ref in self._morton_keys_refs:
                lib.free_morton_keys(ref)
