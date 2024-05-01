import numpy as np
from kifmm import (
    f32_laplace_fft,
    f32_laplace_blas,
    f32_helmholtz_blas,
    f32_helmholtz_fft,
)

CONSTRUCTORS_F32 = {"laplace": {"fft": f32_laplace_fft, "blas": f32_laplace_blas}}
CONSTRUCTORS_F64 = {"laplace": {"fft": f32_helmholtz_fft, "blas": f32_helmholtz_blas}}

CONSTRUCTORS = {
    np.dtypes.Float32DType: CONSTRUCTORS_F32,
    np.dtypes.Float64DType: CONSTRUCTORS_F32,
}


class KiFmm:
    """
    Constructor for KiFmm objects on a single node
    """

    def __init__(
        self,
        expansion_order,
        n_crit,
        sparse,
        eval_type,
        sources,
        targets,
        charges,
        kernel,
        field_translation,
        svd_threshold=None,
        wavenumber=None,
    ):
        """
        Parameters
        ----------


        Returns:
        --------
        """

        # Check validity of each input type

        try:
            assert isinstance(expansion_order, int)
        except:
            raise TypeError(
                f"expansion order of type {type(expansion_order)}, expected 'int'"
            )

        try:
            assert isinstance(n_crit, int)
        except:
            raise TypeError(f"n_crit of type {type(n_crit)}, expected 'int'")

        try:
            assert isinstance(sparse, bool)
        except:
            raise TypeError(f"sparse of type {type(sparse)}, expected 'bool'")

        try:
            assert isinstance(eval_type, bool) & (eval_type == 0 | eval_type == 1)
        except:
            raise TypeError(
                f"eval_type={eval_type} of type {type(eval_type)}, expected 'int' of value either '0' or '1'"
            )

        try:

            assert (
                isinstance(sources, np.ndarray)
                & isinstance(targets, np.ndarray)
                & isinstance(charges, np.ndarray)
            )
        except:
            raise TypeError(f"Sources, targets and charges must of type numpy.ndarray")

        try:
            assert sources.dtype == targets.dtype == charges.dtype
        except:
            raise TypeError(
                f"Sources of type {sources.dtype}, targets of type {targets.dtype}, charges of type {charges.dtype}"
            )

        try:
            assert field_translation == "fft" | field_translation == "blas"
        except:
            raise TypeError(
                f"Unsupported field translation type '{field_translation}', must choose one of {CONSTRUCTORS_F32["laplace"].keys()}"
            )

        try:
            assert kernel == "fft" | kernel == "blas"
        except:
            raise TypeError(
                f"Unsupported kernel type '{kernel}', must choose one of {CONSTRUCTORS_F32.keys()}"
            )

        # Floating point type
        _type = type(sources.dtype)

        # Set constructor
        constructor = CONSTRUCTORS[_type][kernel]

        if field_translation == "fft":
            if wavenumber is None:
                self._fmm = constructor(
                    expansion_order,
                    n_crit,
                    sparse,
                    eval_type,
                    sources,
                    targets,
                    charges,
                )
            else:
                self._fmm = constructor(
                    expansion_order,
                    n_crit,
                    sparse,
                    eval_type,
                    sources,
                    targets,
                    charges,
                    wavenumber,
                )

        if field_translation == "blas":
            if svd_threshold is None:
                raise TypeError("SVD threshold must be set for BLAS field translations")
            if wavenumber is None:
                self._fmm = constructor(
                    expansion_order,
                    n_crit,
                    sparse,
                    eval_type,
                    sources,
                    targets,
                    svd_threshold,
                    charges,
                )
            else:
                self._fmm = constructor(
                    expansion_order,
                    n_crit,
                    sparse,
                    eval_type,
                    sources,
                    targets,
                    charges,
                    svd_threshold,
                    wavenumber,
                )

    def evaluate(self):
        """
        Evaluate potentials due to source charges at targets.
        """
        pass

    def data(self, key):
        """
        examine the data at a key.
        """
        pass
