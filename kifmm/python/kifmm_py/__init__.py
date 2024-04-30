import numpy as np
from kifmm import f32_laplace_fft, f32_laplace_blas


CONSTRUCTORS_F32 = {"laplace": {"fft": f32_laplace_fft, "blas": f32_laplace_blas}}
CONSTRUCTORS_F64 = {"laplace": {"fft": f32_laplace_fft, "blas": f32_laplace_fft}}

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
    ):
        """
        Parameters
        ----------


        Returns:
        --------
        """
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

        _type = type(sources.dtype)

        constructors = CONSTRUCTORS[_type]

        # Call Rust constructor based on type
        if kernel in constructors.keys():
            constructor = constructors[kernel]
            if field_translation in constructor.keys():
                constructor = constructor[field_translation]
                if field_translation == "fft":
                    self.fmm = constructor(
                        expansion_order,
                        n_crit,
                        sparse,
                        eval_type,
                        sources,
                        targets,
                        charges,
                    )
                else:
                    if isinstance(svd_threshold, None):
                        raise TypeError(
                            "SVD threshold must be set for BLAS field translations"
                        )
                    else:
                        self.fmm = constructor(
                            expansion_order,
                            n_crit,
                            sparse,
                            eval_type,
                            sources,
                            targets,
                            charges,
                            svd_threshold,
                        )
            else:
                raise TypeError(
                    f"Unsupported field translation type '{field_translation}', must choose one of {CONSTRUCTORS_F32["laplace"].keys()}"
                )

        else:
            raise TypeError(
                f"Unsupported kernel type '{kernel}', must choose one of {CONSTRUCTORS_F32.keys()}"
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
