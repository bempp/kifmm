//! Types for storing field translation metadata.

use crate::tree::types::morton::MortonKey;
use green_kernels::traits::Kernel;
use num::{Complex, Float};
use rlst::RlstScalar;
use rlst::{rlst_dynamic_array2, Array, BaseArray, VectorContainer};

/// Simple type alias for a 2D `Matrix<f64>`
pub type SvdM2lEntry<T> = Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>;

/// Simple type alias for pre-computed FFT of green's function evaluations computed for each transfer vector in a box's halo
/// Each index corresponds to a halo position, and contains 64 convolutions, one for each of a box's siblings with each child
/// of the halo box.
pub type FftKernelData<C> = Vec<Vec<C>>;

/// FFT field translation for KiFMM
#[derive(Default)]
pub struct FftFieldTranslationKiFmm<T, U>
where
    T: Default + RlstScalar<Real = T> + Float,
    U: Kernel<T = T> + Default,
{
    /// Map between indices of surface convolution grid points.
    pub surf_to_conv_map: Vec<usize>,

    /// Map between indices of convolution and surface grid points.
    pub conv_to_surf_map: Vec<usize>,

    /// Precomputed data required for FFT compressed M2L interaction.
    pub operator_data: FftM2lOperatorData<Complex<T>>,

    /// Unique transfer vectors to lookup m2l unique kernel interactions
    pub transfer_vectors: Vec<TransferVector>,

    /// The associated kernel with this translation operator.
    pub kernel: U,

    /// Expansion order
    pub expansion_order: usize,
}

/// A type to store the M2L field translation meta-data and data for an SVD based sparsification in the kernel independent FMM.
/// Here we take the a SVD over all M2L matrices, but the compressed M2L matrices are recompressed to account for the redundancy
/// in rank due to some M2L matrices being of higher rank. This recompression is controlled by the threshold parameter which is
/// which is computed as the percentage of the energy of the compressed M2L matrix, as measured by the sum of the squares of the
/// singular values.
#[derive(Default)]
pub struct BlasFieldTranslationKiFmm<T, U>
where
    T: RlstScalar<Real = T>,
    U: Kernel<T = T> + Default,
{
    /// Threshold
    pub threshold: T,

    /// Precomputed data required for SVD compressed M2L interaction.
    pub operator_data: BlasSourceToTargetOperatorData<T>,

    /// Unique transfer vectors to lookup m2l unique kernel interactions.
    pub transfer_vectors: Vec<TransferVector>,

    /// The associated kernel with this translation operator.
    pub kernel: U,

    /// Expansion order
    pub expansion_order: usize,

    /// Cutoff rank
    pub cutoff_rank: usize,
}

/// A type to store a transfer vector between a `source` and `target` Morton key.
#[derive(Debug)]
pub struct TransferVector {
    /// Three vector of components.
    pub components: [i64; 3],

    /// Unique identifier for transfer vector, for easy lookup.
    pub hash: usize,

    /// The `source` Morton key associated with this transfer vector.
    pub source: MortonKey,

    /// The `target` Morton key associated with this transfer vector.
    pub target: MortonKey,
}

/// Container for the precomputed data required for FFT field translation.
#[derive(Default)]
pub struct FftM2lOperatorData<C> {
    /// FFT of unique kernel evaluations for each transfer vector in a halo of a sibling set
    pub kernel_data: FftKernelData<C>,

    /// FFT of unique kernel evaluations for each transfer vector in a halo of a sibling set, re-arranged in frequency order
    pub kernel_data_f: FftKernelData<C>,
}

/// Container to store precomputed data required for SVD field translations.
/// See Fong & Darve (2009) for the definitions of 'fat' and 'thin' M2L matrices.
pub struct BlasSourceToTargetOperatorData<T>
where
    T: RlstScalar,
{
    /// Left singular vectors from SVD of fat M2L matrix.
    pub u: SvdM2lEntry<T>,

    /// Right singular vectors from SVD of thin M2L matrix, cutoff to a maximum rank of 'k'.
    pub st_block: SvdM2lEntry<T>,

    /// The quantity $C_{block} = \Sigma \cdot V^T_{block} S_{block} $, where $\Sigma$ is diagonal matrix of singular values
    /// from the SVD of the fat M2L matrix, $V^T_{block}$ is a is a block of the right singular vectors corresponding
    /// to each transfer vector from the same SVD, and $S_{block}$ is a block of the transposed right singular vectors
    /// from the SVD of the thin M2L matrix. $C$ is composed of $C_{block}$, with one for each unique transfer vector. As
    /// we recompress $C_{block}$, this corresponds to the left singular vectors after recompression.
    pub c_u: Vec<SvdM2lEntry<T>>,

    /// The quantity $C_{block} = \Sigma \cdot V^T_{block} S_{block} $, where $\Sigma$ is diagonal matrix of singular values
    /// from the SVD of the fat M2L matrix, $V^T_{block}$ is a is a block of the right singular vectors corresponding
    /// to each transfer vector from the same SVD, and $S_{block}$ is a block of the transposed right singular vectors
    /// from the SVD of the thin M2L matrix. $C$ is composed of $C_{block}$, with one for each unique transfer vector.
    /// Right singular vectors from SVD of thin M2L matrix, cutoff to a maximum rank of 'k'. As we recompress $C_{block}$, t
    /// his corresponds to the right singular vectors after recompression.
    pub c_vt: Vec<SvdM2lEntry<T>>,
}

impl<T> Default for BlasSourceToTargetOperatorData<T>
where
    T: RlstScalar,
{
    fn default() -> Self {
        let u = rlst_dynamic_array2!(T, [1, 1]);
        let st_block = rlst_dynamic_array2!(T, [1, 1]);

        BlasSourceToTargetOperatorData {
            u,
            st_block,
            c_u: Vec::default(),
            c_vt: Vec::default(),
        }
    }
}