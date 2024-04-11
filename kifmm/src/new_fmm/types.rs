//! Data structures for kernel independent FMM
use crate::traits::fftw::RealToComplexFft3D;
use crate::traits::field::ConfigureSourceToTargetData;
use crate::traits::{field::SourceToTargetData, tree::FmmTree};
use crate::tree::types::{Domain, MortonKey, SingleNodeTree};
use crate::{Float, RlstScalarComplexFloat, RlstScalarFloat};
use green_kernels::{traits::Kernel, types::EvalType};
use num::Complex;
use rlst::{rlst_dynamic_array2, Array, BaseArray, RlstScalar, VectorContainer};
use std::collections::HashMap;

#[cfg(feature = "mpi")]
use crate::tree::types::MultiNodeTree;
#[cfg(feature = "mpi")]
use crate::RlstScalarFloatMpi;
#[cfg(feature = "mpi")]
use mpi::traits::Equivalence;

/// Represents charge data in a two-dimensional array with shape `[ncharges, nvecs]`,
/// organized in column-major order.
pub type Charges<T> = Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>;

/// Represents coordinate data in a two-dimensional array with shape `[n_coords, dim]`,
/// stored in column-major order.
pub type Coordinates<T> = Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>;

/// Represents a threadsafe mutable raw pointer to`T`.
///
/// This struct encapsulates a raw mutable pointer (`*mut T`),
/// making it safe to send across threads. It is primarily used
/// in scenarios where direct manipulation of memory across threads
/// is necessary and where Rust's ownership rules for data safety are
/// manually upheld by the programmer.
///
/// # Safety
///
/// The user must ensure that the pointed-to data adheres to Rust's
/// safety rules regarding mutability, lifetimes, and thread safety.
#[derive(Clone, Debug, Copy)]
pub struct SendPtrMut<T> {
    /// Holds the raw mutable pointer to an instance of `T`.
    pub raw: *mut T,
}

/// Represents a threadsafe immutable raw pointer to `T`.
///
/// This struct wraps a raw immutable pointer (`*const T`),
/// allowing it to be safely shared across threads. It is useful
/// for cases where read-only access to underlying data across
/// different threads is required, with the programmer manually
/// managing data safety and synchronization.
///
/// # Safety
///
/// The programmer is responsible for ensuring that the data pointed to
/// by `raw` remains valid and that any concurrency concerns are properly
/// addressed.
#[derive(Clone, Debug, Copy)]
pub struct SendPtr<T> {
    /// Holds the raw immutable pointer to an instance of `T`.
    pub raw: *const T,
}

pub struct KiFmm<Scalar, Kern, SourceToTarget>
where
    Scalar: RlstScalar,
    Kern: Kernel<T = Scalar>,
    SourceToTarget: SourceToTargetData,
    <Scalar as RlstScalar>::Real: Default,
{
    /// Dimension of the FMM
    pub dim: usize,

    /// A single node tree
    pub tree: SingleNodeFmmTree<Scalar::Real>,

    /// The associated kernel function
    pub kernel: Kern,

    /// The charge data at each target leaf box.
    pub charges: Vec<Scalar::Real>,

    /// The expansion order of the FMM
    pub expansion_order: usize,

    /// The number of coefficients, corresponding to points discretising the equivalent surface
    pub ncoeffs: usize,

    /// The kernel evaluation type, either for potentials or potentials and gradients
    pub kernel_eval_type: EvalType,

    /// The FMM evaluation type, either for a vector or matrix of input charges.
    pub fmm_eval_type: FmmEvalType,

    /// Set by the kernel evaluation type, either 1 or 4 corresponding to evaluating potentials or potentials and derivatives
    pub kernel_eval_size: usize,

    /// Index pointer for source coordinates
    pub charge_index_pointer_sources: Vec<(usize, usize)>,

    /// Index pointer for target coordinates
    pub charge_index_pointer_targets: Vec<(usize, usize)>,

    /// Upward surfaces associated with source leaves
    pub leaf_upward_surfaces_sources: Vec<Scalar::Real>,

    /// Upward surfaces associated with target leaves
    pub leaf_upward_surfaces_targets: Vec<Scalar::Real>,

    /// Scales of each source leaf box
    pub leaf_scales_sources: Vec<Scalar::Real>,

    /// The pseudo-inverse of the dense interaction matrix between the upward check and upward equivalent surfaces.
    /// Store in two parts to avoid propagating error from computing pseudo-inverse
    pub uc2e_inv_1: Array<Scalar, BaseArray<Scalar, VectorContainer<Scalar>, 2>, 2>,

    /// The pseudo-inverse of the dense interaction matrix between the upward check and upward equivalent surfaces.
    /// Store in two parts to avoid propagating error from computing pseudo-inverse
    pub uc2e_inv_2: Array<Scalar, BaseArray<Scalar, VectorContainer<Scalar>, 2>, 2>,

    /// The pseudo-inverse of the dense interaction matrix between the downward check and downward equivalent surfaces.
    /// Store in two parts to avoid propagating error from computing pseudo-inverse
    pub dc2e_inv_1: Array<Scalar, BaseArray<Scalar, VectorContainer<Scalar>, 2>, 2>,

    /// The pseudo-inverse of the dense interaction matrix between the downward check and downward equivalent surfaces.
    /// Store in two parts to avoid propagating error from computing pseudo-inverse
    pub dc2e_inv_2: Array<Scalar, BaseArray<Scalar, VectorContainer<Scalar>, 2>, 2>,

    /// Data and metadata for field translations
    pub source_to_target: SourceToTarget,

    /// The multipole translation matrices, for a cluster of eight children and their parent. Stored in Morton order.
    pub source: Array<Scalar, BaseArray<Scalar, VectorContainer<Scalar>, 2>, 2>,

    /// The metadata required for source to source translation
    pub source_vec: Vec<Array<Scalar, BaseArray<Scalar, VectorContainer<Scalar>, 2>, 2>>,

    /// The local to local operator matrices, each index is associated with a child box (in sequential Morton order).
    pub target_vec: Vec<Array<Scalar, BaseArray<Scalar, VectorContainer<Scalar>, 2>, 2>>,

    /// The multipole expansion data at each box.
    pub multipoles: Vec<Scalar>,

    /// The local expansion at each box
    pub locals: Vec<Scalar>,

    /// The evaluated potentials at each target leaf box.
    pub potentials: Vec<Scalar>,

    /// Multipole expansions at leaf level
    pub leaf_multipoles: Vec<Vec<SendPtrMut<Scalar>>>,

    /// Multipole expansions at each level
    pub level_multipoles: Vec<Vec<Vec<SendPtrMut<Scalar>>>>,

    /// Local expansions at the leaf level
    pub leaf_locals: Vec<Vec<SendPtrMut<Scalar>>>,

    /// The local expansion data at each level.
    pub level_locals: Vec<Vec<Vec<SendPtrMut<Scalar>>>>,

    /// Index pointers to each key at a given level, indexed by level.
    pub level_index_pointer_locals: Vec<HashMap<MortonKey<Scalar::Real>, usize>>,

    /// Index pointers to each key at a given level, indexed by level.
    pub level_index_pointer_multipoles: Vec<HashMap<MortonKey<Scalar::Real>, usize>>,

    /// The evaluated potentials at each target leaf box.
    pub potentials_send_pointers: Vec<SendPtrMut<Scalar>>,
}
impl<Scalar, Kern, SourceToTarget> Default for KiFmm<Scalar, Kern, SourceToTarget>
where
    Scalar: RlstScalar,
    Kern: Kernel<T = Scalar> + Default,
    SourceToTarget: SourceToTargetData + Default,
    <Scalar as RlstScalar>::Real: Default,
{
    fn default() -> Self {
        let uc2e_inv_1 = rlst_dynamic_array2!(Scalar, [1, 1]);
        let uc2e_inv_2 = rlst_dynamic_array2!(Scalar, [1, 1]);
        let dc2e_inv_1 = rlst_dynamic_array2!(Scalar, [1, 1]);
        let dc2e_inv_2 = rlst_dynamic_array2!(Scalar, [1, 1]);
        let source = rlst_dynamic_array2!(Scalar, [1, 1]);

        KiFmm {
            tree: SingleNodeFmmTree::default(),
            source_to_target: SourceToTarget::default(),
            kernel: Kern::default(),
            expansion_order: 0,
            fmm_eval_type: FmmEvalType::Vector,
            kernel_eval_type: EvalType::Value,
            kernel_eval_size: 0,
            dim: 0,
            ncoeffs: 0,
            uc2e_inv_1,
            uc2e_inv_2,
            dc2e_inv_1,
            dc2e_inv_2,
            source,
            source_vec: Vec::default(),
            target_vec: Vec::default(),
            multipoles: Vec::default(),
            locals: Vec::default(),
            leaf_multipoles: Vec::default(),
            level_multipoles: Vec::default(),
            leaf_locals: Vec::default(),
            level_locals: Vec::default(),
            level_index_pointer_locals: Vec::default(),
            level_index_pointer_multipoles: Vec::default(),
            potentials: Vec::default(),
            potentials_send_pointers: Vec::default(),
            leaf_upward_surfaces_sources: Vec::default(),
            leaf_upward_surfaces_targets: Vec::default(),
            charges: Vec::default(),
            charge_index_pointer_sources: Vec::default(),
            charge_index_pointer_targets: Vec::default(),
            leaf_scales_sources: Vec::default(),
        }
    }
}
/// Specifies the format of the input data for Fast Multipole Method (FMM) calculations.
///
/// This enum is used to indicate whether the input to the FMM algorithm consists
/// of a single vector of charges or a matrix where each column represents a separate
/// vector of charges. The choice between a vector and a matrix format can affect
/// how the FMM processes the input data, potentially influencing both the performance
/// and the outcome of the calculations.
///
/// When using BLAS based field translations it's likely that you will obtain a cache
/// advantage from using a matrix of input charges, which will not be the case for an
/// FFT based field translation.
///
/// # Variants
///
/// - `Vector`- Indicates that the input is a single vector of charges.
///
/// - `Matrix`- Indicates that the input is a matrix of charges, where each column in the
///   matrix represents a distinct vector of charges.
#[derive(Clone, Copy)]
pub enum FmmEvalType {
    /// Indicates a single vector of charges.
    Vector,
    /// Indicates a matrix of charges, where each column represents a vector of charges.
    /// The `usize` parameter specifies the number of vectors (columns) in the matrix.
    Matrix(usize),
}

#[derive(Default)]
pub struct SingleNodeBuilder<Scalar, Kern, SourceToTarget>
where
    Scalar: RlstScalar + Default,
    Kern: Kernel<T = Scalar> + Clone,
    SourceToTarget: ConfigureSourceToTargetData,
    <Scalar as RlstScalar>::Real: Default,
{
    /// Tree
    pub tree: Option<SingleNodeFmmTree<Scalar::Real>>,

    /// Kernel
    pub kernel: Option<Kern>,

    /// Charges
    pub charges: Option<Charges<Scalar::Real>>,

    /// Data and metadata for field translations
    pub source_to_target: Option<SourceToTarget>,

    /// Domain
    pub domain: Option<Domain<Scalar::Real>>,

    /// Expansion order
    pub expansion_order: Option<usize>,

    /// Number of coefficients
    pub ncoeffs: Option<usize>,

    /// Kernel eval type
    pub kernel_eval_type: Option<EvalType>,

    /// FMM eval type
    pub fmm_eval_type: Option<FmmEvalType>,
}

/// Represents an octree structure for Fast Multipole Method (FMM) calculations on a single node.
///
/// This struct encapsulates octrees for two distributions of points, sources, and targets,
/// along with an associated computational domain.
///
/// # Fields
///
/// - `source_tree`- An octree structure containing the source points. The source points
///   are those from which the potential will be computed.
///
/// - `target_tree`- An octree structure containing the target points. The target points
///   are those at which the potential will be evaluated.
///
/// - `domain`- The computational domain associated with this FMM calculation. This domain
///   defines the spatial extent within which the sources and targets are located and
///   interacts.
#[derive(Default)]
pub struct SingleNodeFmmTree<T: RlstScalar + Float + Default> {
    /// An octree structure containing the source points for the FMM calculation.
    pub source_tree: SingleNodeTree<T>,
    /// An octree structure containing the target points for the FMM calculation.
    pub target_tree: SingleNodeTree<T>,
    /// The computational domain associated with this FMM calculation.
    pub domain: Domain<T>,
}

/// Represents an octree structure for Fast Multipole Method (FMM) calculations on distributed nodes.
///
/// This struct encapsulates octrees for two distributions of points, sources, and targets,
/// along with an associated computational domain.
///
/// # Fields
///
/// - `source_tree`- An octree structure containing the source points. The source points
///   are those from which the potential will be computed.
///
/// - `target_tree`- An octree structure containing the target points. The target points
///   are those at which the potential will be evaluated.
///
/// - `domain`- The computational domain associated with this FMM calculation. This domain
///   defines the spatial extent within which the sources and targets are located and
///   interacts.
#[cfg(feature = "mpi")]
pub struct MultiNodeFmmTree<T: RlstScalar + Float + Equivalence> {
    /// An octree structure containing the source points for the FMM calculation.
    pub source_tree: MultiNodeTree<T>,
    /// An octree structure containing the target points for the FMM calculation.
    pub target_tree: MultiNodeTree<T>,
    /// The computational domain associated with this FMM calculation.
    pub domain: Domain<T>,
}

/// Stores data and metadata for FFT based acceleration scheme for field translation.
///
///  # Fields
///
/// - `surf_to_conv_map`- A mapping from indices of surface grid points to their
///   corresponding points in the convolution grid.
///
/// - `conv_to_surf_map`- A mapping from indices in the convolution grid back to
///   the surface grid points.
///
/// - `metadata`- Stores precomputed metadata required to apply this method.
///
/// - `transfer_vectors`- Contains unique transfer vectors that facilitate lookup of M2L unique kernel interactions.
///
/// - `kernel`- Specifies the kernel to be used for the FMM calculations.
///
/// - `expansion_order`- Specifies the expansion order for the multipole/local expansions,
///   used to control the accuracy and computational complexity of the FMM.
#[derive(Default)]
pub struct FftFieldTranslation<Scalar, Kern>
where
    Scalar: RlstScalar,
    Kern: Kernel<T = Scalar> + Default,
{
    /// Map between indices of surface convolution grid points.
    pub surf_to_conv_map: Vec<usize>,

    /// Map between indices of convolution and surface grid points.
    pub conv_to_surf_map: Vec<usize>,

    /// Precomputed data required for FFT compressed M2L interaction.
    pub metadata: FftMetadata<Scalar>,

    /// Unique transfer vectors to lookup m2l unique kernel interactions
    pub transfer_vectors: Vec<TransferVector<Scalar::Real>>,

    /// The associated kernel with this translation operator.
    pub kernel: Kern,

    /// Expansion order
    pub expansion_order: usize,
}

/// Stores data and metadata for BLAS based acceleration scheme for field translation.
///
/// Our compressions scheme is based on [[Messner et. al, 2012](https://arxiv.org/abs/1210.7292)]. We take the a SVD over
/// interaction matrices corresponding to all unique transfer vectors, and re-compress in a directional manner.
/// This recompression is controlled via the `threshold` parameter, which filters singular vectors with corresponding
/// singular values smaller than this.
///
/// # Fields
///
/// - `threshold`- A value used to filter singular vectors during recompression.
///
/// - `metadata`- Stores precomputed metadata required to apply this method.
///
/// - `transfer_vectors`- Contains unique transfer vectors that facilitate lookup of M2L unique kernel interactions.
///
/// - `kernel`- Specifies the kernel to be used for the FMM calculations.
///
/// - `expansion_order`- Specifies the expansion order for the multipole/local expansions,
///   used to control the accuracy and computational complexity of the FMM.
///
/// - `cutoff_rank`- Determined from the `threshold` parameter as the largest rank over the global SVD over all interaction
///    matrices corresponding to unique transfer vectors.
#[derive(Default)]
pub struct BlasFieldTranslation<Scalar, Kern>
where
    Scalar: RlstScalar,
    Kern: Kernel<T = Scalar> + Default,
{
    /// Threshold
    pub threshold: Scalar::Real,

    /// Precomputed metadata
    pub metadata: BlasMetadata<Scalar>,

    /// Unique transfer vectors corresponding to each metadata
    pub transfer_vectors: Vec<TransferVector<Scalar::Real>>,

    /// The associated kernel with this translation operator.
    pub kernel: Kern,

    /// Expansion order
    pub expansion_order: usize,

    /// Cutoff rank
    pub cutoff_rank: usize,
}

/// Represents the vector between a source and target boxes encoded by Morton keys.
///
/// Encapsulates the directional vector from a source to a target, identified by their Morton keys,
/// providing both the vector components and a unique identifier (hash) for efficient lookup and
/// comparison operations.
///
/// # Fields
///
/// - `components`- The discrete vector components (x, y, z) represented as `[i64; 3]` indicating the
///   directional offset from `source` to `target`.
///
/// - `hash`- A unique identifier for this transfer vector, used to facilitate quick lookups.
///
/// - `source`- The Morton key of the source box.
///
/// - `target`- The Morton key of the target box.
#[derive(Debug)]
pub struct TransferVector<T>
where
    T: RlstScalar + Float,
{
    /// Three vector of components.
    pub components: [i64; 3],

    /// Unique identifier for transfer vector, for easy lookup.
    pub hash: usize,

    /// The `source` Morton key associated with this transfer vector.
    pub source: MortonKey<T>,

    /// The `target` Morton key associated with this transfer vector.
    pub target: MortonKey<T>,
}

/// Stores metadata for FFT based acceleration scheme for field translation.
///
/// A set of eight siblings is called a __cluster__, the M2L translations for a given target cluster then consists
/// only of translations between source clusters which correspond to neighbours of the target cluster's parent.
/// In 3D there are (up to) 26 source clusters for each target cluster, which we call its __halo__.
///
/// Each source/target cluster pair will have 64 unique interaction matrices, for each of which we compute the DFT using FFTW.
/// We store this data as `kernel_data`, a vector of length 26 - corresponding to the size of the halo. Each element contains a vector
/// of all 64 DFT sequences of the interaction matrices between source and target clusters. The outer and inner vectors are in Morton order.
///
/// Each DFT sequence, $K_i$, is of length $P$, such that each element of `kernel_data` is of the form,
///
/// $$ [K_1, K_2, ..., K_{64}] = [[K_1^1, K_1^2, ..., K_1^{P}], [K_2^1, K_2^2, ..., K_2^{P}], ..., [K_{64}^1, K_{64}^2, ..., K_{64}^{P}]] $$
///
/// We also store these in a permuted 'frequency' order for fast application in `kernel_data_f`, based on the techniques presented in [[Malhotra et. al, 2015](https://www.cambridge.org/core/journals/communications-in-computational-physics/article/pvfmm-a-parallel-kernel-independent-fmm-for-particle-and-volume-potentials/365109A4C15B126CD2A184F767D4C957)]
///
/// $$ [[K_1^1, K_2^1, ..., K_{64}^1],  [K_1^2, K_2^2, ..., K_{64}^2], ..., [K_1^{P}, K_2^P, ..., K_{64}^P ] $$
#[derive(Default)]
pub struct FftMetadata<T>
where
    T: RlstScalar,
{
    /// DFT of unique kernel evaluations for each source cluster in a halo of a target cluster
    pub kernel_data: Vec<Vec<T>>,

    /// DFT of unique kernel evaluations for each source cluster in a halo of a target cluster, re-arranged in frequency order
    pub kernel_data_f: Vec<Vec<T>>,
}

/// Stores metadata for BLAS based acceleration scheme for field translation.
///
/// Each interaction, identified by a unique transfer vector, $t \in T$, corresponds to a matrix $K_t$, where $T$ is the set of unique transfer vectors.
///
/// We assemble a single matrix row-wise
///
/// $$ K_{\text{fat}} = \left [ K_1, ..., K_{|T|} \right ]$$
///
/// and column-wise,
///
/// $$ K_{\text{thin}} = \left [ K_1; ...; K_{|T|} \right ]$$
///
/// These larger matrices can be compressed with SVDs, giving
///
/// $$ K_{\text{fat}} = U \Sigma  \left [ V^T_1, ..., V^T_{|T|} \right ] =  U \Sigma \tilde{V}^T$$
// /  = U \Sigma \tilde{V}^T $$
///
/// and
///
/// $$K_{\text{thin}} = \left [ R_1; ...; R_{|T|} \right ]  \Lambda S^T = \tilde{R} \Lambda S^T$$.
///
/// Through algebraic manipulation [[Fong and Darve, 2009](https://www.sciencedirect.com/science/article/pii/S0021999109004665)], $K_t$ can be expressed as,
///
/// $$K_t \approx U C_t S^T $$
///
/// where $C_t = U^T K_t S$ is of size $ k \times k$, which we call the __compressed M2L matrix__ and $U$ and $S$ are unitary matrices, of size
/// $N \times k$, where $N$ corresponds to the number of quadrature points discretising a box and $k$ is a cutoff rank determined by a the user
/// specified `threshold` parameter for smallest compared against the singular values.
///
/// We store $U$ and $S^T$ as `u` and `st` respectively.
///
/// As the value of $k$ in this formulation is dictated by the highest-rank interaction in $T$, the cost of applying most $C_t$ can be lowered
/// with another SVD for each individual $t \in T$ such that
///
/// $$ C_t \approx \bar{U}_t \bar{\Sigma}_t \bar{V}_t^T = \bar{U}_t \bar{V'}_t^{T} $$
///
/// where $\bar{U}_t$ and $\bar{V'}_t$ are of size $k \times k_t$, $\bar{\Sigma}$ is of size $k_t \times k_t$, and $k_t$
/// can be chosen to preserve $\epsilon$.
///
/// The latter terms correspond to the fields `c_u` and `c_vt` respectively.
///
/// # Fields
/// - `u`- Left singular vectors from SVD of $K_{\text{fat}}$, truncated to correspond to a maximum cutoff rank of $k$.
///
/// - `st`- Right singular vectors from the SVD of the $K_{\text{thin}}$, truncated to correspond to a maximum cutoff rank of $k$.
///
/// - `c_u`-  Left singular vectors of re-compressed M2L matrix, one entry for each transfer vector.
///
/// - `c_vt`- Right singular vectors of re-compressed M2L matrix, one entry for each transfer vector.
pub struct BlasMetadata<T>
where
    T: RlstScalar,
{
    /// Left singular vectors from SVD of fat M2L matrix, truncated to a maximum cutoff rank
    pub u: Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>,

    /// Right singular vectors from SVD of thin M2L matrix, truncated to a maximum cutoff rank.
    pub st: Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>,

    /// Left singular vectors of re-compressed M2L matrix, one entry for each transfer vector.
    pub c_u: Vec<Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>>,

    /// Right singular vectors of re-compressed M2L matrix, one entry for each transfer vector.
    pub c_vt: Vec<Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>>,
}

impl<T> Default for BlasMetadata<T>
where
    T: RlstScalar,
{
    fn default() -> Self {
        let u = rlst_dynamic_array2!(T, [1, 1]);
        let st = rlst_dynamic_array2!(T, [1, 1]);

        BlasMetadata {
            u,
            st,
            c_u: Vec::default(),
            c_vt: Vec::default(),
        }
    }
}
