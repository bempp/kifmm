//! Data structures for kernel independent FMM
use crate::traits::{field::SourceToTargetData, tree::FmmTree};
use crate::tree::types::{Domain, MortonKey, SingleNodeTree};
use green_kernels::{traits::Kernel, types::EvalType};
use num::{Complex, Float};
use num_complex::ComplexFloat;
use rlst::{rlst_dynamic_array2, Array, BaseArray, RlstScalar, VectorContainer};
use std::collections::HashMap;

/// Represents charge data in a two-dimensional array with shape `[ncharges, nvecs]`,
/// organized in column-major order.
pub type Charges<T> = Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>;

/// Represents coordinate data in a two-dimensional array with shape `[ncoords, dim]`,
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

/// Holds all required data and metadata for evaluating a kernel independent FMM on a single node.
pub struct KiFmm<
    T: FmmTree<Tree = SingleNodeTree<W>>,
    U: SourceToTargetData<V>,
    V: Kernel,
    W: RlstScalar<Real = W> + Float + Default,
> {
    /// A single node tree
    pub tree: T,

    /// The metadata required for source to target translation
    pub source_to_target_translation_data: U,

    /// The associated kernel function
    pub kernel: V,

    /// The expansion order of the FMM
    pub expansion_order: usize,

    /// The number of coefficients, corresponding to points discretising the equivalent surface
    pub ncoeffs: usize,

    /// The FMM evaluation type, either for a vector or matrix of input charges.
    pub fmm_eval_type: FmmEvalType,

    /// The kernel evaluation type, either for potentials or potentials and derivatives
    pub kernel_eval_type: EvalType,

    /// The metadata required for source to source translation
    pub source_translation_data_vec: Vec<Array<W, BaseArray<W, VectorContainer<W>, 2>, 2>>,

    /// Set by the kernel evaluation type, either 1 or 4 corresponding to evaluating potentials or potentials and derivatives
    pub kernel_eval_size: usize,

    /// Index pointer for source coordinates
    pub charge_index_pointer_sources: Vec<(usize, usize)>,

    /// Index pointer for target coordinates
    pub charge_index_pointer_targets: Vec<(usize, usize)>,

    /// Dimension of the FMM
    pub dim: usize,

    /// Upward surfaces associated with source leaves
    pub leaf_upward_surfaces_sources: Vec<W>,

    /// Upward surfaces associated with target leaves
    pub leaf_upward_surfaces_targets: Vec<W>,

    /// The pseudo-inverse of the dense interaction matrix between the upward check and upward equivalent surfaces.
    /// Store in two parts to avoid propagating error from computing pseudo-inverse
    pub uc2e_inv_1: Array<W, BaseArray<W, VectorContainer<W>, 2>, 2>,

    /// The pseudo-inverse of the dense interaction matrix between the upward check and upward equivalent surfaces.
    /// Store in two parts to avoid propagating error from computing pseudo-inverse
    pub uc2e_inv_2: Array<W, BaseArray<W, VectorContainer<W>, 2>, 2>,

    /// The pseudo-inverse of the dense interaction matrix between the downward check and downward equivalent surfaces.
    /// Store in two parts to avoid propagating error from computing pseudo-inverse
    pub dc2e_inv_1: Array<W, BaseArray<W, VectorContainer<W>, 2>, 2>,

    /// The pseudo-inverse of the dense interaction matrix between the downward check and downward equivalent surfaces.
    /// Store in two parts to avoid propagating error from computing pseudo-inverse
    pub dc2e_inv_2: Array<W, BaseArray<W, VectorContainer<W>, 2>, 2>,

    /// The multipole to multipole operator matrices, each index is associated with a child box (in sequential Morton order),
    pub source_data: Array<W, BaseArray<W, VectorContainer<W>, 2>, 2>,

    /// The local to local operator matrices, each index is associated with a child box (in sequential Morton order).
    pub target_data: Vec<Array<W, BaseArray<W, VectorContainer<W>, 2>, 2>>,

    /// The multipole expansion data at each box.
    pub multipoles: Vec<W>,

    /// Multipole expansions at leaf level
    pub leaf_multipoles: Vec<Vec<SendPtrMut<W>>>,

    /// Multipole expansions at each level
    pub level_multipoles: Vec<Vec<Vec<SendPtrMut<W>>>>,

    /// The local expansion at each box
    pub locals: Vec<W>,

    /// Local expansions at the leaf level
    pub leaf_locals: Vec<Vec<SendPtrMut<W>>>,

    /// The local expansion data at each level.
    pub level_locals: Vec<Vec<Vec<SendPtrMut<W>>>>,

    /// index pointers to each key at a given level, indexed by level.
    pub level_index_pointer_locals: Vec<HashMap<MortonKey, usize>>,

    /// index pointers to each key at a given level, indexed by level.
    pub level_index_pointer_multipoles: Vec<HashMap<MortonKey, usize>>,

    /// The evaluated potentials at each target leaf box.
    pub potentials: Vec<W>,

    /// The evaluated potentials at each target leaf box.
    pub potentials_send_pointers: Vec<SendPtrMut<W>>,

    /// Leaf downward surfaces
    pub leaf_downward_surfaces: Vec<W>,

    /// The charge data at each target leaf box.
    pub charges: Vec<W>,

    /// Scales of each source leaf box
    pub leaf_scales_sources: Vec<W>,
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
/// - `Vector`: Indicates that the input is a single vector of charges.
///
/// - `Matrix`: Indicates that the input is a matrix of charges, where each column in the
///   matrix represents a distinct vector of charges.
#[derive(Clone, Copy)]
pub enum FmmEvalType {
    /// Indicates a single vector of charges.
    Vector,
    /// Indicates a matrix of charges, where each column represents a vector of charges.
    /// The `usize` parameter specifies the number of vectors (columns) in the matrix.
    Matrix(usize),
}

/// A builder for constructing a Kernel-Independent Fast Multipole Method (KiFMM) object
/// for simulations on a single node.
///
/// This builder facilitates the configuration and initialisation of the KiFMM in a step-by-step
/// manner
///
///
/// # Type Parameters
///
/// - `T`: Metadata associated with multipole to local field translation method.
/// - `U`: Scalar type must satisfy `RlstScalar`, `Float`, and `Default`.
/// - `V`: The kernel type, representing the mathematical kernel used in FMM calculations.
///
/// # Fields
///
/// - `tree`: Holds an octree structure (`SingleNodeFmmTree`) representing
///   the sources and targets within the computational domain.
///
/// - `charges`: Holds the charge data associated with the source points.
///
/// - `source_to_target`: Metadata for multipole to local field translation, of type `T`.
///
/// - `domain`: Defines the computational domain for the FMM calculations.
///
/// - `kernel`: Specifies the kernel to be used for the FMM calculations.
///
/// - `expansion_order`: Specifies the expansion order for the multipole/local expansions,
///   used to control the accuracy and computational complexity of the FMM.
///
/// - `ncoeffs`: The number of quadrature points associated with the `exansion_order`.
///
/// - `kernel_eval_type`: Specifies the evaluation type of the kernel, either evaluating potentials
///    or potentials as well as gradients.
///
/// - `fmm_eval_type`: Defines the evaluation type for the FMM algorithm, either for a single charge
///    vector or multiple charge vectors.
///
/// # Example
/// ```
/// # extern crate blas_src;
/// # extern crate lapack_src;
/// use kifmm::{KiFmmBuilderSingleNode, BlasFieldTranslationKiFmm, FftFieldTranslationKiFmm};
/// use kifmm::traits::fmm::Fmm;
/// use kifmm::traits::tree::FmmTree;
/// use kifmm::tree::helpers::points_fixture;
/// use rlst::{rlst_dynamic_array2, RawAccessMut};
/// use green_kernels::{laplace_3d::Laplace3dKernel, types::EvalType};
///
/// /// Particle data
/// let nsources = 1000;
/// let ntargets = 2000;
/// let sources = points_fixture::<f64>(nsources, None, None, Some(0));
/// let targets = points_fixture::<f64>(ntargets, None, None, Some(3));
///
/// // FMM parameters
/// let n_crit = Some(150);
/// let expansion_order = 10;
/// let sparse = true;
///
/// /// Charge data
/// let nvecs = 1;
/// let tmp = vec![1.0; nsources * nvecs];
/// let mut charges = rlst_dynamic_array2!(f64, [nsources, nvecs]);
/// charges.data_mut().copy_from_slice(&tmp);
///
/// /// Create a new builder, and attach a tree
/// let fmm = KiFmmBuilderSingleNode::new()
///     .tree(&sources, &targets, n_crit, sparse)
///     .unwrap();
///
/// /// Specify the FMM parameters, such as the kernel , the kernel evaluation mode, expansion order and charge data
/// let fmm = fmm
///     .parameters(
///         &charges,
///         expansion_order,
///         Laplace3dKernel::new(),
///         EvalType::Value,
///         FftFieldTranslationKiFmm::new(),
///     )
///     .unwrap()
///     .build()
///     .unwrap();
/// ````
/// This example demonstrates creating a new `KiFmmBuilderSingleNode` instance, configuring it
/// with source and target points, charge data, and specifying FMM parameters like the kernel
/// and expansion order, before finally building the KiFMM object.
#[derive(Default)]
pub struct KiFmmBuilderSingleNode<T, U, V>
where
    T: SourceToTargetData<V>,
    U: RlstScalar<Real = U> + Float + Default,
    V: Kernel,
{
    /// Tree
    pub tree: Option<SingleNodeFmmTree<U>>,
    /// Charges
    pub charges: Option<Charges<U>>,
    /// Source to target
    pub source_to_target: Option<T>,
    /// Domain
    pub domain: Option<Domain<U>>,
    /// Kernel
    pub kernel: Option<V>,
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
/// - `source_tree`: An octree structure containing the source points. The source points
///   are those from which the potential will be computed.
///
/// - `target_tree`: An octree structure containing the target points. The target points
///   are those at which the potential will be evaluated.
///
/// - `domain`: The computational domain associated with this FMM calculation. This domain
///   defines the spatial extent within which the sources and targets are located and
///   interacts.
///
/// Note: This example assumes that `SingleNodeTree` and `Domain` have been implemented
/// and provide `Default` implementations. Replace `f64` with the appropriate type
/// that meets the trait bounds of `T`.
#[derive(Default)]
pub struct SingleNodeFmmTree<T: RlstScalar<Real = T> + Float + Default> {
    /// An octree structure containing the source points for the FMM calculation.
    pub source_tree: SingleNodeTree<T>,
    /// An octree structure containing the target points for the FMM calculation.
    pub target_tree: SingleNodeTree<T>,
    /// The computational domain associated with this FMM calculation.
    pub domain: Domain<T>,
}

/// FFT field translation for KiFMM
#[derive(Default)]
pub struct FftFieldTranslationKiFmm<T, U>
where
    T: Default + RlstScalar<Real = T> + Float,
    U: Kernel<T = T> + Default,
    Complex<T>: ComplexFloat
{
    /// Map between indices of surface convolution grid points.
    pub surf_to_conv_map: Vec<usize>,

    /// Map between indices of convolution and surface grid points.
    pub conv_to_surf_map: Vec<usize>,

    /// Precomputed data required for FFT compressed M2L interaction.
    pub metadata: FftMetadata<Complex<T>>,

    /// Unique transfer vectors to lookup m2l unique kernel interactions
    pub transfer_vectors: Vec<TransferVector>,

    /// The associated kernel with this translation operator.
    pub kernel: U,

    /// Expansion order
    pub expansion_order: usize,
}

/// Stores field translation meta-data and data for an SVD based sparsification in the kernel independent FMM.
///
/// Our compressions scheme is based on [Messner et. al, 2012](https://arxiv.org/abs/1210.7292). We take the a SVD over
/// interaction matrices corresponding to all unique transfer vectors, and re-compress in a directional manner.
/// This recompression is controlled via the `threshold` parameter, which filters singular vectors with corresponding
/// singular values smaller than this.
///
/// # Fields
///
/// - `threshold`: A value used to filter singular vectors during recompression.
///
/// - `operator_data`: Stores precomputed metadata required to apply this method.
///
/// - `transfer_vectors`: Contains unique transfer vectors that facilitate lookup of M2L unique kernel interactions.
///
/// - `kernel`: Specifies the kernel to be used for the FMM calculations.
///
/// - `expansion_order`: Specifies the expansion order for the multipole/local expansions,
///   used to control the accuracy and computational complexity of the FMM.
///
/// - `cutoff_rank`: Determined from the `threshold` parameter as the largest rank over the global SVD over all interaction
///    matrices corresponding to unique transfer vectors.
#[derive(Default)]
pub struct BlasFieldTranslationKiFmm<T, U>
where
    T: RlstScalar<Real = T>,
    U: Kernel<T = T> + Default,
{
    /// Threshold
    pub threshold: T,

    /// Precomputed metadata
    pub metadata: BlasMetadata<T>,

    /// Unique transfer vectors corresponding to each metadata
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
pub struct FftMetadata<T>
where
    T: ComplexFloat
{
    /// FFT of unique kernel evaluations for each transfer vector in a halo of a sibling set
    pub kernel_data: Vec<Vec<T>>,

    /// FFT of unique kernel evaluations for each transfer vector in a halo of a sibling set, re-arranged in frequency order
    pub kernel_data_f: Vec<Vec<T>>,
}

/// Stores metadata required for BLAS field translations based on the SVD recompression scheme of [[Messner et. al, 2012](https://arxiv.org/abs/1210.7292)]
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
/// - `u`: Left singular vectors from SVD of $K_{\text{fat}}$, truncated to correspond to a maximum cutoff rank of $k$.
///
/// - `st`: Right singular vectors from the SVD of the $K_{\text{thin}}$, truncated to correspond to a maximum cutoff rank of $k$.
///
/// - `c_u`:  Left singular vectors of re-compressed M2L matrix, one entry for each transfer vector.
///
/// - `c_vt`: Right singular vectors of re-compressed M2L matrix, one entry for each transfer vector.
pub struct BlasMetadata<T>
where
    T: RlstScalar,
{
    /// Left singular vectors from SVD of fat M2L matrix, truncated to a maximum cutoff rank
    pub u: Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>,

    /// Right singular vectors from SVD of thin M2L matrix, truncated to a maximum cutoff rank.
    pub st:  Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>,

    /// Left singular vectors of re-compressed M2L matrix, one entry for each transfer vector.
    pub c_u: Vec< Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>>,

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
