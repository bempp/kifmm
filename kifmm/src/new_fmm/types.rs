//! Data structures for kernel independent FMM
use std::collections::HashMap;

use green_kernels::{traits::Kernel as KernelTrait, types::EvalType};
use num::traits::Float;
use rlst::{rlst_dynamic_array2, Array, BaseArray, RlstScalar, VectorContainer};

use crate::{
    traits::{
        fftw::Dft,
        field::{ConfigureSourceToTargetData, SourceToTargetData as SourceToTargetDataTrait},
        general::AsComplex,
    },
    tree::types::{Domain, MortonKey, SingleNodeTree},
};

#[cfg(feature = "mpi")]
use crate::tree::types::MultiNodeTree;
// #[cfg(feature = "mpi")]
// use crate::RlstScalarFloatMpi;
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

/// Holds all required data and metadata for evaluating a kernel independent FMM on a single node.
///
/// # Fields
///
/// - `dim` - Dimension of FMM, defaults to 3.
///
/// - `tree`- Holds an octree structure (`SingleNodeFmmTree`) representing
/// the sources and targets within the computational domain.
///
/// - `kernel`- Specifies the kernel to be used for the FMM calculations.
///
/// - `charges`- Holds the charge data associated with the source points, stored as a buffer
/// where each item is associated with a leaf in Morton order and looked up using `charge_index_pointer_targets`.
/// The displacement must be correctly calculated if using without the provided trait interface and multiple input charges
/// are used in the FMM.
///
/// - `expansion_order`- Specifies the expansion order for the multipole/local expansions,
/// used to control the accuracy and computational complexity of the FMM.
///
/// - `ncoeffs`- the number of quadrature points associated with the `exansion_order`.
///
/// - `kernel_eval_type`- Specifies the evaluation type of the kernel, either evaluating potentials
/// or potentials as well as gradients.
///
/// - `fmm_eval_type`- Defines the evaluation type for the FMM algorithm, either for a single charge
/// vector or multiple charge vectors.
///
/// - `kernel_eval_size` - Set by the kernel eval type.
///
/// - `charge_index_pointer_sources` - Index pointer providing left and right indices of source points
/// contained within a source leaf. This vector is `n_sources` long.
///
/// - `charge_index_pointer_targets` - Index pointer providing left and right indices of target points
/// contained within a target leaf. This vector is `n_targets` long.
///
/// - `leaf_upward_surfaces_sources` - Upward surface associated with each source leaf, in Morton order, precomputed
/// for performance during loops.
///
/// - `leaf_upward_surfaces_targets` - Upward surface associated with each target leaf, in Morton order, precomputed
/// for performance during loops.
///
/// - `leaf_scales_sources` - Scale factor for operators when applying to a each source leaf box, precomputed for
/// performance during loops.
///
/// - `uc2e_inv_1` - First component of pseudo-inverse of interaction matrix between upward check and equivalent surfaces, stored
/// in two parts for stability purposes.
///
/// - `uc2e_inv_2` - Second component of pseudo-inverse of interaction matrix between upward check and equivalent surfaces, stored
/// in two parts for stability purposes.
///
/// - `dc2e_inv_1` - First component of pseudo-inverse of interaction matrix between downward check and equivalent surfaces, stored
/// in two parts for stability purposes.
///
/// - `dc2e_inv_2` - Second component of pseudo-inverse of interaction matrix between downward check and equivalent surfaces, stored
/// in two parts for stability purposes.
///
/// - `source` -  The multipole translation matrices, for a cluster of eight children and their parent. Stored in Morton order as a single matrix
/// for ease of application.
///
/// - `source_vec` -  The multipole translation matrices, for a cluster of eight children and their parent. Stored in Morton order where each
/// index corresponds to a child box.
///
/// - `target_vec` - The local translation matrices, for a cluster of eight children and their parent. Stored in Morton order where each
/// index corresponds to a child box.
///
/// - `multipoles` - Buffer containing multipole data of all source boxes stored in Morton order. If `n` charge vectors are used in
/// the FMM, their associated multipole data is displaced by `nsources * ncoeffs` in `multipole` where `ncoeffs` is the length of each
/// sequence corresponding to a multipole expansion and there are `nsources` boxes in the source tree.
///
/// - `locals` - Buffer containing local data of all target boxes stored in Morton order. If `n` charge vectors are used in
/// the FMM, their associated local data is displaced by `ntargets * ncoeffs` in `locals` where `ncoeffs` is the length of each
/// sequence corresponding to a local expansion and there are `ntargets` boxes in the target tree.
///
/// `potentials` - Buffer containing evaluated potentials of all target boxes stored in Morton order. If `n` charge vectors are used in
/// the FMM, their associated potential data is displaced by `ntargets * nparticles` in `potentials` where `nparticles` is the number of
/// target particles and there are `ntargets` boxes in the target tree.
///
/// - `leaf_multipoles` - Thread safe pointers to beginning of buffer containing leaf multipole data, where the outer index is set by the number
/// of evaluations being computed by the FMM.
///
/// - `level_multipoles` - Thread safe pointers to beginning of buffer containing multipole data at each level, where the outer index is set by the
/// the level of the source tree, and the inner index is set by the number of evaluations being computed by the FMM.
///
/// - `leaf_locals` - Thread safe pointers to beginning of buffer containing leaf local data, where the outer index is set by the number
/// of evaluations being computed by the FMM.
///
/// - `level_locals` - Thread safe pointers to beginning of buffer containing local data at each level, where the outer index is set by the
/// the level of the target tree, and the inner index is set by the number of evaluations being computed by the FMM.
///
/// - `level_index_pointer_locals - Index of each key in target tree at a given level within the Morton sorted keys at that level.
///
/// - `level_index_pointer_multipoles- Index of each key in source tree at a given level within the Morton sorted keys at that level.
///
/// - `potentials_send_pointers` - Threadsafe mutable pointers corresponding to each evaluated potential for each leaf box, stored in Morton order.
/// If `n` charge vectors are used in the FMM, their associated pointers are displaced by `ntargets` where there are `ntargets` boxes in the target tree.
pub struct KiFmm<Scalar, Kernel, SourceToTargetData>
where
    Scalar: RlstScalar,
    Kernel: KernelTrait<T = Scalar>,
    SourceToTargetData: SourceToTargetDataTrait,
    <Scalar as RlstScalar>::Real: Default,
{
    /// Dimension of the FMM
    pub dim: usize,

    /// A single node tree
    pub tree: SingleNodeFmmTree<Scalar::Real>,

    /// The associated kernel function
    pub kernel: Kernel,

    /// The charge data at each target leaf box.
    pub charges: Vec<Scalar>,

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
    pub leaf_scales_sources: Vec<Scalar>,

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
    pub source_to_target: SourceToTargetData,

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
impl<Scalar, Kernel, SourceToTargetData> Default for KiFmm<Scalar, Kernel, SourceToTargetData>
where
    Scalar: RlstScalar,
    Kernel: KernelTrait<T = Scalar> + Default,
    SourceToTargetData: SourceToTargetDataTrait + Default,
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
            source_to_target: SourceToTargetData::default(),
            kernel: Kernel::default(),
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

/// A builder for constructing a Kernel-Independent Fast Multipole Method (KiFMM) object
/// for simulations on a single node.
///
/// This builder facilitates the configuration and initialisation of the KiFMM in a step-by-step
/// manner
///
/// # Fields
///
/// - `tree`- Holds an octree structure (`SingleNodeFmmTree`) representing
/// the sources and targets within the computational domain.
///
/// - `charges`- Holds the charge data associated with the source points.
///
/// - `source_to_target`- Metadata for multipole to local field translation, of type `T`.
///
/// - `domain`- Defines the computational domain for the FMM calculations.
///
/// - `kernel`- Specifies the kernel to be used for the FMM calculations.
///
/// - `expansion_order`- Specifies the expansion order for the multipole/local expansions,
/// used to control the accuracy and computational complexity of the FMM.
///
/// - `ncoeffs`- the number of quadrature points associated with the `exansion_order`.
///
/// - `kernel_eval_type`- Specifies the evaluation type of the kernel, either evaluating potentials
/// or potentials as well as gradients.
///
/// - `fmm_eval_type`- Defines the evaluation type for the FMM algorithm, either for a single charge
/// vector or multiple charge vectors.
///
/// # Example
/// ```
/// # extern crate blas_src;
/// # extern crate lapack_src;
/// use kifmm::{SingleNodeBuilder, BlasFieldTranslation, FftFieldTranslation};
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
/// let fmm = SingleNodeBuilder::new()
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
///         FftFieldTranslation::new(),
///     )
///     .unwrap()
///     .build()
///     .unwrap();
/// ````
/// This example demonstrates creating a new `KiFmmBuilderSingleNode` instance, configuring it
/// with source and target points, charge data, and specifying FMM parameters like the kernel
/// and expansion order, before finally building the KiFMM object.
#[derive(Default)]
pub struct SingleNodeBuilder<Scalar, Kernel, SourceToTargetData>
where
    Scalar: RlstScalar + Default,
    Kernel: KernelTrait<T = Scalar> + Clone,
    SourceToTargetData: ConfigureSourceToTargetData,
    <Scalar as RlstScalar>::Real: Default,
{
    /// Tree
    pub tree: Option<SingleNodeFmmTree<Scalar::Real>>,

    /// Kernel
    pub kernel: Option<Kernel>,

    /// Charges
    pub charges: Option<Charges<Scalar>>,

    /// Data and metadata for field translations
    pub source_to_target: Option<SourceToTargetData>,

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
pub struct FftFieldTranslation<Scalar, Kernel>
where
    Scalar: RlstScalar + AsComplex + Default + Dft,
    <Scalar as RlstScalar>::Real: RlstScalar + Default,
    Kernel: KernelTrait<T = Scalar> + Default,
{
    /// Map between indices of surface convolution grid points.
    pub surf_to_conv_map: Vec<usize>,

    /// Map between indices of convolution and surface grid points.
    pub conv_to_surf_map: Vec<usize>,

    /// Precomputed data required for FFT compressed M2L interaction.
    pub metadata: FftMetadata<<Scalar as AsComplex>::ComplexType>,

    /// Unique transfer vectors to lookup m2l unique kernel interactions
    pub transfer_vectors: Vec<TransferVector<Scalar::Real>>,

    /// The associated kernel with this translation operator.
    pub kernel: Kernel,

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
pub struct BlasFieldTranslation<Scalar, Kernel>
where
    Scalar: RlstScalar,
    Kernel: KernelTrait<T = Scalar> + Default,
{
    /// Threshold
    pub threshold: Scalar::Real,

    /// Precomputed metadata
    pub metadata: BlasMetadata<Scalar>,

    /// Unique transfer vectors corresponding to each metadata
    pub transfer_vectors: Vec<TransferVector<Scalar::Real>>,

    /// The associated kernel with this translation operator.
    pub kernel: Kernel,

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
