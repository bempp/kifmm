//! Data structures for kernel independent FMM
use std::{collections::HashMap, sync::RwLock};

use green_kernels::{traits::Kernel as KernelTrait, types::GreenKernelEvalType};
use num::traits::Float;
use rlst::{
    rlst_dynamic_array2, Array, BaseArray, RawAccess, RawAccessMut, RlstScalar, Shape,
    SliceContainer, VectorContainer,
};

use crate::{
    linalg::rsvd::Normaliser,
    traits::{
        fftw::Dft,
        field::FieldTranslation as FieldTranslationTrait,
        fmm::HomogenousKernel,
        general::single_node::AsComplex,
        types::{CommunicationTime, FmmOperatorTime, MetadataTime},
    },
    tree::{Domain, MortonKey, SingleNodeTree},
};

#[cfg(feature = "mpi")]
use crate::tree::MultiNodeTree;

#[cfg(feature = "mpi")]
use std::collections::HashSet;

#[cfg(feature = "mpi")]
use mpi::topology::SimpleCommunicator;

#[cfg(feature = "mpi")]
use mpi::{
    traits::{Communicator, Equivalence},
    Count, Rank,
};

/// Represents charge data in a two-dimensional array with shape `[ncharges, nvecs]`,
/// organized in row-major order.
pub type Charges<T> = Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>;

/// Represents coordinate data in a two-dimensional array with shape `[dim, n_coords]`,
/// stored in row-major order.
pub type Coordinates<T> = Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>;

/// Represents coordinate data in a two-dimensional array with shape `[dim, n_coords]`,
/// stored in row-major order.
pub type CoordinatesSlice<'slc, T> = Array<T, BaseArray<T, SliceContainer<'slc, T>, 2>, 2>;

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
pub(crate) struct SendPtrMut<T> {
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
#[allow(dead_code)]
pub(crate) struct SendPtr<T> {
    /// Holds the raw immutable pointer to an instance of `T`.
    pub raw: *const T,
}

/// Holds all required data and metadata for evaluating a kernel independent FMM on a single node.
#[allow(clippy::type_complexity)]
pub struct KiFmm<Scalar, Kernel, FieldTranslation>
where
    Scalar: RlstScalar,
    Kernel: KernelTrait<T = Scalar> + HomogenousKernel,
    FieldTranslation: FieldTranslationTrait,
    <Scalar as RlstScalar>::Real: Default,
{
    /// Operator runtimes
    pub operator_times: Vec<FmmOperatorTime>,

    /// Communication runtimes
    pub communication_times: Vec<CommunicationTime>,

    /// Metadata runtimes
    pub metadata_times: Vec<MetadataTime>,

    /// Whether the object and its methods are timed
    pub(crate) timed: bool,

    /// Dimension of the FMM
    pub(crate) dim: usize,

    /// Instruction set architecture
    pub(crate) isa: Isa,

    /// A single node tree
    pub(crate) tree: SingleNodeFmmTree<Scalar::Real>,

    /// The associated kernel function
    pub(crate) kernel: Kernel,

    /// The charge data at each target leaf box.
    pub(crate) charges: Vec<Scalar>,

    /// Set to true if expansion order varies by level
    pub(crate) variable_expansion_order: bool,

    /// The expansion order of the FMM, used to construct equivalent surfaces.
    pub(crate) equivalent_surface_order: Vec<usize>, // Index corresponds to level

    /// The expansion order used to construct check surfaces
    pub(crate) check_surface_order: Vec<usize>, // index corresponds to level

    /// The number of coefficients, corresponding to points discretising the equivalent surface
    pub(crate) n_coeffs_equivalent_surface: Vec<usize>, // Index corresponds to level

    /// The number of coefficients, corresponding to points discretising the check surface
    pub(crate) n_coeffs_check_surface: Vec<usize>, // Index corresponds to level

    /// The kernel evaluation type, either for potentials or potentials and gradients
    pub(crate) kernel_eval_type: GreenKernelEvalType,

    /// The FMM evaluation type, either for a vector or matrix of input charges.
    pub(crate) fmm_eval_type: FmmEvalType,

    /// Set by the kernel evaluation type, either 1 or 4 corresponding to evaluating potentials or potentials and derivatives
    pub(crate) kernel_eval_size: usize,

    /// Index pointer for source coordinates
    pub(crate) charge_index_pointer_sources: Vec<(usize, usize)>,

    /// Index pointer for target coordinates
    pub(crate) charge_index_pointer_targets: Vec<(usize, usize)>,

    /// Upward surfaces associated with source leaves
    pub(crate) leaf_upward_equivalent_surfaces_sources: Vec<Scalar::Real>,

    /// Upward surfaces associated with source leaves
    pub(crate) leaf_upward_check_surfaces_sources: Vec<Scalar::Real>,

    /// Upward surfaces associated with target leaves
    pub(crate) leaf_downward_equivalent_surfaces_targets: Vec<Scalar::Real>,

    /// Scales of each source leaf box
    pub(crate) leaf_scales_sources: Vec<Scalar>,

    /// The pseudo-inverse of the dense interaction matrix between the upward check and upward equivalent surfaces.
    /// Store in two parts to avoid propagating error from computing pseudo-inverse
    pub(crate) uc2e_inv_1: Vec<Array<Scalar, BaseArray<Scalar, VectorContainer<Scalar>, 2>, 2>>, // index corresponds to level

    /// The pseudo-inverse of the dense interaction matrix between the upward check and upward equivalent surfaces.
    /// Store in two parts to avoid propagating error from computing pseudo-inverse
    pub(crate) uc2e_inv_2: Vec<Array<Scalar, BaseArray<Scalar, VectorContainer<Scalar>, 2>, 2>>, // index corresponds to level

    /// The pseudo-inverse of the dense interaction matrix between the downward check and downward equivalent surfaces.
    /// Store in two parts to avoid propagating error from computing pseudo-inverse
    pub(crate) dc2e_inv_1: Vec<Array<Scalar, BaseArray<Scalar, VectorContainer<Scalar>, 2>, 2>>, // index corresponds to level

    /// The pseudo-inverse of the dense interaction matrix between the downward check and downward equivalent surfaces.
    /// Store in two parts to avoid propagating error from computing pseudo-inverse
    pub(crate) dc2e_inv_2: Vec<Array<Scalar, BaseArray<Scalar, VectorContainer<Scalar>, 2>, 2>>, // index corresponds to level

    /// Data and metadata for field translations
    pub(crate) source_to_target: FieldTranslation,

    /// The multipole translation matrices, for a cluster of eight children and their parent. Stored in Morton order.
    pub(crate) source: Vec<Array<Scalar, BaseArray<Scalar, VectorContainer<Scalar>, 2>, 2>>, // index corresponds to level

    /// The metadata required for source to source translation
    pub(crate) source_vec:
        Vec<Vec<Array<Scalar, BaseArray<Scalar, VectorContainer<Scalar>, 2>, 2>>>, // index corresponds to level

    /// The local to local operator matrices, each index is associated with a child box (in sequential Morton order).
    pub(crate) target_vec:
        Vec<Vec<Array<Scalar, BaseArray<Scalar, VectorContainer<Scalar>, 2>, 2>>>, // index corresponds to level

    /// The multipole expansion data at each box.
    pub(crate) multipoles: Vec<Scalar>,

    /// The local expansion at each box
    pub(crate) locals: Vec<Scalar>,

    /// The evaluated potentials at each target leaf box.
    pub(crate) potentials: Vec<Scalar>,

    /// Multipole expansions at leaf level
    pub(crate) leaf_multipoles: Vec<Vec<SendPtrMut<Scalar>>>,

    /// Multipole expansions at each level
    pub(crate) level_multipoles: Vec<Vec<Vec<SendPtrMut<Scalar>>>>,

    /// Local expansions at the leaf level
    pub(crate) leaf_locals: Vec<Vec<SendPtrMut<Scalar>>>,

    /// The local expansion data at each level.
    pub(crate) level_locals: Vec<Vec<Vec<SendPtrMut<Scalar>>>>,

    /// Index pointers to each key at a given level, indexed by level.
    pub(crate) level_index_pointer_locals: Vec<HashMap<MortonKey<Scalar::Real>, usize>>,

    /// Index pointers to each key at a given level, indexed by level.
    pub(crate) level_index_pointer_multipoles: Vec<HashMap<MortonKey<Scalar::Real>, usize>>,

    /// The evaluated potentials at each target leaf box.
    pub(crate) potentials_send_pointers: Vec<SendPtrMut<Scalar>>,
}

impl<Scalar, Kernel, FieldTranslation> Default for KiFmm<Scalar, Kernel, FieldTranslation>
where
    Scalar: RlstScalar,
    Kernel: KernelTrait<T = Scalar> + HomogenousKernel + Default,
    FieldTranslation: FieldTranslationTrait + Default,
    <Scalar as RlstScalar>::Real: Default,
{
    fn default() -> Self {
        KiFmm {
            timed: false,
            operator_times: Vec::default(),
            communication_times: Vec::default(),
            metadata_times: Vec::default(),
            isa: Isa::default(),
            tree: SingleNodeFmmTree::default(),
            source_to_target: FieldTranslation::default(),
            kernel: Kernel::default(),
            variable_expansion_order: false,
            equivalent_surface_order: Vec::default(),
            check_surface_order: Vec::default(),
            fmm_eval_type: FmmEvalType::Vector,
            kernel_eval_type: GreenKernelEvalType::Value,
            kernel_eval_size: 0,
            dim: 0,
            n_coeffs_equivalent_surface: Vec::default(),
            n_coeffs_check_surface: Vec::default(),
            uc2e_inv_1: Vec::default(),
            uc2e_inv_2: Vec::default(),
            dc2e_inv_1: Vec::default(),
            dc2e_inv_2: Vec::default(),
            source: Vec::default(),
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
            leaf_upward_equivalent_surfaces_sources: Vec::default(),
            leaf_upward_check_surfaces_sources: Vec::default(),
            leaf_downward_equivalent_surfaces_targets: Vec::default(),
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
/// # Example
/// ```
/// use kifmm::{SingleNodeBuilder, BlasFieldTranslationSaRcmp, FftFieldTranslation};
/// use kifmm::traits::fmm::{Evaluate, ChargeHandler};
/// use kifmm::traits::tree::SingleFmmTree;
/// use kifmm::tree::helpers::points_fixture;
/// use rlst::{rlst_dynamic_array2, RawAccessMut, RawAccess};
/// use green_kernels::{laplace_3d::Laplace3dKernel, types::GreenKernelEvalType};
///
/// /// Particle data
/// let n_sources = 1000;
/// let n_targets = 2000;
/// let sources = points_fixture::<f64>(n_sources, None, None, Some(0));
/// let targets = points_fixture::<f64>(n_targets, None, None, Some(3));
///
/// // FMM parameters
/// let n_crit = Some(150); // Constructed from data, using `n_crit` parameter
/// let depth = None; // Must not specify a depth in this case
/// let expansion_order = [10]; // Constructed with `n_crit`, therefore can only use a single expansion order.
/// let prune_empty = true;
///
/// /// Charge data
/// let nvecs = 1;
/// let tmp = vec![1.0; n_sources * nvecs];
/// let mut charges = rlst_dynamic_array2!(f64, [n_sources, nvecs]);
/// charges.data_mut().copy_from_slice(&tmp);
///
/// /// Create a new builder, and attach a tree
/// let fmm = SingleNodeBuilder::new(false) // optionally time operators
///     .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
///     .unwrap();
///
/// /// Specify the FMM parameters, such as the kernel , the kernel evaluation mode, expansion order and charge data
/// let mut fmm = fmm
///     .parameters(
///         charges.data(),
///         &expansion_order,
///         Laplace3dKernel::new(),
///         GreenKernelEvalType::Value,
///         FftFieldTranslation::new(None),
///     )
///     .unwrap()
///     .build()
///     .unwrap();
///
/// // Run the FMM
///
/// fmm.evaluate().unwrap();
///
/// /// Can clear charges on the runtime object, and re-attach
/// let new_charges = vec![2.0; n_sources * nvecs];
/// fmm.attach_charges_unordered(&new_charges).unwrap();
///
/// /// And then can evaluate again, without having to run pre-computation
/// fmm.evaluate().unwrap();
///
/// ````
/// This example demonstrates creating a new `KiFmmBuilderSingleNode` instance, configuring it
/// with source and target points, charge data, and specifying FMM parameters like the kernel
/// and expansion order, before finally building the KiFMM object.
#[derive(Default)]
pub struct SingleNodeBuilder<Scalar, Kernel, FieldTranslation>
where
    Scalar: RlstScalar + Default,
    Kernel: KernelTrait<T = Scalar> + Clone,
    FieldTranslation: FieldTranslationTrait,
    <Scalar as RlstScalar>::Real: Default,
{
    /// Whether construction and operators are timed
    pub timed: Option<bool>,

    /// Instruction set architecture
    pub isa: Option<Isa>,

    /// Tree
    pub tree: Option<SingleNodeFmmTree<Scalar::Real>>,

    /// Kernel
    pub kernel: Option<Kernel>,

    /// Charges
    pub charges: Option<Vec<Scalar>>,

    /// Data and metadata for field translations
    pub source_to_target: Option<FieldTranslation>,

    /// Domain
    pub domain: Option<Domain<Scalar::Real>>,

    /// Variable expansion order by level
    pub variable_expansion_order: Option<bool>,

    /// Expansion order used to discretise equivalent surface
    pub equivalent_surface_order: Option<Vec<usize>>,

    /// Expansion order used to discretise check surface
    pub check_surface_order: Option<Vec<usize>>,

    /// Number of coefficients
    pub n_coeffs_equivalent_surface: Option<Vec<usize>>,

    /// Number of coefficients
    pub n_coeffs_check_surface: Option<Vec<usize>>,

    /// Kernel eval type
    pub kernel_eval_type: Option<GreenKernelEvalType>,

    /// FMM eval type
    pub fmm_eval_type: Option<FmmEvalType>,

    /// Has depth or ncrit been set
    pub depth_set: Option<bool>,

    /// Communication runtimes
    pub communication_times: Option<Vec<CommunicationTime>>,
}

/// Builder for distributed FMM, example usage can be found in the examples directory.
#[derive(Default)]
#[cfg(feature = "mpi")]
pub struct MultiNodeBuilder<Scalar, Kernel, FieldTranslation>
where
    Scalar: RlstScalar + Default + Equivalence,
    Kernel: KernelTrait<T = Scalar> + Clone,
    FieldTranslation: FieldTranslationTrait,
    <Scalar as RlstScalar>::Real: Default + Equivalence,
{
    /// Whether construction and operators are timed
    pub timed: Option<bool>,

    /// Kernel
    pub kernel: Option<Kernel>,

    /// Tree
    pub tree: Option<MultiNodeFmmTree<Scalar::Real, SimpleCommunicator>>,

    /// Associated communicator
    pub communicator: Option<SimpleCommunicator>,

    /// Associated global domain
    pub domain: Option<Domain<Scalar::Real>>,

    /// Associated ISA
    pub isa: Option<Isa>,

    /// Data and metadata for field translations
    pub source_to_target: Option<FieldTranslation>,

    /// Equivalent surface order, variable expansion order not supported
    pub equivalent_surface_order: Option<usize>,

    /// Check surface order, variable expansion order not supported
    pub check_surface_order: Option<usize>,

    /// Number of coefficients
    pub n_coeffs_equivalent_surface: Option<usize>,

    /// Number of coefficients
    pub n_coeffs_check_surface: Option<usize>,

    /// Kernel eval type
    pub kernel_eval_type: Option<GreenKernelEvalType>,

    /// FMM eval type
    pub fmm_eval_type: Option<FmmEvalType>,

    /// Charges associated with each source point
    pub charges: Option<Vec<Scalar>>,

    /// Communication runtimes
    pub communication_times: Option<Vec<CommunicationTime>>,
}

/// Represents an octree structure for Fast Multipole Method (FMM) calculations on a single node.
///
/// This struct encapsulates octrees for two distributions of points, sources, and targets,
/// along with an associated computational domain.

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
#[cfg(feature = "mpi")]
pub struct MultiNodeFmmTree<T: RlstScalar + Float + Equivalence, C: Communicator> {
    /// An octree structure containing the source points for the FMM calculation.
    pub source_tree: MultiNodeTree<T, C>,

    /// An octree structure containing the target points for the FMM calculation.
    pub target_tree: MultiNodeTree<T, C>,

    /// The computational domain associated with this FMM calculation.
    pub domain: Domain<T>,

    /// Layout of sources at each rank
    pub source_layout: Layout<T>,

    /// V list queries
    pub v_list_query: Query,

    /// U list query
    pub u_list_query: Query,
}

/// Stores data and metadata for FFT based acceleration scheme for field translation.
#[derive(Default)]
pub struct FftFieldTranslation<Scalar>
where
    Scalar: RlstScalar + AsComplex + Default + Dft,
{
    /// A  mapping from indices of surface grid points to their  corresponding points in the convolution grid.
    pub surf_to_conv_map: Vec<Vec<usize>>, // Indexed by level

    /// A mapping from indices in the convolution grid back to the surface grid points.
    pub conv_to_surf_map: Vec<Vec<usize>>, // Indexed by level

    /// Maximum block size when grouping interactions
    pub block_size: usize,

    /// Precomputed data required for FFT compressed M2L interaction.
    pub metadata: Vec<FftMetadata<<Scalar as AsComplex>::ComplexType>>, // index corresponds to level

    /// Unique transfer vectors to lookup m2l unique kernel interactions
    pub transfer_vectors: Vec<TransferVector<Scalar::Real>>,

    /// The map between sources/targets in the field translation, indexed by level, then by source index.
    pub displacements: Vec<Vec<RwLock<Vec<usize>>>>,
}

impl<Scalar> Clone for FftFieldTranslation<Scalar>
where
    Scalar: RlstScalar + AsComplex + Default + Dft,
    Scalar::Real: Clone,
    <Scalar as AsComplex>::ComplexType: Clone,
    FftMetadata<<Scalar as AsComplex>::ComplexType>: Clone,
    TransferVector<Scalar::Real>: Clone,
{
    fn clone(&self) -> Self {
        FftFieldTranslation {
            surf_to_conv_map: self.surf_to_conv_map.clone(),
            conv_to_surf_map: self.conv_to_surf_map.clone(),
            block_size: self.block_size, // Copy as it's a primitive type (usize)
            metadata: self.metadata.clone(),
            transfer_vectors: self.transfer_vectors.clone(),
            displacements: self
                .displacements
                .iter()
                .map(|vec| {
                    vec.iter()
                        .map(|lock| {
                            // Lock the RwLock to get access to the inner Vec<usize> and clone it
                            RwLock::new(lock.read().unwrap().clone())
                        })
                        .collect()
                })
                .collect(),
        }
    }
}

/// Stores data and metadata for BLAS based acceleration scheme for field translation.
///
/// Our compressions scheme is based on [[Messner et. al, 2012](https://arxiv.org/abs/1210.7292)]. We take the a SVD over
/// interaction matrices corresponding to all unique transfer vectors, and re-compress in a directional manner. This is
/// termed the Single Approximation Recompression (SaRcmp) by Messner et. al.
/// This recompression is controlled via the `threshold` parameter, which filters singular vectors with corresponding
/// singular values smaller than this.
///
#[derive(Default)]
pub struct BlasFieldTranslationSaRcmp<Scalar>
where
    Scalar: RlstScalar,
{
    /// Threshold
    pub threshold: Scalar::Real,

    /// Precomputed metadata
    pub metadata: Vec<BlasMetadataSaRcmp<Scalar>>, // Indexed by level

    /// Unique transfer vectors corresponding to each metadata
    pub transfer_vectors: Vec<TransferVector<Scalar::Real>>,

    /// Cutoff rank
    pub cutoff_rank: Vec<usize>, // Indexed by level

    /// Directional cutoff ranks
    pub directional_cutoff_ranks: Vec<Vec<usize>>,

    /// The map between sources/targets in the field translation, indexed by level, then by source index.
    pub displacements: Vec<Vec<RwLock<Vec<i32>>>>,

    /// Difference in expansion order between check and equivalent surface, defaults to 0
    pub surface_diff: usize,

    /// Select SVD algorithm for compression, either deterministic or randomised
    pub svd_mode: FmmSvdMode,
}

impl<Scalar> Clone for BlasFieldTranslationSaRcmp<Scalar>
where
    Scalar: RlstScalar,
    Scalar::Real: Clone,
    BlasMetadataSaRcmp<Scalar>: Clone,
    TransferVector<Scalar::Real>: Clone,
    FmmSvdMode: Clone,
{
    fn clone(&self) -> Self {
        BlasFieldTranslationSaRcmp {
            threshold: self.threshold,
            metadata: self.metadata.clone(),
            transfer_vectors: self.transfer_vectors.clone(),
            cutoff_rank: self.cutoff_rank.clone(),
            directional_cutoff_ranks: self.directional_cutoff_ranks.clone(),
            displacements: self
                .displacements
                .iter()
                .map(|vec| {
                    vec.iter()
                        .map(|lock| {
                            // Lock the RwLock to get access to the inner Vec<i32> and clone it
                            RwLock::new(lock.read().unwrap().clone())
                        })
                        .collect()
                })
                .collect(),
            surface_diff: self.surface_diff,
            svd_mode: self.svd_mode,
        }
    }
}

/// Variants of SVD algorithms
#[derive(Default, Clone, Copy)]
pub enum FmmSvdMode {
    /// Use randomised SVD with optional power iteration for additional accuracy
    Random {
        /// Number of singular values/vectors sought
        n_components: Option<usize>,

        /// Set normaliser
        normaliser: Option<Normaliser>,

        /// Set number of additional samples, in addition to n_components
        n_oversamples: Option<usize>,

        /// Set a random state.
        random_state: Option<usize>,
    },

    /// Use DGESVD from Lapack bindings
    #[default]
    Deterministic,
}

impl FmmSvdMode {
    /// Constructor for SVD settings
    pub fn new(
        random: bool,
        n_iter: Option<usize>,
        n_components: Option<usize>,
        n_oversamples: Option<usize>,
        random_state: Option<usize>,
    ) -> Self {
        if random {
            let n_iter = n_iter.unwrap_or_default();
            if n_iter > 0 {
                FmmSvdMode::Random {
                    n_components,
                    normaliser: Some(Normaliser::Qr(n_iter)),
                    n_oversamples,
                    random_state,
                }
            } else {
                FmmSvdMode::Random {
                    n_components,
                    normaliser: None,
                    n_oversamples,
                    random_state,
                }
            }
        } else {
            FmmSvdMode::Deterministic
        }
    }
}

/// Stores data and metadata for BLAS based acceleration scheme for field translation.
///
/// Our compressions scheme is based on [[Messner et. al, 2012](https://arxiv.org/abs/1210.7292)]. We take the a SVD over
/// each interaction matrix at each level, termed the individual approximation (IA) scheme by Messner et. al. This is
/// particularly advantageous for oscillatory kernels where the ranks of each interaction matrix can be large.
#[derive(Default)]
pub struct BlasFieldTranslationIa<Scalar>
where
    Scalar: RlstScalar,
{
    /// Threshold
    pub threshold: Scalar::Real,

    /// Precomputed metadata
    pub metadata: Vec<BlasMetadataIa<Scalar>>,

    /// Unique transfer vectors corresponding to each metadata
    pub transfer_vectors: Vec<Vec<TransferVector<Scalar::Real>>>,

    /// Determined from the `threshold` parameter as the largest rank over the global SVD over all interaction
    /// matrices corresponding to unique transfer vectors. Indexed by level and then by transfer vector.
    pub cutoff_ranks: Vec<Vec<usize>>,

    /// The map between sources/targets in the field translation, indexed by level, then by source index.
    pub displacements: Vec<Vec<RwLock<Vec<i32>>>>,

    /// Difference in expansion order between check and equivalent surface, defaults to 0
    pub surface_diff: usize,

    /// Select SVD algorithm for compression, either deterministic or randomised
    pub svd_mode: FmmSvdMode,
}

/// Represents the vector between a source and target boxes encoded by Morton keys.
///
/// Encapsulates the directional vector from a source to a target, identified by their Morton keys,
/// providing both the vector components and a unique identifier (hash) for efficient lookup and
/// comparison operations.
#[derive(Debug, Clone, Copy)]
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
#[derive(Default, Clone)]
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
pub struct BlasMetadataSaRcmp<T>
where
    T: RlstScalar,
{
    /// Left singular vectors from SVD of $K_{\text{fat}}$, truncated to correspond to a maximum cutoff rank of $k$.
    pub u: Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>,

    /// Right singular vectors from the SVD of the $K_{\text{thin}}$, truncated to correspond to a maximum cutoff rank of $k$.
    pub st: Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>,

    /// Left singular vectors of re-compressed M2L matrix, one entry for each transfer vector.
    pub c_u: Vec<Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>>,

    /// Right singular vectors of re-compressed M2L matrix, one entry for each transfer vector.
    pub c_vt: Vec<Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>>,
}

impl<Scalar> Clone for BlasMetadataSaRcmp<Scalar>
where
    Scalar: RlstScalar + Clone,
{
    fn clone(&self) -> Self {
        let mut u = rlst_dynamic_array2!(Scalar, self.u.shape());
        u.data_mut().copy_from_slice(self.u.data());

        let mut st = rlst_dynamic_array2!(Scalar, self.st.shape());
        st.data_mut().copy_from_slice(self.st.data());

        let mut c_u = Vec::new();
        let mut c_vt = Vec::new();

        for item in self.c_u.iter() {
            let mut tmp = rlst_dynamic_array2!(Scalar, item.shape());
            tmp.data_mut().copy_from_slice(item.data());
            c_u.push(tmp);
        }

        for item in self.c_vt.iter() {
            let mut tmp = rlst_dynamic_array2!(Scalar, item.shape());
            tmp.data_mut().copy_from_slice(item.data());
            c_vt.push(tmp);
        }

        Self { u, st, c_u, c_vt }
    }
}

/// Stores metadata for BLAS based acceleration scheme for field translation.
///
/// Each interaction, identified by a unique transfer vector, $t \in T$, at a given level, $l$, corresponds to
///  a matrix $K_t$, where $T$ is the set of unique transfer vectors.
///
/// We individually compress each $K_t \sim U V^T$, with an SVD. Storing in a vector where each index corresponds to a unique $t$.
#[derive(Default)]
pub struct BlasMetadataIa<T>
where
    T: RlstScalar,
{
    /// Left singular vectors from SVD of compressed M2L matrix, truncated to a maximum cutoff rank
    pub u: Vec<Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>>,

    /// Right singular vectors of compressed M2L matrix, truncated to a maximum cutoff rank
    pub vt: Vec<Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>>,
}

impl<Scalar> Clone for BlasMetadataIa<Scalar>
where
    Scalar: RlstScalar + Clone,
{
    fn clone(&self) -> Self {
        let mut u = Vec::new();
        let mut vt = Vec::new();

        for item in self.u.iter() {
            let mut tmp = rlst_dynamic_array2!(Scalar, item.shape());
            tmp.data_mut().copy_from_slice(item.data());
            u.push(tmp);
        }

        for item in self.vt.iter() {
            let mut tmp = rlst_dynamic_array2!(Scalar, item.shape());
            tmp.data_mut().copy_from_slice(item.data());
            vt.push(tmp);
        }

        Self { u, vt }
    }
}

impl<T> Default for BlasMetadataSaRcmp<T>
where
    T: RlstScalar,
{
    fn default() -> Self {
        let u = rlst_dynamic_array2!(T, [1, 1]);
        let st = rlst_dynamic_array2!(T, [1, 1]);

        BlasMetadataSaRcmp {
            u,
            st,
            c_u: Vec::default(),
            c_vt: Vec::default(),
        }
    }
}

/// Instruction set architecture

#[derive(Default, Clone, Copy, Debug)]
pub enum Isa {
    /// Neon FCMA ISA, extension which provides floating point complex multiply-add instructions.
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    Neon(pulp::aarch64::NeonFcma),

    /// AVX2 ISA
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    Avx(pulp::x86::V3),

    /// Default is no vectorisation
    #[default]
    Default,
}

/// Holds all required data and metadata for evaluating a kernel independent FMM on on multiple nodes.
#[cfg(feature = "mpi")]
#[allow(clippy::type_complexity)]
pub struct KiFmmMulti<Scalar, Kernel, FieldTranslation>
where
    Scalar: RlstScalar + Equivalence + Float,
    <Scalar as RlstScalar>::Real: RlstScalar + Equivalence + Float,
    Kernel: KernelTrait<T = Scalar> + HomogenousKernel,
    FieldTranslation: FieldTranslationTrait,
{
    /// Operator runtimes
    pub operator_times: Vec<FmmOperatorTime>,

    /// Communication runtimes
    pub communication_times: Vec<CommunicationTime>,

    /// Metadata runtimes
    pub metadata_times: Vec<MetadataTime>,

    /// Dimension
    pub(crate) dim: usize,

    /// Whether the object and its methods are timed
    pub(crate) timed: bool,

    /// Instruction set architecture
    pub(crate) isa: Isa,

    /// Associated MPI communicator
    pub(crate) communicator: SimpleCommunicator,

    /// Neighbourhood communicator for V list communication
    pub(crate) neighbourhood_communicator_v: NeighbourhoodCommunicator,

    /// Neighbourhood communicator for U list communication
    pub(crate) neighbourhood_communicator_u: NeighbourhoodCommunicator,

    /// Neighbourhood communicator for charge data
    pub(crate) neighbourhood_communicator_charge: NeighbourhoodCommunicator,

    /// Associated MPI rank
    pub(crate) rank: i32,

    /// The associated kernel function
    pub(crate) kernel: Kernel,

    /// A multi node tree
    pub tree: MultiNodeFmmTree<<Scalar as RlstScalar>::Real, SimpleCommunicator>,

    /// Charges associated with each source tree
    pub(crate) charges: Vec<Scalar>,

    /// The expansion order used to construct check surfaces
    pub(crate) check_surface_order: usize,

    /// The expansion order of the FMM, used to construct equivalent surfaces.
    pub(crate) equivalent_surface_order: usize,

    /// The number of coefficients, corresponding to points discretising the equivalent surface
    pub(crate) n_coeffs_equivalent_surface: usize,

    /// The number of coefficients, corresponding to points discretising the check surface
    pub(crate) n_coeffs_check_surface: usize,

    /// Set by the kernel evaluation type, either 1 or 4 corresponding to evaluating potentials or potentials and derivatives
    pub(crate) kernel_eval_type: GreenKernelEvalType,

    /// The FMM evaluation type, either for a vector or matrix of input charges.
    pub(crate) fmm_eval_type: FmmEvalType,

    /// Set by the kernel evaluation type, either 1 or 4 corresponding to evaluating potentials or potentials and derivatives
    pub(crate) kernel_eval_size: usize,

    /// Index pointer for source coordinates
    pub(crate) charge_index_pointer_sources: Vec<(usize, usize)>,

    /// Index pointer for target coordinates
    pub(crate) charge_index_pointer_targets: Vec<(usize, usize)>,

    /// Upward surfaces associated with source leaves
    pub(crate) leaf_upward_equivalent_surfaces_sources: Vec<Scalar::Real>,

    /// Upward surfaces associated with source leaves
    pub(crate) leaf_upward_check_surfaces_sources: Vec<Scalar::Real>,

    /// Upward surfaces associated with target leaves
    pub(crate) leaf_downward_equivalent_surfaces_targets: Vec<Scalar::Real>,

    /// Scales of each source leaf box
    pub(crate) leaf_scales_sources: Vec<Scalar>,

    /// The pseudo-inverse of the dense interaction matrix between the upward check and upward equivalent surfaces.
    /// Store in two parts to avoid propagating error from computing pseudo-inverse
    pub(crate) uc2e_inv_1: Vec<Array<Scalar, BaseArray<Scalar, VectorContainer<Scalar>, 2>, 2>>,

    /// The pseudo-inverse of the dense interaction matrix between the upward check and upward equivalent surfaces.
    /// Store in two parts to avoid propagating error from computing pseudo-inverse
    pub(crate) uc2e_inv_2: Vec<Array<Scalar, BaseArray<Scalar, VectorContainer<Scalar>, 2>, 2>>,

    /// The pseudo-inverse of the dense interaction matrix between the downward check and downward equivalent surfaces.
    /// Store in two parts to avoid propagating error from computing pseudo-inverse
    pub(crate) dc2e_inv_1: Vec<Array<Scalar, BaseArray<Scalar, VectorContainer<Scalar>, 2>, 2>>,

    /// The pseudo-inverse of the dense interaction matrix between the downward check and downward equivalent surfaces.
    /// Store in two parts to avoid propagating error from computing pseudo-inverse
    pub(crate) dc2e_inv_2: Vec<Array<Scalar, BaseArray<Scalar, VectorContainer<Scalar>, 2>, 2>>,

    /// Data and metadata for field translations
    pub(crate) source_to_target: FieldTranslation,

    /// The multipole translation matrices, for a cluster of eight children and their parent. Stored in Morton order.
    pub(crate) source: Array<Scalar, BaseArray<Scalar, VectorContainer<Scalar>, 2>, 2>,

    /// The metadata required for source to source translation
    pub(crate) source_vec: Vec<Array<Scalar, BaseArray<Scalar, VectorContainer<Scalar>, 2>, 2>>,

    /// The local to local operator matrices, each index is associated with a child box (in sequential Morton order).
    pub(crate) target_vec: Vec<Array<Scalar, BaseArray<Scalar, VectorContainer<Scalar>, 2>, 2>>,

    /// Multipoles associated with locally owned data
    pub(crate) multipoles: Vec<Scalar>,

    /// Locals associated with locally owned data
    pub(crate) locals: Vec<Scalar>,

    /// Potentials associated with locally owned data
    pub(crate) potentials: Vec<Scalar>,

    /// Multipole expansions at leaf level
    pub(crate) leaf_multipoles: Vec<SendPtrMut<Scalar>>,

    /// Multipole expansions at each level
    pub(crate) level_multipoles: Vec<Vec<SendPtrMut<Scalar>>>,

    /// Local expansions at the leaf level
    pub(crate) leaf_locals: Vec<SendPtrMut<Scalar>>, // Same as leaf multipoles

    /// The local expansion data at each level.
    pub(crate) level_locals: Vec<Vec<SendPtrMut<Scalar>>>, // same as level multipoles

    /// Index pointers to each key at a given level, indexed by level.
    pub(crate) level_index_pointer_locals: Vec<HashMap<MortonKey<Scalar::Real>, usize>>, // outer fmm

    /// Index pointers to each key at a given level, indexed by level.
    pub(crate) level_index_pointer_multipoles: Vec<HashMap<MortonKey<Scalar::Real>, usize>>, // outer fmm

    /// The evaluated potentials at each target leaf box.
    pub(crate) potentials_send_pointers: Vec<SendPtrMut<Scalar>>, // outer fmm

    /// Object holding global FMM, to be run on nominated node
    pub global_fmm: KiFmm<Scalar, Kernel, FieldTranslation>,

    /// Object holding ghost V list data
    pub ghost_fmm_v: KiFmm<Scalar, Kernel, FieldTranslation>,

    /// Object holding ghost U list data
    pub ghost_fmm_u: KiFmm<Scalar, Kernel, FieldTranslation>,

    /// Buffer to store received V list queries for runtime use
    pub(crate) ghost_received_queries_v: Vec<u64>,

    /// Store received V list queries counts
    pub(crate) ghost_received_queries_counts_v: Vec<Count>,

    /// Store received V list queries displacements
    pub(crate) ghost_received_queries_displacements_v: Vec<Count>,

    /// Store requested V list queries counts
    pub(crate) ghost_requested_queries_counts_v: Vec<Count>,

    /// Requested V list queries index map of ghost keys from V list queries
    pub(crate) ghost_requested_queries_key_to_index_v: HashMap<MortonKey<Scalar::Real>, usize>,

    /// Number of input charges (initial input, unordered by Morton sort)
    pub(crate) local_count_charges: u64,

    /// Displacement of input charges among global input charge vector
    pub(crate) local_displacement_charges: u64,

    /// All global indices to send new (unordered) charge data to
    pub(crate) ghost_received_queries_charge: Vec<u64>,

    /// Counts of all global indices to send new (unordered) charge data to
    pub(crate) ghost_received_queries_charge_counts: Vec<i32>,

    /// Displacements of all global indices to send new (unordered) charge data to
    pub(crate) ghost_received_queries_charge_displacements: Vec<i32>,

    /// Store charge queries counts to send
    pub(crate) charge_send_queries_counts: Vec<Count>,

    /// Store charge queries displacments to send
    pub(crate) charge_send_queries_displacements: Vec<Count>,

    /// Store charge queries counts to receive
    pub(crate) charge_receive_queries_counts: Vec<Count>,

    /// Store charge queries displacments to receive
    pub(crate) charge_receive_queries_displacements: Vec<Count>,
}

/// Specified owned range defined by owned roots of local trees at each rank.
#[cfg(feature = "mpi")]
#[derive(Default)]
pub struct Layout<T: RlstScalar + Float> {
    /// Owned roots in terms of Morton keys
    pub raw: Vec<MortonKey<T>>,

    /// Owned roots as a set for inclusion testing
    pub raw_set: HashSet<MortonKey<T>>,

    /// Counts of the raw buffer of all roots
    pub counts: Vec<i32>,

    /// Displacements of the raw buffer of all roots
    pub displacements: Vec<i32>,

    /// Ranks of all roots
    pub ranks: Vec<i32>,

    /// Map between range and associated rank of all roots
    pub range_to_rank: HashMap<MortonKey<T>, i32>,
}

/// Defines a subset of the global communicator, useful for ghost data exchange.
#[cfg(feature = "mpi")]
pub struct NeighbourhoodCommunicator {
    /// Neighbour ranks
    pub neighbours: Vec<i32>,

    /// Wrapper around a simple communicator type
    pub raw: SimpleCommunicator,
}

/// For the storage of queries sent in ghost exchange.
#[derive(Default)]
#[cfg(feature = "mpi")]
pub struct Query {
    /// Queries sorted into rank order
    pub queries: Vec<u64>,

    /// Associated ranks, keys.len() long
    pub ranks: Vec<Rank>,

    /// Send counts for each rank in global communicator
    pub send_counts: Vec<Count>,

    /// Each index marks a rank in the global communicator that is involved in this query
    pub send_marker: Vec<Rank>,

    /// Receive counts for each rank in global communicator
    pub receive_counts: Vec<Count>,

    /// Each index marks a rank in the global communicator that is involved in this query
    pub receive_marker: Vec<Rank>,
}
