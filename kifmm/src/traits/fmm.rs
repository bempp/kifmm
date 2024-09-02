//! FMM traits
use crate::{
    fmm::types::CommunicationMode, traits::tree::SingleNodeFmmTreeTrait, tree::types::MortonKey,
};
use green_kernels::{traits::Kernel, types::EvalType};
use rlst::RlstScalar;

use super::{
    tree::{MultiNodeFmmTreeTrait, MultiNodeTreeTrait, SingleNodeTreeTrait},
    types::FmmError,
};

/// Interface for source field translations.
pub trait SourceTranslation {
    /// Particle to multipole translations, applied at leaf level over all source boxes.
    fn p2m(&self) -> Result<(), FmmError>;

    /// Multipole to multipole translations, applied during upward pass. Defined over each level of a tree.
    ///
    /// # Arguments
    /// * `level` - The child level at which this translation is being applied.
    fn m2m(&self, level: u64) -> Result<(), FmmError>;
}

/// Interface for target field translations.
pub trait TargetTranslation {
    /// Local to local translations, applied during downward pass. Defined over each level of a tree.
    ///
    /// # Arguments
    /// * `level` - The child level at which this translation is being applied.
    fn l2l(&self, level: u64) -> Result<(), FmmError>;

    /// Multipole to particle translations, applies to leaf boxes when a source box is within
    /// the near field of a target box, but is small enough that a multipole expansion converges
    /// at the target box. Defined over all leaf target boxes.
    fn m2p(&self) -> Result<(), FmmError>;

    /// Local to particle translations, applies the local expansion accumulated at each leaf box to the
    /// target particles it contains. Defined over all leaf target boxes.
    fn l2p(&self) -> Result<(), FmmError>;

    /// Near field particle to particle (direct) potential contributions to particles in a given leaf box's
    /// near field where the `p2l` and `m2p` do not apply. Defined over all leaf target boxes.
    fn p2p(&self) -> Result<(), FmmError>;
}

/// Interface for the source to target (multipole to local / M2L) field translations.
pub trait SourceToTargetTranslation {
    /// Interface for multipole to local translation, defined over each level of a tree.
    ///
    /// # Arguments
    /// * `level` - The level of the tree at which this translation is being applied.
    fn m2l(&self, level: u64) -> Result<(), FmmError>;

    /// Particle to local translations, applies to leaf boxes when a source box is within
    /// the far field of a target box, but is too large for the multipole expansion to converge
    /// at the target, so instead its contribution is computed directly. Defined over each level of a tree.
    ///
    /// # Arguments
    /// * `level` - The level of the tree at which this translation is being applied.
    fn p2l(&self, level: u64) -> Result<(), FmmError>;
}

/// Interface for a Kernel-Independent Fast Multipole Method (FMM).
///
/// This trait abstracts the core functionalities of FMM, allowing for different underlying
/// data structures, kernels, and precision types. It supports operations essential for
/// executing FMM calculations, including accessing multipole and local expansions, evaluating
/// potentials, and managing the underlying tree structure and kernel functions.
pub trait Fmm
where
    Self::Scalar: RlstScalar,
{
    /// Data associated with FMM, must implement RlstScalar.
    type Scalar;

    /// Type of tree, must implement `FmmTree`, allowing for separate source and target trees.
    type Tree: SingleNodeFmmTreeTrait;

    /// Kernel associated with this FMMl
    type Kernel: Kernel<T = Self::Scalar>;

    /// Get the multipole expansion data associated with a node as a slice
    /// # Arguments
    /// * `key` - The source node.
    fn multipole(
        &self,
        key: &<<Self::Tree as SingleNodeFmmTreeTrait>::Tree as SingleNodeTreeTrait>::Node,
    ) -> Option<&[Self::Scalar]>;

    /// Get the multipole expansion data associated with a tree level as a slice
    /// # Arguments
    /// * `level` - The tree level.
    fn multipoles(&self, level: u64) -> Option<&[Self::Scalar]>;

    /// Get the local expansion data associated with a node as a slice
    /// # Arguments
    /// * `key` - The target node.
    fn local(
        &self,
        key: &<<Self::Tree as SingleNodeFmmTreeTrait>::Tree as SingleNodeTreeTrait>::Node,
    ) -> Option<&[Self::Scalar]>;

    /// Get the local expansion data associated with a tree level as a slice
    /// # Arguments
    /// * `level` - The tree level.
    fn locals(&self, level: u64) -> Option<&[Self::Scalar]>;

    /// Get the potential data associated with the particles contained at a given node
    /// # Arguments
    /// * `key` - The target leaf node.
    fn potential(
        &self,
        leaf: &<<Self::Tree as SingleNodeFmmTreeTrait>::Tree as SingleNodeTreeTrait>::Node,
    ) -> Option<Vec<&[Self::Scalar]>>;

    /// Get all potential data at all particles, stored in order by global index
    fn potentials(&self) -> Option<&Vec<Self::Scalar>>;

    /// Get the expansion order associated with this FMM, used to discretise the equivalent surface.
    fn equivalent_surface_order(&self, level: u64) -> usize;

    /// Get the expansion order associated with this FMM, used to discretise the check surface.
    fn check_surface_order(&self, level: u64) -> usize;

    /// Check whether or not expansion order is set variably
    fn variable_expansion_order(&self) -> bool;

    /// Get the number of multipole/local coefficients associated with this FMM
    fn ncoeffs_equivalent_surface(&self, level: u64) -> usize;

    /// Get the number of multipole/local coefficients associated with this FMM
    fn ncoeffs_check_surface(&self, level: u64) -> usize;

    /// Get the tree associated with this FMM
    fn tree(&self) -> &Self::Tree;

    /// Get the kernel associated with this FMM
    fn kernel(&self) -> &Self::Kernel;

    /// Get the dimension of the data in this FMM
    fn dim(&self) -> usize;

    /// Evaluate the potentials, or potential gradients, for this FMM
    fn evaluate(&mut self, timed: bool) -> Result<(), FmmError>;

    /// Clear the data buffers and add new charge data for re-evaluation.
    ///
    /// # Arguments
    /// * `charges` - new charge data.
    fn clear(&mut self, charges: &[Self::Scalar]);
}

pub trait MultiNodeFmm
where
    Self::Scalar: RlstScalar,
{
    /// Data associated with FMM, must implement RlstScalar.
    type Scalar;

    /// Type of tree, must implement `FmmTree`, allowing for separate source and target trees.
    type Tree: MultiNodeFmmTreeTrait;

    /// Kernel associated with this FMMl
    type Kernel: Kernel<T = Self::Scalar>;

    /// Get the kernel associated with this FMM
    fn kernel(&self) -> &Self::Kernel;

    /// Get the dimension of the data in this FMM
    fn dim(&self) -> usize;

    /// Evaluate the potentials, or potential gradients, for this FMM
    fn evaluate(&mut self, timed: bool) -> Result<(), FmmError>;

    fn equivalent_surface_order(&self, level: u64) -> usize;

    fn check_surface_order(&self, level: u64) -> usize;

    fn ncoeffs_equivalent_surface(&self, level: u64) -> usize;

    fn ncoeffs_check_surface(&self, level: u64) -> usize;

    fn multipole(
        &self,
        fmm_idx: usize,
        key: &<<<Self::Tree as MultiNodeFmmTreeTrait>::Tree as MultiNodeTreeTrait>::Tree as SingleNodeTreeTrait>::Node,
    ) -> Option<&[Self::Scalar]>;

    fn multipoles(&self, fmm_idx: usize, level: u64) -> Option<&[Self::Scalar]>;
}

/// Set all metadata required for FMMs
pub trait FmmMetadata {
    /// Associated scalar
    type Scalar: RlstScalar;

    type Charges;

    /// Compute all metadata required for FMM.
    /// TODO: Breakup into smaller pieces of functionality for clarity.
    fn metadata<'a>(&mut self, eval_type: EvalType, charges: &'a [Self::Charges]);
}

pub trait FmmMetadataMultiNode {
    type Scalar;

    // After ghost exchange, insert new charges and associated leaf data into local trees
    fn update_charges(&mut self, new_charges: &[Self::Scalar]);

    // After ghost exchange and local upward passses, insert new multipole data, and associated keys into local trees
    fn update_multipoles(
        &mut self,
        new_multipoles: &[Self::Scalar],
        keys: &[Self::Scalar],
        expansion_order: usize,
    );

    /// Called after multipoles updated, creates new displacement data structure for looking up appropriate M2L data, which in a multi-node setting must be done at
    /// runtime due to ghost exchange.
    fn displacements(&mut self);
}

/// Defines how metadata associated with field translations is looked up at runtime.
/// Defined by kernel type, as well as field translation method.
pub trait FmmOperatorData
where
    Self: FmmMetadata,
{
    /// Lookup convolution grid map for FFT based M2L operator
    fn fft_map_index(&self, level: u64) -> usize;

    /// Lookup expansion order
    fn expansion_index(&self, level: u64) -> usize;

    /// Lookup c2e operator
    fn c2e_operator_index(&self, level: u64) -> usize;

    /// Lookup m2m operator
    fn m2m_operator_index(&self, level: u64) -> usize;

    /// Lookup m2l operator
    fn m2l_operator_index(&self, level: u64) -> usize;

    /// Lookup l2l operator
    fn l2l_operator_index(&self, level: u64) -> usize;

    /// Displacement index
    fn displacement_index(&self, level: u64) -> usize;
}

/// Marker trait for homogenous kernels
pub trait HomogenousKernel
where
    Self: Kernel,
{
    /// Homogeneity check
    fn is_homogenous(&self) -> bool;
}

pub trait GhostExchange {
    /// Gather ranges controlled by each local tree
    fn gather_ranges(&mut self);

    /// Exchange V list data
    fn v_list_exchange(&mut self);

    /// Exchange U list data
    fn u_list_exchange(&mut self);

    /// Gather root multipoles from local trees at nominated node
    fn gather_root_multipoles(&mut self);

    /// Scatter root locals back to local trees
    fn scatter_root_locals(&mut self);
}
