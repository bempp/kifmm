//! FMM traits
use crate::traits::tree::FmmTree;
use green_kernels::{traits::Kernel, types::GreenKernelEvalType};
use rlst::RlstScalar;

use super::{tree::Tree, types::FmmError};

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
    type Tree: FmmTree;

    /// Kernel associated with this FMMl
    type Kernel: Kernel<T = Self::Scalar>;

    /// Get the multipole expansion data associated with a node as a slice
    /// # Arguments
    /// * `key` - The source node.
    fn multipole(
        &self,
        key: &<<Self::Tree as FmmTree>::Tree as Tree>::Node,
    ) -> Option<&[Self::Scalar]>;

    /// Get the multipole expansion data associated with a tree level as a slice
    /// # Arguments
    /// * `level` - The tree level.
    fn multipoles(&self, level: u64) -> Option<&[Self::Scalar]>;

    /// Get the local expansion data associated with a node as a slice
    /// # Arguments
    /// * `key` - The target node.
    fn local(&self, key: &<<Self::Tree as FmmTree>::Tree as Tree>::Node)
        -> Option<&[Self::Scalar]>;

    /// Get the local expansion data associated with a tree level as a slice
    /// # Arguments
    /// * `level` - The tree level.
    fn locals(&self, level: u64) -> Option<&[Self::Scalar]>;

    /// Get the potential data associated with the particles contained at a given node
    /// # Arguments
    /// * `key` - The target leaf node.
    fn potential(
        &self,
        leaf: &<<Self::Tree as FmmTree>::Tree as Tree>::Node,
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

/// Set all metadata required for FMMs
pub trait FmmMetadata {
    /// Associated scalar
    type Scalar: RlstScalar;

    /// Compute all metadata required for FMM.
    /// TODO: Breakup into smaller pieces of functionality for clarity.
    fn metadata(&mut self, eval_type: GreenKernelEvalType, charges: &[Self::Scalar]);
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
