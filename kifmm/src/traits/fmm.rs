//! FMM traits
use crate::{fmm::types::Charges, traits::tree::FmmTree};
use green_kernels::{traits::Kernel, types::EvalType};
use rlst::RlstScalar;

use super::tree::Tree;

/// Interface for source field translations.
pub trait SourceTranslation {
    /// Particle to multipole translations, applied at leaf level over all source boxes.
    fn p2m(&self);

    /// Multipole to multipole translations, applied during upward pass. Defined over each level of a tree.
    ///
    /// # Arguments
    /// * `level` - The child level at which this translation is being applied.
    fn m2m(&self, level: u64);
}

/// Interface for target field translations.
pub trait TargetTranslation {
    /// Local to local translations, applied during downward pass. Defined over each level of a tree.
    ///
    /// # Arguments
    /// * `level` - The child level at which this translation is being applied.
    fn l2l(&self, level: u64);

    /// Multipole to particle translations, applies to leaf boxes when a source box is within
    /// the near field of a target box, but is small enough that a multipole expansion converges
    /// at the target box. Defined over all leaf target boxes.
    fn m2p(&self);

    /// Local to particle translations, applies the local expansion accumulated at each leaf box to the
    /// target particles it contains. Defined over all leaf target boxes.
    fn l2p(&self);

    /// Near field particle to particle (direct) potential contributions to particles in a given leaf box's
    /// near field where the `p2l` and `m2p` do not apply. Defined over all leaf target boxes.
    fn p2p(&self);
}

/// Interface for the source to target (multipole to local / M2L) field translations.
pub trait SourceToTargetTranslation {
    /// Interface for multipole to local translation, defined over each level of a tree.
    ///
    /// # Arguments
    /// * `level` - The level of the tree at which this translation is being applied.
    fn m2l(&self, level: u64);

    /// Particle to local translations, applies to leaf boxes when a source box is within
    /// the far field of a target box, but is too large for the multipole expansion to converge
    /// at the target, so instead its contribution is computed directly. Defined over each level of a tree.
    ///
    /// # Arguments
    /// * `level` - The level of the tree at which this translation is being applied.
    fn p2l(&self, level: u64);
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

    /// Get the local expansion data associated with a node as a slice
    /// # Arguments
    /// * `key` - The target node.
    fn local(&self, key: &<<Self::Tree as FmmTree>::Tree as Tree>::Node)
        -> Option<&[Self::Scalar]>;

    /// Get the potential data associated with the particles contained at a given node
    /// # Arguments
    /// * `key` - The target leaf node.
    fn potential(
        &self,
        leaf: &<<Self::Tree as FmmTree>::Tree as Tree>::Node,
    ) -> Option<Vec<&[Self::Scalar]>>;

    /// Get the expansion order associated with this FMM
    fn expansion_order(&self) -> usize;

    /// Get the tree associated with this FMM
    fn tree(&self) -> &Self::Tree;

    /// Get the kernel associated with this FMM
    fn kernel(&self) -> &Self::Kernel;

    /// Get the dimension of the data in this FMM
    fn dim(&self) -> usize;

    /// Evaluate the potentials, or potential gradients, for this FMM
    fn evaluate(&self);

    /// Clear the data buffers and add new charge data for re-evaluation.
    ///
    /// # Arguments
    /// * `charges` - new charge data.
    fn clear(&mut self, charges: &Charges<Self::Scalar>);
}

/// Set all metadata required for FMMs
pub trait FmmMetadata {
    /// Associated scalar
    type Scalar: RlstScalar;

    /// TODO: Breakup into smaller pieces of functionality for clarity.
    fn metadata(&mut self, eval_type: EvalType, charges: &Charges<Self::Scalar>);
}

/// Kernels compatible with our implementation
pub trait FmmKernel
where
    Self: Kernel,
{
    /// Homogeneity check
    fn homogenous(&self) -> bool;

    /// Lookup p2m operator
    fn p2m_operator_index(&self, level: u64) -> usize;

    /// Lookup m2m operator
    fn m2m_operator_index(&self, level: u64) -> usize;
}
