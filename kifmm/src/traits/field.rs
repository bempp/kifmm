//! Field Traits

use super::types::FmmError;
/// Marker trait for field translations
pub trait FieldTranslation {
    /// Metadata for applying each to source to target translation, depends on both the kernel
    /// and translation method
    type Metadata;

    /// Does this field translation support overdetermined check surfaces?
    fn overdetermined(&self) -> bool;

    /// Calculated as equivalent_surface_order - check_surface_order = surface_diff
    fn surface_diff(&self) -> usize;
}

/// Source tree translation metadata
pub trait SourceTranslationMetadata {
    /// Constructor
    fn source(&mut self);
}

/// Target tree translation metadata
pub trait TargetTranslationMetadata {
    /// Constructor
    fn target(&mut self);
}

/// Set M2L metadata associated with a kernel
pub trait SourceToTargetTranslationMetadata {
    /// Source to target field translation
    fn source_to_target(&mut self);

    /// Map between source/target nodes, indexed by level.
    fn displacements(&mut self, start_level: Option<u64>);
}

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
