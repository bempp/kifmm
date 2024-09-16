//! Field Traits

use rlst::RlstScalar;

use crate::tree::SingleNodeTree;
/// Marker trait for field translations
pub trait SourceToTargetData {
    /// Metadata for applying each to source to target translation, depends on both the kernel
    /// and translation method
    type Metadata;

    /// Can check and equivalent surfaces be overdetermined?
    fn overdetermined(&self) -> bool;

    /// Calculated as equivalent_surface_order - check_surface_order = surface_diff
    fn surface_diff(&self) -> usize;
}

/// Set M2M and L2L metadata associated with a kernel
pub trait SourceAndTargetTranslationMetadata {
    /// Source field translations
    fn source(&mut self);

    /// Target field translations
    fn target(&mut self);
}

/// Set M2L metadata associated with a kernel
pub trait SourceToTargetTranslationMetadata {
    /// Source to target field translation
    fn source_to_target(&mut self);

    /// Map between source/target nodes, indexed by level.
    fn displacements(&mut self);
}

/// Set M2L metadata associated with a kernel, for the Ghost data
pub trait SourceToTargetTranslationMetadataGhostTrees {
    /// Associated scalar type
    type Scalar: RlstScalar;

    /// Map between source/target nodes, sources are from Ghost tree, target are local target trees
    fn displacements(
        &mut self,
        target_trees: &[SingleNodeTree<<Self::Scalar as RlstScalar>::Real>],
        total_depth: u64,
        global_depth: u64,
    );
}
