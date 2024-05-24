//! Field Traits
/// Marker trait for field translations
pub trait SourceToTargetData {
    /// Metadata for applying each to source to target translation, depends on both the kernel
    /// and translation method
    type Metadata;
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

    fn displacements(&mut self);
}
