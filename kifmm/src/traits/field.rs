//! Field Traits
/// Marker trait for field translations
pub trait SourceToTargetData {
    /// Metadata for applying each to source to target translation, depends on both the kernel
    /// and translation method
    type Metadata;
}

/// Set metadata associated with a kernel
pub trait SourceAndTargetTranslationMetadata {
    // Source field translations
    fn source(&mut self);

    // Target field translations
    fn target(&mut self);
}

/// Set metadata associated with a kernel
pub trait SourcetoTargetTranslationMetadata {
    fn field_translation(&mut self);
}
