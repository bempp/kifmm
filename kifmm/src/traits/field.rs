//! Field traits
use green_kernels::traits::Kernel;

/// Interface for handling field translation data and metadata
pub trait SourceToTargetData<T>
where
    T: Kernel,
{
    /// Metadata for applying each to source to target translation, depends on both the kernel
    /// and translation method
    type OperatorData;

    /// The computational domain defining the tree.
    type Domain;

    /// Set the field translation operators corresponding to each unique transfer vector.
    ///
    /// # Arguments
    /// * `expansion_order` - The expansion order of the FMM
    /// * `domain` - Domain associated with the global point set.
    fn operator_data(&mut self, expansion_order: usize, domain: Self::Domain);

    /// Set expansion order
    ///
    /// # Arguments
    /// * `expansion_order` - The expansion order of the FMM
    fn expansion_order(&mut self, expansion_order: usize);

    /// Set the associated kernel
    ///
    /// # Arguments
    /// * `kernel` - The kernel being used
    fn kernel(&mut self, kernel: T);
}
