//! Field Traits
use green_kernels::traits::Kernel;
use rlst::RlstScalar;

use super::tree::Domain;

/// Marker trait for field translations
pub trait SourceToTargetData {
    /// Metadata for applying each to source to target translation, depends on both the kernel
    /// and translation method
    type Metadata;
}

/// Interface for configuration of field translation data used during FMM construction
pub trait ConfigureSourceToTargetData
where
    Self: SourceToTargetData,
    Self::Scalar: RlstScalar,
{
    /// Scalar type
    type Scalar;

    /// Kernel function associated with field translation
    type Kernel: Kernel<T = Self::Scalar>;

    /// The computational domain defining the tree.
    type Domain: Domain<Scalar = <Self::Scalar as RlstScalar>::Real>;

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
    fn kernel(&mut self, kernel: Self::Kernel);
}
