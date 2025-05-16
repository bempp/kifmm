//! Traits for handling P2P kernel on GPUs and CPUs
use rlst::RlstScalar;

/// Trait for evaluating P2P between source/target boxes where all displacements and counts
/// Are provided
pub trait P2PKernel {

    /// Associated scalar type
    type Scalar: RlstScalar;

    /// Actual function TODO docs
    fn p2p_kernel(
        targets: &[<Self::Scalar as RlstScalar>::Real],
        targets_counts: &[usize],
        targets_displacements: &[usize],
        sources: &[<Self::Scalar as RlstScalar>::Real],
        sources_counts: &[usize],
        sources_displacements: &[usize],
        charges: &[Self::Scalar],
        result: &mut [Self::Scalar]
    );
}