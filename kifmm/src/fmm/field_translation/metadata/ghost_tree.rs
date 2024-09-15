//! Metadata for Ghost data

use mpi::traits::Equivalence;
use num::Float;
use rlst::RlstScalar;

use crate::{
    fmm::types::FftFieldTranslationMultiNode,
    traits::{fftw::Dft, field::SourceToTargetTranslationMetadataGhostTrees, general::AsComplex},
    tree::types::GhostTreeV,
};

impl<Scalar> SourceToTargetTranslationMetadataGhostTrees
    for GhostTreeV<Scalar, FftFieldTranslationMultiNode<Scalar>>
where
    Scalar: RlstScalar
        + Equivalence
        + AsComplex
        + Default
        + Float
        + Dft<InputType = Scalar, OutputType = <Scalar as AsComplex>::ComplexType>,
    <Scalar as RlstScalar>::Real: RlstScalar + Default + Equivalence + Float,
{
    fn displacements<T: RlstScalar + Float>(
        &mut self,
        target_trees: &[crate::tree::SingleNodeTree<T>],
    ) {
        let mut displacements = Vec::new();

        for target_tree_index in 0..target_trees.len() {}

        self.source_to_target.displacements = displacements;
    }
}
