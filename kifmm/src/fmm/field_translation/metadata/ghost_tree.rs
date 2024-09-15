//! Metadata for Ghost data

use std::{collections::HashSet, sync::RwLock};

use itertools::Itertools;
use mpi::traits::Equivalence;
use num::Float;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rlst::RlstScalar;

use crate::{
    fmm::types::FftFieldTranslationMultiNode,
    traits::{
        fftw::Dft, field::SourceToTargetTranslationMetadataGhostTrees, general::AsComplex,
        tree::SingleNodeTreeTrait,
    },
    tree::{
        constants::{NHALO, NSIBLINGS},
        types::{GhostTreeV, MortonKey},
    },
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
    type Scalar = Scalar;

    fn displacements(
        &mut self,
        target_trees: &[crate::tree::SingleNodeTree<<Self::Scalar as RlstScalar>::Real>],
        total_depth: u64,
        global_depth: u64,
    ) {
        let mut displacements = Vec::new();

        for target_tree_index in 0..target_trees.len() {
            let target_tree = &target_trees[target_tree_index];
            let mut result_t = Vec::new();

            for level in global_depth..total_depth {
                let result_l = vec![Vec::new(); NHALO];
                let result_l = result_l.into_iter().map(RwLock::new).collect_vec();

                let targets = target_tree.keys(level).unwrap();
                let targets_parents: HashSet<MortonKey<_>> =
                    targets.iter().map(|target| target.parent()).collect();
                let mut targets_parents = targets_parents.into_iter().collect_vec();
                targets_parents.sort();
                let ntargets_parents = targets_parents.len();

                let sources = self.keys(level).unwrap();

                let sources_parents: HashSet<MortonKey<_>> =
                    sources.iter().map(|source| source.parent()).collect();
                let mut sources_parents = sources_parents.into_iter().collect_vec();
                sources_parents.sort();
                let nsources_parents = sources_parents.len();

                let targets_parents_neighbors = targets_parents
                    .iter()
                    .map(|parent| parent.all_neighbors())
                    .collect_vec();

                let zero_displacement = nsources_parents * NSIBLINGS;
                (0..NHALO).into_par_iter().for_each(|i| {
                    let mut result_li = result_l[i].write().unwrap();
                    for all_neighbors in targets_parents_neighbors.iter().take(ntargets_parents) {
                        // Check if neighbor exists in a valid tree
                        if let Some(neighbor) = all_neighbors[i] {
                            // If it does, check if first child exists in the source tree
                            let first_child = neighbor.first_child();
                            if let Some(neighbor_displacement) = self.level_index_pointer_multipoles
                                [level as usize]
                                .get(&first_child)
                            {
                                result_li.push(*neighbor_displacement)
                            } else {
                                result_li.push(zero_displacement)
                            }
                        } else {
                            result_li.push(zero_displacement)
                        }
                    }
                });
                result_t.push(result_l)
            }

            displacements.push(result_t)
        }

        // Wrap by another one as there is only one source tree, a HACK
        // TODO: Remove hack
        self.source_to_target.displacements = vec![displacements];
    }
}
