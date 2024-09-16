//! Implementation of FMM compatible trees
use mpi::traits::{Communicator, Equivalence};
use num::traits::Float;
use rlst::RlstScalar;

use crate::{
    fmm::types::SingleNodeFmmTree,
    traits::tree::{MultiNodeFmmTreeTrait, SingleNodeFmmTreeTrait, SingleNodeTreeTrait},
    tree::types::MultiNodeTree,
    tree::types::SingleNodeTree,
    MultiNodeFmmTree,
};

impl<T, C> MultiNodeFmmTreeTrait for MultiNodeFmmTree<T, C>
where
    T: RlstScalar + Float + Default + Equivalence,
    C: Communicator,
{
    type Tree = MultiNodeTree<T, C>;

    fn rank(&self) -> i32 {
        self.source_tree.rank
    }

    fn target_tree(&self) -> &Self::Tree {
        &self.target_tree
    }

    fn source_tree(&self) -> &Self::Tree {
        &self.source_tree
    }

    fn n_source_trees(&self) -> usize {
        self.source_tree.trees.len()
    }

    fn n_target_trees(&self) -> usize {
        self.target_tree.trees.len()
    }
}

impl<T> SingleNodeFmmTreeTrait for SingleNodeFmmTree<T>
where
    T: RlstScalar + Float + Default,
{
    type Tree = SingleNodeTree<T>;

    fn source_tree(&self) -> &Self::Tree {
        &self.source_tree
    }

    fn target_tree(&self) -> &Self::Tree {
        &self.target_tree
    }

    fn domain(&self) -> &<Self::Tree as SingleNodeTreeTrait>::Domain {
        &self.domain
    }

    fn near_field(
        &self,
        leaf: &<Self::Tree as SingleNodeTreeTrait>::Node,
    ) -> Option<Vec<<Self::Tree as SingleNodeTreeTrait>::Node>> {
        // Get the all_keys_set if it exists
        self.source_tree().all_keys_set().map(|all_keys_set| {
            // Collect neighbors that exist in the all_keys_set and push the leaf into the vector
            let mut u_list: Vec<_> = leaf
                .neighbors()
                .into_iter()
                .filter(|neighbor| all_keys_set.contains(neighbor))
                .collect();

            // Push the leaf into the vector
            u_list.push(*leaf);

            u_list
        })
    }
}

unsafe impl<T: RlstScalar + Float + Default> Send for SingleNodeFmmTree<T> {}
unsafe impl<T: RlstScalar + Float + Default> Sync for SingleNodeFmmTree<T> {}
