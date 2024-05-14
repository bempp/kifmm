//! Implementation of FMM compatible trees
use num::traits::Float;
use rlst::RlstScalar;

use crate::{
    fmm::types::SingleNodeFmmTree,
    traits::tree::{FmmTree, Tree},
    tree::types::SingleNodeTree,
};

impl<T> FmmTree for SingleNodeFmmTree<T>
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

    fn domain(&self) -> &<Self::Tree as Tree>::Domain {
        &self.domain
    }

    fn near_field(&self, leaf: &<Self::Tree as Tree>::Node) -> Vec<<Self::Tree as Tree>::Node> {
        // Reserve capacity for the vector based on the expected number of neighbors
        // let mut u_list = Vec::with_capacity(27);

        let all_keys_set = self.source_tree().all_keys_set().unwrap();
        // Collect neighbors that exist in the all_keys_set and the leaf itself
        let mut u_list: Vec<_> = leaf.neighbors()
            .into_iter()
            .filter(|neighbor| all_keys_set.contains(neighbor))
            .collect();

        // Push the leaf into the vector
        u_list.push(leaf.clone());

        // Return the vector if it contains more than just the leaf
        u_list
    }

}

unsafe impl<T: RlstScalar + Float + Default> Send for SingleNodeFmmTree<T> {}
unsafe impl<T: RlstScalar + Float + Default> Sync for SingleNodeFmmTree<T> {}
