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

    fn near_field(
        &self,
        leaf: &<Self::Tree as Tree>::Node,
    ) -> Option<Vec<<Self::Tree as Tree>::Node>> {
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
