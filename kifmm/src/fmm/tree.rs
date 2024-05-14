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
        let mut u_list = Vec::new();
        let neighbours = leaf.neighbors();

        // Key level
        let mut neighbors_adj = neighbours
            .iter()
            .filter(|n| {
                self.source_tree().all_keys_set().unwrap().contains(n) && leaf.is_adjacent(n)
            })
            .cloned()
            .collect();

        u_list.append(&mut neighbors_adj);
        u_list.push(*leaf);

        if !u_list.is_empty() {
            Some(u_list)
        } else {
            None
        }
    }
}

unsafe impl<T: RlstScalar + Float + Default> Send for SingleNodeFmmTree<T> {}
unsafe impl<T: RlstScalar + Float + Default> Sync for SingleNodeFmmTree<T> {}
