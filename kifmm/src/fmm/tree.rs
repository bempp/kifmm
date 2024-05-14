use num::traits::Float;
use rlst::RlstScalar;

use crate::{
    fmm::types::SingleNodeFmmTree,
    traits::tree::{FmmTree, Tree},
    tree::types::{MortonKey, SingleNodeTree},
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
        let u_list = leaf.neighbors();

        // Key level
        let mut u_list: Vec<MortonKey<_>> = u_list
            .into_iter()
            .filter(|n| {self.source_tree().all_keys_set().unwrap().contains(n)})
            .collect();

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
