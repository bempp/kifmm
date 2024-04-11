//! Implementation of FmmTree Trait

use super::types::SingleNodeFmmTree;
use crate::traits::tree::{FmmTree, Tree};
use crate::tree::types::{MortonKey, SingleNodeTree};
use crate::{Float, RlstScalarFloat};

impl<T> FmmTree for SingleNodeFmmTree<T>
where
    T: RlstScalarFloat<Real = T> + Float,
{
    type Scalar = T;
    type Node = MortonKey<T::Real>;
    type Tree = SingleNodeTree<T::Real>;

    fn source_tree(&self) -> &Self::Tree {
        &self.source_tree
    }

    fn target_tree(&self) -> &Self::Tree {
        &self.target_tree
    }

    fn domain(&self) -> &<Self::Tree as Tree>::Domain {
        &self.domain
    }

    fn near_field(&self, leaf: &Self::Node) -> Option<Vec<Self::Node>> {
        let mut u_list = Vec::new();
        let neighbours = leaf.neighbors();

        // Child level
        let mut neighbors_children_adj = neighbours
            .iter()
            .flat_map(|n| n.children())
            .filter(|nc| {
                self.source_tree().all_keys_set().unwrap().contains(nc) && leaf.is_adjacent(nc)
            })
            .collect();

        // Key level
        let mut neighbors_adj = neighbours
            .iter()
            .filter(|n| {
                self.source_tree().all_keys_set().unwrap().contains(n) && leaf.is_adjacent(n)
            })
            .cloned()
            .collect();

        // Parent level
        let mut parent_neighbours_adj = leaf
            .parent()
            .neighbors()
            .into_iter()
            .filter(|pn| {
                self.source_tree().all_keys_set().unwrap().contains(pn) && leaf.is_adjacent(pn)
            })
            .collect();

        u_list.append(&mut neighbors_children_adj);
        u_list.append(&mut neighbors_adj);
        u_list.append(&mut parent_neighbours_adj);
        u_list.push(*leaf);

        if !u_list.is_empty() {
            Some(u_list)
        } else {
            None
        }
    }
}

unsafe impl<T: RlstScalarFloat<Real = T> + Float> Send for SingleNodeFmmTree<T> {}
unsafe impl<T: RlstScalarFloat<Real = T> + Float> Sync for SingleNodeFmmTree<T> {}
