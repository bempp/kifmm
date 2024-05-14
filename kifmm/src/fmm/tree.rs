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

    // fn near_field(
    //     &self,
    //     leaf: &<Self::Tree as Tree>::Node,
    // ) -> Option<Vec<<Self::Tree as Tree>::Node>> {
    // // Collect the filtered neighbors directly into a vector
    // let all_keys_set = self.source_tree().all_keys_set().unwrap();
    // let mut u_list: Vec<_> = leaf
    //     .neighbors()
    //     .into_iter()
    //     .filter(|n| all_keys_set.contains(n))
    //     .collect();

    // // Conditionally push leaf if u_list is not empty
    // if !u_list.is_empty() {
    //     u_list.push(*leaf);
    //     Some(u_list)
    // } else {
    //     None
    // }
    // }
    fn near_field(&self, leaf: &<Self::Tree as Tree>::Node) -> Vec<<Self::Tree as Tree>::Node> {
        // Reserve capacity for the vector based on the expected number of neighbors
        let mut u_list = Vec::with_capacity(27); // Assuming a max of 26 neighbors plus the leaf

        let all_keys_set = self.source_tree().all_keys_set().unwrap();

        // Collect neighbors that exist in the all_keys_set
        for neighbor in leaf.neighbors() {
            if all_keys_set.contains(&neighbor) {
                u_list.push(neighbor);
            }
        }

        // Push the leaf into the vector
        u_list.push(*leaf);

        // Return the vector if it contains more than just the leaf
        u_list
    }

    // fn near_field(
    //     &self,
    //     leaf: &<Self::Tree as Tree>::Node,
    // ) -> Option<Vec<<Self::Tree as Tree>::Node>> {
    //     let all_keys_set = match self.source_tree().all_keys_set() {
    //         Some(set) => set,
    //         None => return None, // Handle the error appropriately
    //     };

    //     // Collect neighbors directly into the vector
    //     let mut u_list = Vec::with_capacity(28); // Reserve capacity for neighbors + leaf

    //     let neighbors = leaf.neighbors();
    //     if neighbors.is_empty() {
    //         return None; // Early return if there are no neighbors
    //     }

    //     for neighbor in neighbors {
    //         if all_keys_set.contains(&neighbor) {
    //             u_list.push(neighbor);
    //         }
    //     }

    //     // Push the leaf into the vector
    //     u_list.push(*leaf);

    //     // Return the vector if it contains more than just the leaf
    //     if u_list.len() > 1 {
    //         Some(u_list)
    //     } else {
    //         None
    //     }
    // }
}

unsafe impl<T: RlstScalar + Float + Default> Send for SingleNodeFmmTree<T> {}
unsafe impl<T: RlstScalar + Float + Default> Sync for SingleNodeFmmTree<T> {}
