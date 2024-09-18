use std::collections::HashMap;

use mpi::traits::{Communicator, Equivalence};
use num::Float;
use rlst::RlstScalar;

use crate::{
    traits::tree::MultiTree,
    tree::{types::MortonKey, MultiNodeTree},
};

/// Create index pointers for each key at each level of an octree
pub fn level_index_pointer_multi_node<T, C>(
    tree: &MultiNodeTree<T, C>,
) -> Vec<HashMap<MortonKey<T>, usize>>
where
    T: RlstScalar + Float + Equivalence,
    C: Communicator,
{
    let mut result = vec![HashMap::new(); (tree.total_depth() + 1).try_into().unwrap()];

    for level in 0..=tree.total_depth() {
        if let Some(keys) = tree.keys(level) {
            for (level_idx, key) in keys.iter().enumerate() {
                result[level as usize].insert(*key, level_idx);
            }
        }
    }
    result
}
