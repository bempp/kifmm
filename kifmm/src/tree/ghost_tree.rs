use std::collections::{HashMap, HashSet};

use itertools::Itertools;
use num::Float;
use pulp::Scalar;
use rlst::RlstScalar;

use crate::fmm::types::IndexPointer;

use super::{
    domain,
    types::{Domain, GhostTreeU, GhostTreeV, MortonKey, MortonKeys},
};

impl<T> GhostTreeU<T>
where
    T: RlstScalar + Float,
{
    /// Constructor
    pub fn from_ghost_data(
        mut ghost_leaves: Vec<MortonKey<T>>,
        ghost_index_pointers: Vec<IndexPointer>,
        ghost_coordinates: Vec<T::Real>,
    ) -> Result<Self, std::io::Error> {
        let mut result = Self::default();

        let leaves = MortonKeys::from(ghost_leaves);
        let leaves_set: HashSet<_> = leaves.iter().cloned().collect();

        let mut leaf_to_index = HashMap::new();

        for (i, key) in leaves.iter().enumerate() {
            leaf_to_index.insert(*key, i);
        }

        let mut leaves_to_coordinates = HashMap::new();

        for (key, index_pointer) in leaves.iter().zip(ghost_index_pointers) {
            leaves_to_coordinates
                .insert(*key, (index_pointer.0 as usize, index_pointer.1 as usize));
        }

        result.leaves = leaves;
        result.leaves_set = leaves_set;
        result.coordinates = ghost_coordinates;
        result.leaf_to_index = leaf_to_index;
        result.leaves_to_coordinates = leaves_to_coordinates;
        Ok(result)
    }
}

impl<T> GhostTreeV<T>
where
    T: RlstScalar + Float,
{
    /// Convert ghost data into a tree like object
    pub fn from_ghost_data(
        mut ghost_keys: Vec<MortonKey<T::Real>>,
        ghost_multipoles: Vec<T>,
        depth: u64,
        ncoeffs_equivalent_surface: usize,
    ) -> Result<Self, std::io::Error> {
        let mut result = Self::default();
        let mut keys_map = HashMap::new();
        let nkeys = ghost_keys.len();

        // Need current index in order to re-allocate received multipole data into Morton order
        for (old_idx, &key) in ghost_keys.iter().enumerate() {
            keys_map.insert(key, old_idx);
        }

        // Sort ghost keys into Morton order
        ghost_keys.sort();

        // Create a new index map for sorted octants
        let mut new_keys_map = HashMap::new();

        for (new_idx, &key) in ghost_keys.iter().enumerate() {
            new_keys_map.insert(key, new_idx);
        }

        // Re-allocate ghost multipole data in Morton order
        let mut multipoles = vec![T::default(); nkeys * ncoeffs_equivalent_surface];

        for (key, &key_idx) in keys_map.iter() {
            // Lookup old multipole index from ghost data
            let old_multipole = &ghost_multipoles
                [key_idx * ncoeffs_equivalent_surface..(key_idx + 1) * ncoeffs_equivalent_surface];

            // Assign new sorted index
            let &new_idx = new_keys_map.get(key).unwrap();

            // Copy to sorted multipoles buffer
            multipoles
                [new_idx * ncoeffs_equivalent_surface..(new_idx + 1) * ncoeffs_equivalent_surface]
                .copy_from_slice(old_multipole);
        }

        // Create morton keys objects
        let mut keys = MortonKeys::from(ghost_keys);
        let keys_set: HashSet<MortonKey<_>> = keys_map.keys().cloned().collect();

        // Group by level to perform efficient lookup
        keys.sort_by_key(|a| a.level());

        let mut levels_to_keys = HashMap::new();
        let mut curr = keys[0];
        let mut curr_idx = 0;
        for (i, key) in keys.iter().enumerate() {
            if key.level() != curr.level() {
                levels_to_keys.insert(curr.level(), (curr_idx, i));
                curr_idx = i;
                curr = *key;
            }
        }

        levels_to_keys.insert(curr.level(), (curr_idx, keys.len()));

        // Return tree in sorted order, by level and then by Morton key
        for l in 0..=depth {
            if let Some(&(l, r)) = levels_to_keys.get(&l) {
                let subset = &mut keys[l..r];
                subset.sort();
            }
        }

        // Compute key to level index
        let mut key_to_level_index = HashMap::new();
        for l in 0..=depth {
            if let Some(&(l, r)) = levels_to_keys.get(&l) {
                let keys = &keys[l..r];
                for (i, key) in keys.iter().enumerate() {
                    key_to_level_index.insert(*key, i);
                }
            }
        }

        // Map between keys/leaves and their respective indices
        let mut key_to_index = HashMap::new();

        for (i, key) in keys.iter().enumerate() {
            key_to_index.insert(*key, i);
        }

        // TODO: GLOBAL INDICES
        result.depth = depth;
        result.keys = keys;
        result.levels_to_keys = levels_to_keys;
        result.key_to_index = key_to_index;
        result.key_to_level_index = key_to_level_index;
        result.keys_set = keys_set;
        result.multipoles = multipoles;

        Ok(result)
    }
}