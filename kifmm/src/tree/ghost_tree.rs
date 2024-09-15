use std::collections::{HashMap, HashSet};

use itertools::Itertools;
use mpi::traits::Source;
use num::{Float, One, Zero};
use pulp::Scalar;
use rlst::RlstScalar;

use crate::{
    fmm::types::IndexPointer,
    traits::{field::SourceToTargetData, tree::SingleNodeTreeTrait},
};

use super::{
    constants::DEEPEST_LEVEL,
    domain,
    types::{Domain, GhostTreeU, GhostTreeV, MortonKey, MortonKeys, Point, Points},
};

impl<T> GhostTreeU<T>
where
    T: RlstScalar + Float,
{
    /// Constructor
    pub fn from_ghost_data(
        depth: u64,
        domain: &Domain<T::Real>,
        ghost_coordinates: Vec<T::Real>,
    ) -> Result<Self, std::io::Error> {
        let mut result = Self::default();

        let dim = 3;
        let n_coords = ghost_coordinates.len() / dim;

        // Convert column major coordinate into `Point`, containing Morton encoding
        let mut points: Vec<Point<T::Real>> = Points::default();
        for i in 0..n_coords {
            let coord: &[T::Real; 3] = &ghost_coordinates[i * dim..(i + 1) * dim]
                .try_into()
                .unwrap();

            let base_key = MortonKey::<T::Real>::from_point(coord, &domain, DEEPEST_LEVEL, None);
            let encoded_key = MortonKey::<T::Real>::from_point(coord, &domain, depth, None);
            points.push(Point {
                coordinate: *coord,
                base_key,
                encoded_key,
                global_index: 0,
            })
        }

        // Morton sort over points
        points.sort();
        // Group coordinates by leaves
        let mut leaves_to_coordinates = HashMap::new();
        let mut curr = points[0];
        let mut curr_idx = 0;

        for (i, point) in points.iter().enumerate() {
            if point.encoded_key != curr.encoded_key {
                leaves_to_coordinates.insert(curr.encoded_key, (curr_idx, i));
                curr_idx = i;
                curr = *point;
            }
        }

        leaves_to_coordinates.insert(curr.encoded_key, (curr_idx, points.len()));

        // Ensure that final leaf set contains siblings of all encoded keys
        let leaves_set: HashSet<MortonKey<_>> = leaves_to_coordinates
            .keys()
            .flat_map(|k| k.siblings())
            .collect();

        // Sort leaves before returning
        let mut leaves = leaves_set.iter().cloned().collect_vec();
        leaves.sort();
        let leaves = MortonKeys::from(leaves);

        let mut leaf_to_index = HashMap::new();

        for (i, key) in leaves.iter().enumerate() {
            leaf_to_index.insert(*key, i);
        }

        // TODO global indices
        let charges = vec![T::one(); n_coords];

        result.leaves = leaves;
        result.leaves_set = leaves_set;
        result.coordinates = ghost_coordinates;
        result.leaf_to_index = leaf_to_index;
        result.leaves_to_coordinates = leaves_to_coordinates;
        result.charges = charges;
        Ok(result)
    }
}

impl<T, V> GhostTreeV<T, V>
where
    T: RlstScalar + Float,
    V: SourceToTargetData + Default,
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

impl<T: RlstScalar + Float> SingleNodeTreeTrait for GhostTreeU<T> {
    type Domain = Domain<T::Real>;
    type Node = MortonKey<T::Real>;
    type Scalar = T::Real;

    fn all_coordinates(&self) -> Option<&[Self::Scalar]> {
        Some(&self.coordinates)
    }

    fn all_global_indices(&self) -> Option<&[usize]> {
        None
    }

    fn all_keys(&self) -> Option<&[Self::Node]> {
        None
    }

    fn all_keys_set(&self) -> Option<&'_ HashSet<Self::Node>> {
        None
    }

    fn all_leaves(&self) -> Option<&[Self::Node]> {
        Some(&self.leaves)
    }

    fn all_leaves_set(&self) -> Option<&'_ HashSet<Self::Node>> {
        Some(&self.leaves_set)
    }

    fn contributing_range(&self) -> Option<[Self::Node; 2]> {
        None
    }

    fn coordinates(&self, leaf: &Self::Node) -> Option<&[Self::Scalar]> {
        if let Some(&(l, r)) = self.leaves_to_coordinates.get(leaf) {
            Some(&self.coordinates[l * 3..r * 3])
        } else {
            None
        }
    }

    fn depth(&self) -> u64 {
        self.depth
    }

    fn domain(&self) -> Option<&Self::Domain> {
        None
    }

    fn global_indices(&self, leaf: &Self::Node) -> Option<&[usize]> {
        None
    }

    fn index(&self, key: &Self::Node) -> Option<&usize> {
        None
    }

    fn level_index(&self, key: &Self::Node) -> Option<&usize> {
        None
    }

    fn keys(&self, level: u64) -> Option<&[Self::Node]> {
        None
    }

    fn leaf_index(&self, leaf: &Self::Node) -> Option<&usize> {
        self.leaf_to_index.get(leaf)
    }

    fn n_coordinates(&self, leaf: &Self::Node) -> Option<usize> {
        self.coordinates(leaf).map(|coords| coords.len() / 3)
    }

    fn n_coordinates_tot(&self) -> Option<usize> {
        self.all_coordinates().map(|coords| coords.len() / 3)
    }

    fn n_keys(&self, level: u64) -> Option<usize> {
        None
    }

    fn n_keys_tot(&self) -> Option<usize> {
        None
    }

    fn n_leaves(&self) -> Option<usize> {
        Some(self.leaves.len())
    }

    fn node(&self, idx: usize) -> Option<&Self::Node> {
        None
    }

    fn owned_range(&self) -> Option<Self::Node> {
        None
    }
}

impl<T: RlstScalar + Float, V: SourceToTargetData> SingleNodeTreeTrait for GhostTreeV<T, V> {
    type Domain = Domain<T::Real>;
    type Node = MortonKey<T::Real>;
    type Scalar = T::Real;

    fn all_coordinates(&self) -> Option<&[Self::Scalar]> {
        None
    }

    fn all_global_indices(&self) -> Option<&[usize]> {
        None
    }

    fn all_keys(&self) -> Option<&[Self::Node]> {
        Some(&self.keys)
    }

    fn all_keys_set(&self) -> Option<&'_ HashSet<Self::Node>> {
        Some(&self.keys_set)
    }

    fn all_leaves(&self) -> Option<&[Self::Node]> {
        None
    }

    fn all_leaves_set(&self) -> Option<&'_ HashSet<Self::Node>> {
        None
    }

    fn contributing_range(&self) -> Option<[Self::Node; 2]> {
        None
    }

    fn coordinates(&self, leaf: &Self::Node) -> Option<&[Self::Scalar]> {
        None
    }

    fn depth(&self) -> u64 {
        self.depth
    }

    fn domain(&self) -> Option<&Self::Domain> {
        None
    }

    fn global_indices(&self, leaf: &Self::Node) -> Option<&[usize]> {
        None
    }

    fn index(&self, key: &Self::Node) -> Option<&usize> {
        self.key_to_index.get(key)
    }

    fn keys(&self, level: u64) -> Option<&[Self::Node]> {
        if let Some(&(l, r)) = self.levels_to_keys.get(&level) {
            Some(&self.keys[l..r])
        } else {
            None
        }
    }

    fn leaf_index(&self, leaf: &Self::Node) -> Option<&usize> {
        None
    }

    fn level_index(&self, key: &Self::Node) -> Option<&usize> {
        self.key_to_level_index.get(key)
    }

    fn n_coordinates(&self, leaf: &Self::Node) -> Option<usize> {
        None
    }

    fn n_coordinates_tot(&self) -> Option<usize> {
        None
    }

    fn n_keys(&self, level: u64) -> Option<usize> {
        if let Some(&(l, r)) = self.levels_to_keys.get(&level) {
            Some(r - l)
        } else {
            None
        }
    }

    fn n_keys_tot(&self) -> Option<usize> {
        Some(self.keys.len())
    }

    fn n_leaves(&self) -> Option<usize> {
        None
    }

    fn node(&self, idx: usize) -> Option<&Self::Node> {
        Some(&self.keys[idx])
    }

    fn owned_range(&self) -> Option<Self::Node> {
        None
    }
}
