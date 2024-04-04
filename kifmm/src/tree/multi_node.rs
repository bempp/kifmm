//! Implementation of constructors for multi node trees from distributed point data.

use crate::{
    traits::tree::Tree,
    tree::{
        constants::DEEPEST_LEVEL,
        morton::encode_anchor,
        types::{Domain, MortonKey, MortonKeys, MultiNodeTree, Point, Points, SingleNodeTree},
    },
};

use crate::hyksort::hyksort;
use itertools::Itertools;
use mpi::{
    topology::UserCommunicator,
    traits::{Communicator, CommunicatorCollectives, Equivalence},
};
use num::traits::Float;
use rlst::RlstScalar;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;

impl<T> MultiNodeTree<T>
where
    T: Float + Default + Equivalence + Debug + RlstScalar<Real = T>,
{
    /// Constructor for uniform trees.
    ///
    /// # Arguments
    /// * `world` - A global communicator for the tree.
    /// * `k` - Size of subcommunicator used in Hyksort. Must be a power of 2.
    /// * `points` - Cartesian point data in column major order.
    /// * `domain` - Domain associated with the global point set.
    /// * `depth` - The maximum depth of recursion for the tree.
    /// * `global_idxs` - Globally unique indices for point data.
    pub fn uniform_tree(
        world: &UserCommunicator,
        k: i32,
        points: &[T],
        domain: &Domain<T>,
        depth: u64,
        global_idxs: &[usize],
    ) -> MultiNodeTree<T> {
        let rank = world.rank();

        // Encode points at deepest level, and map to specified depth.
        let dim = 3;
        let npoints = points.len() / dim;

        let mut tmp = Points::default();
        for i in 0..npoints {
            let point = [points[i], points[i + npoints], points[i + 2 * npoints]];
            let base_key = MortonKey::from_point(&point, domain, DEEPEST_LEVEL);
            let encoded_key = MortonKey::from_point(&point, domain, depth);
            tmp.points.push(Point {
                coordinate: point,
                base_key,
                encoded_key,
                global_idx: global_idxs[i],
            })
        }

        let mut points = tmp;

        // 2.i Perform parallel Morton sort over encoded points
        let comm = world.duplicate();
        hyksort(&mut points.points, k, comm);

        // Define range owned by each processor

        // 2.ii Find leaf keys on each processor
        let mut min = points.points.iter().min().unwrap().encoded_key;
        if rank == 0 {
            min = min.siblings().into_iter().min().unwrap();
        }

        let mut max = points.points.iter().max().unwrap().encoded_key;
        if rank == world.size() - 1 {
            max = max.siblings().into_iter().max().unwrap();
        }

        let diameter = 1 << (DEEPEST_LEVEL - depth);

        // Find all leaves within this processor's range
        let leaves = MortonKeys {
            keys: (min.anchor[0]..max.anchor[0])
                .step_by(diameter)
                .flat_map(|i| {
                    (min.anchor[1]..max.anchor[1])
                        .step_by(diameter)
                        .map(move |j| (i, j))
                })
                .flat_map(|(i, j)| {
                    (min.anchor[2]..max.anchor[2])
                        .step_by(diameter)
                        .map(move |k| [i, j, k])
                })
                .map(|anchor| {
                    let morton = encode_anchor(&anchor, depth);
                    MortonKey { anchor, morton }
                })
                .collect(),
            index: 0,
        };

        // 3. Assign leaves to points
        let unmapped = SingleNodeTree::assign_nodes_to_points(&leaves, &mut points);

        // Group points by leaves
        points.sort();

        let mut leaves_to_coordinates = HashMap::new();
        let mut curr = points.points[0];
        let mut curr_idx = 0;

        for (i, point) in points.points.iter().enumerate() {
            if point.encoded_key != curr.encoded_key {
                leaves_to_coordinates.insert(curr.encoded_key, (curr_idx, i));
                curr_idx = i;
                curr = *point;
            }
        }
        leaves_to_coordinates.insert(curr.encoded_key, (curr_idx, points.points.len()));

        // // For sparse trees need to ensure that final leaf set contains all siblings of encoded trees if they
        // // are in range.
        // {
        //     let leaves: HashSet<MortonKey> = leaves_to_coordinates
        //         .keys()
        //         .flat_map(|k| k.siblings())
        //         .filter(|&sib| min <= sib && sib <= max)
        //         .collect();
        // }

        // Add unmapped leaves
        let leaves = MortonKeys {
            keys: leaves_to_coordinates
                .keys()
                .cloned()
                .chain(unmapped.iter().copied())
                .collect_vec(),
            index: 0,
        };

        // Find all keys in tree
        let tmp: HashSet<MortonKey> = leaves
            .iter()
            .flat_map(|leaf| leaf.ancestors().into_iter())
            .collect();

        let mut keys = MortonKeys {
            keys: tmp.into_iter().collect_vec(),
            index: 0,
        };

        let leaves_set: HashSet<MortonKey> = leaves.iter().cloned().collect();
        let keys_set: HashSet<MortonKey> = keys.iter().cloned().collect();

        let min = leaves.iter().min().unwrap();
        let max = leaves.iter().max().unwrap();
        let range = [world.rank() as u64, min.morton, max.morton];

        // Group by level to perform efficient lookup of nodes
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

        // Return tree in sorted order
        for l in 0..=depth {
            let &(l, r) = levels_to_keys.get(&l).unwrap();
            let subset = &mut keys[l..r];
            subset.sort();
        }

        let coordinates = points
            .points
            .iter()
            .map(|p| p.coordinate)
            .flat_map(|[x, y, z]| vec![x, y, z])
            .collect_vec();
        let global_indices = points.points.iter().map(|p| p.global_idx).collect_vec();

        let mut key_to_index = HashMap::new();

        for (i, key) in keys.iter().enumerate() {
            key_to_index.insert(*key, i);
        }

        let mut leaf_to_index = HashMap::new();

        for (i, key) in leaves.iter().enumerate() {
            leaf_to_index.insert(*key, i);
        }

        MultiNodeTree {
            world: world.duplicate(),
            depth,
            domain: *domain,
            coordinates,
            global_indices,
            leaves,
            keys,
            leaves_to_coordinates,
            key_to_index,
            leaf_to_index,
            levels_to_keys,
            leaves_set,
            keys_set,
            range,
        }
    }

    /// TODO: Docs
    pub fn uniform_tree_sparse(
        world: &UserCommunicator,
        k: i32,
        points: &[T],
        domain: &Domain<T>,
        depth: u64,
        global_idxs: &[usize],
    ) -> MultiNodeTree<T> {
        let rank = world.rank();

        // Encode points at deepest level, and map to specified depth.
        let dim = 3;
        let npoints = points.len() / dim;

        let mut tmp = Points::default();
        for i in 0..npoints {
            let point = [points[i], points[i + npoints], points[i + 2 * npoints]];
            let base_key = MortonKey::from_point(&point, domain, DEEPEST_LEVEL);
            let encoded_key = MortonKey::from_point(&point, domain, depth);
            tmp.points.push(Point {
                coordinate: point,
                base_key,
                encoded_key,
                global_idx: global_idxs[i],
            })
        }

        let mut points = tmp;

        // 2.i Perform parallel Morton sort over encoded points
        let comm = world.duplicate();
        hyksort(&mut points.points, k, comm);

        // Define range owned by each processor

        // 2.ii Find leaf keys on each processor
        let mut min = points.points.iter().min().unwrap().encoded_key;
        if rank == 0 {
            min = min.siblings().into_iter().min().unwrap();
        }

        let mut max = points.points.iter().max().unwrap().encoded_key;
        if rank == world.size() - 1 {
            max = max.siblings().into_iter().max().unwrap();
        }

        let diameter = 1 << (DEEPEST_LEVEL - depth);

        // Find all leaves within this processor's range
        let leaves = MortonKeys {
            keys: (min.anchor[0]..max.anchor[0])
                .step_by(diameter)
                .flat_map(|i| {
                    (min.anchor[1]..max.anchor[1])
                        .step_by(diameter)
                        .map(move |j| (i, j))
                })
                .flat_map(|(i, j)| {
                    (min.anchor[2]..max.anchor[2])
                        .step_by(diameter)
                        .map(move |k| [i, j, k])
                })
                .map(|anchor| {
                    let morton = encode_anchor(&anchor, depth);
                    MortonKey { anchor, morton }
                })
                .collect(),
            index: 0,
        };

        // 3. Assign leaves to points
        let unmapped = SingleNodeTree::assign_nodes_to_points(&leaves, &mut points);

        // Group points by leaves
        points.sort();

        let mut leaves_to_coordinates = HashMap::new();
        let mut curr = points.points[0];
        let mut curr_idx = 0;

        for (i, point) in points.points.iter().enumerate() {
            if point.encoded_key != curr.encoded_key {
                leaves_to_coordinates.insert(curr.encoded_key, (curr_idx, i));
                curr_idx = i;
                curr = *point;
            }
        }
        leaves_to_coordinates.insert(curr.encoded_key, (curr_idx, points.points.len()));

        // // For sparse trees need to ensure that final leaf set contains all siblings of encoded trees if they
        // // are in range.
        // {
        //     let leaves: HashSet<MortonKey> = leaves_to_coordinates
        //         .keys()
        //         .flat_map(|k| k.siblings())
        //         .filter(|&sib| min <= sib && sib <= max)
        //         .collect();
        // }

        // Add unmapped leaves
        let leaves = MortonKeys {
            keys: leaves_to_coordinates
                .keys()
                .cloned()
                .chain(unmapped.iter().copied())
                .collect_vec(),
            index: 0,
        };

        // Find all keys in tree
        let tmp: HashSet<MortonKey> = leaves
            .iter()
            .flat_map(|leaf| leaf.ancestors().into_iter())
            .collect();

        let mut keys = MortonKeys {
            keys: tmp.into_iter().collect_vec(),
            index: 0,
        };

        let leaves_set: HashSet<MortonKey> = leaves.iter().cloned().collect();
        let keys_set: HashSet<MortonKey> = keys.iter().cloned().collect();

        let min = leaves.iter().min().unwrap();
        let max = leaves.iter().max().unwrap();
        let range = [world.rank() as u64, min.morton, max.morton];

        // Group by level to perform efficient lookup of nodes
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

        // Return tree in sorted order
        for l in 0..=depth {
            let &(l, r) = levels_to_keys.get(&l).unwrap();
            let subset = &mut keys[l..r];
            subset.sort();
        }

        let coordinates = points
            .points
            .iter()
            .map(|p| p.coordinate)
            .flat_map(|[x, y, z]| vec![x, y, z])
            .collect_vec();
        let global_indices = points.points.iter().map(|p| p.global_idx).collect_vec();

        let mut key_to_index = HashMap::new();

        for (i, key) in keys.iter().enumerate() {
            key_to_index.insert(*key, i);
        }

        let mut leaf_to_index = HashMap::new();

        for (i, key) in leaves.iter().enumerate() {
            leaf_to_index.insert(*key, i);
        }

        MultiNodeTree {
            world: world.duplicate(),
            depth,
            domain: *domain,
            coordinates,
            global_indices,
            leaves,
            keys,
            leaves_to_coordinates,
            key_to_index,
            leaf_to_index,
            levels_to_keys,
            leaves_set,
            keys_set,
            range,
        }
    }

    /// Create a new multi-node tree. If non-adaptive (uniform) trees are created, they are specified
    /// by a user defined maximum depth, if an adaptive tree is created it is specified by only
    /// by the user defined maximum leaf maximum occupancy n_crit.

    /// # Arguments
    /// * `world` - A global communicator for the tree.
    /// * `hyksort_subcomm_size` - Size of subcommunicator used in Hyksort. Must be a power of 2.
    /// * `points` - Cartesian point data in column major order.
    /// * `domain` - Domain associated with the global point set.
    /// * `n_crit` - Maximum number of particles in a leaf node.
    /// * `global_idxs` - Globally unique indices for point data.
    pub fn new(
        points: &[T],
        depth: u64,
        sparse: bool,
        domain: Option<Domain<T>>,
        world: &UserCommunicator,
        hyksort_subcomm_size: i32,
    ) -> MultiNodeTree<T> {
        let dim = 3;
        let domain = domain.unwrap_or(Domain::from_global_points(points, world));
        let npoints = points.len() / dim;

        // Assign global indices
        let global_idxs = global_indices(npoints, world);

        if sparse {
            MultiNodeTree::uniform_tree_sparse(
                world,
                hyksort_subcomm_size,
                points,
                &domain,
                depth,
                &global_idxs,
            )
        } else {
            MultiNodeTree::uniform_tree(
                world,
                hyksort_subcomm_size,
                points,
                &domain,
                depth,
                &global_idxs,
            )
        }
    }
}

/// Assign global indices to points owned by each process
fn global_indices(npoints: usize, comm: &UserCommunicator) -> Vec<usize> {
    // Gather counts of coordinates at each process
    let rank = comm.rank() as usize;

    let nprocs = comm.size() as usize;
    let mut counts = vec![0usize; nprocs];
    comm.all_gather_into(&npoints, &mut counts[..]);

    // Compute displacements
    let mut displacements = vec![0usize; nprocs];

    for i in 1..nprocs {
        displacements[i] = displacements[i - 1] + counts[i - 1]
    }

    // Assign unique global indices to all coordinates
    let mut global_indices = vec![0usize; npoints];

    for (i, global_index) in global_indices.iter_mut().enumerate().take(npoints) {
        *global_index = displacements[rank] + i;
    }

    global_indices
}

impl<T> Tree for MultiNodeTree<T>
where
    T: Float + Default + RlstScalar<Real = T>,
{
    type Scalar = T;
    type Domain = Domain<T>;
    type Node = MortonKey;
    type NodeSlice<'a> = &'a [MortonKey]
        where T: 'a;
    type Nodes = MortonKeys;

    fn ncoordinates(&self, key: &Self::Node) -> Option<usize> {
        self.coordinates(key).map(|coords| coords.len() / 3)
    }

    fn ncoordinates_tot(&self) -> Option<usize> {
        self.all_coordinates().map(|coords| coords.len() / 3)
    }

    fn node(&self, idx: usize) -> Option<&Self::Node> {
        Some(&self.keys[idx])
    }

    fn nkeys_tot(&self) -> Option<usize> {
        Some(self.keys.len())
    }

    fn nkeys(&self, level: u64) -> Option<usize> {
        if let Some((l, r)) = self.levels_to_keys.get(&level) {
            Some(r - l)
        } else {
            None
        }
    }

    fn nleaves(&self) -> Option<usize> {
        Some(self.leaves.len())
    }

    fn depth(&self) -> u64 {
        self.depth
    }

    fn domain(&self) -> &'_ Self::Domain {
        &self.domain
    }

    fn keys(&self, level: u64) -> Option<Self::NodeSlice<'_>> {
        if let Some(&(l, r)) = self.levels_to_keys.get(&level) {
            Some(&self.keys[l..r])
        } else {
            None
        }
    }

    fn all_keys(&self) -> Option<Self::NodeSlice<'_>> {
        Some(&self.keys)
    }

    fn all_keys_set(&self) -> Option<&'_ HashSet<Self::Node>> {
        Some(&self.keys_set)
    }

    fn all_leaves_set(&self) -> Option<&'_ HashSet<Self::Node>> {
        Some(&self.leaves_set)
    }

    fn all_leaves(&self) -> Option<Self::NodeSlice<'_>> {
        Some(&self.leaves)
    }

    fn coordinates(&self, key: &Self::Node) -> Option<&[Self::Scalar]> {
        if let Some(&(l, r)) = self.leaves_to_coordinates.get(key) {
            Some(&self.coordinates[l * 3..r * 3])
        } else {
            None
        }
    }

    fn all_coordinates(&self) -> Option<&[Self::Scalar]> {
        Some(&self.coordinates)
    }

    fn global_indices(&self, key: &Self::Node) -> Option<&[usize]> {
        if let Some(&(l, r)) = self.leaves_to_coordinates.get(key) {
            Some(&self.global_indices[l..r])
        } else {
            None
        }
    }

    fn all_global_indices(&self) -> Option<&[usize]> {
        Some(&self.global_indices)
    }

    fn index(&self, key: &Self::Node) -> Option<&usize> {
        self.key_to_index.get(key)
    }

    fn leaf_index(&self, key: &Self::Node) -> Option<&usize> {
        self.leaf_to_index.get(key)
    }
}
