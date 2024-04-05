//! Implementation of constructors for multi node trees from distributed point data.

use crate::{
    traits::tree::Tree,
    tree::{
        constants::DEEPEST_LEVEL,
        types::{Domain, MortonKey, MortonKeys, MultiNodeTree, Point, Points, SingleNodeTree},
    },
};

use crate::hyksort::hyksort;
use itertools::Itertools;
use mpi::{
    topology::UserCommunicator,
    traits::{Communicator, CommunicatorCollectives, Destination, Equivalence, Source},
    Rank,
};
use num::traits::Float;
use rlst::RlstScalar;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;

use super::{constants::ROOT, morton::complete_region};

impl<T> MultiNodeTree<T>
where
    T: Float + Default + Equivalence + Debug + RlstScalar<Real = T>,
{
    /// Constructor for uniform trees.
    ///
    /// # Arguments
    /// * `world` - A global communicator for the tree.
    /// * `hyksort_subcomm_size` - Size of subcommunicator used in Hyksort. Must be a power of 2.
    /// * `points` - Cartesian point data in column major order.
    /// * `domain` - Domain associated with the global point set.
    /// * `depth` - The maximum depth of recursion for the tree.
    /// * `global_idxs` - Globally unique indices for point data.
    pub fn uniform_tree(
        world: &UserCommunicator,
        hyksort_subcomm_size: i32,
        coordinates: &[T],
        domain: &Domain<T>,
        depth: u64,
        global_idxs: &[usize],
    ) -> Result<MultiNodeTree<T>, std::io::Error> {
        let size = world.size();
        let rank = world.rank();

        // Encode points at deepest level, and map to specified depth.
        let dim = 3;
        let n_points = coordinates.len() / dim;

        let mut points = Points::default();
        for i in 0..n_points {
            let point = [
                coordinates[i],
                coordinates[i + n_points],
                coordinates[i + 2 * n_points],
            ];
            let base_key = MortonKey::from_point(&point, domain, DEEPEST_LEVEL);
            let encoded_key = MortonKey::from_point(&point, domain, depth);
            points.push(Point {
                coordinate: point,
                base_key,
                encoded_key,
                global_idx: global_idxs[i],
            })
        }

        // 2.i Perform parallel Morton sort over encoded points
        let comm = world.duplicate();
        hyksort(&mut points, hyksort_subcomm_size, comm)?;

        let leaves: HashSet<MortonKey> = points.iter().map(|p| p.encoded_key).collect();
        let mut leaves = MortonKeys::from(leaves);
        leaves.complete();

        // 2.ii Find leaf keys on each processor
        let mut seeds = SingleNodeTree::<T>::find_seeds(&leaves);

        let blocktree = Self::complete_blocktree(&mut seeds, rank, size, world);

        Self::transfer_points_to_blocktree(world, &points, &seeds, &rank, &size);

        // Split blocks to required depth
        let mut leaves = MortonKeys::new();
        for block in blocktree.iter() {
            let level_diff = depth - block.level();
            leaves.append(&mut block.descendants(level_diff).unwrap())
        }

        let min = leaves.iter().min().unwrap();
        let max = leaves.iter().max().unwrap();

        // 3. Assign leaves to points
        let _unmapped = SingleNodeTree::assign_nodes_to_points(&leaves, &mut points);

        // Group points by leaves
        points.sort();

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

        // Find all keys in tree
        let range = [world.rank() as u64, min.morton, max.morton];

        let tmp: HashSet<MortonKey> = leaves
            .iter()
            .flat_map(|leaf| leaf.ancestors().into_iter())
            .collect();

        let tmp: HashSet<MortonKey> = tmp
            .iter()
            .flat_map(|key| {
                if key.level() != 0 {
                    key.siblings()
                } else {
                    vec![*key]
                }
            })
            .collect();

        let mut keys = MortonKeys::from(tmp);

        let leaves_set: HashSet<MortonKey> = leaves.iter().cloned().collect();
        let keys_set: HashSet<MortonKey> = keys.iter().cloned().collect();

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

        let coordinates_row_major = points
            .iter()
            .map(|p| p.coordinate)
            .flat_map(|[x, y, z]| vec![x, y, z])
            .collect_vec();
        let global_indices = points.iter().map(|p| p.global_idx).collect_vec();

        let mut key_to_index = HashMap::new();

        for (i, key) in keys.iter().enumerate() {
            key_to_index.insert(*key, i);
        }

        let mut leaf_to_index = HashMap::new();

        for (i, key) in leaves.iter().enumerate() {
            leaf_to_index.insert(*key, i);
        }

        Ok(MultiNodeTree {
            world: world.duplicate(),
            depth,
            domain: *domain,
            coordinates: coordinates_row_major,
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
        })
    }

    /// TODO: Docs
    pub fn uniform_tree_sparse(
        world: &UserCommunicator,
        hyksort_subcomm_size: i32,
        coordinates: &[T],
        domain: &Domain<T>,
        depth: u64,
        global_idxs: &[usize],
    ) -> Result<MultiNodeTree<T>, std::io::Error> {
        let size = world.size();
        let rank = world.rank();

        // Encode points at deepest level, and map to specified depth.
        let dim = 3;
        let n_points = coordinates.len() / dim;

        let mut tmp = Points::default();
        for i in 0..n_points {
            let point = [
                coordinates[i],
                coordinates[i + n_points],
                coordinates[i + 2 * n_points],
            ];
            let base_key = MortonKey::from_point(&point, domain, DEEPEST_LEVEL);
            let encoded_key = MortonKey::from_point(&point, domain, depth);
            tmp.push(Point {
                coordinate: point,
                base_key,
                encoded_key,
                global_idx: global_idxs[i],
            })
        }

        let mut points = tmp;

        // 2.i Perform parallel Morton sort over encoded points
        let comm = world.duplicate();
        hyksort(&mut points, hyksort_subcomm_size, comm)?;

        let leaves: HashSet<MortonKey> = points.iter().map(|p| p.encoded_key).collect();
        let mut leaves = MortonKeys::from(leaves);
        leaves.complete();

        // 2.ii Find leaf keys on each processor
        let mut seeds = SingleNodeTree::<T>::find_seeds(&leaves);

        let blocktree = Self::complete_blocktree(&mut seeds, rank, size, world);

        Self::transfer_points_to_blocktree(world, &points, &seeds, &rank, &size);

        // Split blocks to required depth
        let mut leaves = MortonKeys::new();
        for block in blocktree.iter() {
            let level_diff = depth - block.level();
            leaves.append(&mut block.descendants(level_diff).unwrap())
        }

        let min = leaves.iter().min().unwrap();
        let max = leaves.iter().max().unwrap();

        // 3. Assign leaves to points
        let _unmapped = SingleNodeTree::assign_nodes_to_points(&leaves, &mut points);

        // Leaves are those that are mapped and their siblings if they exist in the processors range
        let leaves: HashSet<MortonKey> = points
            .iter()
            .map(|p| p.encoded_key)
            .flat_map(|k| k.siblings())
            .filter(|k| min <= k && k <= max)
            .collect();
        let mut leaves = MortonKeys::from(leaves);
        leaves.sort();

        // Group points by leaves
        points.sort();

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

        // Find all keys in tree
        let range = [world.rank() as u64, min.morton, max.morton];

        let tmp: HashSet<MortonKey> = leaves
            .iter()
            .flat_map(|leaf| leaf.ancestors().into_iter())
            .collect();

        let tmp: HashSet<MortonKey> = tmp
            .iter()
            .flat_map(|key| {
                if key.level() != 0 {
                    key.siblings()
                } else {
                    vec![*key]
                }
            })
            .collect();

        let mut keys = MortonKeys::from(tmp);

        let leaves_set: HashSet<MortonKey> = leaves.iter().cloned().collect();
        let keys_set: HashSet<MortonKey> = keys.iter().cloned().collect();

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
            .iter()
            .map(|p| p.coordinate)
            .flat_map(|[x, y, z]| vec![x, y, z])
            .collect_vec();
        let global_indices = points.iter().map(|p| p.global_idx).collect_vec();

        let mut key_to_index = HashMap::new();

        for (i, key) in keys.iter().enumerate() {
            key_to_index.insert(*key, i);
        }

        let mut leaf_to_index = HashMap::new();

        for (i, key) in leaves.iter().enumerate() {
            leaf_to_index.insert(*key, i);
        }

        Ok(MultiNodeTree {
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
        })
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
    ) -> Result<MultiNodeTree<T>, std::io::Error> {
        let dim = 3;
        let points_len = points.len();

        if !points.is_empty() && points_len & dim == 0 {
            let domain = domain.unwrap_or(Domain::from_global_points(points, world));
            let n_points = points_len / dim;

            // Calculate subcommunicator size for hyksort
            let hyksort_subcomm_size = 2;

            // Assign global indices
            let global_idxs = global_indices(n_points, world);

            if sparse {
                return MultiNodeTree::uniform_tree_sparse(
                    world,
                    hyksort_subcomm_size,
                    points,
                    &domain,
                    depth,
                    &global_idxs,
                );
            } else {
                return MultiNodeTree::uniform_tree(
                    world,
                    hyksort_subcomm_size,
                    points,
                    &domain,
                    depth,
                    &global_idxs,
                );
            }
        }

        Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "Invalid points format",
        ))
    }

    fn complete_blocktree(
        seeds: &mut MortonKeys,
        rank: i32,
        size: i32,
        world: &UserCommunicator,
    ) -> MortonKeys {
        // Define the tree's global domain with the finest first/last descendants
        if rank == 0 {
            let ffc_root = ROOT.finest_first_child();
            let min = seeds.iter().min().unwrap();
            let fa = ffc_root.finest_ancestor(min);
            let first_child = fa.children().into_iter().min().unwrap();
            seeds.push(first_child);
            seeds.sort();
        }

        if rank == (size - 1) {
            let flc_root = ROOT.finest_last_child();
            let max = seeds.iter().max().unwrap();
            let fa = flc_root.finest_ancestor(max);
            let last_child = fa.children().into_iter().max().unwrap();
            seeds.push(last_child);
        }

        let next_rank = if rank + 1 < size { rank + 1 } else { 0 };
        let previous_rank = if rank > 0 { rank - 1 } else { size - 1 };

        let previous_process = world.process_at_rank(previous_rank);
        let next_process = world.process_at_rank(next_rank);

        // Send required data to partner process.
        if rank > 0 {
            let min = *seeds.iter().min().unwrap();
            previous_process.send(&min);
        }

        let mut boundary = MortonKey::default();

        if rank < (size - 1) {
            next_process.receive_into(&mut boundary);
            seeds.push(boundary);
        }

        // Complete region between seeds at each process
        let mut complete = MortonKeys::new();

        for i in 0..(seeds.iter().len() - 1) {
            let a = seeds[i];
            let b = seeds[i + 1];

            let mut tmp: MortonKeys = complete_region(&a, &b).into();
            complete.keys.push(a);
            complete.keys.append(&mut tmp);
        }

        if rank == (size - 1) {
            complete.keys.push(seeds.last().unwrap());
        }

        complete.sort();
        complete
    }

    // Transfer points based on the coarse distributed blocktree.
    fn transfer_points_to_blocktree(
        world: &UserCommunicator,
        points: &[Point<T>],
        seeds: &[MortonKey],
        &rank: &Rank,
        &size: &Rank,
    ) -> Points<T> {
        let mut received_points: Points<T> = Vec::new();

        let min_seed = if rank == 0 {
            points.iter().min().unwrap().encoded_key
        } else {
            *seeds.iter().min().unwrap()
        };

        let prev_rank = if rank > 0 { rank - 1 } else { size - 1 };
        let next_rank = if rank + 1 < size { rank + 1 } else { 0 };

        if rank > 0 {
            let msg: Points<T> = points
                .iter()
                .filter(|&p| p.encoded_key < min_seed)
                .cloned()
                .collect();

            let msg_size: Rank = msg.len() as Rank;
            world.process_at_rank(prev_rank).send(&msg_size);
            world.process_at_rank(prev_rank).send(&msg[..]);
        }

        if rank < (size - 1) {
            let mut bufsize = 0;
            world.process_at_rank(next_rank).receive_into(&mut bufsize);
            let mut buffer = vec![Point::default(); bufsize as usize];
            world
                .process_at_rank(next_rank)
                .receive_into(&mut buffer[..]);
            received_points.append(&mut buffer);
        }

        // Filter out local points that have been sent to partner
        let mut points: Points<T> = points
            .iter()
            .filter(|&p| p.encoded_key >= min_seed)
            .cloned()
            .collect();

        received_points.append(&mut points);
        received_points.sort();

        received_points
    }
}

/// Assign global indices to points owned by each process
fn global_indices(n_points: usize, comm: &UserCommunicator) -> Vec<usize> {
    // Gather counts of coordinates at each process
    let rank = comm.rank() as usize;

    let nprocs = comm.size() as usize;
    let mut counts = vec![0usize; nprocs];
    comm.all_gather_into(&n_points, &mut counts[..]);

    // Compute displacements
    let mut displacements = vec![0usize; nprocs];

    for i in 1..nprocs {
        displacements[i] = displacements[i - 1] + counts[i - 1]
    }

    // Assign unique global indices to all coordinates
    let mut global_indices = vec![0usize; n_points];

    for (i, global_index) in global_indices.iter_mut().enumerate().take(n_points) {
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

    fn n_keys_tot(&self) -> Option<usize> {
        Some(self.keys.len())
    }

    fn n_keys(&self, level: u64) -> Option<usize> {
        if let Some((l, r)) = self.levels_to_keys.get(&level) {
            Some(r - l)
        } else {
            None
        }
    }

    fn n_leaves(&self) -> Option<usize> {
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
