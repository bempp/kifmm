//! Implementation of constructors for MPI distributed multi node trees, from distributed point data.
use crate::{
    traits::tree::Tree,
    tree::{
        constants::DEEPEST_LEVEL,
        types::{Domain, MortonKey, MortonKeys, MultiNodeTree, Point, Points, SingleNodeTree},
    },
};

use crate::hyksort::hyksort;
use itertools::Itertools;
use mpi::topology::SimpleCommunicator;
use mpi::{
    traits::{Communicator, CommunicatorCollectives, Destination, Equivalence, Source},
    Rank,
};
use num::Float;
use rlst::RlstScalar;
use std::collections::{HashMap, HashSet};

impl<T, C: Communicator> MultiNodeTree<T, C>
where
    T: RlstScalar + Float + Equivalence + Default,
{
    /// Constructor for uniform trees, distributed with MPI, node refined to a user defined depth.
    ///
    /// The input point data is also assumed to be distributed across each node.
    ///
    /// # Arguments
    /// * `coordinates_row_major` - A slice of point coordinates, expected in row major order.
    /// [x_1, y_1, z_1,...x_N, y_N, z_N]
    /// * `domain` - The (global) physical domain with which Morton Keys are being constructed with respect to.
    /// * `depth` - The maximum depth of the tree, defines the level of recursion.
    /// * `global_indices` - A slice of indices to uniquely identify the points.
    /// * `world` - a global communicator for the tree.
    /// * `hyksort_subcomm_size` - size of sub-communicator used by
    /// [hyksort](https://dl.acm.org/doi/abs/10.1145/2464996.2465442?casa_token=vfaxtoyb_xsaaaaa:dqq1hfnp_gokaatn_d0svex37v_xooiqevdrong-4lyn_pmsuphmr3cp-0qvbisxtbwvuucaua).
    /// must be a power of 2.
    pub fn uniform_tree(
        coordinates_row_major: &[T],
        &domain: &Domain<T>,
        depth: u64,
        global_indices: &[usize],
        world: &C,
        hyksort_subcomm_size: i32,
    ) -> Result<MultiNodeTree<T, SimpleCommunicator>, std::io::Error> {
        let size = world.size();
        let rank = world.rank();
        let dim = 3;
        let n_coords = coordinates_row_major.len() / dim;
        let mut points = Points::default();

        for i in 0..n_coords {
            let coord: &[T; 3] = &coordinates_row_major[i * dim..(i + 1) * dim]
                .try_into()
                .unwrap();
            let base_key = MortonKey::from_point(coord, &domain, DEEPEST_LEVEL);
            let encoded_key = MortonKey::from_point(coord, &domain, depth);

            points.push(Point {
                coordinate: *coord,
                base_key,
                encoded_key,
                global_index: global_indices[i],
            })
        }

        // Perform parallel Morton sort over encoded points
        let comm = world.duplicate();
        hyksort(&mut points, hyksort_subcomm_size, comm)?;

        // Find unique leaves specified by points on each processor
        let leaves: HashSet<MortonKey<_>> = points.iter().map(|p| p.encoded_key).collect();
        let mut leaves = MortonKeys::from(leaves);
        leaves.complete();

        // Find the seeds (coarsest leaves) on each processor
        let mut seeds = SingleNodeTree::<T>::find_seeds(&leaves);

        // Compute the minimum spanning block tree
        let block_tree = Self::complete_block_tree(&mut seeds, rank, size, world);

        // Transfer points below minimum seed to previous processor
        Self::transfer_points_to_blocktree(world, &points, &seeds, &rank, &size);

        // Morton sort over local points after transfer
        points.sort();

        // Split blocks to required depth, defines leaves
        let mut leaves = MortonKeys::new();
        for block in block_tree.iter() {
            let level_diff = depth - block.level();
            leaves.append(&mut block.descendants(level_diff).unwrap())
        }

        // Find the minimum and maximum owned leaves
        let min = leaves.iter().min().unwrap();
        let max = leaves.iter().max().unwrap();

        // Assign leaves to points, disregard unmapped as they are included by definition in leaves buffer
        let _unmapped = SingleNodeTree::<T>::assign_nodes_to_points(&leaves, &mut points);

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

        // Find all keys in tree
        let range = [world.rank() as u64, min.morton, max.morton];

        let tmp: HashSet<MortonKey<_>> = leaves
            .iter()
            .flat_map(|leaf| leaf.ancestors().into_iter())
            .collect();

        // This additional step is needed in distributed trees to ensure that siblings of ancestors
        // are contained on each processor
        let tmp: HashSet<MortonKey<_>> = tmp
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

        // Create sets for inclusion testing
        let leaves_set: HashSet<MortonKey<_>> = leaves.iter().cloned().collect();
        let keys_set: HashSet<MortonKey<_>> = keys.iter().cloned().collect();

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
            let &(l, r) = levels_to_keys.get(&l).unwrap();
            let subset = &mut keys[l..r];
            subset.sort();
        }

        // Collect coordinates in row-major order, for ease of lookup
        let coordinates = points
            .iter()
            .map(|p| p.coordinate)
            .flat_map(|[x, y, z]| vec![x, y, z])
            .collect_vec();

        // Collect global indices, in Morton sorted order
        let global_indices = points.iter().map(|p| p.global_index).collect_vec();

        // Map between keys/leaves and their respective indices
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
            domain,
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

    /// Constructor for uniform trees, distributed with MPI, refined to a user defined depth, however excludes
    /// empty nodes which don't contain particles and their ancestors.
    ///
    /// The input point data is also assumed to be distributed across each node.
    ///
    /// # Arguments
    /// * `coordinates_row_major` - A slice of point coordinates, expected in row major order.
    /// [x_1, y_1, z_1,...x_N, y_N, z_N]
    /// * `domain` - The physical domain with which Morton Keys are being constructed with respect to.
    /// * `depth` - The maximum depth of the tree, defines the level of recursion.
    /// * `global_indices` - A slice of indices to uniquely identify the points.
    /// * `world` - a global communicator for the tree.
    /// * `hyksort_subcomm_size` - size of sub-communicator used by
    /// [hyksort](https://dl.acm.org/doi/abs/10.1145/2464996.2465442?casa_token=vfaxtoyb_xsaaaaa:dqq1hfnp_gokaatn_d0svex37v_xooiqevdrong-4lyn_pmsuphmr3cp-0qvbisxtbwvuucaua).
    /// must be a power of 2.
    pub fn uniform_tree_pruned(
        coordinates_row_major: &[T],
        &domain: &Domain<T>,
        depth: u64,
        global_indices: &[usize],
        world: &C,
        hyksort_subcomm_size: i32,
    ) -> Result<MultiNodeTree<T, SimpleCommunicator>, std::io::Error> {
        let size = world.size();
        let rank = world.rank();
        let dim = 3;
        let n_coords = coordinates_row_major.len() / dim;
        let mut points = Points::default();

        for i in 0..n_coords {
            let coord: &[T; 3] = &coordinates_row_major[i * dim..(i + 1) * dim]
                .try_into()
                .unwrap();
            let base_key = MortonKey::from_point(coord, &domain, DEEPEST_LEVEL);
            let encoded_key = MortonKey::from_point(coord, &domain, depth);

            points.push(Point {
                coordinate: *coord,
                base_key,
                encoded_key,
                global_index: global_indices[i],
            })
        }

        // Perform parallel Morton sort over encoded points
        let comm = world.duplicate();
        hyksort(&mut points, hyksort_subcomm_size, comm)?;

        // Find unique leaves specified by points on each processor
        let leaves: HashSet<MortonKey<_>> = points.iter().map(|p| p.encoded_key).collect();
        let mut leaves = MortonKeys::from(leaves);
        leaves.complete();

        // Find the seeds (coarsest leaves) on each processor
        let mut seeds = SingleNodeTree::<T>::find_seeds(&leaves);

        // Compute the minimum spanning block tree
        let block_tree = Self::complete_block_tree(&mut seeds, rank, size, world);

        // Transfer points below minimum seed to previous processor
        Self::transfer_points_to_blocktree(world, &points, &seeds, &rank, &size);

        // Morton sort over local points after transfer
        points.sort();

        // Split blocks to required depth
        let mut leaves = MortonKeys::new();
        for block in block_tree.iter() {
            let level_diff = depth - block.level();
            leaves.append(&mut block.descendants(level_diff).unwrap())
        }

        // Find the minimum and maximum owned leaves
        let min = leaves.iter().min().unwrap();
        let max = leaves.iter().max().unwrap();

        // Assign leaves to points, disregard unmapped
        let _unmapped = SingleNodeTree::<T>::assign_nodes_to_points(&leaves, &mut points);

        // Leaves are those that are mapped and their siblings if they exist in the processors range
        let leaves: HashSet<MortonKey<_>> = points
            .iter()
            .map(|p| p.encoded_key)
            .flat_map(|k| k.siblings())
            .filter(|k| min <= k && k <= max)
            .collect();
        let mut leaves = MortonKeys::from(leaves);
        leaves.sort();

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

        // Find all keys in tree
        let range = [world.rank() as u64, min.morton, max.morton];

        let tmp: HashSet<MortonKey<_>> = leaves
            .iter()
            .flat_map(|leaf| leaf.ancestors().into_iter())
            .collect();

        // This additional step is needed in distributed trees to ensure that siblings of ancestors
        // are contained on each processor
        let tmp: HashSet<MortonKey<_>> = tmp
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

        // Create sets for inclusion testing
        let leaves_set: HashSet<MortonKey<_>> = leaves.iter().cloned().collect();
        let keys_set: HashSet<MortonKey<_>> = keys.iter().cloned().collect();

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

        // Return tree in sorted order, by level and then by Morton key
        for l in 0..=depth {
            let &(l, r) = levels_to_keys.get(&l).unwrap();
            let subset = &mut keys[l..r];
            subset.sort();
        }

        // Collect coordinates in row-major order, for ease of lookup
        let coordinates = points
            .iter()
            .map(|p| p.coordinate)
            .flat_map(|[x, y, z]| vec![x, y, z])
            .collect_vec();

        // Collect global indices, in Morton sorted order
        let global_indices = points.iter().map(|p| p.global_index).collect_vec();

        // Map between keys/leaves and their respective indices
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
            domain,
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

    // TODO: Convert to row major
    /// Constructs a new multi-node tree distributed with MPI with uniform refinement up to a specified depth.
    ///
    /// This method initializes a single-node tree, uniformly subdivided to a user-defined maximum
    /// depth. If 'prune_empty' is used, the tree will exclude empty leaf nodes and their empty
    /// ancestors, optimizing memory usage and potentially improving query performance by eliminating
    /// unoccupied regions of the spatial domain.
    ///
    /// # Arguments

    /// - `coordinates_row_major` - A slice of coordinates in row major order, structured as
    ///   [x_1, y_1, z_1,...x_N, y_N, z_N]. This ordering facilitates
    ///   efficient spatial indexing and operations within the tree.
    ///
    /// - `depth` - Defines the maximum recursion level of the tree, determining the granularity of
    ///   spatial division. A greater depth results in a finer partitioning of the spatial domain.
    ///
    /// - `prune_empty` - Specifies whether to prune empty leaf nodes and their unoccupied ancestors from the tree.
    ///   Enabling this option streamlines the tree by removing nodes that do not contain any point data, potentially
    ///   enhancing query efficiency and reducing memory usage by focusing the tree structure on regions with actual data.
    ///
    /// - `domain` - Optionally specifies the spatial domain of the tree. If provided, this domain is
    ///   used directly; otherwise, it is computed from the point data, ensuring the tree encompasses
    ///   all points.
    ///
    /// - `domain` - The spatial domain covered by the tree's associated point data, if not provided estimated from global point data.
    ///
    /// - `world` - The global MPI communicator for this tree.
    pub fn new(
        coordinates_row_major: &[T],
        depth: u64,
        prune_empty: bool,
        domain: Option<Domain<T>>,
        world: &C,
    ) -> Result<MultiNodeTree<T, SimpleCommunicator>, std::io::Error> {
        let dim = 3;
        let coords_len = coordinates_row_major.len();

        if !coordinates_row_major.is_empty() && coords_len & dim == 0 {
            let domain = domain.unwrap_or(Domain::from_global_points(coordinates_row_major, world));
            let n_coords = coords_len / dim;

            // Calculate subcommunicator size for hyksort
            let hyksort_subcomm_size = 2;

            // Assign global indices
            let global_indices = global_indices(n_coords, world);

            if prune_empty {
                return MultiNodeTree::uniform_tree_pruned(
                    coordinates_row_major,
                    &domain,
                    depth,
                    &global_indices,
                    world,
                    hyksort_subcomm_size,
                );
            } else {
                return MultiNodeTree::uniform_tree(
                    coordinates_row_major,
                    &domain,
                    depth,
                    &global_indices,
                    world,
                    hyksort_subcomm_size,
                );
            }
        }

        Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "Invalid points format",
        ))
    }

    fn complete_block_tree(
        seeds: &mut MortonKeys<T>,
        rank: i32,
        size: i32,
        world: &C,
    ) -> MortonKeys<T> {
        let root = MortonKey::root();
        // Define the tree's global domain with the finest first/last descendants
        if rank == 0 {
            let ffc_root = root.finest_first_child();
            let min = seeds.iter().min().unwrap();
            let fa = ffc_root.finest_ancestor(min);
            let first_child = fa.children().into_iter().min().unwrap();
            seeds.push(first_child);
            seeds.sort();
        }

        if rank == (size - 1) {
            let flc_root = root.finest_last_child();
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

            let mut tmp: MortonKeys<T> = MortonKeys::complete_region(&a, &b).into();
            complete.keys.push(a);
            complete.keys.append(&mut tmp);
        }

        if rank == (size - 1) {
            complete.keys.push(seeds.last().unwrap());
        }

        complete.sort();
        complete
    }

    // Transfer points based on the coarse distributed block_tree.
    fn transfer_points_to_blocktree(
        world: &C,
        points: &[Point<T>],
        seeds: &[MortonKey<T>],
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
fn global_indices(n_points: usize, comm: &impl Communicator) -> Vec<usize> {
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

impl<T, C: Communicator> Tree for MultiNodeTree<T, C>
where
    T: RlstScalar + Float + Equivalence,
{
    type Scalar = T;
    type Domain = Domain<T>;
    type Node = MortonKey<T>;
    type NodeSlice<'a> = &'a [MortonKey<T>]
        where T: 'a, C: 'a;
    type Nodes = MortonKeys<T>;

    fn n_coordinates(&self, leaf: &Self::Node) -> Option<usize> {
        self.coordinates(leaf).map(|coords| coords.len() / 3)
    }

    fn n_coordinates_tot(&self) -> Option<usize> {
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

    fn coordinates(&self, leaf: &Self::Node) -> Option<&[Self::Scalar]> {
        if let Some(&(l, r)) = self.leaves_to_coordinates.get(leaf) {
            Some(&self.coordinates[l * 3..r * 3])
        } else {
            None
        }
    }

    fn all_coordinates(&self) -> Option<&[Self::Scalar]> {
        Some(&self.coordinates)
    }

    fn global_indices(&self, leaf: &Self::Node) -> Option<&[usize]> {
        if let Some(&(l, r)) = self.leaves_to_coordinates.get(leaf) {
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

    fn leaf_index(&self, leaf: &Self::Node) -> Option<&usize> {
        self.leaf_to_index.get(leaf)
    }
}
