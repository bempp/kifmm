//! Implementation of constructors for single node trees.
use crate::traits::tree::Tree;
use crate::tree::{
    constants::{DEEPEST_LEVEL, LEVEL_SIZE},
    morton::encode_anchor,
    types::{Domain, MortonKey, MortonKeys, Point, Points, SingleNodeTree},
};
use crate::{Float, RlstScalarFloat};
use itertools::Itertools;
use rlst::RlstScalar;

use std::collections::{HashMap, HashSet};

impl<T> SingleNodeTree<T>
where
    T: RlstScalarFloat<Real = T> + Float,
{
    /// Constructor for uniform trees on a single node refined to a user defined depth.
    ///
    /// # Arguments
    /// * `coordinates_col_major` - A slice of point coordinates, expected in column major order.
    /// \[x1, x2, ... xn, y1, y2, ..., yn, z1, z2, ..., zn\].
    /// * `domain` - The physical domain with which Morton Keys are being constructed with respect to.
    /// * `depth` - The maximum depth of the tree, defines the level of recursion.
    /// * `global_indices` - A slice of indices to uniquely identify the points.
    fn uniform_tree(
        coordinates_col_major: &[T],
        &domain: &Domain<T>,
        depth: u64,
        global_indices: &[usize],
    ) -> Result<SingleNodeTree<T>, std::io::Error> {
        let dim = 3;
        let n_coords = coordinates_col_major.len() / dim;

        // Convert column major coordinate into `Point`, containing Morton encoding
        let mut points: Vec<Point<T>> = Points::default();
        for i in 0..n_coords {
            let coord: [T; 3] = [
                coordinates_col_major[i],
                coordinates_col_major[i + n_coords],
                coordinates_col_major[i + 2 * n_coords],
            ];
            let base_key = MortonKey::<T>::from_point(&coord, &domain, DEEPEST_LEVEL);
            let encoded_key = MortonKey::<T>::from_point(&coord, &domain, depth);
            points.push(Point {
                coordinate: coord,
                base_key,
                encoded_key,
                global_index: global_indices[i],
            })
        }

        // Morton sort over points
        points.sort();

        // Generate complete tree at specified depth
        let diameter = 1 << (DEEPEST_LEVEL - depth);

        let leaves = MortonKeys::from_iter(
            (0..LEVEL_SIZE)
                .step_by(diameter)
                .flat_map(|i| (0..LEVEL_SIZE).step_by(diameter).map(move |j| (i, j)))
                .flat_map(|(i, j)| (0..LEVEL_SIZE).step_by(diameter).map(move |k| [i, j, k]))
                .map(|anchor| {
                    let morton = encode_anchor(&anchor, depth);
                    MortonKey::<T>::new(&anchor, morton)
                }),
        );

        // Assign keys to points
        let unmapped = SingleNodeTree::<T>::assign_nodes_to_points(&leaves, &mut points);

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

        // Add unmapped leaves
        let mut leaves = MortonKeys::from(
            leaves_to_coordinates
                .keys()
                .cloned()
                .chain(unmapped.iter().copied())
                .collect_vec(),
        );

        // Sort leaves before returning
        leaves.sort();

        // Find all keys in tree
        let tmp: HashSet<MortonKey<_>> = leaves
            .iter()
            .flat_map(|leaf| leaf.ancestors().into_iter())
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
        let coordinates_row_major = points
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

        Ok(SingleNodeTree {
            depth,
            coordinates: coordinates_row_major,
            global_indices,
            domain,
            leaves,
            keys,
            leaves_to_coordinates,
            key_to_index,
            leaf_to_index,
            leaves_set,
            keys_set,
            levels_to_keys,
        })
    }

    /// Constructor for uniform trees on a single node refined to a user defined depth, however excludes
    /// empty nodes which don't contain particles and their ancestors.
    ///
    /// # Arguments
    /// * `coordinates_col_major` - A slice of point coordinates, expected in column major order
    /// \[x1, x2, ... xn, y1, y2, ..., yn, z1, z2, ..., zn\].
    /// * `domain` - The physical domain with which Morton Keys are being constructed with respect to.
    /// * `depth` - The maximum depth of the tree, defines the level of recursion.
    /// * `global_indices` - A slice of indices to uniquely identify the points.
    fn uniform_tree_pruned(
        coordinates_col_major: &[T],
        &domain: &Domain<T>,
        depth: u64,
        global_indices: &[usize],
    ) -> Result<SingleNodeTree<T>, std::io::Error> {
        let dim = 3;
        let n_coords = coordinates_col_major.len() / dim;

        // Convert column major coordinate into `Point`, containing Morton encoding
        let mut points = Points::default();
        for i in 0..n_coords {
            let point = [
                coordinates_col_major[i],
                coordinates_col_major[i + n_coords],
                coordinates_col_major[i + 2 * n_coords],
            ];
            let base_key = MortonKey::from_point(&point, &domain, DEEPEST_LEVEL);
            let encoded_key = MortonKey::from_point(&point, &domain, depth);
            points.push(Point {
                coordinate: point,
                base_key,
                encoded_key,
                global_index: global_indices[i],
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
        let leaves: HashSet<MortonKey<_>> = leaves_to_coordinates
            .keys()
            .flat_map(|k| k.siblings())
            .collect();

        // Sort leaves before returning
        let mut leaves = MortonKeys::from(leaves);
        leaves.sort();

        // Find all keys in tree
        let tmp: HashSet<MortonKey<_>> = leaves
            .iter()
            .flat_map(|leaf| leaf.ancestors().into_iter())
            .collect();

        // Ensure all siblings of ancestors are included
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
        let coordinates_row_major = points
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

        Ok(SingleNodeTree {
            depth,
            coordinates: coordinates_row_major,
            global_indices,
            domain,
            leaves,
            keys,
            leaves_to_coordinates,
            key_to_index,
            leaf_to_index,
            leaves_set,
            keys_set,
            levels_to_keys,
        })
    }

    /// Calculates the minimum depth of a tree required to ensure that each leaf box contains at most `n_crit` points,
    /// assuming a uniform distribution of points.
    ///
    /// This function determines the minimum depth of a tree necessary to achieve a specified maximum occupancy (`n_crit`)
    /// per leaf node, given a total number of points (`n_points`). It assumes a uniform distribution of points across
    /// the spatial domain. The calculation is based on the principle that each level of depth in the tree divides the
    /// space into 8 equally sized octants (or leaf boxes), thereby reducing the number of points in each box by a factor
    /// of 8 as the tree depth increases.
    ///
    /// # Arguments
    ///
    /// - `n_points` - The total number of points uniformly distributed within the domain.
    ///
    /// - `n_crit` - The maximum desired number of points per leaf box.
    pub fn minimum_depth(n_points: u64, n_crit: u64) -> u64 {
        let mut tmp = n_points;
        let mut level = 0;
        while tmp > n_crit {
            level += 1;
            tmp /= 8;
        }

        level as u64
    }

    /// Constructs a new single-node tree with uniform refinement up to a specified depth.
    ///
    /// This method initializes a single-node tree, uniformly subdivided to a user-defined maximum
    /// depth. If 'prune_empty' is used, the tree will exclude empty leaf nodes and their empty
    /// ancestors, optimizing memory usage and potentially improving query performance by eliminating
    /// unoccupied regions of the spatial domain.
    ///
    /// # Arguments
    ///
    /// - `coordinates_col_major` - A slice of coordinates in column major order, structured as
    ///   [x_1, x_2, ... x_N, y_1, y_2, ..., y_N, z_1, z_2, ..., z_N]. This ordering facilitates
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
    pub fn new(
        coordinates_col_major: &[T],
        depth: u64,
        prune_empty: bool,
        domain: Option<Domain<T>>,
    ) -> Result<SingleNodeTree<T>, std::io::Error> {
        let dim = 3;
        let coords_len = coordinates_col_major.len();

        let valid_len = !coordinates_col_major.is_empty();
        let valid_dim = coords_len % dim == 0;
        let valid_depth = depth <= DEEPEST_LEVEL;

        match (valid_depth, valid_dim, valid_len, prune_empty) {
            (true, true, true, true) => {
                let n_coords = coords_len / dim;
                let domain = domain.unwrap_or(Domain::from_local_points(coordinates_col_major));
                let global_indices = (0..n_coords).collect_vec();

                SingleNodeTree::uniform_tree_pruned(
                    coordinates_col_major,
                    &domain,
                    depth,
                    &global_indices,
                )
            }
            (true, true, true, false) => {
                let n_coords = coords_len / dim;
                let domain = domain.unwrap_or(Domain::from_local_points(coordinates_col_major));
                let global_indices = (0..n_coords).collect_vec();

                SingleNodeTree::uniform_tree(coordinates_col_major, &domain, depth, &global_indices)
            }

            (false, _, _, _) => {
                let msg = format!(
                    "Invalid depth, depth={} > max allowed depth={}",
                    depth, DEEPEST_LEVEL
                );
                Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, msg))
            }

            (_, false, _, _) => {
                let msg = format!("Coordinates must be three dimensional Cartesian, found `coords.len() % 3 = {:?} != 0`", coords_len % 3);
                Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, msg))
            }

            (_, _, false, _) => {
                let msg = "Empty coordinate vector".to_string();
                Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, msg))
            }
        }
    }

    /// Assigns points to their corresponding octree nodes based on Morton encoding, identifying nodes
    /// without points.
    ///
    /// Iterates through a collection of points, assigning each to the appropriate node by Morton key.
    /// Nodes that end up without any points are collected and returned, indicating areas of the space
    /// that are unoccupied.
    ///
    /// # Arguments
    ///
    /// - `nodes` - Morton keys discretising a spatial domain being considered.
    /// - `points` - Mutable reference to points, allowing for assignment of their encoded Morton keys, if any.
    pub fn assign_nodes_to_points(nodes: &MortonKeys<T>, points: &mut Points<T>) -> MortonKeys<T> {
        let mut map: HashMap<MortonKey<_>, bool> = HashMap::new();
        for node in nodes.iter() {
            map.insert(*node, false);
        }

        for point in points.iter_mut() {
            // Ancestor could be the key itself
            if let Some(ancestor) = point
                .base_key
                .ancestors()
                .into_iter()
                .sorted()
                .rev()
                .find(|a| map.contains_key(a))
            {
                point.encoded_key = ancestor;
                map.insert(ancestor, true);
            }
        }

        let mut unmapped = MortonKeys::new();

        for (node, is_mapped) in map.iter() {
            if !is_mapped {
                unmapped.push(*node)
            }
        }

        unmapped
    }

    /// Completes a minimal tree structure by filling gaps between a given set of `seed` octants.
    ///
    /// Seeds are a set of octants that serve as the starting points for building the complete tree,
    /// and the function may modify this collection to include the newly added nodes necessary to fill
    /// in the gaps.
    ///
    /// Operates in-place to ensure that the tree structure represents a continuous space, without gaps,
    /// encompassing all the provided seed octants. This process involves adding any necessary intermediate
    /// nodes to bridge the gaps between the initially disjoint seed octants, resulting in a coherent,
    /// minimal 'block' tree that spans the seeds' collective domain, by a set of 'blocks' which are the
    /// largest possible nodes, specified by Morton key, that span the domain specified by the seeds.
    ///
    /// This method is based on Algorithm 3 in
    /// [[Sundar et. al, 2007](https://epubs.siam.org/doi/abs/10.1137/070681727?casa_token=HjC61E-77RMAAAAA:2_Y9GNftaYBzGB-Y7-AiuGPNR6-RWvHoJ6DLpIUJ8lf1F2wPJHJWo8IicuGkYCNrrw72DP8)]
    ///
    /// # Arguments
    ///
    /// - `seeds` - Mutable reference to a collection of seed octants.
    pub fn complete_block_tree(seeds: &mut MortonKeys<T>) -> MortonKeys<T> {
        let root = MortonKey::root();

        let ffc_root = root.finest_first_child();
        let min = seeds.iter().min().unwrap();
        let fa = ffc_root.finest_ancestor(min);
        let first_child = fa.children().into_iter().min().unwrap();

        // Check for overlap
        if first_child < *min {
            seeds.push(first_child)
        }

        let flc_root = root.finest_last_child();
        let max = seeds.iter().max().unwrap();
        let fa = flc_root.finest_ancestor(max);
        let last_child = fa.children().into_iter().max().unwrap();

        if last_child > *max
            && !max.ancestors().contains(&last_child)
            && !last_child.ancestors().contains(max)
        {
            seeds.push(last_child);
        }

        seeds.sort();

        let mut block_tree = MortonKeys::new();

        for i in 0..(seeds.iter().len() - 1) {
            let a = seeds[i];
            let b = seeds[i + 1];
            let mut tmp = MortonKeys::complete_region(&a, &b);
            block_tree.keys.push(a);
            block_tree.keys.append(&mut tmp);
        }

        block_tree.keys.push(seeds.last().unwrap());

        block_tree.sort();

        block_tree
    }

    /// Identifies and returns the coarsest level nodes, called 'seeds', from a set of leaf boxes.
    ///
    /// This function examines a collection of leaf boxes, each represented by a Morton Key, to determine
    /// the seeds of the collection. Seeds are defined as the leaf boxes at the coarsest level of granularity
    /// present within the given set, serving as the starting points for further operations or tree construction.
    /// The resulting seeds are returned in a sorted order based on their Morton Keys.
    ///
    /// # Arguments
    ///
    /// - `leaves` - A reference to a collection of Morton Keys, representing leaf nodes.
    pub fn find_seeds(leaves: &MortonKeys<T>) -> MortonKeys<T> {
        let coarsest_level = leaves.iter().map(|k| k.level()).min().unwrap();

        let mut seeds = MortonKeys::from(
            leaves
                .iter()
                .filter(|k| k.level() == coarsest_level)
                .cloned()
                .collect_vec(),
        );

        seeds.sort();
        seeds
    }

    /// Splits tree blocks to meet a specified maximum occupancy constraint per leaf node.
    ///
    /// Processes the `block_tree` in-place, subdividing blocks as necessary to ensure that no block
    /// exceeds the `n_crit` maximum number of points allowed per leaf node.
    ///
    /// # Arguments
    ///
    /// - `points` - Mutable reference to points, used to count occupancy within blocks, contain
    /// Morton encoding.
    /// - `block_tree` - Initially constructed block_tree, subject to refinement through splitting.
    /// - `n_crit` - Defines the occupancy limit for leaf nodes, triggering splits when exceeded.
    pub fn split_blocks(
        points: &mut Points<T>,
        mut block_tree: MortonKeys<T>,
        n_crit: usize,
    ) -> MortonKeys<T> {
        let split_block_tree;
        let mut blocks_to_points;
        loop {
            let mut new_block_tree = MortonKeys::new();

            // Map between blocks and the leaves they contain
            let unmapped = SingleNodeTree::<T>::assign_nodes_to_points(&block_tree, points);
            blocks_to_points = points
                .iter()
                .enumerate()
                .fold(
                    (HashMap::new(), 0, points[0]),
                    |(mut blocks_to_points, curr_idx, curr), (i, point)| {
                        if point.encoded_key != curr.encoded_key {
                            blocks_to_points.insert(curr.encoded_key, (curr_idx, i));

                            (blocks_to_points, i, *point)
                        } else {
                            (blocks_to_points, curr_idx, curr)
                        }
                    },
                )
                .0;

            // Collect all blocks, including those which haven't been mapped
            let mut blocks = blocks_to_points.keys().cloned().collect_vec();
            // Add empty nodes to blocks.
            for key in unmapped.iter() {
                blocks.push(*key)
            }

            // Generate a new block_tree with a block's children if they violate the n_crit parameter
            let mut check = 0;

            for block in blocks.iter() {
                if let Some((l, r)) = blocks_to_points.get(block) {
                    if (r - l) > n_crit {
                        let mut children = block.children();
                        new_block_tree.append(&mut children);
                    } else {
                        new_block_tree.push(*block);
                        check += 1;
                    }
                } else {
                    // Retain unmapped blocks
                    new_block_tree.push(*block);
                    check += 1;
                }
            }

            // Return if we cycle through all blocks without splitting
            if check == blocks.len() {
                split_block_tree = new_block_tree;
                break;
            } else {
                block_tree = new_block_tree;
            }
        }
        split_block_tree
    }
}

impl<T> Tree for SingleNodeTree<T>
where
    T: RlstScalarFloat<Real = T> + Float,
{
    type Scalar = T;
    type Domain = Domain<T>;
    type Node = MortonKey<T>;
    type NodeSlice<'a> = &'a [MortonKey<T>]
        where T: 'a;
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
        if let Some(&(l, r)) = self.levels_to_keys.get(&level) {
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

    fn coordinates(&self, leaf: &Self::Node) -> Option<&[<Self::Scalar as RlstScalar>::Real]> {
        if let Some(&(l, r)) = self.leaves_to_coordinates.get(leaf) {
            Some(&self.coordinates[l * 3..r * 3])
        } else {
            None
        }
    }

    fn all_coordinates(&self) -> Option<&[<Self::Scalar as RlstScalar>::Real]> {
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

#[cfg(test)]
mod test {

    use super::*;
    use crate::tree::helpers::{points_fixture, points_fixture_col};
    use rlst::RawAccess;

    #[test]
    pub fn test_uniform_tree() {
        let n_points = 100;
        let depth = 2;

        // Test uniformly distributed data
        let points = points_fixture(n_points, Some(-1.0), Some(1.0), None);
        let tree = SingleNodeTree::<f64>::new(points.data(), depth, false, None);

        // Test that the tree really is uniform
        let levels: Vec<u64> = tree
            .unwrap()
            .all_leaves()
            .unwrap()
            .iter()
            .map(|node| node.level())
            .collect();
        let first = levels[0];
        assert!(levels.iter().all(|key| *key == first));

        // Test that max level constraint is satisfied
        assert!(first == depth);

        // Test a column distribution of data
        let points = points_fixture_col::<f64>(n_points);
        let tree = SingleNodeTree::<f64>::new(points.data(), depth, false, None).unwrap();

        // Test that the tree really is uniform
        let levels: Vec<u64> = tree
            .all_leaves()
            .unwrap()
            .iter()
            .map(|node| node.level())
            .collect();
        let first = levels[0];
        assert!(levels.iter().all(|key| *key == first));

        // Test that max level constraint is satisfied
        assert!(first == depth);

        let mut unique_leaves = HashSet::new();

        // Test that only a subset of the leaves contain any points
        for leaf in tree.all_leaves_set().unwrap().iter() {
            if let Some(_points) = tree.coordinates(leaf) {
                unique_leaves.insert(leaf.morton);
            }
        }

        let expected = 2u64.pow(depth.try_into().unwrap()) as usize; // Number of octants at encoding level that should be filled
        assert_eq!(unique_leaves.len(), expected);
    }

    pub fn test_no_overlaps_helper<T>(nodes: &[MortonKey<T>])
    where
        T: RlstScalarFloat<Real = T> + Float,
    {
        let key_set: HashSet<MortonKey<_>> = nodes.iter().cloned().collect();

        for node in key_set.iter() {
            let ancestors = node.ancestors();
            let int = key_set.intersection(&ancestors).collect_vec();
            assert!(int.len() == 1);
        }
    }

    #[test]
    pub fn test_assign_nodes_to_points() {
        let root = MortonKey::<f64>::root();

        // Generate points in a single octant of the domain
        let n_points = 10;
        let points = points_fixture::<f64>(n_points, Some(0.), Some(0.5), None);
        let domain = Domain::new(&[0., 0., 0.], &[1., 1., 1.]);

        let mut tmp = Points::default();
        for i in 0..n_points {
            let point = [points[[i, 0]], points[[i, 1]], points[[i, 2]]];
            let key = MortonKey::from_point(&point, &domain, DEEPEST_LEVEL);
            tmp.push(Point {
                coordinate: point,
                base_key: key,
                encoded_key: key,
                global_index: i,
            })
        }
        let mut points = tmp;
        let keys = MortonKeys::from(root.children());

        SingleNodeTree::<f64>::assign_nodes_to_points(&keys, &mut points);

        let leaves_to_points = points
            .iter()
            .enumerate()
            .fold(
                (HashMap::new(), 0, points[0]),
                |(mut leaves_to_points, curr_idx, curr), (i, point)| {
                    if point.encoded_key != curr.encoded_key {
                        leaves_to_points.insert(curr.encoded_key, (curr_idx, i + 1));

                        (leaves_to_points, i + 1, *point)
                    } else {
                        (leaves_to_points, curr_idx, curr)
                    }
                },
            )
            .0;

        println!("HERE {:?}", leaves_to_points);

        // Test that a single octant contains all the points
        for (_, (l, r)) in leaves_to_points.iter() {
            // println!("HERE {:?} {:?}", r, l);
            if (r - l) > 0 {
                assert!((r - l) == n_points);
            }
        }
    }

    #[test]
    pub fn test_split_blocks() {
        let root = MortonKey::root();
        let domain = Domain::<f64>::new(&[0., 0., 0.], &[1., 1., 1.]);
        let n_points = 10000;
        let points = points_fixture(n_points, None, None, None);

        let mut tmp = Points::default();

        for i in 0..n_points {
            let point = [points[[i, 0]], points[[i, 1]], points[[i, 2]]];
            let key = MortonKey::from_point(&point, &domain, DEEPEST_LEVEL);
            tmp.push(Point {
                coordinate: point,
                base_key: key,
                encoded_key: key,
                global_index: i,
            })
        }
        let mut points = tmp;

        let n_crit = 15;

        // Test case where blocks span the entire domain
        let block_tree = MortonKeys::from(vec![root]);

        SingleNodeTree::<f64>::split_blocks(&mut points, block_tree, n_crit);
        let split_block_tree = MortonKeys::from(points.iter().map(|p| p.encoded_key).collect_vec());

        test_no_overlaps_helper::<f64>(&split_block_tree);

        // Test case where the block_tree only partially covers the area
        let mut children = root.children();
        children.sort();

        let a = children[0];
        let b = children[6];

        let mut seeds = MortonKeys::from(vec![a, b]);

        let block_tree = SingleNodeTree::<f64>::complete_block_tree(&mut seeds);

        SingleNodeTree::<f64>::split_blocks(&mut points, block_tree, 25);
        let split_block_tree = MortonKeys::from(points.iter().map(|p| p.encoded_key).collect_vec());
        test_no_overlaps_helper::<f64>(&split_block_tree);
    }

    #[test]
    fn test_complete_blocktree() {
        let root = MortonKey::root();

        let a = root.first_child();
        let b = *root.children().last().unwrap();

        let mut seeds = MortonKeys::from(vec![a, b]);

        let mut block_tree = SingleNodeTree::<f64>::complete_block_tree(&mut seeds);

        block_tree.sort();

        let mut children = root.children();
        children.sort();
        // Test that the block_tree is completed
        assert_eq!(block_tree.len(), 8);

        for (a, b) in children.iter().zip(block_tree.iter()) {
            assert_eq!(a, b)
        }
    }

    #[test]
    pub fn test_levels_to_keys() {
        // Uniform tree
        let n_points = 10000;
        let points = points_fixture::<f64>(n_points, None, None, None);
        let depth = 3;
        let tree = SingleNodeTree::<f64>::new(points.data(), depth, false, None).unwrap();

        let keys = tree.all_keys().unwrap();

        let depth = tree.depth();

        let mut tot = 0;
        for level in (0..=depth).rev() {
            // Get keys at this level
            if let Some(tmp) = tree.keys(level) {
                tot += tmp.len();
            }
        }
        assert_eq!(tot, keys.len());

        let mut tot = 0;
        for level in (0..=depth).rev() {
            // Get all points at this level
            if let Some(nodes) = tree.keys(level) {
                for node in nodes.iter() {
                    if let Some(points) = tree.coordinates(node) {
                        tot += points.len() / 3
                    }
                }
            }
        }
        assert_eq!(tot, n_points);
    }

    #[test]
    fn test_siblings_in_tree() {
        // Test that siblings lie adjacently in a tree.

        let n_points = 10000;
        let points = points_fixture::<f64>(n_points, None, None, None);
        let depth = 3;
        let tree = SingleNodeTree::<f64>::new(points.data(), depth, false, None).unwrap();

        let keys = tree.keys(3).unwrap();

        // Pick out some random indices to test
        let idx_vec = [0, 8, 16, 32];

        for idx in idx_vec {
            let found = keys[idx].siblings();

            for i in 0..8 {
                assert!(found[i] == keys[idx + i])
            }
        }
    }
}
