//! Implementation of constructors for single node trees.
use crate::traits::tree::Tree;
use crate::tree::{
    constants::{DEEPEST_LEVEL, LEVEL_SIZE, ROOT},
    implementations::impl_morton::encode_anchor,
    types::{
        domain::Domain,
        morton::{MortonKey, MortonKeys},
        point::{Point, PointType, Points},
        single_node::SingleNodeTree,
    },
};
use itertools::Itertools;
use num::Float;
use rlst::RlstScalar;
use std::collections::{HashMap, HashSet};

use super::impl_morton::complete_region;

impl<T> SingleNodeTree<T>
where
    T: Float + Default + RlstScalar<Real = T>,
{
    /// Constructor for uniform trees on a single node refined to a user defined depth.
    /// Returns a SingleNodeTree, with the leaves in sorted order.
    ///
    /// # Arguments
    /// * `points` - A slice of point coordinates, expected in column major order  [x_1, x_2, ... x_N, y_1, y_2, ..., y_N, z_1, z_2, ..., z_N].
    /// * `domain` - The physical domain with which Morton Keys are being constructed with respect to.
    /// * `depth` - The maximum depth of the tree, defines the level of recursion.
    /// * `global_idxs` - A slice of indices to uniquely identify the points.
    fn uniform_tree(
        points: &[T],
        domain: &Domain<T>,
        depth: u64,
        global_idxs: &[usize],
    ) -> SingleNodeTree<T> {
        // Encode points at deepest level, and map to specified depth

        // TODO: Automatically infer dimension
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
        points.sort();

        // Generate complete tree at specified depth
        let diameter = 1 << (DEEPEST_LEVEL - depth);

        let leaves = MortonKeys {
            keys: (0..LEVEL_SIZE)
                .step_by(diameter)
                .flat_map(|i| (0..LEVEL_SIZE).step_by(diameter).map(move |j| (i, j)))
                .flat_map(|(i, j)| (0..LEVEL_SIZE).step_by(diameter).map(move |k| [i, j, k]))
                .map(|anchor| {
                    let morton = encode_anchor(&anchor, depth);
                    MortonKey { anchor, morton }
                })
                .collect(),
            index: 0,
        };
        // Assign keys to points
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

        // Add unmapped leaves
        let mut leaves = MortonKeys {
            keys: leaves_to_coordinates
                .keys()
                .cloned()
                .chain(unmapped.iter().copied())
                .collect_vec(),
            index: 0,
        };

        // Sort leaves before returning
        leaves.sort();

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

        SingleNodeTree {
            depth,
            adaptive: false,
            coordinates,
            global_indices,
            domain: *domain,
            leaves,
            keys,
            leaves_to_coordinates,
            key_to_index,
            leaf_to_index,
            leaves_set,
            keys_set,
            levels_to_keys,
        }
    }

    /// Constructor for uniform trees on a single node refined to a user defined depth, however excludes empty nodes which don't contain particles.
    /// Returns a SingleNodeTree, with the leaves in sorted order.
    ///
    /// # Arguments
    /// * `points` - A slice of point coordinates, expected in column major order  [x_1, x_2, ... x_N, y_1, y_2, ..., y_N, z_1, z_2, ..., z_N].
    /// * `domain` - The physical domain with which Morton Keys are being constructed with respect to.
    /// * `depth` - The maximum depth of the tree, defines the level of recursion.
    /// * `global_idxs` - A slice of indices to uniquely identify the points.
    fn uniform_tree_sparse(
        points: &[PointType<T>],
        domain: &Domain<T>,
        depth: u64,
        global_idxs: &[usize],
    ) -> SingleNodeTree<T> {
        let dim = 3;
        let npoints = points.len() / dim;

        // Encode points at specified depth
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

        // Ensure that final leaf set contains siblings of all encoded keys
        let leaves: HashSet<MortonKey> = leaves_to_coordinates
            .keys()
            .flat_map(|k| k.siblings())
            .collect();

        // Store in a sorted vector
        let mut leaves = MortonKeys {
            keys: leaves.into_iter().collect_vec(),
            index: 0,
        };
        leaves.sort();

        // Find all keys in tree
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

        let mut keys = MortonKeys {
            keys: tmp.into_iter().collect_vec(),
            index: 0,
        };

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
        SingleNodeTree {
            depth,
            adaptive: false,
            coordinates,
            global_indices,
            domain: *domain,
            leaves,
            keys,
            leaves_to_coordinates,
            key_to_index,
            leaf_to_index,
            leaves_set,
            keys_set,
            levels_to_keys,
        }
    }

    /// Minimum depth
    pub fn minimum_depth(npoints: u64, n_crit: u64) -> u64 {
        let mut tmp = npoints;
        let mut level = 0;
        while tmp > n_crit {
            level += 1;
            tmp /= 8;
        }

        level as u64
    }

    /// Create a new single-node tree. If non-adaptive (uniform) trees are created, they are specified
    /// by a user defined maximum `depth`, if an adaptive tree is created it is specified by only by the
    /// user defined maximum leaf maximum occupancy `n_crit`.
    ///
    /// # Arguments
    /// * `points` - A slice of point coordinates, expected in column major order  [x_1, x_2, ... x_N, y_1, y_2, ..., y_N, z_1, z_2, ..., z_N].
    /// * `depth` - The maximum depth of the tree, defines the level of recursion.
    /// * `sparse` - Whether or not to create a 'sparse' tree, in which empty boxes are dropped.
    /// * `domain` - Optionally specify the domain, otherwise calculated from point data
    pub fn new(
        points: &[T],
        depth: u64,
        sparse: bool,
        domain: Option<Domain<T>>,
    ) -> SingleNodeTree<T> {
        let dim = 3;
        let domain = domain.unwrap_or(Domain::from_local_points(points));
        let npoints = points.len() / dim;
        let global_idxs = (0..npoints).collect_vec();

        if sparse {
            SingleNodeTree::uniform_tree_sparse(points, &domain, depth, &global_idxs)
        } else {
            SingleNodeTree::uniform_tree(points, &domain, depth, &global_idxs)
        }
    }
    /// Create a mapping between octree nodes and the points they contain, assumed to overlap.
    /// Return any keys that are unmapped.
    ///
    /// # Arguments
    /// * `nodes` - A reference to a container of MortonKeys.
    /// * `points` - A mutable reference to a container of points.
    pub fn assign_nodes_to_points(nodes: &MortonKeys, points: &mut Points<T>) -> MortonKeys {
        let mut map: HashMap<MortonKey, bool> = HashMap::new();
        for node in nodes.iter() {
            map.insert(*node, false);
        }

        for point in points.points.iter_mut() {
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

    /// Complete the minimal tree between a set of `seed` octants in some domain. Computed
    /// in place.
    ///
    /// # Arguments
    /// * `seeds` - A mutable reference to a container of `seed' octants, with gaps between them.
    pub fn complete_blocktree(seeds: &mut MortonKeys) -> MortonKeys {
        let ffc_root = ROOT.finest_first_child();
        let min = seeds.iter().min().unwrap();
        let fa = ffc_root.finest_ancestor(min);
        let first_child = fa.children().into_iter().min().unwrap();

        // Check for overlap
        if first_child < *min {
            seeds.push(first_child)
        }

        let flc_root = ROOT.finest_last_child();
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

        let mut blocktree = MortonKeys::new();

        for i in 0..(seeds.iter().len() - 1) {
            let a = seeds[i];
            let b = seeds[i + 1];
            let mut tmp = complete_region(&a, &b);
            blocktree.keys.push(a);
            blocktree.keys.append(&mut tmp);
        }

        blocktree.keys.push(seeds.last().unwrap());

        blocktree.sort();

        blocktree
    }

    /// Seeds are defined as the coarsest boxes in a set of non-uniform leaf boxes.
    /// Returns an owned vector of seeds in sorted order.
    ///
    /// # Arguments
    /// * `leaves` - A reference to a container of Morton Keys containing the leaf boxes.
    pub fn find_seeds(leaves: &MortonKeys) -> MortonKeys {
        let coarsest_level = leaves.iter().map(|k| k.level()).min().unwrap();

        let mut seeds: MortonKeys = MortonKeys {
            keys: leaves
                .iter()
                .filter(|k| k.level() == coarsest_level)
                .cloned()
                .collect_vec(),
            index: 0,
        };

        seeds.sort();
        seeds
    }

    /// Split tree coarse blocks by counting how many particles they contain until they satisfy
    /// a constrants specified by `n_crit` for the maximum number of particles that they can contain.
    ///
    /// # Arguments
    /// * `points` - A mutable reference to a container of points.
    /// * `blocktree` - An owned container of the `blocktree`, created by completing the space between seeds.
    /// * `n_crit` - The maximum number of points per leaf node.
    pub fn split_blocks(
        points: &mut Points<T>,
        mut blocktree: MortonKeys,
        n_crit: usize,
    ) -> MortonKeys {
        let split_blocktree;
        let mut blocks_to_points;
        loop {
            let mut new_blocktree = MortonKeys::new();

            // Map between blocks and the leaves they contain
            let unmapped = SingleNodeTree::assign_nodes_to_points(&blocktree, points);
            blocks_to_points = points
                .points
                .iter()
                .enumerate()
                .fold(
                    (HashMap::new(), 0, points.points[0]),
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

            // Generate a new blocktree with a block's children if they violate the n_crit parameter
            let mut check = 0;

            for block in blocks.iter() {
                if let Some((l, r)) = blocks_to_points.get(block) {
                    if (r - l) > n_crit {
                        let mut children = block.children();
                        new_blocktree.append(&mut children);
                    } else {
                        new_blocktree.push(*block);
                        check += 1;
                    }
                } else {
                    // Retain unmapped blocks
                    new_blocktree.push(*block);
                    check += 1;
                }
            }

            // Return if we cycle through all blocks without splitting
            if check == blocks.len() {
                split_blocktree = new_blocktree;
                break;
            } else {
                blocktree = new_blocktree;
            }
        }
        split_blocktree
    }
}

impl<T> Tree for SingleNodeTree<T>
where
    T: Float + Default + RlstScalar<Real = T>,
{
    type Precision = T;
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
        if let Some(&(l, r)) = self.levels_to_keys.get(&level) {
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

    fn coordinates(&self, key: &Self::Node) -> Option<&[Self::Precision]> {
        if let Some(&(l, r)) = self.leaves_to_coordinates.get(key) {
            Some(&self.coordinates[l * 3..r * 3])
        } else {
            None
        }
    }

    fn all_coordinates(&self) -> Option<&[Self::Precision]> {
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

#[cfg(test)]
mod test {
    use super::*;
    use crate::tree::implementations::helpers::{points_fixture, points_fixture_col};
    use rlst::RawAccess;

    #[test]
    pub fn test_uniform_tree() {
        let npoints = 100;
        let depth = 2;

        // Test uniformly distributed data
        let points = points_fixture(npoints, Some(-1.0), Some(1.0), None);
        let tree = SingleNodeTree::new(points.data(), depth, false, None);

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

        // Test a column distribution of data
        let points = points_fixture_col::<f64>(npoints);
        let tree = SingleNodeTree::new(points.data(), depth, false, None);

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

    pub fn test_no_overlaps_helper(nodes: &[MortonKey]) {
        let key_set: HashSet<MortonKey> = nodes.iter().cloned().collect();

        for node in key_set.iter() {
            let ancestors = node.ancestors();
            let int: Vec<&MortonKey> = key_set.intersection(&ancestors).collect();
            assert!(int.len() == 1);
        }
    }

    #[test]
    pub fn test_assign_nodes_to_points() {
        // Generate points in a single octant of the domain
        let npoints = 10;
        let points = points_fixture::<f64>(npoints, Some(0.), Some(0.5), None);

        let domain = Domain {
            origin: [0.0, 0.0, 0.0],
            diameter: [1.0, 1.0, 1.0],
        };

        let mut tmp = Points::default();
        for i in 0..npoints {
            let point = [points[[i, 0]], points[[i, 1]], points[[i, 2]]];
            let key = MortonKey::from_point(&point, &domain, DEEPEST_LEVEL);
            tmp.points.push(Point {
                coordinate: point,
                base_key: key,
                encoded_key: key,
                global_idx: i,
            })
        }
        let mut points = tmp;

        let keys = MortonKeys {
            keys: ROOT.children(),
            index: 0,
        };

        SingleNodeTree::assign_nodes_to_points(&keys, &mut points);

        let leaves_to_points = points
            .points
            .iter()
            .enumerate()
            .fold(
                (HashMap::new(), 0, points.points[0]),
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

        // Test that a single octant contains all the points
        for (_, (l, r)) in leaves_to_points.iter() {
            if (r - l) > 0 {
                assert!((r - l) == npoints);
            }
        }
    }

    #[test]
    pub fn test_split_blocks() {
        let domain = Domain {
            origin: [0., 0., 0.],
            diameter: [1.0, 1.0, 1.0],
        };
        let npoints = 10000;
        let points = points_fixture(npoints, None, None, None);

        let mut tmp = Points::default();

        for i in 0..npoints {
            let point = [points[[i, 0]], points[[i, 1]], points[[i, 2]]];
            let key = MortonKey::from_point(&point, &domain, DEEPEST_LEVEL);
            tmp.points.push(Point {
                coordinate: point,
                base_key: key,
                encoded_key: key,
                global_idx: i,
            })
        }
        let mut points = tmp;

        let n_crit = 15;

        // Test case where blocks span the entire domain
        let blocktree = MortonKeys {
            keys: vec![ROOT],
            index: 0,
        };

        SingleNodeTree::split_blocks(&mut points, blocktree, n_crit);
        let split_blocktree = MortonKeys {
            keys: points.points.iter().map(|p| p.encoded_key).collect_vec(),
            index: 0,
        };

        test_no_overlaps_helper(&split_blocktree);

        // Test case where the blocktree only partially covers the area
        let mut children = ROOT.children();
        children.sort();

        let a = children[0];
        let b = children[6];

        let mut seeds = MortonKeys {
            keys: vec![a, b],
            index: 0,
        };

        let blocktree = SingleNodeTree::<f64>::complete_blocktree(&mut seeds);

        SingleNodeTree::split_blocks(&mut points, blocktree, 25);
        let split_blocktree = MortonKeys {
            keys: points.points.iter().map(|p| p.encoded_key).collect_vec(),
            index: 0,
        };
        test_no_overlaps_helper(&split_blocktree);
    }

    #[test]
    fn test_complete_blocktree() {
        let a = ROOT.first_child();
        let b = *ROOT.children().last().unwrap();

        let mut seeds = MortonKeys {
            keys: vec![a, b],
            index: 0,
        };

        let mut blocktree = SingleNodeTree::<f64>::complete_blocktree(&mut seeds);

        blocktree.sort();

        let mut children = ROOT.children();
        children.sort();
        // Test that the blocktree is completed
        assert_eq!(blocktree.len(), 8);

        for (a, b) in children.iter().zip(blocktree.iter()) {
            assert_eq!(a, b)
        }
    }

    #[test]
    pub fn test_levels_to_keys() {
        // Uniform tree
        let npoints = 10000;
        let points = points_fixture::<f64>(npoints, None, None, None);
        let depth = 3;
        let tree = SingleNodeTree::new(points.data(), depth, false, None);

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
        assert_eq!(tot, npoints);
    }

    #[test]
    fn test_siblings_in_tree() {
        // Test that siblings lie adjacently in a tree.

        let npoints = 10000;
        let points = points_fixture::<f64>(npoints, None, None, None);
        let depth = 3;
        let tree = SingleNodeTree::new(points.data(), depth, false, None);

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