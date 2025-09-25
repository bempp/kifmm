//! Implementation of constructors for single node trees.

use std::collections::{HashMap, HashSet};

use itertools::Itertools;
use num::traits::Float;
use rlst::RlstScalar;

use crate::{
    traits::tree::SingleTree,
    tree::{
        constants::{DEEPEST_LEVEL, LEVEL_SIZE},
        morton::encode_anchor,
        Domain, MortonKey, MortonKeys, Point, Points, SingleNodeTree,
    },
};

impl<T> SingleNodeTree<T>
where
    T: RlstScalar + Float,
{
    /// Constructor for uniform trees on a single node refined to a user defined depth.
    ///
    /// # Arguments
    /// * `coordinates_row_major` - A slice of point coordinates, expected in row major order.
    /// [x_1, y_1, z_1,...x_N, y_N, z_N]
    /// * `domain` - The physical domain with which Morton Keys are being constructed with respect to.
    /// * `depth` - The maximum depth of the tree, defines the level of recursion.
    /// * `global_indices` - A slice of indices to uniquely identify the points.
    fn uniform_tree(
        coordinates_row_major: &[T],
        &domain: &Domain<T>,
        depth: u64,
        global_indices: &[usize],
    ) -> Result<SingleNodeTree<T>, std::io::Error> {
        let dim = 3;
        let n_coords = coordinates_row_major.len() / dim;
        let root = MortonKey::root();

        // Convert column major coordinate into `Point`, containing Morton encoding
        let mut points: Vec<Point<T>> = Points::default();
        for i in 0..n_coords {
            let coord: &[T; 3] = &coordinates_row_major[i * dim..(i + 1) * dim]
                .try_into()
                .unwrap();

            let base_key = MortonKey::<T>::from_point(coord, &domain, DEEPEST_LEVEL);
            let encoded_key = MortonKey::<T>::from_point(coord, &domain, depth);
            points.push(Point {
                coordinate: *coord,
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

        let mut key_to_level_index = HashMap::new();
        // Compute key to level index
        for l in 0..=depth {
            let &(l, r) = levels_to_keys.get(&l).unwrap();
            let keys = &keys[l..r];
            for (i, key) in keys.iter().enumerate() {
                key_to_level_index.insert(*key, i);
            }
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

        Ok(SingleNodeTree {
            root,
            depth,
            points,
            coordinates,
            global_indices,
            domain,
            leaves,
            keys,
            leaves_to_coordinates,
            key_to_index,
            key_to_level_index,
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
    /// * `coordinates_row_major` - A slice of point coordinates, expected in row major order.
    /// [x_1, y_1, z_1,...x_N, y_N, z_N]
    /// * `domain` - The physical domain with which Morton Keys are being constructed with respect to.
    /// * `depth` - The maximum depth of the tree, defines the level of recursion.
    /// * `global_indices` - A slice of indices to uniquely identify the points.
    fn uniform_tree_pruned(
        coordinates_row_major: &[T],
        &domain: &Domain<T>,
        depth: u64,
        global_indices: &[usize],
    ) -> Result<SingleNodeTree<T>, std::io::Error> {
        let root = MortonKey::root();
        let dim = 3;
        let n_coords = coordinates_row_major.len() / dim;

        // Convert column major coordinate into `Point`, containing Morton encoding
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

        let mut key_to_level_index = HashMap::new();
        // Compute key to level index
        for l in 0..=depth {
            let &(l, r) = levels_to_keys.get(&l).unwrap();
            let keys = &keys[l..r];
            for (i, key) in keys.iter().enumerate() {
                key_to_level_index.insert(*key, i);
            }
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

        Ok(SingleNodeTree {
            root,
            depth,
            points,
            coordinates,
            global_indices,
            domain,
            leaves,
            keys,
            leaves_to_coordinates,
            key_to_index,
            key_to_level_index,
            leaf_to_index,
            leaves_set,
            keys_set,
            levels_to_keys,
        })
    }

    /// Constructor for uniform trees on a single node refined to a user defined depth, however excludes
    /// empty nodes which don't contain particles and their ancestors, specify root node explicitly. This
    /// is useful for constructing trees that begin at different levels.
    ///
    /// # Arguments
    /// * `coordinates_row_major` - A slice of point coordinates, expected in row major order.
    /// [x_1, y_1, z_1,...x_N, y_N, z_N]
    /// * `domain` - The physical domain with which Morton Keys are being constructed with respect to.
    /// * `depth` - The maximum depth of the tree, defines the level of recursion.
    /// * `global_indices` - A slice of indices to uniquely identify the points.
    /// * `root` - The root node for this tree.
    fn uniform_tree_pruned_with_root(
        coordinates_row_major: &[T],
        &domain: &Domain<T>,
        depth: u64,
        global_indices: &[usize],
        root: MortonKey<T>,
    ) -> Result<SingleNodeTree<T>, std::io::Error> {
        let dim = 3;
        let n_coords = coordinates_row_major.len() / dim;

        // Convert column major coordinate into `Point`, containing Morton encoding
        let mut points = Points::default();
        for i in 0..n_coords {
            let coord: &[T; 3] = &coordinates_row_major[i * dim..(i + 1) * dim]
                .try_into()
                .unwrap();

            let encoded_key = MortonKey::from_point(coord, &domain, depth);
            let base_key = MortonKey::from_point(coord, &domain, DEEPEST_LEVEL);
            let ancestors = base_key.ancestors();

            if ancestors.contains(&root) {
                points.push(Point {
                    coordinate: *coord,
                    base_key,
                    encoded_key,
                    global_index: global_indices[i],
                })
            }
        }

        // Morton sort over points
        points.sort();

        // Ensure that final leaf set contains siblings of all encoded keys
        let leaves: HashSet<MortonKey<_>> = points
            .iter()
            .flat_map(|p| p.encoded_key.siblings())
            .collect();
        let leaves = MortonKeys::from(leaves);

        // Find all keys in tree up to root (if specified) or level 0 otherwise
        let root_level = root.level();
        if root.level() > depth {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "root level cannot exceed depth",
            ));
        }

        // Find all keys in tree
        let tmp: HashSet<MortonKey<_>> = leaves
            .iter()
            .flat_map(|leaf| leaf.ancestors_to_level(root_level).into_iter())
            .collect();

        // Ensure all siblings of ancestors are included
        let tmp: HashSet<MortonKey<_>> = tmp
            .iter()
            .flat_map(|key| {
                if key.level() != root_level {
                    key.siblings()
                } else {
                    vec![*key]
                }
            })
            .collect();

        let keys = MortonKeys::from(tmp);

        // Collect coordinates in row-major order, for ease of lookup
        let coordinates = points
            .iter()
            .map(|p| p.coordinate)
            .flat_map(|[x, y, z]| vec![x, y, z])
            .collect_vec();

        Ok(SingleNodeTree {
            root,
            depth,
            points,
            coordinates,
            domain,
            leaves,
            keys,
            ..Default::default()
        })
    }

    /// Construct multiple single node trees, specified by `roots` for the points specified by `points`.
    /// This is useful for constructing 'split' trees, whereby the global octree is split into 'local' portions
    /// - which are specified by a root that defines a subset of the global domain, and a 'global' portion which
    /// specifies the shared ancestors shared by all roots, and the roots form the leaves of the global portion.
    ///
    /// # Arguments
    /// * `roots` - The roots of the single node trees.
    /// * `points` - The points being mapped to the trees specified by the `roots`.
    /// * `domain` - The domain that contains all the `points`
    /// * `local_depth` - The depth of each single node tree, specified by a root.
    /// * `global_depth` - The difference between the level root node of all `MortonKeys` i.e. 0 and the
    /// * local depth.
    /// * `prune_empty` - Specifies whether to prune empty leaf nodes and their unoccupied ancestors from the tree.
    ///   Enabling this option streamlines the tree by removing nodes that do not contain any point data, potentially
    ///   enhancing query efficiency and reducing memory usage by focusing the tree structure on regions with actual data.
    #[allow(unused)]
    pub(crate) fn from_roots(
        roots: &MortonKeys<T>,
        points: &mut Points<T>,
        domain: &Domain<T>,
        global_depth: u64,
        local_depth: u64,
        prune_empty: bool,
    ) -> Vec<SingleNodeTree<T>> {
        let mut result = Vec::new();

        let (_unmapped, index_map) =
            SingleNodeTree::<T>::assign_nodes_to_points_with_index_map(roots, points);

        let depth = local_depth + global_depth;

        for root in roots.iter() {
            if let Some(indices) = index_map.get(root) {
                let mut local_points = indices.iter().map(|&i| points[i]).collect_vec();
                local_points.sort();

                // Create new buffer containing local coordinates on this local tree
                let local_coordinates =
                    local_points.iter().flat_map(|p| p.coordinate).collect_vec();
                let local_indices = Some(local_points.iter().map(|p| p.global_index).collect_vec());

                let tree = SingleNodeTree::new(
                    &local_coordinates,
                    depth,
                    prune_empty,
                    Some(*domain),
                    Some(*root),
                    local_indices,
                )
                .unwrap();

                result.push(tree)
            }
        }

        result
    }

    /// Construct a tree form specified leaves.
    ///
    /// # Arguments
    /// * `leaves` - The leaves being used to construct the tree.
    /// * `domain` - The domain being used to construct the tree.
    /// * `depth` - The depth of the tree.
    #[allow(unused)]
    pub(crate) fn from_leaves(
        leaves: Vec<MortonKey<T>>,
        domain: &Domain<T>,
        depth: u64,
    ) -> SingleNodeTree<T> {
        let mut result = Self::default();

        // First step is to find all keys

        // Need to insert sibling and their ancestors data if its missing in multipole data received
        // also need to ensure that siblings of ancestors are included
        let mut leaves_set: HashSet<_> = leaves.iter().cloned().collect();

        let mut keys_set = HashSet::new();

        for leaf in leaves.iter() {
            let siblings = leaf.siblings();
            let ancestors_siblings = leaf
                .ancestors()
                .iter()
                .flat_map(|a| {
                    if a.level() != 0 {
                        a.siblings()
                    } else {
                        vec![*a]
                    }
                })
                .collect_vec();

            for &key in siblings.iter() {
                keys_set.insert(key);
                leaves_set.insert(key);
            }

            for &key in ancestors_siblings.iter() {
                keys_set.insert(key);
            }
        }

        let mut keys = keys_set.iter().cloned().collect_vec();

        // Group by level to perform efficient lookup
        keys.sort_by_key(|k| k.level());

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

        let mut key_to_level_index = HashMap::new();
        // Compute key to level index
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

        let mut leaf_to_index = HashMap::new();

        for (i, key) in leaves.iter().enumerate() {
            leaf_to_index.insert(*key, i);
        }

        // Add data
        result.root = MortonKey::root();
        result.depth = depth;
        result.domain = *domain;
        result.leaves = leaves.into();
        result.keys = keys.into();
        result.key_to_index = key_to_index;
        result.key_to_level_index = key_to_level_index;
        result.leaf_to_index = leaf_to_index;
        result.keys_set = keys_set;
        result.leaves_set = leaves_set;
        result.levels_to_keys = levels_to_keys;

        result
    }

    /// Construct a single node tree from U list ghost octants, specified by their coordinates
    /// TODO: To deprecate with cleaner interface
    #[cfg(feature = "mpi")]
    pub(crate) fn from_ghost_octants_u(
        domain: &Domain<T>,
        depth: u64,
        coordinates_row_major: Vec<T>,
    ) -> (SingleNodeTree<T>, Vec<usize>) {
        let mut result = SingleNodeTree::default();

        let dim = 3;
        let mut leaves_to_coordinates = HashMap::new();
        let mut leaf_to_index = HashMap::new();
        let mut leaves = MortonKeys::default();
        let mut global_indices = Vec::new();
        let mut coordinates = Vec::new();
        let mut sort_indices = Vec::new();

        if !coordinates_row_major.is_empty() {
            let n_coords = coordinates_row_major.len() / dim;

            // Convert column major coordinate into `Point`, containing Morton encoding
            let mut points = Points::default();
            for i in 0..n_coords {
                let coord: &[T; 3] = &coordinates_row_major[i * dim..(i + 1) * dim]
                    .try_into()
                    .unwrap();

                let base_key = MortonKey::from_point(coord, domain, DEEPEST_LEVEL);
                let encoded_key = MortonKey::from_point(coord, domain, depth);
                points.push(Point {
                    coordinate: *coord,
                    base_key,
                    encoded_key,
                    global_index: 0, // TODO: Update with real global index
                })
            }

            // Sort points by Morton key, and return indices that sort the points
            sort_indices = (0..points.len()).collect_vec();
            sort_indices.sort_by_key(|&i| &points[i]);
            let points = sort_indices.iter().map(|&i| points[i]).collect_vec();

            // Group coordinates by leaves
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
            let tmp: HashSet<MortonKey<_>> = leaves_to_coordinates
                .keys()
                .flat_map(|k| k.siblings())
                .collect();

            // Sort leaves before returning
            leaves = MortonKeys::from(tmp);
            leaves.sort();

            // Collect global indices, in Morton sorted order
            global_indices = points.iter().map(|p| p.global_index).collect_vec();

            // Map between leaves and their respective indices
            for (i, key) in leaves.iter().enumerate() {
                leaf_to_index.insert(*key, i);
            }

            // Collect coordinates in row-major order, for ease of lookup
            coordinates = points
                .iter()
                .map(|p| p.coordinate)
                .flat_map(|[x, y, z]| vec![x, y, z])
                .collect_vec();
        }

        result.coordinates = coordinates;
        result.leaves_to_coordinates = leaves_to_coordinates;
        result.leaves_set = leaves.iter().cloned().collect();
        result.leaves = leaves;
        result.global_indices = global_indices;
        result.leaf_to_index = leaf_to_index;
        result.depth = depth;

        (result, sort_indices)
    }

    /// Construct a single node tree from V list ghost octants, ensure that provided keys are in Morton order and contain sibling data
    #[cfg(feature = "mpi")]
    pub(crate) fn from_ghost_octants_v(
        global_depth: u64,
        total_depth: u64,
        mut keys: Vec<MortonKey<T>>,
        keys_set: HashSet<MortonKey<T>>,
    ) -> SingleNodeTree<T> {
        let mut result = SingleNodeTree::default();

        // Set index pointers
        let mut levels_to_keys = HashMap::new();
        let mut key_to_level_index = HashMap::new();
        let mut key_to_index = HashMap::new();

        if !keys.is_empty() {
            // Group by level to perform efficient lookup
            keys.sort_by_key(|k| k.level());

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
            for l in global_depth..=total_depth {
                if let Some(&(l, r)) = levels_to_keys.get(&l) {
                    let subset = &mut keys[l..r];
                    subset.sort();
                }
            }

            // Compute key to level index
            for l in global_depth..=total_depth {
                if let Some(&(l, r)) = levels_to_keys.get(&l) {
                    let keys = &keys[l..r];
                    for (i, key) in keys.iter().enumerate() {
                        key_to_level_index.insert(*key, i);
                    }
                }
            }

            // Map between keys and their respective indices
            for (i, key) in keys.iter().enumerate() {
                key_to_index.insert(*key, i);
            }
        }

        result.depth = total_depth;
        result.keys = keys.into();
        result.keys_set = keys_set;
        result.levels_to_keys = levels_to_keys;
        result.key_to_level_index = key_to_level_index;
        result.key_to_index = key_to_index;

        result
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
    pub(crate) fn minimum_depth(n_points: u64, n_crit: u64) -> u64 {
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
    /// - `coordinates_row_major` - A slice of coordinates in column major order, structured as
    ///   [x_1, y_1, z_1,...x_N, y_N, z_N]. This ordering facilitates efficient spatial indexing
    ///   and operations within the tree.
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
    /// - `root` - Optionally specify a root node for the tree, defaults to the global root node.
    ///
    /// - `global_indices` - Optionally specify a set of global indices to map the specified points to,
    ///  defaults to assigning index based on ordering of input points
    pub fn new(
        coordinates_row_major: &[T],
        depth: u64,
        prune_empty: bool,
        domain: Option<Domain<T>>,
        root: Option<MortonKey<T>>,
        global_indices: Option<Vec<usize>>,
    ) -> Result<SingleNodeTree<T>, std::io::Error> {
        let dim = 3;

        let valid_len = !coordinates_row_major.is_empty();
        let coords_len = coordinates_row_major.len();
        let valid_dim = coords_len.is_multiple_of(dim);
        let valid_depth = depth <= DEEPEST_LEVEL;
        let root_specified = match root {
            Some(_root) => true,
            None => false,
        };

        match (
            valid_depth,
            valid_dim,
            valid_len,
            prune_empty,
            root_specified,
        ) {
            (true, true, true, true, true) => {
                let n_coords = coords_len / dim;
                let domain = domain.unwrap_or(Domain::from_local_points(coordinates_row_major));
                let global_indices = global_indices.unwrap_or((0..n_coords).collect_vec());

                SingleNodeTree::uniform_tree_pruned_with_root(
                    coordinates_row_major,
                    &domain,
                    depth,
                    &global_indices,
                    root.unwrap(),
                )
            }

            (true, true, true, false, true) => {
                let n_coords = coords_len / dim;
                let domain = domain.unwrap_or(Domain::from_local_points(coordinates_row_major));
                let global_indices = global_indices.unwrap_or((0..n_coords).collect_vec());
                SingleNodeTree::uniform_tree_pruned_with_root(
                    coordinates_row_major,
                    &domain,
                    depth,
                    &global_indices,
                    root.unwrap(),
                )
            }

            (true, true, true, true, false) => {
                let n_coords = coords_len / dim;
                let domain = domain.unwrap_or(Domain::from_local_points(coordinates_row_major));
                let global_indices = global_indices.unwrap_or((0..n_coords).collect_vec());

                SingleNodeTree::uniform_tree_pruned(
                    coordinates_row_major,
                    &domain,
                    depth,
                    &global_indices,
                )
            }
            (true, true, true, false, false) => {
                let n_coords = coords_len / dim;
                let domain = domain.unwrap_or(Domain::from_local_points(coordinates_row_major));
                let global_indices = global_indices.unwrap_or((0..n_coords).collect_vec());

                SingleNodeTree::uniform_tree(coordinates_row_major, &domain, depth, &global_indices)
            }

            (false, _, _, _, _) => {
                let msg =
                    format!("Invalid depth, depth={depth} > max allowed depth={DEEPEST_LEVEL}");
                Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, msg))
            }

            (_, false, _, _, _) => {
                let msg = format!("Coordinates must be three dimensional Cartesian, found `coords.len() % 3 = {:?} != 0`", coords_len % 3);
                Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, msg))
            }

            (_, _, false, _, _) => {
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
    pub(crate) fn assign_nodes_to_points(
        nodes: &MortonKeys<T>,
        points: &mut Points<T>,
    ) -> MortonKeys<T> {
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

    /// Also returns the indices of mapped points,
    pub(crate) fn assign_nodes_to_points_with_index_map(
        nodes: &MortonKeys<T>,
        points: &mut Points<T>,
    ) -> (MortonKeys<T>, HashMap<MortonKey<T>, Vec<usize>>) {
        let mut map: HashMap<MortonKey<_>, bool> = HashMap::new();

        let mut index_map = HashMap::new();

        for node in nodes.iter() {
            map.insert(*node, false);
        }

        for node in nodes.iter() {
            index_map.insert(*node, Vec::new());
        }

        for (point_index, point) in points.iter_mut().enumerate() {
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
                index_map.get_mut(&ancestor).unwrap().push(point_index)
            }
        }

        index_map.retain(|_, v| !v.is_empty());

        let mut unmapped = MortonKeys::new();

        for (node, is_mapped) in map.iter() {
            if !is_mapped {
                unmapped.push(*node)
            }
        }

        (unmapped, index_map)
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
    /// - `seeds` - Mutable reference to a collection of seed octants.i
    #[allow(unused)]
    pub(crate) fn complete_block_tree(seeds: &mut MortonKeys<T>) -> MortonKeys<T> {
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
    #[allow(unused)]
    pub(crate) fn find_seeds(leaves: &MortonKeys<T>) -> MortonKeys<T> {
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
    ///  Morton encoding.
    /// - `block_tree` - Initially constructed block_tree, subject to refinement through splitting.
    /// - `n_crit` - Defines the occupancy limit for leaf nodes, triggering splits when exceeded.
    #[allow(unused)]
    pub(crate) fn split_blocks(
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

impl<T> SingleTree for SingleNodeTree<T>
where
    T: RlstScalar + Float,
{
    type Scalar = T;
    type Domain = Domain<T>;
    type Node = MortonKey<T>;

    fn root(&self) -> Self::Node {
        self.root
    }

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

    fn keys(&self, level: u64) -> Option<&[Self::Node]> {
        if let Some(&(l, r)) = self.levels_to_keys.get(&level) {
            if r - l > 0 {
                Some(&self.keys[l..r])
            } else {
                None
            }
        } else {
            None
        }
    }

    fn all_keys(&self) -> Option<&[Self::Node]> {
        Some(&self.keys)
    }

    fn all_keys_set(&self) -> Option<&'_ HashSet<Self::Node>> {
        Some(&self.keys_set)
    }

    fn all_leaves_set(&self) -> Option<&'_ HashSet<Self::Node>> {
        Some(&self.leaves_set)
    }

    fn all_leaves(&self) -> Option<&[Self::Node]> {
        Some(&self.leaves)
    }

    fn coordinates(&self, leaf: &Self::Node) -> Option<&[Self::Scalar]> {
        if let Some(&(l, r)) = self.leaves_to_coordinates.get(leaf) {
            if r - l > 0 {
                Some(&self.coordinates[l * 3..r * 3])
            } else {
                None
            }
        } else {
            None
        }
    }

    fn points(&self, leaf: &Self::Node) -> Option<&[Point<Self::Scalar>]> {
        if let Some(&(l, r)) = self.leaves_to_coordinates.get(leaf) {
            if r - l > 0 {
                Some(&self.points[l..r])
            } else {
                None
            }
        } else {
            None
        }
    }

    fn all_coordinates(&self) -> Option<&[Self::Scalar]> {
        Some(&self.coordinates)
    }

    fn all_points(&self) -> Option<&[Point<Self::Scalar>]> {
        Some(&self.points)
    }

    fn global_indices(&self, leaf: &Self::Node) -> Option<&[usize]> {
        if let Some(&(l, r)) = self.leaves_to_coordinates.get(leaf) {
            if r - l > 0 {
                Some(&self.global_indices[l..r])
            } else {
                None
            }
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

    fn level_index(&self, key: &Self::Node) -> Option<&usize> {
        self.key_to_level_index.get(key)
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
        let tree = SingleNodeTree::<f64>::new(points.data(), depth, false, None, None, None);

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
        let tree =
            SingleNodeTree::<f64>::new(points.data(), depth, false, None, None, None).unwrap();

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

    #[test]
    fn test_uniform_tree_from_root() {
        let n_points = 1000;
        let depth = 2;

        // Use first child of global root as root node
        let root = MortonKey::root().children()[0];

        // Test data distributed over domain
        let points = points_fixture(n_points, Some(0.), Some(1.0), None);
        let domain = Domain::new(&[0., 0., 0.], &[1., 1., 1.]);
        let tree =
            SingleNodeTree::<f64>::new(points.data(), depth, false, Some(domain), Some(root), None)
                .unwrap();

        // Test that only points contained within specified root node are mapped to this tree.
        assert!(tree.all_points().unwrap().len() < n_points);

        // Test data contained in first child
        let points = points_fixture(n_points, Some(0.), Some(0.5), None);
        let domain = Domain::new(&[0., 0., 0.], &[1., 1., 1.]);
        let tree =
            SingleNodeTree::<f64>::new(points.data(), depth, false, Some(domain), Some(root), None)
                .unwrap();

        // Test that only points contained within specified root node are mapped to this tree.
        assert_eq!(tree.all_points().unwrap().len(), n_points);
    }

    pub fn test_no_overlaps_helper<T>(nodes: &[MortonKey<T>])
    where
        T: RlstScalar + Float,
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
            let point = [points[[0, i]], points[[1, i]], points[[2, i]]];
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

        // Test that a single octant contains all the points
        for (_, (l, r)) in leaves_to_points.iter() {
            if (r - l) > 0 {
                assert!((r - l) == n_points);
            }
        }
    }

    #[test]
    pub fn test_assign_nodes_to_points_with_index_map() {
        let root = MortonKey::<f64>::root();

        // Generate points in a single octant of the domain
        let n_points = 1234;
        let points = points_fixture::<f64>(n_points, None, None, None);
        let domain = Domain::new(&[0., 0., 0.], &[1., 1., 1.]);

        let mut tmp = Points::default();
        for i in 0..n_points {
            let point = [points[[0, i]], points[[1, i]], points[[2, i]]];
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
        let (_unmapped, index_map) =
            SingleNodeTree::assign_nodes_to_points_with_index_map(&keys, &mut points);

        // Test that all points have been assigned a root
        let mut found = 0;
        for (_key, indices) in index_map.iter() {
            found += indices.len();
        }

        assert_eq!(n_points, found);
    }

    #[test]
    pub fn test_split_blocks() {
        let root = MortonKey::root();
        let domain = Domain::<f64>::new(&[0., 0., 0.], &[1., 1., 1.]);
        let n_points = 10000;
        let points = points_fixture(n_points, None, None, None);

        let mut tmp = Points::default();

        for i in 0..n_points {
            let point = [points[[0, i]], points[[1, i]], points[[2, i]]];
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
        let tree =
            SingleNodeTree::<f64>::new(points.data(), depth, false, None, None, None).unwrap();

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
        let tree =
            SingleNodeTree::<f64>::new(points.data(), depth, false, None, None, None).unwrap();

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

    #[test]
    fn test_from_roots() {
        let n_points = 10000;
        let points = points_fixture::<f64>(n_points, None, None, None);
        let mut tmp = Points::default();
        let domain = Domain::new(&[0., 0., 0.], &[1., 1., 1.]);
        let global_depth = 2;
        let local_depth = 3;

        for i in 0..n_points {
            let point = [points[[0, i]], points[[1, i]], points[[2, i]]];
            let base_key = MortonKey::from_point(&point, &domain, DEEPEST_LEVEL);
            let encoded_key = MortonKey::from_point(&point, &domain, global_depth + local_depth);

            tmp.push(Point {
                coordinate: point,
                base_key,
                encoded_key,
                global_index: i,
            })
        }
        let mut points = tmp;

        let roots = MortonKey::root().descendants(global_depth).unwrap();

        let n_roots = roots.len();
        let trees = SingleNodeTree::from_roots(
            &roots.into(),
            &mut points,
            &domain,
            global_depth,
            local_depth,
            false,
        );

        // Test that the number of roots matches the number expected
        assert_eq!(trees.len(), n_roots);

        let total_depth = global_depth + local_depth;

        for tree in trees.iter() {
            // Test that the depth of the leaves matches the total depth
            for leaf in tree.leaves.iter() {
                assert_eq!(leaf.level(), total_depth)
            }

            // Test that the root level matches the global depth
            assert_eq!(tree.root().level(), global_depth);

            // Test that no keys are at a level higher than the global depth.
            let mut min_level = tree.keys[0].level();

            for key in tree.keys.iter() {
                if key.level() < min_level {
                    min_level = key.level()
                }
            }

            assert_eq!(min_level, global_depth);
        }
    }
}
