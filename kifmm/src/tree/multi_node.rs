//! Implementation of constructors for MPI distributed multi node trees, from distributed point data.
use crate::{
    sorting::{hyksort, samplesort, simplesort},
    traits::tree::{MultiFmmTree, MultiTree, SingleTree},
    tree::{
        constants::DEEPEST_LEVEL,
        types::{Domain, MortonKey, MortonKeys, Point, Points, SingleNodeTree},
    },
    MultiNodeFmmTree,
};

use itertools::Itertools;
use mpi::topology::SimpleCommunicator;
use mpi::traits::{Communicator, CommunicatorCollectives, Equivalence};
use num::Float;
use pulp::Scalar;
use rlst::RlstScalar;
use std::collections::{HashMap, HashSet};

use super::types::{MultiNodeTree, SortKind};

impl<T, C: Communicator> MultiNodeTree<T, C>
where
    T: RlstScalar + Float + Equivalence + Default,
{
    /// Construct uniform tree, pruned by default
    pub fn uniform_tree(
        coordinates_row_major: &[T],
        &domain: &Domain<T>,
        local_depth: u64,
        global_depth: u64,
        global_indices: &[usize],
        world: &C,
        sort_kind: SortKind,
        prune_empty: bool,
    ) -> Result<MultiNodeTree<T, SimpleCommunicator>, std::io::Error> {
        let rank = world.rank();

        let dim = 3;
        let n_coords = coordinates_row_major.len() / dim;

        let mut points = Points::default();
        for i in 0..n_coords {
            let coord: &[T; 3] = &coordinates_row_major[i * dim..(i + 1) * dim]
                .try_into()
                .unwrap();
            let base_key = MortonKey::from_point(coord, &domain, DEEPEST_LEVEL);
            let encoded_key = MortonKey::from_point(coord, &domain, global_depth);

            points.push(Point {
                coordinate: *coord,
                base_key,
                encoded_key,
                global_index: global_indices[i],
            })
        }

        // Perform parallel Morton sort over encoded points
        let comm = world.duplicate();

        match sort_kind {
            SortKind::Hyksort { k } => hyksort(&mut points, k, comm)?,
            SortKind::Samplesort { k } => samplesort(&mut points, &comm, k)?,
            SortKind::Simplesort => {
                let splitters = MortonKey::root().descendants(global_depth).unwrap();
                let mut splitters = splitters
                    .into_iter()
                    .map(|m| Point {
                        coordinate: [T::zero(); 3],
                        global_index: 0,
                        encoded_key: m,
                        base_key: m,
                    })
                    .collect_vec();
                splitters.sort();
                let splitters = &splitters[1..];
                simplesort(&mut points, &comm, &splitters)?;
            }
        }

        // Find unique leaves specified by points on each processor
        let leaves: HashSet<MortonKey<_>> = points.iter().map(|p| p.encoded_key).collect();
        let mut leaves = MortonKeys::from(leaves);
        leaves.sort();

        // These define all the single node trees to be constructed
        let trees = SingleNodeTree::from_roots(
            &leaves,
            &mut points,
            &domain,
            global_depth,
            local_depth,
            prune_empty,
        );

        // MultiTree parameters
        let total_depth = local_depth + global_depth;
        let mut keys = Vec::new();
        let mut leaves = Vec::new();
        let mut roots = Vec::new();
        let mut points = Vec::new();

        for tree in trees.iter() {
            // Morton data
            keys.extend_from_slice(tree.keys.as_slice());
            leaves.extend_from_slice(tree.leaves.as_slice());
            roots.push(tree.root);

            // coordinate data
            points.extend_from_slice(tree.points.as_slice());
        }

        keys.sort();
        leaves.sort();
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

        let leaves = MortonKeys::from(leaves);
        let mut keys = MortonKeys::from(keys);

        // Sets for inclusion testing
        let keys_set = keys.iter().cloned().collect();
        let leaves_set = leaves.iter().cloned().collect();

        // Number of subtrees
        let n_trees = roots.len();

        // Group key by level for efficient lookup
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
        for l in global_depth..=total_depth {
            let &(l, r) = levels_to_keys.get(&l).unwrap();
            let subset = &mut keys[l..r];
            subset.sort();
        }

        let mut key_to_level_index = HashMap::new();
        // Compute key to level index
        for l in global_depth..=total_depth {
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

        Ok(MultiNodeTree {
            comm: world.duplicate(),
            rank,
            global_depth,
            local_depth,
            total_depth,
            n_trees,
            trees,
            roots,
            coordinates,
            points,
            keys_set,
            leaves_set,
            key_to_index,
            leaf_to_index,
            key_to_level_index,
            global_indices,
            leaves,
            keys,
            leaves_to_coordinates,
            levels_to_keys,
        })
    }

    /// Constructor for multinode trees
    pub fn new(
        comm: &C,
        coordinates_row_major: &[T],
        local_depth: u64,
        global_depth: u64,
        domain: Option<Domain<T>>,
        sort_kind: SortKind,
        prune_empty: bool,
    ) -> Result<MultiNodeTree<T, SimpleCommunicator>, std::io::Error> {
        let dim = 3;
        let coords_len = coordinates_row_major.len();

        if !coordinates_row_major.is_empty() && coords_len & dim == 0 {
            let domain = domain.unwrap_or(Domain::from_global_points(coordinates_row_major, comm));
            let n_coords = coords_len / dim;

            // Assign global indices
            let global_indices = global_indices(n_coords, comm);

            return MultiNodeTree::uniform_tree(
                coordinates_row_major,
                &domain,
                local_depth,
                global_depth,
                &global_indices,
                comm,
                sort_kind,
                prune_empty,
            );
        }

        Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "Invalid points format",
        ))
    }
}

impl<T, C> MultiTree for MultiNodeTree<T, C>
where
    T: RlstScalar + Default + Float + Equivalence,
    C: Communicator,
{
    type SingleTree = SingleNodeTree<T>;

    fn rank(&self) -> i32 {
        self.rank
    }

    fn trees<'a>(&'a self) -> &'a [Self::SingleTree] {
        self.trees.as_ref()
    }

    fn n_trees(&self) -> usize {
        self.n_trees
    }

    fn roots<'a>(&'a self) -> &'a [<Self::SingleTree as SingleTree>::Node] {
        self.roots.as_ref()
    }

    fn n_coordinates(&self, leaf: &<Self::SingleTree as SingleTree>::Node) -> Option<usize> {
        self.coordinates(leaf).map(|coords| coords.len() / 3)
    }

    fn n_coordinates_tot(&self) -> Option<usize> {
        self.all_coordinates().map(|coords| coords.len() / 3)
    }

    fn node(&self, idx: usize) -> Option<&<Self::SingleTree as SingleTree>::Node> {
        Some(&self.keys[idx])
    }

    fn n_keys(&self, level: u64) -> Option<usize> {
        Some(self.keys.len())
    }

    fn n_leaves(&self) -> Option<usize> {
        Some(self.leaves.len())
    }

    fn n_keys_tot(&self) -> Option<usize> {
        Some(self.keys.len())
    }

    fn total_depth(&self) -> u64 {
        self.global_depth + self.local_depth
    }

    fn global_depth(&self) -> u64 {
        self.global_depth
    }

    fn local_depth(&self) -> u64 {
        self.global_depth
    }

    fn keys(&self, level: u64) -> Option<&[<Self::SingleTree as SingleTree>::Node]> {
        if let Some(&(l, r)) = self.levels_to_keys.get(&level) {
            Some(&self.keys[l..r])
        } else {
            None
        }
    }

    fn all_keys(&self) -> Option<&[<Self::SingleTree as SingleTree>::Node]> {
        Some(&self.keys)
    }

    fn all_keys_set(&self) -> Option<&'_ HashSet<<Self::SingleTree as SingleTree>::Node>> {
        Some(&self.keys_set)
    }

    fn all_leaves(&self) -> Option<&[<Self::SingleTree as SingleTree>::Node]> {
        Some(&self.leaves)
    }

    fn all_leaves_set(&self) -> Option<&'_ HashSet<<Self::SingleTree as SingleTree>::Node>> {
        Some(&self.leaves_set)
    }

    fn coordinates(
        &self,
        leaf: &<Self::SingleTree as SingleTree>::Node,
    ) -> Option<&[<Self::SingleTree as SingleTree>::Scalar]> {
        if let Some(&(l, r)) = self.leaves_to_coordinates.get(leaf) {
            Some(&self.coordinates[l * 3..r * 3])
        } else {
            None
        }
    }

    fn all_coordinates(&self) -> Option<&[<Self::SingleTree as SingleTree>::Scalar]> {
        Some(&self.coordinates)
    }

    fn index(&self, key: &<Self::SingleTree as SingleTree>::Node) -> Option<&usize> {
        self.key_to_index.get(key)
    }

    fn level_index(&self, key: &<Self::SingleTree as SingleTree>::Node) -> Option<&usize> {
        self.key_to_level_index.get(key)
    }

    fn leaf_index(&self, leaf: &<Self::SingleTree as SingleTree>::Node) -> Option<&usize> {
        self.leaf_to_index.get(leaf)
    }
}

impl<T, C> MultiFmmTree for MultiNodeFmmTree<T, C>
where
    T: RlstScalar + Default + Float + Equivalence,
    C: Communicator,
{
    type Tree = MultiNodeTree<T, C>;

    fn n_source_trees(&self) -> usize {
        self.source_tree().n_trees()
    }

    fn n_target_trees(&self) -> usize {
        self.target_tree().n_trees()
    }

    fn source_tree<'a>(&'a self) -> &'a Self::Tree {
        &self.source_tree
    }

    fn target_tree<'a>(&'a self) -> &'a Self::Tree {
        &self.target_tree
    }
}

/// Assign global indices to points owned by each process
pub fn global_indices(n_points: usize, comm: &impl Communicator) -> Vec<usize> {
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
