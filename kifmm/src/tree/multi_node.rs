//! Implementation of constructors for MPI distributed multi node trees, from distributed point data.
use crate::{
    samplesort::samplesort,
    traits::tree::{MultiNodeTreeTrait, SingleNodeTreeTrait},
    tree::{
        constants::DEEPEST_LEVEL,
        types::{Domain, MortonKey, MortonKeys, Point, Points, SingleNodeTree},
    },
};

use crate::hyksort::hyksort;
use crate::simplesort::simplesort;

use itertools::Itertools;
use mpi::topology::SimpleCommunicator;
use mpi::traits::{Communicator, CommunicatorCollectives, Equivalence};
use num::Float;
use rlst::RlstScalar;
use std::collections::HashSet;

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
    ) -> Result<MultiNodeTree<T, SimpleCommunicator>, std::io::Error> {
        let rank = world.rank();

        let dim = 3;
        let n_coords = coordinates_row_major.len() / dim;

        let mut points = Points::default();
        for i in 0..n_coords {
            let coord: &[T; 3] = &coordinates_row_major[i * dim..(i + 1) * dim]
                .try_into()
                .unwrap();
            let base_key = MortonKey::from_point(coord, &domain, DEEPEST_LEVEL, Some(rank));
            let encoded_key = MortonKey::from_point(coord, &domain, global_depth, Some(rank));

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
                let splitters = MortonKey::root(None).descendants(global_depth).unwrap();
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
        let trees =
            SingleNodeTree::from_roots(&leaves, &mut points, &domain, global_depth, local_depth);

        let mut keys_set = HashSet::new();
        let mut roots = Vec::new();

        for tree in trees.iter() {
            keys_set.extend(&mut tree.keys.clone());
            roots.push(tree.root)
        }

        let total_depth = local_depth + global_depth;
        let ntrees = roots.len();

        Ok(MultiNodeTree {
            comm: world.duplicate(),
            rank,
            global_depth,
            local_depth,
            total_depth,
            n_trees: ntrees,
            trees,
            roots,
            keys_set,
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
            );
        }

        Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "Invalid points format",
        ))
    }
}

impl<T, C> MultiNodeTreeTrait for MultiNodeTree<T, C>
where
    T: RlstScalar + Default + Float + Equivalence,
    C: Communicator,
{
    type Tree = SingleNodeTree<T>;

    fn total_depth(&self) -> u64 {
        self.global_depth + self.local_depth
    }

    fn global_depth(&self) -> u64 {
        self.global_depth
    }

    fn local_depth(&self) -> u64 {
        self.global_depth
    }

    fn rank(&self) -> i32 {
        self.rank
    }

    fn trees<'a>(&'a self) -> &'a [Self::Tree] {
        self.trees.as_ref()
    }

    fn n_trees(&self) -> usize {
        self.n_trees
    }

    fn roots<'a>(&'a self) -> &'a [<Self::Tree as SingleNodeTreeTrait>::Node] {
        self.roots.as_ref()
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
