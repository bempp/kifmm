//! Implementation of constructors for MPI distributed multi node trees, from distributed point data.
use crate::{
    samplesort::samplesort,
    simplesort,
    traits::tree::{MultiNodeTreeTrait, SingleNodeTreeTrait},
    tree::{
        constants::DEEPEST_LEVEL,
        types::{Domain, MortonKey, MortonKeys, MultiNodeTree, Point, Points, SingleNodeTree},
    },
};

use crate::hyksort::hyksort;
use crate::simplesort::simplesort;

use itertools::Itertools;
use mpi::topology::SimpleCommunicator;
use mpi::{
    traits::{Communicator, CommunicatorCollectives, Destination, Equivalence, Source},
    Rank,
};
use num::{zero, Float, Zero};
use pulp::Scalar;
use rlst::RlstScalar;
use std::{
    collections::{HashMap, HashSet},
    fmt::Debug,
};

use super::types::MultiNodeTreeNew;

#[derive(Clone)]
pub enum SortKind {
    Hyksort { k: i32 },
    Samplesort { k: usize },
    Simplesort,
}

pub fn splitters<T: RlstScalar + Float>(root: MortonKey<T>, global_depth: u64) -> Vec<Point<T>> {
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
    splitters.to_vec()
}

impl<T, C: Communicator> MultiNodeTreeNew<T, C>
where
    T: RlstScalar + Float + Equivalence + Default,
{
    pub fn uniform_tree(
        coordinates_row_major: &[T],
        &domain: &Domain<T>,
        local_depth: u64,
        global_depth: u64,
        global_indices: &[usize],
        world: &C,
        sort_kind: SortKind,
    ) -> Result<MultiNodeTreeNew<T, SimpleCommunicator>, std::io::Error> {
        let size = world.size();
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
                let root = MortonKey::root(Some(rank));
                let splitters = splitters(root, global_depth);
                simplesort(&mut points, &comm, &splitters)?;
            }
        }

        // hyksort(&mut points, hyksort_subcomm_size, comm)?;
        // samplesort(&mut points, &comm, 500).unwrap();

        // let splitters = MortonKey::root(None).descendants(global_depth)?;
        // let mut splitters = splitters
        //     .into_iter()
        //     .map(|m| Point {
        //         coordinate: [T::zero(); 3],
        //         global_index: 0,
        //         encoded_key: m,
        //         base_key: m,
        //     })
        //     .collect_vec();
        // splitters.sort();
        // let splitters = &splitters[1..];
        // simplesort(&mut points, &comm, splitters)?;

        // Find unique leaves specified by points on each processor
        let leaves: HashSet<MortonKey<_>> = points.iter().map(|p| p.encoded_key).collect();
        let mut leaves = MortonKeys::from_hashset(leaves, rank);
        leaves.sort();
        // leaves.complete();

        // These define all the single node trees to be constructed
        let trees = SingleNodeTree::from_roots(
            rank,
            &leaves,
            &mut points,
            &domain,
            global_depth,
            local_depth,
            true,
        );

        let mut keys_set = HashSet::new();

        for tree in trees.iter() {
            keys_set.extend(&mut tree.keys.clone());
        }

        Ok(MultiNodeTreeNew {
            world: world.duplicate(),
            rank,
            global_depth,
            local_depth,
            trees,
            keys_set,
        })
    }

    pub fn new(
        coordinates_row_major: &[T],
        local_depth: u64,
        global_depth: u64,
        prune_empty: bool,
        domain: Option<Domain<T>>,
        world: &C,
        sort_kind: SortKind,
    ) -> Result<MultiNodeTreeNew<T, SimpleCommunicator>, std::io::Error> {
        let dim = 3;
        let coords_len = coordinates_row_major.len();

        if !coordinates_row_major.is_empty() && coords_len & dim == 0 {
            let domain = domain.unwrap_or(Domain::from_global_points(coordinates_row_major, world));
            let n_coords = coords_len / dim;

            // Assign global indices
            let global_indices = global_indices(n_coords, world);

            return MultiNodeTreeNew::uniform_tree(
                coordinates_row_major,
                &domain,
                local_depth,
                global_depth,
                &global_indices,
                world,
                sort_kind,
            );
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
        let root = MortonKey::root(None);
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

impl<T, C> MultiNodeTreeTrait for MultiNodeTreeNew<T, C>
where
    T: RlstScalar + Default + Float + Equivalence,
    C: Communicator,
{
    type Tree = SingleNodeTree<T>;

    fn rank(&self) -> i32 {
        self.rank
    }

    fn trees<'a>(&'a self) -> &'a [Self::Tree] {
        self.trees.as_ref()
    }

    fn n_roots(&self) -> usize {
        self.trees.len()
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
