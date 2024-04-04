//! Data structures to create distributed octrees with MPI.

#[cfg(feature = "mpi")]
use mpi::topology::UserCommunicator;

use num::traits::Float;
use rlst::RlstScalar;
use std::collections::{HashMap, HashSet};

/// A domain is a box defined aby an origin coordinate and its diameter along all three Cartesian axes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct Domain<T>
where
    T: Float + Default,
{
    /// The lower left corner of the domain, defined by the point distribution.
    pub origin: [T; 3],

    /// The diameter of the domain along the [x, y, z] axes respectively, defined
    /// by the maximum width of the point distribution along a given axis.
    pub diameter: [T; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
/// Representation of a Morton key with an 'anchor' specifying the origin of the node it encodes
/// with respect to the deepest level of the octree, as well as 'morton', a bit-interleaved single
/// integer representation.
pub struct MortonKey {
    /// The anchor is the index coordinate of the key, with respect to the origin of the Domain.
    pub anchor: [u64; 3],
    /// The Morton encoded anchor.
    pub morton: u64,
}

/// Iterable container of `MortonKey` data
#[derive(Clone, Debug, Default)]
pub struct MortonKeys {
    /// A vector of MortonKeys
    pub keys: Vec<MortonKey>,

    /// index for implementing the Iterator trait.
    pub index: usize,
}

/// Distributed trees created with MPI.
#[cfg(feature = "mpi")]
pub struct MultiNodeTree<T>
where
    T: Default + Float + RlstScalar<Real = T>,
{
    /// Global communicator for this Tree
    pub world: UserCommunicator,

    /// Depth of the tree
    pub depth: u64,

    /// Domain spanned by the points.
    pub domain: Domain<T>,

    /// All coordinates
    pub coordinates: Vec<T>,

    /// All global indices
    pub global_indices: Vec<usize>,

    /// The leaves that span the tree.
    pub leaves: MortonKeys,

    /// All nodes in tree.
    pub keys: MortonKeys,

    /// Associate leaves with point indices.
    pub leaves_to_coordinates: HashMap<MortonKey, (usize, usize)>,

    /// Associate levels with key indices.
    pub levels_to_keys: HashMap<u64, (usize, usize)>,

    /// Map between a key and its index
    pub key_to_index: HashMap<MortonKey, usize>,

    /// Map between a leaf and its index
    pub leaf_to_index: HashMap<MortonKey, usize>,

    /// All leaves, returned as a set.
    pub leaves_set: HashSet<MortonKey>,

    /// All keys, returned as a set.
    pub keys_set: HashSet<MortonKey>,

    /// Range of Morton keys at this processor, and their current rank [rank, min, max]
    pub range: [u64; 3],
}

/// A 3D cartesian point, described by coordinate, a unique global index, and the Morton Key for
/// the octree node in which it lies. Each Point as an associated 'base key', which is its matching
/// Morton encoding at the lowest possible level of discretization (DEEPEST_LEVEL), and an 'encoded key'
/// specifiying its encoding at a given level of discretization. Points also have associated data
#[repr(C)]
#[derive(Clone, Debug, Default, Copy)]
pub struct Point<T>
where
    T: RlstScalar<Real = T>,
{
    /// Physical coordinate in Cartesian space.
    pub coordinate: [T; 3],

    /// Global unique index.
    pub global_idx: usize,

    /// Key at finest level of encoding.
    pub base_key: MortonKey,

    /// Key at a given level of encoding, strictly an ancestor of 'base_key'.
    pub encoded_key: MortonKey,
}

/// Iterable container of `Point` data
#[derive(Clone, Debug, Default)]
pub struct Points<T>
where
    T: RlstScalar<Real = T>,
{
    /// A vector of Points
    pub points: Vec<Point<T>>,

    /// index for implementing the Iterator trait.
    pub index: usize,
}

/// Single Node Trees
#[derive(Default)]
pub struct SingleNodeTree<T>
where
    T: Float + Default + RlstScalar<Real = T>,
{
    /// Depth of a tree.
    pub depth: u64,

    /// Domain spanned by the points.
    pub domain: Domain<T>,

    /// All coordinates
    pub coordinates: Vec<T>,

    /// All global indices
    pub global_indices: Vec<usize>,

    /// The leaves that span the tree, and associated Point data.
    pub leaves: MortonKeys,

    /// All nodes in tree, and associated Node data.
    pub keys: MortonKeys,

    /// Associate leaves with coordinate indices.
    pub leaves_to_coordinates: HashMap<MortonKey, (usize, usize)>,

    /// Associate levels with key indices.
    pub levels_to_keys: HashMap<u64, (usize, usize)>,

    /// Map between a key and its index
    pub key_to_index: HashMap<MortonKey, usize>,

    /// Map between a leaf and its index
    pub leaf_to_index: HashMap<MortonKey, usize>,

    /// All leaves, returned as a set.
    pub leaves_set: HashSet<MortonKey>,

    /// All keys, returned as a set.
    pub keys_set: HashSet<MortonKey>,
}
