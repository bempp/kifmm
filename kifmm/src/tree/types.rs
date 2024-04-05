//! Data structures to create distributed octrees with MPI.

#[cfg(feature = "mpi")]
use mpi::topology::UserCommunicator;

use num::traits::Float;
use rlst::RlstScalar;
use std::collections::{HashMap, HashSet};

/// Represents a three-dimensional box characterized by its origin and side-length along the Cartesian axes.
///
/// # Fields
/// - `origin` - Defines the lower left corner (minimum x, y, z values) of the domain. This point serves as
/// the reference from which the domain extends in the positive direction along the Cartesian axes.
///
/// - `side_length` - Specifies the length of the domain along each of the Cartesian axes ([x, y, z] respectively).
/// This represents the domain's size and how far it extends from the origin along each axis.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct Domain<T>
where
    T: Float + Default,
{
    /// The lower left corner of the domain, minimum of x, y, z values.
    pub origin: [T; 3],

    /// The extent of the point distribution along the x, y, z axes respectively.
    pub side_length: [T; 3],
}

/// Represents a Morton key associated with a node within an octree structure.
///
/// A Morton key, or Z-order curve value, efficiently encodes multi-dimensional data into a single
/// dimension while preserving locality. This struct pairs the Morton key (`morton`) with its
/// 'anchor' point (`anchor`), which specifies the origin of the node it encodes in relation to the
/// deepest level of the octree. The anchor acts as a spatial reference point, indicating the
/// position of the node within the broader domain.
///
/// # Fields
/// - `anchor` - The index coordinate of the key's anchor point relative to the origin
///   of the domain. It represents the spatial starting point of the node in the octree, defined in
///   three-dimensional space (x, y, z).
///
/// - `morton` - The Morton-encoded value of the anchor. This single integer represents the
///   three-dimensional position of the node in a bit-interleaved format, enabling efficient
///   spatial indexing and operations.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct MortonKey {
    /// The anchor is the index coordinate of the key, with respect to the origin of the Domain.
    pub anchor: [u64; 3],
    /// The Morton encoded anchor.
    pub morton: u64,
}

/// A collection that stores and allows iteration over a sequence of `MortonKey` values.
#[derive(Clone, Debug, Default)]
pub struct MortonKeys {
    /// A vector of Morton_keys
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

    /// All points coordinates in row major format, such that [x1, y1, z1, ..., xn, yn, zn]
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

/// Represents a 3D point within an octree structure, enriched with Morton encoding information.
///
/// This struct defines a point in 3D Cartesian space, along with additional attributes relevant
/// for spatial indexing in octree-based data structures. It includes both a `base_key`, representing
/// the Morton encoding at the finest allowed level of discretization (16), and an `encoded_key`, which
/// corresponds to a specified level of discretization. The point is uniquely identified by a global
/// index, facilitating tracking and operations across different spatial contexts.
///
/// # Fields
///
/// - `coordinate` - The position of the point in 3D space, given as Cartesian coordinates.
///
/// - `global_idx` - A unique identifier for the point across the entire dataset or domain,
///   enabling efficient reference and retrieval.
///
/// - `base_key` - The Morton key encoding of the point's position at the deepest level
///   of octree discretization. This provides a fine-grained spatial index for the point.
///
/// - `encoded_key` - The Morton key encoding at a specific, potentially coarser level
///   of discretization than `base_key`. This key is always an ancestor of `base_key` in the octree.
#[repr(C)]
#[derive(Clone, Debug, Default, Copy)]
pub struct Point<T>
where
    T: RlstScalar<Real = T>,
{
    /// Physical coordinate in Cartesian space.
    pub coordinate: [T; 3],

    /// Global unique index.
    pub global_index: usize,

    /// Key at finest level of encoding.
    pub base_key: MortonKey,

    /// Key at a given level of encoding, strictly an ancestor of 'base_key'.
    pub encoded_key: MortonKey,
}

/// A collection of `Point` instances, each representing a 3D point within an octree structure.
pub type Points<T> = Vec<Point<T>>;

/// Single node trees.
#[derive(Default)]
pub struct SingleNodeTree<T>
where
    T: Float + Default + RlstScalar<Real = T>,
{
    /// Depth of a tree.
    pub depth: u64,

    /// Domain spanned by the points.
    pub domain: Domain<T>,

    /// All points coordinates in row major format, such that [x1, y1, z1, ..., xn, yn, zn]
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
