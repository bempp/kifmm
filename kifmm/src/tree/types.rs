//! Data structures to create distributed octrees with MPI.
use std::{
    collections::{HashMap, HashSet},
    marker::PhantomData,
};

#[cfg(feature = "mpi")]
use mpi::{
    traits::{Communicator, Equivalence},
    Count, Rank,
};

use num::Float;
use rlst::RlstScalar;

/// Represents a three-dimensional box characterized by its origin and side-length along the Cartesian axes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct Domain<T>
where
    T: RlstScalar,
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
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct MortonKey<T>
where
    T: RlstScalar,
{
    /// The anchor is the index coordinate of the key, with respect to the origin of the Domain.
    pub anchor: [u64; 3],
    /// The Morton encoded anchor.
    pub morton: u64,
    /// Scalar type of coordinate data associated with the key
    pub scalar: PhantomData<T>,
}

/// A collection that stores and allows iteration over a sequence of `MortonKey` values.
#[derive(Clone, Debug, Default)]
pub struct MortonKeys<T>
where
    T: RlstScalar,
{
    /// A vector of Morton_keys
    pub keys: Vec<MortonKey<T>>,

    /// index for implementing the Iterator trait.
    pub index: usize,
}

/// Represents an MPI distributed tree structure equipped with spatial indexing capabilities for 3D
/// particle data.
///
/// `MultiNodeTrees` are split into 'local' and 'global' portions, where the local trees correspond
/// to trees that are defined by roots that form the leaves of the global tree, such that that leaves
/// of the local trees do not intersect with each other. Each rank can contain multiple local trees, the
/// total number of local trees across all ranks is controlled with the `global_depth` parameter, as the
/// leaves correspond to the local trees' roots.
///
/// The global tree, which contains common ancestor nodes of all local trees distributed across nodes,
/// is stored on a single nominated node at rank 0 in the global communicator. Data from the global tree
/// is shared with the local trees via collective operations.
///
/// This split between local and global trees allows one to achieve optimal scaling in the distributed FMM
/// for all the ghost data exchanges.
///
/// # Fields
/// - `roots` - The roots associated with the local trees at this rank
///
/// - `all_roots` - All the roots associated with local trees in the distributed MultiNodeTree.
///
/// - `all_roots_ranks` - The associated ranks of `all_roots`.
///
/// - `all_roots_displacements` - The displacements of all `all_roots`, for collective communications
/// with the global tree.
///
/// - `all_roots_counts` - The counts of all `all_roots`, for collective communications
/// with the global tree.
///
/// - `global_depth` - The depth of the global tree.
///
/// - `local_depth` - The depth of the local trees.
///
/// - `total_depth` - The local depth + the global depth
///
/// - `domain` - The spatial domain covered by the globally distributed associated point data.
///
/// - `coordinates` - A flat vector storing the coordinates of all points managed by the tree,
///   in row-major format (e.g., `[x1, y1, z1, ..., xn, yn, zn]`), for efficient retrieval.
///
/// - `points` - Coordinates stored as `Points`.
///
/// - `global_indices` - A vector of unique global indices corresponding to each point, allowing
///   for efficient identification and lookup of points across different parts of the system.
///
/// - `leaves` - `MortonKeys` representing the leaves of all local trees, each associated with specific
///   point data encoded via Morton encoding, in Morton sorted order
///
/// - `keys` - `MortonKeys` representing all leaves at this rank from all local trees, in level then Morton order.
///
/// - `leaves_to_coordinates` - A mapping from Morton-encoded leaves to their corresponding
///   indices in the `coordinates` vector.
///
/// - `key_to_index` - A hash map linking each node key to its index within the
///   `keys` vector, enabling efficient node lookup and operations.
///
/// - `key_to_level_index` - Maps a key to the associated index pointer of the key for level data.
///
/// - `leaf_to_index` - A hash map linking each leaf key (`MortonKey`) to its index within the
///   `leaves` vector, supporting efficient leaf operations and data retrieval.
///
/// - `leaves_set` - A set of all `MortonKeys` representing the leaves, used for quick existence
///   checks and deduplication.
///
/// - `keys_set` - A set of all `MortonKeys` representing the nodes, used for quick existence
///   checks and deduplication.
///
/// - `levels_to_keys` - A mapping from tree levels to indices in the `keys` vector,
///   allowing for level-wise traversal and manipulation of the tree structure.
///
/// - `rank` - The rank of this tree.
///
/// - `trees` - The single node trees that are the local trees at this rank, that together
/// constitute the MultiNodeTree at this rank.
///
/// - `n_trees` - The number of local trees at this rank.
///
/// - `communicator` - The global MPI communicator for this tree.
#[cfg(feature = "mpi")]
#[derive(Default, Clone)]
pub struct MultiNodeTree<T, C: Communicator>
where
    T: RlstScalar + Float + Equivalence,
{
    /// Roots associated with trees at this rank
    pub roots: Vec<MortonKey<T>>,

    /// All global roots, only populated at nominated rank
    pub all_roots: Vec<MortonKey<T>>,

    /// All global roots origin ranks, only populated at nominated rank
    pub all_roots_ranks: Vec<Rank>,

    /// All global roots displacements, only populated at nominated rank
    pub all_roots_displacements: Vec<Count>,

    /// All global root origin counts, only populated at nominated rank
    pub all_roots_counts: Vec<Count>,

    /// Depth of the global tree
    pub global_depth: u64,

    /// Depth of each local tree
    pub local_depth: u64,

    /// Total depth of the tree, sum of local and global depths
    pub total_depth: u64,

    /// Get domain defined by the points, gets global domain in multi node setting.
    pub domain: Domain<T>,

    /// All points coordinates in row major format, such that [x1, y1, z1, ..., xn, yn, zn]
    pub coordinates: Vec<T>,

    /// All points coordinate with associated morton key
    pub points: Points<T>,

    /// Associated global indices
    pub global_indices: Vec<usize>,

    /// All associated leaves
    pub leaves: MortonKeys<T>,

    /// All associated keys
    pub keys: MortonKeys<T>,

    /// Associate leaves with coordinate indices.
    pub leaves_to_coordinates: HashMap<MortonKey<T>, (usize, usize)>,

    /// Map between key and index
    pub key_to_index: HashMap<MortonKey<T>, usize>,

    /// Map between a key and its index at a level
    pub key_to_level_index: HashMap<MortonKey<T>, usize>,

    /// Map between leaf key and leaf index
    pub leaf_to_index: HashMap<MortonKey<T>, usize>,

    /// All associated keys, for rapid inclusion checking
    pub leaves_set: HashSet<MortonKey<T>>,

    /// All associated keys, for rapid inclusion checking
    pub keys_set: HashSet<MortonKey<T>>,

    /// Associate levels with key indices.
    pub levels_to_keys: HashMap<u64, (usize, usize)>,

    /// MPI Rank
    pub rank: i32,

    /// Single node trees at this rank
    pub trees: Vec<SingleNodeTree<T>>,

    /// Number of single node trees at this rank
    pub n_trees: usize,

    /// Communicator associated with this tree
    pub communicator: C,
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
    T: RlstScalar + Float,
{
    /// Physical coordinate in Cartesian space.
    pub coordinate: [T; 3],

    /// Global unique index.
    pub global_index: usize,

    /// Key at finest level of encoding.
    pub base_key: MortonKey<T>,

    /// Key at a given level of encoding, strictly an ancestor of 'base_key'.
    pub encoded_key: MortonKey<T>,
}

/// A collection of `Point` instances, each representing a 3D point within an octree structure.
pub type Points<T> = Vec<Point<T>>;

/// Represents a single node tree structure equipped with spatial indexing capabilities for 3D
/// particle data.
///
/// This struct is designed for efficient representation and manipulation of spatial data within
/// a specified domain. It leverages Morton encoding to index points and nodes, facilitating
/// operations such as insertion, deletion, and querying.
///
/// # Fields
/// - `root` - The root node of the tree.
///
/// - `depth` - The depth of the tree.
///
/// - `domain` - The spatial domain covered by the tree's associated point data.
///
/// - `coordinates` - A flat vector storing the coordinates of all points managed by the tree,
///   in row-major format (e.g., `[x1, y1, z1, ..., xn, yn, zn]`), for efficient retrieval.
///
/// - `points` - Coordinates stored as `Points`.
///
/// - `global_indices` - A vector of unique global indices corresponding to each point, allowing
///   for efficient identification and lookup of points across different parts of the system.
///
/// - `leaves` - `MortonKeys` representing the leaves of the tree, each associated with specific
///   point data encoded via Morton encoding, in Morton sorted order
///
/// - `keys` - `MortonKeys` representing all leaves and their ancestors, in level and then Morton order.
///
/// - `leaves_to_coordinates` - A mapping from Morton-encoded leaves to their corresponding
///   indices in the `coordinates` vector.
///
/// - `levels_to_keys` - A mapping from tree levels to indices in the `keys` vector,
///   allowing for level-wise traversal and manipulation of the tree structure.
///
/// - `key_to_index` - A hash map linking each node key to its index within the
///   `keys` vector, enabling efficient node lookup and operations.
///
/// - `key_to_level_index` - Maps a key to the associated index pointer of the key for level data.
///
/// - `leaf_to_index` - A hash map linking each leaf key (`MortonKey`) to its index within the
///   `leaves` vector, supporting efficient leaf operations and data retrieval.
///
/// - `leaves_set` - A set of all `MortonKeys` representing the leaves, used for quick existence
///   checks and deduplication.
///
/// - `keys_set` - A set of all `MortonKeys` representing the nodes, used for quick existence
///   checks and deduplication.
#[derive(Default, Clone)]
pub struct SingleNodeTree<T>
where
    T: RlstScalar + Float,
{
    /// Root node of the tree
    pub root: MortonKey<T>,

    /// Depth of a tree.
    pub depth: u64,

    /// Domain spanned by the points.
    pub domain: Domain<T>,

    /// All points coordinates in row major format, such that [x1, y1, z1, ..., xn, yn, zn]
    pub coordinates: Vec<T>,

    /// All points coordinate with associated morton key
    pub points: Points<T>,

    /// All global indices
    pub global_indices: Vec<usize>,

    /// The leaves that span the tree, and associated Point data.
    pub leaves: MortonKeys<T>,

    /// All nodes in tree, and associated Node data.
    pub keys: MortonKeys<T>,

    /// Associate leaves with coordinate indices.
    pub leaves_to_coordinates: HashMap<MortonKey<T>, (usize, usize)>,

    /// Associate levels with key indices.
    pub levels_to_keys: HashMap<u64, (usize, usize)>,

    /// Map between a key and its index
    pub key_to_index: HashMap<MortonKey<T>, usize>,

    /// Map between a key and its index at a level
    pub key_to_level_index: HashMap<MortonKey<T>, usize>,

    /// Map between a leaf and its index
    pub leaf_to_index: HashMap<MortonKey<T>, usize>,

    /// All leaves, returned as a set.
    pub leaves_set: HashSet<MortonKey<T>>,

    /// All keys, returned as a set.
    pub keys_set: HashSet<MortonKey<T>>,
}

/// Parallel sort variants for constructing multi-node trees. Used to sort points
/// by Morton index.
#[cfg(feature = "mpi")]
#[derive(Clone)]
pub enum SortKind {
    /// Hypercube communication scheme based quicksort, based on Sundar et. al. 2013.
    Hyksort {
        /// Subcommunicator size, restricted to being a power of 2
        subcomm_size: i32,
    },

    /// Sample sort.
    Samplesort {
        /// The size of each sample from each MPI process
        n_samples: usize,
    },

    /// A variant of bucket sort
    Simplesort,
}
