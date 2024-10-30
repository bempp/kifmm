//! Tree Traits
use std::{collections::HashSet, hash::Hash};

use rlst::RlstScalar;

use num::traits::Float;

use crate::tree::types::Point;

/// Trees on a single node
pub trait SingleTree {
    /// Scalar type
    type Scalar: RlstScalar + Float;

    /// The computational domain defining the tree.
    type Domain: Domain<Scalar = Self::Scalar>;

    /// A tree node.
    type Node: TreeNode<Scalar = Self::Scalar, Domain = Self::Domain> + Clone + Copy;

    /// Root node
    fn root(&self) -> Self::Node;

    /// Number of leaves
    fn n_leaves(&self) -> Option<usize>;

    /// Total number of keys
    fn n_keys_tot(&self) -> Option<usize>;

    /// Number of keys at a given tree level
    fn n_keys(&self, level: u64) -> Option<usize>;

    /// Get depth of tree.
    fn depth(&self) -> u64;

    /// Get a reference to all leaves.
    fn all_leaves(&self) -> Option<&[Self::Node]>;

    /// Get a reference to keys at a given level.
    fn keys(&self, level: u64) -> Option<&[Self::Node]>;

    /// Get a reference to all keys.
    fn all_keys(&self) -> Option<&[Self::Node]>;

    /// Get a reference to all keys as a set.
    fn all_keys_set(&self) -> Option<&'_ HashSet<Self::Node>>;

    /// Get a reference to all leaves as a set.
    fn all_leaves_set(&self) -> Option<&'_ HashSet<Self::Node>>;

    /// Gets a reference to the coordinates contained with a leaf node.
    ///
    /// # Arguments
    /// - `leaf` - node being queried.
    fn coordinates(&self, leaf: &Self::Node) -> Option<&[Self::Scalar]>;

    /// Gets a reference to the points contained with a leaf node.
    ///
    /// # Arguments
    /// - `leaf` - node being queried.
    fn points(&self, leaf: &Self::Node) -> Option<&[Point<Self::Scalar>]>;

    /// Query number of coordinates contained at a given leaf node
    ///
    /// # Arguments
    /// - `leaf` - node being queried.
    fn n_coordinates(&self, leaf: &Self::Node) -> Option<usize>;

    /// Gets a reference to the coordinates contained in across tree
    fn all_coordinates(&self) -> Option<&[Self::Scalar]>;

    /// Gets a reference to the points contained in across tree
    fn all_points(&self) -> Option<&[Point<Self::Scalar>]>;

    /// Total number of coordinates (local in a multi node setting)
    fn n_coordinates_tot(&self) -> Option<usize>;

    /// Gets global indices at a leaf node
    ///
    /// # Arguments
    /// - `leaf` - node being queried.
    fn global_indices(&self, leaf: &Self::Node) -> Option<&[usize]>;

    /// gets all global indices (local in mult inode setting)
    fn all_global_indices(&self) -> Option<&[usize]>;

    /// Get domain defined by the points, gets global domain in multi node setting.
    fn domain(&self) -> &Self::Domain;

    /// Map from the key to index position in sorted keys
    ///
    /// # Arguments
    /// - `key` - Node being queried.
    fn index(&self, key: &Self::Node) -> Option<&usize>;

    /// Map from the key to index position in sorted keys at a given level
    ///
    /// # Arguments
    /// - `key` - Node being queried.
    fn level_index(&self, key: &Self::Node) -> Option<&usize>;

    /// Map from the leaf to its index position in sorted leaves
    ///
    /// # Arguments
    /// - `leaf` - Node being queried.
    fn leaf_index(&self, leaf: &Self::Node) -> Option<&usize>;

    /// Map from an index position to a node
    ///
    /// # Arguments
    /// - `idx` - Index being query.
    fn node(&self, idx: usize) -> Option<&Self::Node>;
}

/// Trees distributed with MPI
#[cfg(feature = "mpi")]
pub trait MultiTree {
    /// Associated single node trees
    type SingleTree: SingleTree;

    /// Associated MPI rank
    fn rank(&self) -> i32;

    /// Roots associated with trees at this rank
    fn roots(&self) -> &[<Self::SingleTree as SingleTree>::Node];

    /// All the single node trees associated with this rank
    fn trees(&self) -> &[Self::SingleTree];

    /// Number of single node trees associated with this rank
    fn n_trees(&self) -> usize;

    /// Number of leaves at this rank
    fn n_leaves(&self) -> Option<usize>;

    /// Total number of keys associated with this MPI rank
    fn n_keys_tot(&self) -> Option<usize>;

    /// Number of keys at a given tree level at this rank
    fn n_keys(&self, level: u64) -> Option<usize>;

    /// Total depth of tree
    fn total_depth(&self) -> u64;

    /// Depth of local trees
    fn local_depth(&self) -> u64;

    /// Depth of global tree
    fn global_depth(&self) -> u64;

    /// Get a reference to all leaves at this rank
    fn all_leaves(&self) -> Option<&[<Self::SingleTree as SingleTree>::Node]>;

    /// Get a reference to keys at a given level at this rank.
    fn keys(&self, level: u64) -> Option<&[<Self::SingleTree as SingleTree>::Node]>;

    /// Get a reference to all keys at this rank.
    fn all_keys(&self) -> Option<&[<Self::SingleTree as SingleTree>::Node]>;

    /// Get a reference to all keys as a set at this rank.
    fn all_keys_set(&self) -> Option<&'_ HashSet<<Self::SingleTree as SingleTree>::Node>>;

    /// Get a reference to all leaves as a set at this rank.
    fn all_leaves_set(&self) -> Option<&'_ HashSet<<Self::SingleTree as SingleTree>::Node>>;

    /// Gets a reference to the coordinates contained with a leaf node.
    ///
    /// # Arguments
    /// - `leaf` - node being queried.
    fn coordinates(
        &self,
        leaf: &<Self::SingleTree as SingleTree>::Node,
    ) -> Option<&[<Self::SingleTree as SingleTree>::Scalar]>;

    /// Gets a reference to the points contained with a leaf node.
    ///
    /// # Arguments
    /// - `leaf` - node being queried.
    fn points(
        &self,
        leaf: &<Self::SingleTree as SingleTree>::Node,
    ) -> Option<&[Point<<Self::SingleTree as SingleTree>::Scalar>]>;

    /// Query number of coordinates contained at a given leaf node
    ///
    /// # Arguments
    /// - `leaf` - node being queried.
    fn n_coordinates(&self, leaf: &<Self::SingleTree as SingleTree>::Node) -> Option<usize>;

    /// Gets a reference to the coordinates contained in this rank.
    fn all_coordinates(&self) -> Option<&[<Self::SingleTree as SingleTree>::Scalar]>;

    /// Total number of coordinates at this rank.
    fn n_coordinates_tot(&self) -> Option<usize>;

    /// Map from the key to index position in sorted keys
    ///
    /// # Arguments
    /// - `key` - Node being queried.
    fn index(&self, key: &<Self::SingleTree as SingleTree>::Node) -> Option<&usize>;

    /// Map from the key to index position in sorted keys at a given level
    ///
    /// # Arguments
    /// - `key` - Node being queried.
    fn level_index(&self, key: &<Self::SingleTree as SingleTree>::Node) -> Option<&usize>;

    /// Map from the leaf to its index position in sorted leaves
    ///
    /// # Arguments
    /// - `leaf` - Node being queried.
    fn leaf_index(&self, leaf: &<Self::SingleTree as SingleTree>::Node) -> Option<&usize>;

    /// Map from an index position to a node
    ///
    /// # Arguments
    /// - `idx` - Index being query.
    fn node(&self, idx: usize) -> Option<&<Self::SingleTree as SingleTree>::Node>;

    /// Get domain defined by the points across all nodes.
    fn domain(&self) -> &<Self::SingleTree as SingleTree>::Domain;
}

/// Defines FMM compatible trees, where the source and target trees are associated with multipole/local data respectively.
pub trait SingleFmmTree {
    /// Tree associated with this FMM tree
    type Tree: SingleTree;

    /// Get the source tree
    fn source_tree(&self) -> &Self::Tree;

    /// Get the target tree
    fn target_tree(&self) -> &Self::Tree;

    /// Get the domain
    fn domain(&self) -> &<Self::Tree as SingleTree>::Domain;
}

/// Defines FMM compatible trees, where the source and target trees are associated with multipole/local data respectively.
/// The difference with respect to `SingleFmmTrees` is that in a multi-node setting, each rank can contain multiple or no
/// source/target trees.
#[cfg(feature = "mpi")]
pub trait MultiFmmTree {
    /// Tree associated with FMM tree
    type Tree: MultiTree;

    /// Global domain
    fn domain(&self) -> &<<Self::Tree as MultiTree>::SingleTree as SingleTree>::Domain;

    /// The source tree, defined by a number of single node trees
    fn source_tree(&self) -> &Self::Tree;

    /// The target tree, defined by a number of single node trees
    fn target_tree(&self) -> &Self::Tree;

    /// Number of single node trees associated with the source tree
    fn n_source_trees(&self) -> usize;

    /// Number of single trees associated with target tree
    fn n_target_trees(&self) -> usize;
}

/// Defines a tree node.
pub trait TreeNode
where
    Self: Hash + Eq + PartialOrd,
    Self::Scalar: RlstScalar,
{
    /// Scalar type
    type Scalar;

    /// The computational domain defining the tree.
    type Domain: Domain<Scalar = Self::Scalar>;

    /// Copy of nodes
    type Nodes: IntoIterator<Item = Self>;

    /// Raw representation of a node
    fn raw(&self) -> u64;

    /// The parent of this node
    fn parent(&self) -> Self;

    /// The level of this node
    fn level(&self) -> u64;

    /// Neighbours of this node defined by nodes sharing a vertex, edge, or face
    fn neighbors(&self) -> Self::Nodes;

    /// Children of this node
    fn children(&self) -> Self::Nodes;

    /// Checks adjacency, defined by sharing a vertex, edge, or face, between this node and another
    ///
    /// # Arguments
    /// - `other` - Node being queried.
    fn is_adjacent(&self, other: &Self) -> bool;
}

/// Defines a tree node that provides functionality required by the FMM
pub trait FmmTreeNode
where
    Self: TreeNode,
{
    /// Scale a surface centered at this node, used in the discretisation of the kernel independent fast nultipole
    /// method
    ///
    /// # Arguments
    /// * `surface` - A general surface grid, computed for a given expansion order computed with the
    /// associated function `surface_grid`.
    /// * `domain` - The physical domain with which nodes are being constructed with respect to.
    /// * `alpha` - The multiplier being used to modify the diameter of the surface grid uniformly along each coordinate axis.
    fn scale_surface(
        &self,
        surface: Vec<Self::Scalar>,
        domain: &Self::Domain,
        alpha: Self::Scalar,
    ) -> Vec<Self::Scalar>;

    /// Compute the convolution grid, centered at this node. This method is used in the FFT acceleration of
    /// the field translation operator for kernel independent fast multipole method.
    ///
    /// Returns an owned vector corresponding to the coordinates of the
    /// convolution grid in column major order \[x1, x2, ... xn, y1, y2, ..., yn, z1, z2, ..., zn], as well as a
    /// vector of grid indices.
    ///
    /// # Arguments
    /// * `expansion_order` - The expansion order of the FMM
    /// * `domain` - The physical domain with which nodes are being constructed with respect to.
    /// * `alpha` - The multiplier being used to modify the diameter of the surface grid uniformly along each coordinate axis.
    /// * `conv_point` - The corner point on the surface grid against which the surface and  convolution grids are aligned.
    /// * `conv_point_corner_index` - The index of the corner of the surface grid that the conv point lies on.
    fn convolution_grid(
        &self,
        expansion_order: usize,
        domain: &Self::Domain,
        alpha: Self::Scalar,
        conv_point_corner: &[Self::Scalar],
        conv_point_corner_index: usize,
    ) -> (Vec<Self::Scalar>, Vec<usize>);

    /// Compute the surface grid, centered at this node, for a given expansion order and alpha parameter. This is used
    /// in the discretisation of the kernel independent fast multipole method
    ///
    /// # Arguments
    /// * `domain` - The physical domain with which node are being constructed with respect to.
    /// * `expansion_order` - The expansion order of the FMM
    /// * `alpha` - The multiplier being used to modify the diameter of the surface grid uniformly along each coordinate axis.
    fn surface_grid(
        &self,
        expansion_order: usize,
        domain: &Self::Domain,
        alpha: Self::Scalar,
    ) -> Vec<Self::Scalar>;
}

/// Defines a computational domain
pub trait Domain
where
    Self::Scalar: RlstScalar + Float,
{
    /// Scalar type
    type Scalar;

    /// Origin of computational domain.
    fn origin(&self) -> &[Self::Scalar; 3];

    /// Side length along each axis
    fn side_length(&self) -> &[Self::Scalar; 3];
}
