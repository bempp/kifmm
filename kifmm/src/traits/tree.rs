//! Tree Traits
use std::{collections::HashSet, hash::Hash};

use rlst::RlstScalar;

use num::traits::Float;

/// Interface for single and multi-node trees
pub trait Tree {
    /// Scalar type
    type Scalar: RlstScalar + Float;

    /// The computational domain defining the tree.
    type Domain: Domain<Scalar = Self::Scalar>;

    /// A tree node.
    type Node: TreeNode<Scalar = Self::Scalar, Domain = Self::Domain> + Clone + Copy;

    /// Slice of nodes.
    type NodeSlice<'a>: IntoIterator<Item = &'a Self::Node>
    where
        Self: 'a;

    /// Copy of nodes
    type Nodes: IntoIterator<Item = Self::Node>;

    /// Number of leaves
    fn n_leaves(&self) -> Option<usize>;

    /// Total number of keys
    fn n_keys_tot(&self) -> Option<usize>;

    /// Number of keys at a given tree level
    fn n_keys(&self, level: u64) -> Option<usize>;

    /// Get depth of tree.
    fn depth(&self) -> u64;

    /// Get a reference to all leaves, gets local keys in multi-node setting.
    fn all_leaves(&self) -> Option<Self::NodeSlice<'_>>;

    /// Get a reference to keys at a given level, gets local keys in a multi-node setting.
    fn keys(&self, level: u64) -> Option<Self::NodeSlice<'_>>;

    /// Get a reference to all keys, gets local keys in a multi-node setting.
    fn all_keys(&self) -> Option<Self::NodeSlice<'_>>;

    /// Get a reference to all keys as a set, gets local keys in a multi-node setting.
    fn all_keys_set(&self) -> Option<&'_ HashSet<Self::Node>>;

    /// Get a reference to all leaves as a set, gets local keys in a multi-node setting.
    fn all_leaves_set(&self) -> Option<&'_ HashSet<Self::Node>>;

    /// Gets a reference to the coordinates contained with a leaf node.
    ///
    /// # arguments
    /// - `leaf` - node being query.
    fn coordinates(&self, leaf: &Self::Node) -> Option<&[Self::Scalar]>;

    /// Query number of coordinates contained at a given leaf node
    ///
    /// # arguments
    /// - `leaf` - node being query.
    fn n_coordinates(&self, leaf: &Self::Node) -> Option<usize>;

    /// Gets a reference to the coordinates contained in across tree (local in multi node setting)
    fn all_coordinates(&self) -> Option<&[Self::Scalar]>;

    /// Total number of coordinates (local in a multi node setting)
    fn n_coordinates_tot(&self) -> Option<usize>;

    /// Gets global indices at a leaf node (local in multi node setting)
    ///
    /// # Arguments
    /// - `leaf` - Node being query.
    fn global_indices(&self, leaf: &Self::Node) -> Option<&[usize]>;

    /// Gets all global indices (local in mult inode setting)
    fn all_global_indices(&self) -> Option<&[usize]>;

    /// Get domain defined by the points, gets global domain in multi node setting.
    fn domain(&self) -> &Self::Domain;

    /// Map from the key to index position in sorted keys
    ///
    /// # Arguments
    /// - `key` - Node being query.
    fn index(&self, key: &Self::Node) -> Option<&usize>;

    /// Map from the key to index position in sorted keys at a given level
    ///
    /// # Arguments
    /// - `key` - Node being query.
    fn level_index(&self, key: &Self::Node) -> Option<&usize>;

    /// Map from the leaf to its index position in sorted leaves
    ///
    /// # Arguments
    /// - `leaf` - Node being query.
    fn leaf_index(&self, leaf: &Self::Node) -> Option<&usize>;

    /// Map from an index position to a node
    ///
    /// # Arguments
    /// - `idx` - Index being query.
    fn node(&self, idx: usize) -> Option<&Self::Node>;
}

/// Interface for trees required by the FMM, which requires separate trees for the source and target particle data
pub trait FmmTree
where
    Self::Tree: Tree,
{
    /// Tree associated with this FMM tree
    type Tree;

    /// Get the source tree
    fn source_tree(&self) -> &Self::Tree;

    /// Get the target tree
    fn target_tree(&self) -> &Self::Tree;

    /// Get the domain
    fn domain(&self) -> &<Self::Tree as Tree>::Domain;

    /// Get the near field of a leaf node
    fn near_field(
        &self,
        leaf: &<Self::Tree as Tree>::Node,
    ) -> Option<Vec<<Self::Tree as Tree>::Node>>;
}

/// Interface for tree nodes
pub trait TreeNode
where
    Self: Hash + Eq,
    Self::Scalar: RlstScalar + Float,
{
    /// Scalar type
    type Scalar;

    /// The computational domain defining the tree.
    type Domain: Domain<Scalar = Self::Scalar>;

    /// Copy of nodes
    type Nodes: IntoIterator<Item = Self>;

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
    /// - `other` - Node being query.
    fn is_adjacent(&self, other: &Self) -> bool;
}

/// Interface for a tree node that provides functionality required by the FMM
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

/// Interface for computational domain
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
