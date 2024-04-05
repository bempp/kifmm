//! Tree Traits
use std::{collections::HashSet, hash::Hash};

use num::Float;
use rlst::RlstScalar;

/// Interface for single and multi-node trees
pub trait Tree {
    /// The computational domain defining the tree.
    type Domain: Domain;

    /// Scalar type
    type Scalar: RlstScalar<Real = Self::Scalar> + Float + Default;

    /// A tree node.
    type Node: TreeNode<Self::Scalar, Domain = Self::Domain> + Clone + Copy;

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
    fn coordinates(&self, key: &Self::Node) -> Option<&[Self::Scalar]>;

    /// Number of coordinates
    fn ncoordinates(&self, key: &Self::Node) -> Option<usize>;

    /// Gets a reference to the coordinates contained in across tree (local in multinode setting)
    fn all_coordinates(&self) -> Option<&[Self::Scalar]>;

    /// Total number of coordinates
    fn ncoordinates_tot(&self) -> Option<usize>;

    /// Gets global indices at a leaf (local in multinode setting)
    fn global_indices<'a>(&'a self, key: &Self::Node) -> Option<&'a [usize]>;

    /// Gets all global indices (local in multinode setting)
    fn all_global_indices(&self) -> Option<&[usize]>;

    /// Get domain defined by the points, gets global domain in multi-node setting.
    fn domain(&self) -> &'_ Self::Domain;

    /// Get a map from the key to index position in sorted keys
    fn index(&self, key: &Self::Node) -> Option<&usize>;

    /// Get a node
    fn node(&self, idx: usize) -> Option<&Self::Node>;

    /// Get a map from the key to leaf index position in sorted leaves
    fn leaf_index(&self, key: &Self::Node) -> Option<&usize>;
}

/// Interface for trees required by the fast multipole method (FMM), which requires
/// separate trees for the source and target particle data.
pub trait FmmTree {
    /// Scalar type
    type Scalar;

    /// Node type
    type Node;

    /// Tree type
    type Tree: Tree<Scalar = Self::Scalar, Node = Self::Node>;

    /// Get the source tree
    fn source_tree(&self) -> &Self::Tree;

    /// Get the target tree
    fn target_tree(&self) -> &Self::Tree;

    /// Get the domain
    fn domain(&self) -> &<Self::Tree as Tree>::Domain;

    /// Get the near field of a leaf node
    fn near_field(&self, leaf: &Self::Node) -> Option<Vec<Self::Node>>;
}

/// Interface for tree nodes
pub trait TreeNode<T>
where
    Self: Hash + Eq,
    T: RlstScalar,
{
    /// The computational domain defining the tree.
    type Domain: Domain;

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
    fn is_adjacent(&self, other: &Self) -> bool;
}

/// Interface for a tree node that provides functionality required by the FMM
pub trait FmmTreeNode<T>
where
    Self: TreeNode<T>,
    T: RlstScalar,
{
    /// Compute the convolution grid centered at a given node and its respective surface grid. This method is used
    /// in the FFT acceleration of the field translation operator for kernel independent fast multipole method.
    /// Returns an owned vector corresponding to the coordinates of the
    /// convolution grid in column major order [x_1, x_2, ... x_N, y_1, y_2, ..., y_N, z_1, z_2, ..., z_N], as well as a
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
        alpha: T,
        conv_point_corner: &[T],
        conv_point_corner_index: usize,
    ) -> (Vec<T>, Vec<usize>);

    /// Compute surface grid for a given expansion order used in the kernel independent fast multipole method
    /// returns a tuple, the first element is an owned vector of the physical coordinates of the
    /// surface grid in column major order [x_1, x_2, ... x_n, y_1, y_2, ..., y_n, z_1, z_2, ..., z_n].
    /// the second element is a vector of indices corresponding to each of these coordinates.
    ///
    /// # Arguments
    /// * `expansion_order` - the expansion order of the fmm
    fn surface_grid(expansion_order: usize) -> Vec<T>;

    /// Scale a surface grid centered at this node, used in the discretisation of the kernel independent fast nultipole
    /// method
    ///
    /// # Arguments
    /// * `surface` - A general surface grid, computed for a given expansion order computed with the
    /// associated function `surface_grid`.
    /// * `domain` - The physical domain with which nodes are being constructed with respect to.
    /// * `alpha` - The multiplier being used to modify the diameter of the surface grid uniformly along each coordinate axis.
    fn scale_surface(&self, surface: Vec<T::Real>, domain: &Self::Domain, alpha: T)
        -> Vec<T::Real>;

    /// Compute the surface grid, centered at this node, for a given expansion order and alpha parameter. This is used
    /// in the discretisation of the kernel independent fast multipole method
    ///
    /// # Arguments
    /// * `domain` - The physical domain with which node are being constructed with respect to.
    /// * `expansion_order` - The expansion order of the FMM
    /// * `alpha` - The multiplier being used to modify the diameter of the surface grid uniformly along each coordinate axis.
    fn compute_surface(
        &self,
        domain: &Self::Domain,
        expansion_order: usize,
        alpha: T,
    ) -> Vec<T::Real>;
}

/// Interface for computational domain
pub trait Domain {
    /// Scalar type
    type Scalar: RlstScalar;

    /// Origin of computational domain.
    fn origin(&self) -> &[Self::Scalar; 3];

    /// Side length along each axis
    fn diameter(&self) -> &[Self::Scalar; 3];
}
