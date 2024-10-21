//! FMM traits
use crate::traits::tree::SingleFmmTree;
use green_kernels::{traits::Kernel, types::GreenKernelEvalType};
use rlst::RlstScalar;

use super::{tree::SingleTree, types::FmmError};

#[cfg(feature = "mpi")]
use super::tree::MultiFmmTree;

#[cfg(feature = "mpi")]
use crate::traits::tree::MultiTree;

/// Multinode FMM data access
#[cfg(feature = "mpi")]
pub trait DataAccessMulti {
    /// Data associated with FMM, must implement RlstScalar.
    type Scalar;

    /// Type of tree, must implement `FmmTree`, allowing for separate source and target trees.
    type Tree: MultiFmmTree;

    /// Kernel associated with this FMMl
    type Kernel: Kernel<T = Self::Scalar>;

    /// Get the potential data associated with the particles contained at a given node
    /// # Arguments
    /// * `key` - The target leaf node.
    fn potential(
        &self,
        leaf: &<<<Self::Tree as MultiFmmTree>::Tree as MultiTree>::SingleTree as SingleTree>::Node,
    ) -> Option<Vec<&[Self::Scalar]>>;

    /// Get the multipole expansion data associated with a node as a slice
    /// # Arguments
    /// * `key` - The source node.
    fn multipole(
        &self,
        key: &<<<Self::Tree as MultiFmmTree>::Tree as MultiTree>::SingleTree as SingleTree>::Node,
    ) -> Option<&[Self::Scalar]>;

    /// Get the multipole expansion data associated with a node as a slice
    /// # Arguments
    /// * `key` - The source node.
    fn multipole_mut(
        &self,
        key: &<<<Self::Tree as MultiFmmTree>::Tree as MultiTree>::SingleTree as SingleTree>::Node,
    ) -> Option<&mut [Self::Scalar]>;

    /// Get the local expansion data associated with a node as a slice
    /// # Arguments
    /// * `key` - The source node.
    fn local(
        &self,
        key: &<<<Self::Tree as MultiFmmTree>::Tree as MultiTree>::SingleTree as SingleTree>::Node,
    ) -> Option<&[Self::Scalar]>;

    /// Get the local expansion data associated with a node as a slice
    /// # Arguments
    /// * `key` - The source node.
    fn local_mut(
        &self,
        key: &<<<Self::Tree as MultiFmmTree>::Tree as MultiTree>::SingleTree as SingleTree>::Node,
    ) -> Option<&mut [Self::Scalar]>;

    /// Get the multipole expansion data associated with a tree level as a slice
    /// # Arguments
    /// * `level` - The tree level.
    fn multipoles(&self, level: u64) -> Option<&[Self::Scalar]>;

    /// Get the local expansion data associated with a tree level as a slice
    /// # Arguments
    /// * `level` - The tree level.
    fn locals(&self, level: u64) -> Option<&[Self::Scalar]>;

    /// Get the expansion order associated with this FMM, used to discretise the equivalent surface.
    fn equivalent_surface_order(&self, level: u64) -> usize;

    /// Get the expansion order associated with this FMM, used to discretise the check surface.
    fn check_surface_order(&self, level: u64) -> usize;

    /// Check whether or not expansion order is set variably
    fn variable_expansion_order(&self) -> bool;

    /// Get the number of multipole/local coefficients associated with this FMM
    fn n_coeffs_equivalent_surface(&self, level: u64) -> usize;

    /// Get the number of multipole/local coefficients associated with this FMM
    fn n_coeffs_check_surface(&self, level: u64) -> usize;

    /// Get the tree associated with this FMM
    fn tree(&self) -> &Self::Tree;

    /// Get the kernel associated with this FMM
    fn kernel(&self) -> &Self::Kernel;

    /// Get the dimension of the data in this FMM
    fn dim(&self) -> usize;
}

/// Access data for FMM objects
pub trait DataAccess {
    /// Data associated with FMM, must implement RlstScalar.
    type Scalar: RlstScalar;

    /// Type of tree, must implement `FmmTree`, allowing for separate source and target trees.
    type Tree: SingleFmmTree;

    /// Kernel associated with this FMMl
    type Kernel: Kernel<T = Self::Scalar>;

    /// Get the multipole expansion data associated with a node as a slice
    /// # Arguments
    /// * `key` - The source node.
    fn multipole(
        &self,
        key: &<<Self::Tree as SingleFmmTree>::Tree as SingleTree>::Node,
    ) -> Option<&[Self::Scalar]>;

    /// Get the multipole expansion data associated with a tree level as a slice
    /// # Arguments
    /// * `level` - The tree level.
    fn multipoles(&self, level: u64) -> Option<&[Self::Scalar]>;

    /// Get the local expansion data associated with a node as a slice
    /// # Arguments
    /// * `key` - The target node.
    fn local(
        &self,
        key: &<<Self::Tree as SingleFmmTree>::Tree as SingleTree>::Node,
    ) -> Option<&[Self::Scalar]>;

    /// Get the local expansion data associated with a tree level as a slice
    /// # Arguments
    /// * `level` - The tree level.
    fn locals(&self, level: u64) -> Option<&[Self::Scalar]>;

    /// Get the potential data associated with the particles contained at a given node
    /// # Arguments
    /// * `key` - The target leaf node.
    fn potential(
        &self,
        leaf: &<<Self::Tree as SingleFmmTree>::Tree as SingleTree>::Node,
    ) -> Option<Vec<&[Self::Scalar]>>;

    /// Get all potential data at all particles, stored in order by global index
    fn potentials(&self) -> Option<&Vec<Self::Scalar>>;

    /// Get the expansion order associated with this FMM, used to discretise the equivalent surface.
    fn equivalent_surface_order(&self, level: u64) -> usize;

    /// Get the expansion order associated with this FMM, used to discretise the check surface.
    fn check_surface_order(&self, level: u64) -> usize;

    /// Check whether or not expansion order is set variably
    fn variable_expansion_order(&self) -> bool;

    /// Get the number of multipole/local coefficients associated with this FMM
    fn n_coeffs_equivalent_surface(&self, level: u64) -> usize;

    /// Get the number of multipole/local coefficients associated with this FMM
    fn n_coeffs_check_surface(&self, level: u64) -> usize;

    /// Get the tree associated with this FMM
    fn tree(&self) -> &Self::Tree;

    /// Get the kernel associated with this FMM
    fn kernel(&self) -> &Self::Kernel;

    /// Get the dimension of the data in this FMM
    fn dim(&self) -> usize;

    /// Clear the data buffers and add new charge data for re-evaluation.
    ///
    /// # Arguments
    /// * `charges` - new charge data.
    fn clear(&mut self, charges: &[<Self as DataAccess>::Scalar]);
}

/// Interface for a Kernel-Independent Fast Multipole Method (FMM).
///
/// This trait abstracts the core functionalities of FMM, allowing for different underlying
/// data structures, kernels, and precision types. It supports operations essential for
/// executing FMM calculations, including accessing multipole and local expansions, evaluating
/// potentials, and managing the underlying tree structure and kernel functions.
pub trait Evaluate
where
    Self: DataAccess,
{
    /// Evaluate the leaf level operations for source tree
    fn evaluate_leaf_sources(&mut self, timed: bool) -> Result<(), FmmError>;

    /// Evaluate the leaf level operations for target tree
    fn evaluate_leaf_targets(&mut self, timed: bool) -> Result<(), FmmError>;

    /// Evaluate the upward pass
    fn evaluate_upward_pass(&mut self, timed: bool) -> Result<(), FmmError>;

    /// Evaluate the downward pass
    fn evaluate_downward_pass(&mut self, timed: bool) -> Result<(), FmmError>;

    /// Evaluate the potentials, or potential gradients, for this FMM
    fn evaluate(&mut self, timed: bool) -> Result<(), FmmError>;
}

/// Interface for multi node FMM
#[cfg(feature = "mpi")]
pub trait EvaluateMulti
where
    Self: DataAccessMulti,
{
    /// Evaluate the leaf level operations for source tree
    fn evaluate_leaf_sources(&mut self, timed: bool) -> Result<(), FmmError>;

    /// Evaluate the leaf level operations for target tree
    fn evaluate_leaf_targets(&mut self, timed: bool) -> Result<(), FmmError>;

    /// Evaluate the upward pass
    fn evaluate_upward_pass(&mut self, timed: bool) -> Result<(), FmmError>;

    /// Evaluate the downward pass
    fn evaluate_downward_pass(&mut self, timed: bool) -> Result<(), FmmError>;

    /// Evaluate the potentials, or potential gradients, for this FMM
    fn evaluate(&mut self, timed: bool) -> Result<(), FmmError>;
}

/// Set all metadata required for FMMs
pub trait Metadata {
    /// Associated scalar
    type Scalar: RlstScalar;

    /// Compute all metadata required for FMM.
    /// TODO: Breakup into smaller pieces of functionality for clarity.
    fn metadata(&mut self, eval_type: GreenKernelEvalType, charges: &[Self::Scalar]);
}

/// Defines how metadata associated with field translations is looked up at runtime.
/// Defined by kernel type, as well as field translation method.
pub trait MetadataAccess
where
    Self: Metadata,
{
    /// Lookup convolution grid map for FFT based M2L operator
    fn fft_map_index(&self, level: u64) -> usize;

    /// Lookup expansion order
    fn expansion_index(&self, level: u64) -> usize;

    /// Lookup c2e operator
    fn c2e_operator_index(&self, level: u64) -> usize;

    /// Lookup m2m operator
    fn m2m_operator_index(&self, level: u64) -> usize;

    /// Lookup m2l operator
    fn m2l_operator_index(&self, level: u64) -> usize;

    /// Lookup l2l operator
    fn l2l_operator_index(&self, level: u64) -> usize;

    /// Displacement index
    fn displacement_index(&self, level: u64) -> usize;
}

/// Marker trait for homogenous kernels
pub trait HomogenousKernel
where
    Self: Kernel,
{
    /// Homogeneity check
    fn is_homogenous(&self) -> bool;
}
