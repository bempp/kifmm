//! # Single and Multi Node Octrees
//!
//!
//! # Example Usage
//!
//! Create a single node tree.
//!
//! ```rust
//! use rlst::RawAccess;
//! use kifmm::tree::{SingleNodeTree, helpers::points_fixture};
//!
//! // Create some test points
//! let n_points = 1000;
//! let points = points_fixture::<f32>(n_points, None, None, None);
//!
//! // Set tree parameters
//! let prune_empty = true; // Setting to prune_empty mode drops empty leaves and ancestors from final tree
//! let domain = None; // Not explicitly setting a domain leads to calculation from point data
//! let depth = 3; // The depth of the tree
//!
//! // Create a single node tree
//! let single_node = SingleNodeTree::<f32>::new(
//!     points.data(),
//!     depth,
//!     prune_empty,
//!     domain
//! ).unwrap();
//!  ```
//!
//! For more examples, including the creation of multi-node trees, see the `examples` folder

pub mod constants;
pub mod types;

mod domain;
pub mod helpers;
pub mod morton;
mod point;
mod single_node;

#[cfg(feature = "mpi")]
pub mod multi_node;

// Public API
#[doc(inline)]
pub use crate::tree::types::SingleNodeTree;

#[cfg(feature = "mpi")]
#[doc(inline)]
pub use crate::tree::types::MultiNodeTree;
