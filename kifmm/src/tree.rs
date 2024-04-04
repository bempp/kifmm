//! # Single and Multi Node Octrees
//!
//!
//! # Example Usage
//!
//! Create a single node tree. Multi Node examples can be found in the `examples` folder.
//!
//! ```rust
//! use kifmm::tree::helpers::points_fixture;
//! use kifmm::{SingleNodeTree, MultiNodeTree};
//!
//! // Create some test points
//! let npoints = 1000;
//! let points = points_fixture(npoints, None, None, None);
//!
//! // Set tree parameters
//! let sparse = true; // Setting to sparse mode drops empty leaves and ancestors from final tree
//! let domain = None; // Not explicitly setting a domain leads to calculation from point data
//! let depth = 3; // The depth of the tree
//!
//! // Create a single node tree
//! let single_node = SingleNodeTree::new(
//!     points.data(),
//!     depth,
//!     sparse,
//!     domain
//! );
//!  ```

pub mod constants;
pub mod types;

mod domain;
pub mod helpers;
pub mod morton;
mod point;
mod single_node;

#[cfg(feature = "mpi")]
mod multi_node;

// Public API
#[doc(inline)]
pub use crate::tree::types::SingleNodeTree;

#[cfg(feature = "mpi")]
#[doc(inline)]
pub use crate::tree::types::MultiNodeTree;
