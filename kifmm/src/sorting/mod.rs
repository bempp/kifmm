//! Parallel sorting algorithms
pub mod hyksort;
pub mod samplesort;
pub mod simplesort;

pub use hyksort::hyksort;
pub use samplesort::samplesort;
pub use simplesort::simplesort;
