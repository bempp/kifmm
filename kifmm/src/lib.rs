//! KiFMM
//!
//! Kernel-independent fast multipole method
#![cfg_attr(feature = "strict", deny(warnings))]
#![warn(missing_docs)]

pub mod fftw;
pub mod field;
pub mod fmm;
pub mod helpers;
#[cfg(feature = "mpi_support")]
pub mod hyksort;
pub mod traits;
pub mod tree;
