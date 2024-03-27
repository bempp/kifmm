//! KiFMM
//!
//! Kernel-independent fast multipole method
#![cfg_attr(feature = "strict", deny(warnings))]
#![warn(missing_docs)]

pub mod field;
pub mod fmm;
pub mod helpers;
pub mod hyksort;
pub mod traits;
pub mod tree;
