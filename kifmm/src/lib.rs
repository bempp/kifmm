//! # Kernel Independent Fast Multipole Method (KiFMM)
//!
//! A Kernel Independent Fast Multipole method designed for portability, and flexible algorithmic construction based on \[1\].
//!
//! Notable features of this library are:
//! * Support for single and multinode simulations via MPI.
//! * Heterogenous acceleration for the the field translations (M2L) and direct summation (P2P) steps.
//! * Flexible trait based interface for developing alternative operator implementations, or indeed related fast algorithms
//!
//!
//! ## References
//! \[1\] Ying, L., Biros, G., & Zorin, D. (2004). A kernel-independent adaptive fast multipole algorithm in two and three dimensions. Journal of Computational Physics, 196(2), 591-626.
#![cfg_attr(feature = "strict", deny(warnings))]
#![warn(missing_docs)]

pub mod fftw;
pub mod fmm;
pub mod helpers;
#[cfg(feature = "mpi_support")]
pub mod hyksort;
pub mod traits;
pub mod tree;

// Public API
#[doc(inline)]
pub use fmm::types::KiFmmBuilderSingleNode as KiFmmBuilderSingleNode;
#[doc(inline)]
pub use fmm::types::BlasFieldTranslationKiFmm as BlasFieldTranslationKiFmm;
#[doc(inline)]
pub use fmm::types::FftFieldTranslationKiFmm as FftFieldTranslationKiFmm;
#[doc(inline)]
pub use fmm::types::SingleNodeFmmTree as SingleNodeFmmTree;


