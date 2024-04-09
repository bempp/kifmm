//! # Kernel Independent Fast Multipole Method (KiFMM)
//!
//! A Kernel Independent Fast Multipole method designed for portability, and flexible algorithmic construction based on \[1\].
//!
//! Notable features of this library are:
//! * Support for single and multinode simulations via MPI.
//! * Heterogenous acceleration for the the field translations (M2L) and direct summation (P2P) steps.
//! * Flexible trait based interface for developing alternative operator implementations, or indeed related fast algorithms
//!
//! # Example Usage
//!
//! `Fmm` objects are built using the `SingleNodeBuilder` objects, for which we give a single node example below. These objects implement
//! the `Fmm` trait, which allows for the evaluation of the algorithm and interaction with the results.
//!
//! Basic usage for evaluating an FMM between a set of source and target points
//!
//! ```rust
//! # extern crate blas_src;
//! # extern crate lapack_src;
//!
//! use green_kernels::{laplace_3d::Laplace3dKernel, types::EvalType};
//! use kifmm::{Fmm, BlasFieldTranslation, FftFieldTranslation, SingleNodeBuilder};
//! use kifmm::tree::helpers::points_fixture;
//! use rlst::{rlst_dynamic_array2, RawAccessMut};
//!
//! // Setup random sources and targets
//! let nsources = 1000;
//! let ntargets = 2000;
//! let sources = points_fixture::<f32>(nsources, None, None, Some(0));
//! let targets = points_fixture::<f32>(ntargets, None, None, Some(1));
//!
//! // FMM parameters
//! let n_crit = Some(150); // Threshold for number of particles in a leaf box
//! let expansion_order = 5; // Expansion order of multipole/local expansions
//! let sparse = true; // Whether to exclude empty boxes in octrees
//!
//! // FFT based Field Translation
//! {
//!     let nvecs = 1;
//!     let tmp = vec![1.0; nsources * nvecs];
//!     let mut charges = rlst_dynamic_array2!(f32, [nsources, nvecs]);
//!     charges.data_mut().copy_from_slice(&tmp);
//!
//!     // Build FMM object, with a given kernel and field translation
//!     let mut fmm_fft = SingleNodeBuilder::new()
//!         .tree(&sources, &targets, n_crit, sparse)
//!         .unwrap()
//!         .parameters(
//!             &charges,
//!             expansion_order,
//!             Laplace3dKernel::new(), // Set the kernel
//!             EvalType::Value, // Set the type of evaluation, either just potentials or potentials + potential gradients
//!             FftFieldTranslation::new(), // Choose a field translation method, could replace with BLAS field translation
//!         )
//!         .unwrap()
//!         .build()
//!         .unwrap();
//!
//!     // Run the FMM
//!     fmm_fft.evaluate();
//!
//!     // Optionally clear, to re-evaluate with new charges
//!     let nvecs = 1;
//!     let tmp = vec![1.0; nsources * nvecs];
//!     let mut new_charges = rlst_dynamic_array2!(f32, [nsources, nvecs]);
//!     new_charges.data_mut().copy_from_slice(&tmp);
//!     fmm_fft.clear(&new_charges);
//! }
//!
//! ````
//!
//!
//! More sophisticated examples, such as setting up FMMs to operate on multiple input charge vectors, can be found in the `examples` folder.
//!
//! ## References
//! \[1\] Ying, L., Biros, G., & Zorin, D. (2004). A kernel-independent adaptive fast multipole algorithm in two and three dimensions. Journal of Computational Physics, 196(2), 591-626.
#![cfg_attr(feature = "strict", deny(warnings))]
#![warn(missing_docs)]

pub mod fftw;
pub mod fmm;
#[cfg(feature = "mpi")]
pub mod hyksort;
pub mod traits;
pub mod tree;

// Public API
#[doc(inline)]
pub use fmm::types::BlasFieldTranslation;
#[doc(inline)]
pub use fmm::types::FftFieldTranslation;
#[doc(inline)]
pub use fmm::types::SingleNodeBuilder;
#[doc(inline)]
pub use fmm::types::SingleNodeFmmTree;

#[cfg(feature = "mpi")]
#[doc(inline)]
pub use fmm::types::MultiNodeFmmTree;

#[cfg(feature = "mpi")]
use mpi::traits::Equivalence;

use num::Float;
use num_complex::ComplexFloat;
use rlst::RlstScalar;
use rlst::{c32, c64};
#[doc(inline)]
pub use traits::fmm::Fmm;

/// Super trait of RlstScalar and  Float trait
pub trait RlstScalarFloat: RlstScalar + Float + Default {}

/// Super trait of RlstScalar and Complex Float trait
pub trait RlstScalarComplexFloat: RlstScalar + ComplexFloat + Default {}

#[cfg(feature = "mpi")]
/// Super trait of RlstScalar and Float trait, and MPI Equivalence trait
pub trait RlstScalarFloatMpi: RlstScalarFloat + Equivalence {}

#[cfg(feature = "mpi")]
/// Super trait of RlstScalar and Complex Float trait, and MPI Equivalence trait
pub trait RlstScalarComplexFloatMpi: RlstScalarFloatMpi + Equivalence {}


impl RlstScalarFloat for f64 {}
impl RlstScalarFloat for f32 {}

#[cfg(feature = "mpi")]
impl RlstScalarFloatMpi for f64 {}
#[cfg(feature = "mpi")]
impl RlstScalarFloatMpi for f32 {}

impl RlstScalarComplexFloat for c64 {}
impl RlstScalarComplexFloat for c32 {}
