//! # Kernel Independent Fast Multipole Method (KiFMM)
//!
//! A Kernel Independent Fast Multipole method designed for portability, and flexible algorithmic construction based on \[1\].
//!
//! Notable features of this library are:
//! * Highly competitive single-node implementation of the kernel independent fast multipole method, with a Laplace/Helmholtz implementation provided.
//! * BLAS and FFT acceleration for the the field translations (M2L)
//! * The ability to handle multiple right hand sides when using BLAS based M2L
//! * Overdetermined check and equivalent surface construction when using BLAS based M2L
//! * The ability to vary expansion orders by level, useful for oscillatory problems
//!
//! # Example Usage
//!
//! `Fmm` objects are built using the `SingleNodeBuilder` objects, for which we give a single node example below. These objects implement
//! the `Fmm` trait, which allows for the evaluation of the algorithm and interaction with the results.
//!
//! Basic usage for evaluating an FMM between a set of source and target points
//!
//! ```rust
//! use green_kernels::{laplace_3d::Laplace3dKernel, types::EvalType};
//! use kifmm::{SingleFmm, BlasFieldTranslationSaRcmp, FftFieldTranslation, SingleNodeBuilder};
//! use kifmm::tree::helpers::points_fixture;
//! use rlst::{rlst_dynamic_array2, RawAccessMut, RawAccess};
//!
//! // Setup random sources and targets
//! let n_sources = 1000;
//! let n_targets = 2000;
//! let sources = points_fixture::<f32>(n_sources, None, None, Some(0));
//! let targets = points_fixture::<f32>(n_targets, None, None, Some(1));
//!
//! // FMM parameters
//! let n_crit = Some(150); // Threshold for number of particles in a leaf box
//! let depth = None; //
//! let expansion_order = [5]; // Expansion order of multipole/local expansions
//! let prune_empty = true; // Whether to exclude empty boxes in octrees
//!
//! // FFT based Field Translation
//! {
//!     let nvecs = 1;
//!     let tmp = vec![1.0; n_sources * nvecs];
//!     let mut charges = rlst_dynamic_array2!(f32, [n_sources, nvecs]);
//!     charges.data_mut().copy_from_slice(&tmp);
//!
//!     // Build FMM object, with a given kernel and field translation
//!     let mut fmm_fft = SingleNodeBuilder::new()
//!         .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
//!         .unwrap()
//!         .parameters(
//!             charges.data(),
//!             &expansion_order,
//!             Laplace3dKernel::new(), // Set the kernel
//!             EvalType::Value, // Set the type of evaluation, either just potentials or potentials + potential gradients
//!             FftFieldTranslation::new(None), // Choose a field translation method, could replace with BLAS field translation
//!         )
//!         .unwrap()
//!         .build()
//!         .unwrap();
//!
//!     // Run the FMM
//!     fmm_fft.evaluate(false);
//!
//!     // Optionally clear, to re-evaluate with new charges
//!     let nvecs = 1;
//!     let tmp = vec![1.0; n_sources * nvecs];
//!     let mut new_charges = rlst_dynamic_array2!(f32, [n_sources, nvecs]);
//!     new_charges.data_mut().copy_from_slice(&tmp);
//!     fmm_fft.clear(charges.data());
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
#![allow(clippy::doc_lazy_continuation)]
#![allow(clippy::macro_metavars_in_unsafe)]
pub mod fftw;
pub mod fmm;
pub mod linalg;
#[cfg(feature = "mpi")]
pub mod sorting;
pub mod traits;
pub mod tree;

// Public API
#[doc(inline)]
pub use fmm::types::BlasFieldTranslationIa;
#[doc(inline)]
pub use fmm::types::BlasFieldTranslationSaRcmp;
#[doc(inline)]
pub use fmm::types::FftFieldTranslation;
#[doc(inline)]
pub use fmm::types::FmmSvdMode;
#[doc(inline)]
pub use fmm::types::SingleNodeBuilder;
#[doc(inline)]
pub use fmm::types::SingleNodeFmmTree;

#[cfg(feature = "mpi")]
#[doc(inline)]
pub use fmm::types::MultiNodeFmmTree;

#[doc(inline)]
pub use traits::fmm::SingleFmm;
#[cfg_attr(feature = "strict", deny(warnings))]
#[warn(missing_docs)]
pub mod bindings;
