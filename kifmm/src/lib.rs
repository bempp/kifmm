//! # Kernel Independent Fast Multipole Method (KiFMM)
//!
//! A Kernel Independent Fast Multipole method designed for portability, and flexible algorithmic construction based on \[1\].
//!
//! Notable features of this library are:
//! * Highly optimised single-node implementation of the kernel independent fast multipole method, with a Laplace/Helmholtz implementation provided.
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
//! use kifmm::{Fmm, BlasFieldTranslationSaRcmp, FftFieldTranslation, SingleNodeBuilder};
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
pub use fmm::types::BlasFieldTranslationIa;
#[doc(inline)]
pub use fmm::types::BlasFieldTranslationSaRcmp;
#[doc(inline)]
pub use fmm::types::FftFieldTranslation;
#[doc(inline)]
pub use fmm::types::SingleNodeBuilder;
#[doc(inline)]
pub use fmm::types::SingleNodeFmmTree;

#[cfg(feature = "mpi")]
#[doc(inline)]
pub use fmm::types::MultiNodeFmmTree;

#[doc(inline)]
pub use traits::fmm::Fmm;

/// Python API
mod python_api {

    use self::fmm::KiFmm;

    use super::*;

    use green_kernels::laplace_3d::Laplace3dKernel;
    use green_kernels::types::EvalType;
    use pyo3::prelude::*;
    use rlst::{rlst_array_from_slice2, rlst_dynamic_array2};
    use std::collections::HashMap;
    use std::sync::Mutex;

    // Global store for MyStruct instances
    lazy_static::lazy_static! {
        static ref STORE_LAPLACE_FFT_F32: Mutex<HashMap<usize, KiFmm<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>>>> = Mutex::new(HashMap::new());
        static ref STORE_LAPLACE_BLAS_F32: Mutex<HashMap<usize, KiFmm<f32, Laplace3dKernel<f32>, BlasFieldTranslationSaRcmp<f32>>>> = Mutex::new(HashMap::new());
        static ref NEXT_ID: Mutex<usize> = Mutex::new(0);
    }

    /// Formats the sum of two numbers as string.
    #[pyfunction]
    fn f32_laplace_blas(
        expansion_order: usize,
        n_crit: u64,
        sparse: bool,
        eval_type: usize,
        sources: Vec<f32>,
        targets: Vec<f32>,
        charges: Vec<f32>,
        svd_threshold: f32,
    ) -> PyResult<usize> {
        let mut next_id = NEXT_ID.lock().unwrap();
        let mut store = STORE_LAPLACE_BLAS_F32.lock().unwrap();
        let struct_id = *next_id;

        // Copy source/target/charge data
        let dim = 3;
        let n_sources = sources.len() / dim;
        let n_targets = targets.len() / dim;

        let mut sources_arr = rlst_dynamic_array2!(f32, [n_sources, dim]);
        let mut targets_arr = rlst_dynamic_array2!(f32, [n_targets, dim]);
        let mut charges_arr = rlst_dynamic_array2!(f32, [n_sources, 1]);
        let sources_slice = rlst_array_from_slice2!(sources.as_slice(), [n_sources, dim]);
        let targets_slice = rlst_array_from_slice2!(targets.as_slice(), [n_targets, dim]);
        let charges_slice = rlst_array_from_slice2!(charges.as_slice(), [n_sources, 1]);
        sources_arr.view_mut().fill_from(sources_slice.view());
        targets_arr.view_mut().fill_from(targets_slice.view());
        charges_arr.view_mut().fill_from(charges_slice.view());

        // Set FMM parameters
        let kernel = Laplace3dKernel::new();
        let eval_type = if eval_type == 0 {
            EvalType::Value
        } else if eval_type == 1 {
            EvalType::ValueDeriv
        } else {
            EvalType::Value
        };
        let source_to_target = BlasFieldTranslationSaRcmp::new(Some(svd_threshold));

        let fmm = SingleNodeBuilder::new()
            .tree(&sources_arr, &targets_arr, Some(n_crit), sparse)
            .unwrap()
            .parameters(
                &charges_arr,
                expansion_order,
                kernel,
                eval_type,
                source_to_target,
            )
            .unwrap()
            .build()
            .unwrap();

        store.insert(struct_id, fmm);
        *next_id += 1;
        Ok(struct_id)
    }

    /// Formats the sum of two numbers as string.
    #[pyfunction]
    fn f32_laplace_fft(
        expansion_order: usize,
        n_crit: u64,
        sparse: bool,
        eval_type: usize,
        sources: Vec<f32>,
        targets: Vec<f32>,
        charges: Vec<f32>,
    ) -> PyResult<usize> {
        let mut next_id = NEXT_ID.lock().unwrap();
        let mut store = STORE_LAPLACE_FFT_F32.lock().unwrap();
        let struct_id = *next_id;

        // Copy source/target/charge data
        let dim = 3;
        let n_sources = sources.len() / dim;
        let n_targets = targets.len() / dim;

        let mut sources_arr = rlst_dynamic_array2!(f32, [n_sources, dim]);
        let mut targets_arr = rlst_dynamic_array2!(f32, [n_targets, dim]);
        let mut charges_arr = rlst_dynamic_array2!(f32, [n_sources, 1]);
        let sources_slice = rlst_array_from_slice2!(sources.as_slice(), [n_sources, dim]);
        let targets_slice = rlst_array_from_slice2!(targets.as_slice(), [n_targets, dim]);
        let charges_slice = rlst_array_from_slice2!(charges.as_slice(), [n_sources, 1]);
        sources_arr.view_mut().fill_from(sources_slice.view());
        targets_arr.view_mut().fill_from(targets_slice.view());
        charges_arr.view_mut().fill_from(charges_slice.view());

        // Set FMM parameters
        let kernel = Laplace3dKernel::new();
        let eval_type = if eval_type == 0 {
            EvalType::Value
        } else if eval_type == 1 {
            EvalType::ValueDeriv
        } else {
            EvalType::Value
        };
        let source_to_target = FftFieldTranslation::new();

        let fmm = SingleNodeBuilder::new()
            .tree(&sources_arr, &targets_arr, Some(n_crit), sparse)
            .unwrap()
            .parameters(
                &charges_arr,
                expansion_order,
                kernel,
                eval_type,
                source_to_target,
            )
            .unwrap()
            .build()
            .unwrap();

        store.insert(struct_id, fmm);
        *next_id += 1;
        Ok(struct_id)
    }

    /// A Python module implemented in Rust. The name of this function must match
    /// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
    /// import the module.
    #[pymodule]
    fn kifmm(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(f32_laplace_fft, m)?)?;
        m.add_function(wrap_pyfunction!(f32_laplace_blas, m)?)?;
        Ok(())
    }
}

#[doc(inline)]
#[allow(unused_imports)]
pub use python_api::*;
