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

    use green_kernels::{helmholtz_3d::Helmholtz3dKernel, laplace_3d::Laplace3dKernel};
    use green_kernels::types::EvalType;
    use pyo3::prelude::*;
    use rlst::{rlst_array_from_slice2, rlst_dynamic_array2, c32};
    use std::collections::HashMap;
    use std::sync::Mutex;

    // Global store for MyStruct instances
    lazy_static::lazy_static! {
        static ref STORE_LAPLACE_FFT_F32: Mutex<HashMap<usize, KiFmm<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>>>> = Mutex::new(HashMap::new());
        static ref STORE_LAPLACE_BLAS_F32: Mutex<HashMap<usize, KiFmm<f32, Laplace3dKernel<f32>, BlasFieldTranslationSaRcmp<f32>>>> = Mutex::new(HashMap::new());
        static ref STORE_HELMHOLTZ_BLAS_F32: Mutex<HashMap<usize, KiFmm<c32, Helmholtz3dKernel<c32>, BlasFieldTranslationIa<c32>>>> = Mutex::new(HashMap::new());
        static ref STORE_HELMHOLTZ_FFT_F32: Mutex<HashMap<usize, KiFmm<c32, Helmholtz3dKernel<c32>, FftFieldTranslation<c32>>>> = Mutex::new(HashMap::new());
        static ref NEXT_ID: Mutex<usize> = Mutex::new(0);
    }

    /// Constructor interface for Python
    pub mod constructors {
        use green_kernels::helmholtz_3d::Helmholtz3dKernel;
        use num_complex::Complex64;
        use super::*;
        use pyo3::{buffer::PyBuffer, conversion::FromPyObjectBound};
        use pyo3::types::{PyAny, PyComplex};
        use rlst::{c32, c64, Shape};


        /// Helmholtz BLAS constructor
        #[pyfunction]
        pub fn f32_helmholtz_blas(
            py: Python,
            expansion_order: usize,
            n_crit: u64,
            sparse: bool,
            eval_type: usize,
            sources: Bound<'_, PyAny>,
            targets: Bound<'_, PyAny>,
            charges: Bound<'_, PyAny>,
            svd_threshold: f32,
            wavenumber: f32
        ) -> PyResult<usize> {
            let dim = 3;
            let sources_buf = PyBuffer::<f32>::get_bound(&sources)?;
            let tmp = sources_buf.as_slice(py).unwrap();

            let n_tmp = tmp.len() / dim;
            let sources_slice = unsafe {
                let _slice = std::slice::from_raw_parts(tmp.as_ptr() as *const f32, n_tmp * dim);
                rlst_array_from_slice2!(_slice, [n_tmp, 3])
            };

            let targets_buf = PyBuffer::<f32>::get_bound(&targets)?;
            let tmp = targets_buf.as_slice(py).unwrap();
            let n_tmp = tmp.len() / dim;
            let targets_slice = unsafe {
                let _slice = std::slice::from_raw_parts(tmp.as_ptr() as *const f32, n_tmp * dim);
                rlst_array_from_slice2!(_slice, [n_tmp, 3])
            };

            let charges_buf = PyBuffer::<f32>::get_bound(&charges)?;
            let tmp = charges_buf.as_slice(py).unwrap();
            let n_tmp = tmp.len() * 2;
            let charges_slice = unsafe {
                let _slice = std::slice::from_raw_parts(tmp.as_ptr() as *const c32, n_tmp);
                rlst_array_from_slice2!(_slice, [n_tmp, 1])
            };

            let mut next_id = NEXT_ID.lock().unwrap();
            let mut store = STORE_HELMHOLTZ_BLAS_F32.lock().unwrap();
            let struct_id = *next_id;

            // Copy source/target/charge data
            let n_sources = sources_slice.shape()[0];
            let n_targets = targets_slice.shape()[0];
            let n_charges = charges_slice.shape()[0];

            let mut sources_arr = rlst_dynamic_array2!(f32, [n_sources, dim]);
            let mut targets_arr = rlst_dynamic_array2!(f32, [n_targets, dim]);
            let mut charges_arr = rlst_dynamic_array2!(c32, [n_charges, 1]);

            sources_arr.view_mut().fill_from(sources_slice.view());
            targets_arr.view_mut().fill_from(targets_slice.view());
            charges_arr.view_mut().fill_from(charges_slice.view());

            // Set FMM parameters
            let kernel = Helmholtz3dKernel::new(wavenumber);

            let eval_type = if eval_type == 0 {
                EvalType::Value
            } else if eval_type == 1 {
                EvalType::ValueDeriv
            } else {
                EvalType::Value
            };
            let source_to_target = BlasFieldTranslationIa::new(Some(svd_threshold));

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
        pub fn f32_helmholtz_fft(
            py: Python,
            expansion_order: usize,
            n_crit: u64,
            sparse: bool,
            eval_type: usize,
            sources: Bound<'_, PyAny>,
            targets: Bound<'_, PyAny>,
            charges: Bound<'_, PyAny>,
            wavenumber: f32
        ) -> PyResult<usize> {

            let dim = 3;
            let sources_buf = PyBuffer::<f32>::get_bound(&sources)?;
            let tmp = sources_buf.as_slice(py).unwrap();

            let n_tmp = tmp.len() / dim;
            let sources_slice = unsafe {
                let _slice = std::slice::from_raw_parts(tmp.as_ptr() as *const f32, n_tmp * dim);
                rlst_array_from_slice2!(_slice, [n_tmp, 3])
            };

            let targets_buf = PyBuffer::<f32>::get_bound(&targets)?;
            let tmp = targets_buf.as_slice(py).unwrap();
            let n_tmp = tmp.len() / dim;
            let targets_slice = unsafe {
                let _slice = std::slice::from_raw_parts(tmp.as_ptr() as *const f32, n_tmp * dim);
                rlst_array_from_slice2!(_slice, [n_tmp, 3])
            };

            let charges_buf = PyBuffer::<f32>::get_bound(&charges)?;
            let tmp = charges_buf.as_slice(py).unwrap();
            let n_tmp = tmp.len() * 2;
            let charges_slice = unsafe {
                let _slice = std::slice::from_raw_parts(tmp.as_ptr() as *const c32, n_tmp);
                rlst_array_from_slice2!(_slice, [n_tmp, 1])
            };

            let mut next_id = NEXT_ID.lock().unwrap();
            let mut store = STORE_HELMHOLTZ_FFT_F32.lock().unwrap();
            let struct_id = *next_id;

            // Copy source/target/charge data
            let n_sources = sources_slice.shape()[0];
            let n_targets = targets_slice.shape()[0];
            let n_charges = charges_slice.shape()[0];

            let mut sources_arr = rlst_dynamic_array2!(f32, [n_sources, dim]);
            let mut targets_arr = rlst_dynamic_array2!(f32, [n_targets, dim]);
            let mut charges_arr = rlst_dynamic_array2!(c32, [n_charges, 1]);

            sources_arr.view_mut().fill_from(sources_slice.view());
            targets_arr.view_mut().fill_from(targets_slice.view());
            charges_arr.view_mut().fill_from(charges_slice.view());

            // Set FMM parameters
            let kernel = Helmholtz3dKernel::new(wavenumber);
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

        /// Formats the sum of two numbers as string.
        #[pyfunction]
        pub fn f32_laplace_blas(
            py: Python,
            expansion_order: usize,
            n_crit: u64,
            sparse: bool,
            eval_type: usize,
            sources: Bound<'_, PyAny>,
            targets: Bound<'_, PyAny>,
            charges: Bound<'_, PyAny>,
            svd_threshold: f32,
        ) -> PyResult<usize> {
            let dim = 3;
            let sources_buf = PyBuffer::<f32>::get_bound(&sources)?;
            let tmp = sources_buf.as_slice(py).unwrap();

            let n_tmp = tmp.len() / dim;
            let sources_slice = unsafe {
                let _slice = std::slice::from_raw_parts(tmp.as_ptr() as *const f32, n_tmp * dim);
                rlst_array_from_slice2!(_slice, [n_tmp, 3])
            };

            let targets_buf = PyBuffer::<f32>::get_bound(&targets)?;
            let tmp = targets_buf.as_slice(py).unwrap();
            let n_tmp = tmp.len() / dim;
            let targets_slice = unsafe {
                let _slice = std::slice::from_raw_parts(tmp.as_ptr() as *const f32, n_tmp * dim);
                rlst_array_from_slice2!(_slice, [n_tmp, 3])
            };

            let charges_buf = PyBuffer::<f32>::get_bound(&charges)?;
            let tmp = charges_buf.as_slice(py).unwrap();
            let n_tmp = tmp.len();
            let charges_slice = unsafe {
                let _slice = std::slice::from_raw_parts(tmp.as_ptr() as *const f32, n_tmp);
                rlst_array_from_slice2!(_slice, [n_tmp, 1])
            };

            let mut next_id = NEXT_ID.lock().unwrap();
            let mut store = STORE_LAPLACE_BLAS_F32.lock().unwrap();
            let struct_id = *next_id;

            // Copy source/target/charge data
            let n_sources = sources_slice.shape()[0];
            let n_targets = targets_slice.shape()[0];
            let n_charges = charges_slice.shape()[0];

            let mut sources_arr = rlst_dynamic_array2!(f32, [n_sources, dim]);
            let mut targets_arr = rlst_dynamic_array2!(f32, [n_targets, dim]);
            let mut charges_arr = rlst_dynamic_array2!(f32, [n_charges, 1]);

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
        pub fn f32_laplace_fft(
            py: Python,
            expansion_order: usize,
            n_crit: u64,
            sparse: bool,
            eval_type: usize,
            sources: Bound<'_, PyAny>,
            targets: Bound<'_, PyAny>,
            charges: Bound<'_, PyAny>,
        ) -> PyResult<usize> {
            let dim = 3;
            let sources_buf = PyBuffer::<f32>::get_bound(&sources)?;
            let tmp = sources_buf.as_slice(py).unwrap();

            let n_tmp = tmp.len() / dim;
            let sources_slice = unsafe {
                let _slice = std::slice::from_raw_parts(tmp.as_ptr() as *const f32, n_tmp * dim);
                rlst_array_from_slice2!(_slice, [n_tmp, 3])
            };

            let targets_buf = PyBuffer::<f32>::get_bound(&targets)?;
            let tmp = targets_buf.as_slice(py).unwrap();
            let n_tmp = tmp.len() / dim;
            let targets_slice = unsafe {
                let _slice = std::slice::from_raw_parts(tmp.as_ptr() as *const f32, n_tmp * dim);
                rlst_array_from_slice2!(_slice, [n_tmp, 3])
            };

            let charges_buf = PyBuffer::<f32>::get_bound(&charges)?;
            let tmp = charges_buf.as_slice(py).unwrap();
            let n_tmp = tmp.len();
            let charges_slice = unsafe {
                let _slice = std::slice::from_raw_parts(tmp.as_ptr() as *const f32, n_tmp);
                rlst_array_from_slice2!(_slice, [n_tmp, 1])
            };

            let mut next_id = NEXT_ID.lock().unwrap();
            let mut store = STORE_LAPLACE_FFT_F32.lock().unwrap();
            let struct_id = *next_id;

            // Copy source/target/charge data
            let n_sources = sources_slice.shape()[0];
            let n_targets = targets_slice.shape()[0];
            let n_charges = charges_slice.shape()[0];

            let mut sources_arr = rlst_dynamic_array2!(f32, [n_sources, dim]);
            let mut targets_arr = rlst_dynamic_array2!(f32, [n_targets, dim]);
            let mut charges_arr = rlst_dynamic_array2!(f32, [n_charges, 1]);

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
    }

    // pub mod data {
    //     use pyo3::{PyResult, PyErr};

    //     use super::STORE_LAPLACE_FFT_F32;

    //     fn potentials(struct_id: usize) -> PyResult<*mut f32> {
    //         let mut store =  STORE_LAPLACE_FFT_F32.lock().unwrap();
    //         if let Some(fmm) = store.get_mut(&struct_id) {
    //             Ok(fmm.potentials.as_mut_ptr())
    //         } else {
    //             Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>("FMM not constructed"))
    //         }
    //     }
    // }

    use constructors::{f32_laplace_blas, f32_laplace_fft, f32_helmholtz_blas, f32_helmholtz_fft};

    /// Functions exposed at the module level
    #[pymodule]
    fn kifmm(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(f32_laplace_fft, m)?)?;
        m.add_function(wrap_pyfunction!(f32_laplace_blas, m)?)?;
        m.add_function(wrap_pyfunction!(f32_helmholtz_fft, m)?)?;
        m.add_function(wrap_pyfunction!(f32_helmholtz_blas, m)?)?;
        Ok(())
    }
}

#[doc(inline)]
#[allow(unused_imports)]
pub use python_api::*;
