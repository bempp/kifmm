//! # FFTW Bindings
//!
//! A subset of the FFTW library relevant for the kernel independent fast multipole method. Specifically the functionality
//! for computing real-to-complex transforms on 3D data.
//!
//! The bindings are generated
//! # Features
//! * The `r2c` and `c2r` transforms implemented on buffers representing 3D data, expected in column major order.
//! * Optionally parallel `r2c` and `c2r` which share a plan to compute batched transforms of multiple input data sets.
//!
mod r2c;
pub mod types;
