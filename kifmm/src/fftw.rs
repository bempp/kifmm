//! # FFTW Bindings
//!
//! A subset of the FFTW library relevant for the kernel independent fast multipole method. Specifically the functionality
//! for computing 3D real-to-complex and complex-to-complex DFTs.
//!
pub mod array;
mod c2c;
mod dft;
mod helpers;
mod r2c;
pub mod types;
