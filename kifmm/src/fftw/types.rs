//! # FFTW Types

use fftw_sys as ffi;

/// A threadsafe wrapper for a FFT plan operating on double precision data
#[derive(Clone, Copy)]
pub struct Plan64(pub *mut ffi::fftw_plan_s);

/// A threadsafe wrapper for a FFT plan operating on single precision data
#[derive(Clone, Copy)]
pub struct Plan32(pub *mut ffi::fftwf_plan_s);

/// Information about length of input and output sequences in real-to-complex DFTs
pub struct ShapeInfo {
    /// Length of the real input sequence
    pub n: usize,

    /// Length of the complex output sequence
    pub n_sub: usize,
}

/// Error type for handling FFTW wrapper, arise from creating plans and using data of incorrect dimension for real-to-complex transforms.
#[derive(Debug)]
pub enum FftError {
    /// Failed to create a valid plan using FFTW library
    InvalidPlanError,

    /// The input and output buffers are of incompatible sizes for real-to-complex transforms.
    InvalidDimensionError,
}