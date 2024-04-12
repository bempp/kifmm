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
mod c2c;
mod r2c;
pub mod types;

use self::types::FftError;

/// Exclusive call of FFTW interface.
#[macro_export]
macro_rules! excall {
    ($call:expr) => {{
        let _lock = FFTW_MUTEX.lock().expect("Cannot get lock");
        unsafe { $call }
    }};
}

/// Validate a DFT plan created with FFTW
///
/// # Arguments
/// * `plan` - Raw pointer for a plan
pub fn validate_plan<T: Sized>(plan: *mut T) -> Result<*mut T, FftError> {
    if plan.is_null() {
        Err(FftError::InvalidPlanError {})
    } else {
        Ok(plan)
    }
}
