//! Helper functions
use super::types::FftError;
use kifmm_fftw_sys as ffi;

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

/// Exclusive call of FFTW interface.
#[macro_export]
macro_rules! excall {
    ($call:expr) => {{
        use crate::fftw::types::FFTW_MUTEX;
        let _lock = FFTW_MUTEX.lock().expect("Cannot get lock");
        unsafe { $call }
    }};
}

pub unsafe fn fftw_malloc<T>(size: usize) -> Vec<T> {
    let ptr = ffi::fftw_malloc(size * std::mem::size_of::<T>()) as *mut T;

    if ptr.is_null() {
        panic!("FFTW Failed to Allocate")
    }

    Vec::from_raw_parts(ptr, size, size)
}

pub unsafe fn fftw_free<T>(ptr: *mut T) {
    ffi::fftwf_free(ptr as *mut _)
}
