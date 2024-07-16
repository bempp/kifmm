//! Helper functions
use crate::fftw::types::FftError;

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
        use $crate::fftw::types::FFTW_MUTEX;
        let _lock = FFTW_MUTEX.lock().expect("Cannot get lock");
        unsafe { $call }
    }};
}
