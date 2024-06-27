//! # FFTW Types

use std::{marker::PhantomData, sync::Mutex};

use kifmm_fftw_sys as ffi;
use lazy_static::lazy_static;
use rlst::RlstScalar;

/// A threadsafe wrapper for a FFT plan operating on double precision data
#[derive(Clone, Copy)]
pub struct Plan64(pub *mut ffi::fftw_plan_s);

/// A threadsafe wrapper for a FFT plan operating on single precision data
#[derive(Clone, Copy)]
pub struct Plan32(pub *mut ffi::fftwf_plan_s);

pub trait FftwPlan {
    type Plan;
    fn plan(&self) -> *mut Self::Plan;
}

pub struct Plan<T: RlstScalar, P: FftwPlan + Send + Sync> {
    scalar: PhantomData<T>,
    pub plan: P,
}

impl<T: RlstScalar, P: FftwPlan + Send + Sync> Plan<T, P> {
    pub fn new(plan: P) -> Self {
        Self {
            scalar: PhantomData,
            plan,
        }
    }
}

pub struct BatchSize(pub i32);

impl FftwPlan for Plan32 {
    type Plan = ffi::fftwf_plan_s;

    fn plan(&self) -> *mut Self::Plan {
        self.0
    }
}

impl FftwPlan for Plan64 {
    type Plan = ffi::fftw_plan_s;

    fn plan(&self) -> *mut Self::Plan {
        self.0
    }
}

/// Information about length of input and output sequences in real-to-complex DFTs
pub struct ShapeInfo {
    /// Length of the input sequence
    pub n_input: usize,

    /// Length of the output sequence
    pub n_output: usize,
}

/// Error type for handling FFTW wrapper, arise from creating plans and using data of incorrect dimension for real-to-complex transforms.
#[derive(Debug)]
pub enum FftError {
    /// Failed to create a valid plan using FFTW library
    InvalidPlanError,

    /// The input and output buffers are of incompatible sizes for real-to-complex transforms.
    InvalidDimensionError,
}

unsafe impl Send for Plan32 {}
unsafe impl Send for Plan64 {}
unsafe impl Sync for Plan32 {}
unsafe impl Sync for Plan64 {}
unsafe impl<T: RlstScalar, P: FftwPlan + Send + Sync> Send for Plan<T, P> {}
unsafe impl<T: RlstScalar, P: FftwPlan + Send + Sync> Sync for Plan<T, P> {}

/// FFTW in 'estimate' mode. A sub-optimal heuristic is used to create FFT plan.
/// input/output arrays are not overwritten during planning, see [original doc](https://www.fftw.org/fftw3_doc/Planner-Flags.html) for detail
pub const FFTW_ESTIMATE: u32 = 1 << 6;

lazy_static! {
    /// Mutex for FFTW call.
    ///
    /// This mutex is necessary because most of calls in FFTW are not thread-safe.
    /// See the [original document](http://www.fftw.org/fftw3_doc/Thread-safety.html) for detail
    pub static ref FFTW_MUTEX: Mutex<()> = Mutex::new(());
}

/// Direction of complex-to-complex transform
#[repr(i32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Sign {
    /// Forward transform
    Forward = -1,

    /// Backward transform
    Backward = 1,
}
