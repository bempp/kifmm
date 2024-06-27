use crate::traits::fftw::FftPlan;
use kifmm_fftw_sys as ffi;

use super::types::{Plan, Plan32, Plan64};


impl FftPlan for Plan<f32> {
    type Plan = ffi::fftwf_plan_s;

    fn plan(&self) -> *mut Self::Plan {
        self.plan.0
    }
}


impl FftPlan for Plan64 {
    type Plan = ffi::fftw_plan_s;

    fn plan(&self) -> *mut Self::Plan {
        self.0
    }
}