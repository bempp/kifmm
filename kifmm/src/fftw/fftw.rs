//! # FFTW Implementations
use fftw_sys as ffi;

use itertools::Itertools;
use num_complex::Complex;

use lazy_static::lazy_static;
use rayon::prelude::*;
use std::sync::Mutex;

use super::traits::RealToComplexFft3D;
use super::types::{FftError, Plan32, Plan64, ShapeInfo};

unsafe impl Send for Plan32 {}
unsafe impl Send for Plan64 {}
unsafe impl Sync for Plan32 {}
unsafe impl Sync for Plan64 {}

/// FFTW in 'estimate' mode. A sub-optimal heuristic is used to create FFT plan.
/// input/output arrays are not overwritten during planning, see [original doc](https://www.fftw.org/fftw3_doc/Planner-Flags.html) for detail
const FFTW_ESTIMATE: u32 = 1 << 6;

lazy_static! {
    /// Mutex for FFTW call.
    ///
    /// This mutex is necessary because most of calls in FFTW are not thread-safe.
    /// See the [original document](http://www.fftw.org/fftw3_doc/Thread-safety.html) for detail
    pub static ref FFTW_MUTEX: Mutex<()> = Mutex::new(());
}

#[macro_export]
/// Exclusive call of FFTW interface.
macro_rules! excall {
    ($call:expr) => {{
        let _lock = FFTW_MUTEX.lock().expect("Cannot get lock");
        unsafe { $call }
    }};
}

/// Validate the dimensions of the (batch) input and output sequences in real-to-complex DFTs
///
/// # Arguments
/// * `shape` - Shape of the input sequences.
/// * `in_len` - Length of the input sequence
/// * `out_len` - Length of the output sequence
fn validate_shape_r2c(
    in_shape: &[usize],
    in_len: usize,
    out_len: usize,
) -> Result<ShapeInfo, FftError> {
    let n: usize = in_shape.iter().product();
    let n_d = in_shape.last().unwrap();
    let n_sub = (n / n_d) * (n_d / 2 + 1);

    let valid = in_shape.len() == 3 && in_len % n == 0 && out_len % n_sub == 0;
    if valid {
        Ok(ShapeInfo { n, n_sub })
    } else {
        Err(FftError::InvalidDimensionError)
    }
}

/// Validate the dimensions of the (batch) input and output sequences in real-to-complex DFTs
///
/// # Arguments
/// * `shape` - Shape of the input sequences.
/// * `in_len` - Length of the input sequence
/// * `out_len` - Length of the output sequence
fn validate_shape_c2r(
    in_shape: &[usize],
    in_len: usize,
    out_len: usize,
) -> Result<ShapeInfo, FftError> {
    let n: usize = in_shape.iter().product();
    let n_d = in_shape.last().unwrap();
    let n_sub = (n / n_d) * (n_d / 2 + 1);

    let valid = in_shape.len() == 3 && in_len % n_sub == 0 && out_len % n == 0;
    if valid {
        Ok(ShapeInfo { n, n_sub })
    } else {
        Err(FftError::InvalidDimensionError)
    }
}

/// Validate a DFT plan created with FFTW
///
/// # Arguments
/// * `plan` - Raw pointer for a plan
fn validate_plan<T: Sized>(plan: *mut T) -> Result<*mut T, FftError> {
    if plan.is_null() {
        Err(FftError::InvalidPlanError {})
    } else {
        Ok(plan)
    }
}

impl RealToComplexFft3D for f32 {
    fn r2c_batch_par(
        in_: &mut [Self],
        out: &mut [Complex<Self>],
        shape: &[usize],
    ) -> Result<(), FftError> {
        let info = validate_shape_r2c(shape, in_.len(), out.len())?;
        let plan = Plan32(validate_plan(excall!(ffi::fftwf_plan_dft_r2c(
            shape.len() as i32,
            shape.iter().map(|&x| x as i32).collect_vec().as_mut_ptr() as *mut _,
            in_.as_mut_ptr(),
            out.as_mut_ptr(),
            FFTW_ESTIMATE,
        )))?);

        let it_in_ = in_.par_chunks_exact_mut(info.n);
        let it_out = out.par_chunks_exact_mut(info.n_sub);

        it_in_.zip(it_out).for_each(|(in_, out)| {
            let p = plan;
            unsafe { ffi::fftwf_execute_dft_r2c(p.0, in_.as_mut_ptr(), out.as_mut_ptr()) };
        });

        unsafe {
            ffi::fftwf_destroy_plan(plan.0);
        };

        Ok(())
    }

    fn c2r_batch_par(
        in_: &mut [Complex<Self>],
        out: &mut [Self],
        shape: &[usize],
    ) -> Result<(), FftError> {
        let info = validate_shape_c2r(shape, in_.len(), out.len())?;
        let plan = Plan32(validate_plan(excall!(ffi::fftwf_plan_dft_c2r(
            shape.len() as i32,
            shape.iter().map(|&x| x as i32).collect_vec().as_mut_ptr() as *mut _,
            in_.as_mut_ptr(),
            out.as_mut_ptr(),
            FFTW_ESTIMATE,
        )))?);

        let it_in_ = in_.par_chunks_exact_mut(info.n_sub);
        let it_out = out.par_chunks_exact_mut(info.n);

        it_in_.zip(it_out).for_each(|(in_, out)| {
            let p = plan;
            unsafe { ffi::fftwf_execute_dft_c2r(p.0, in_.as_mut_ptr(), out.as_mut_ptr()) }

            // Normalise output
            out.iter_mut()
                .for_each(|value| *value *= 1.0 / (info.n as f32));
        });

        unsafe {
            ffi::fftwf_destroy_plan(plan.0);
        };

        Ok(())
    }

    fn r2c_batch(
        in_: &mut [Self],
        out: &mut [Complex<Self>],
        shape: &[usize],
    ) -> Result<(), FftError> {
        // let size: usize = shape.iter().product();
        // let size_d = shape.last().unwrap();
        // let size_real = (size / size_d) * (size_d / 2 + 1);

        let info = validate_shape_r2c(shape, in_.len(), out.len())?;

        let plan = Plan32(validate_plan(excall!(ffi::fftwf_plan_dft_r2c(
            shape.len() as i32,
            shape.iter().map(|&x| x as i32).collect_vec().as_mut_ptr() as *mut _,
            in_.as_mut_ptr(),
            out.as_mut_ptr(),
            FFTW_ESTIMATE
        )))?);

        let it_in_ = in_.chunks_exact_mut(info.n);
        let it_out = out.chunks_exact_mut(info.n_sub);

        it_in_.zip(it_out).for_each(|(in_, out)| {
            unsafe { ffi::fftwf_execute_dft_r2c(plan.0, in_.as_mut_ptr(), out.as_mut_ptr()) };
        });

        unsafe {
            ffi::fftwf_destroy_plan(plan.0);
        };

        Ok(())
    }

    fn c2r_batch(
        in_: &mut [Complex<Self>],
        out: &mut [Self],
        shape: &[usize],
    ) -> Result<(), FftError> {
        let info = validate_shape_c2r(shape, in_.len(), out.len())?;
        let plan = Plan32(validate_plan(excall!(ffi::fftwf_plan_dft_c2r(
            shape.len() as i32,
            shape.iter().map(|&x| x as i32).collect_vec().as_mut_ptr() as *mut _,
            in_.as_mut_ptr(),
            out.as_mut_ptr(),
            FFTW_ESTIMATE
        )))?);

        let it_in_ = in_.chunks_exact_mut(info.n_sub);
        let it_out = out.chunks_exact_mut(info.n);

        it_in_.zip(it_out).for_each(|(in_, out)| {
            unsafe { ffi::fftwf_execute_dft_c2r(plan.0, in_.as_mut_ptr(), out.as_mut_ptr()) }

            // Normalise output
            out.iter_mut()
                .for_each(|value| *value *= 1.0 / (info.n as f32));
        });

        unsafe {
            ffi::fftwf_destroy_plan(plan.0);
        };

        Ok(())
    }

    fn r2c(in_: &mut [Self], out: &mut [Complex<Self>], shape: &[usize]) -> Result<(), FftError> {
        let _info = validate_shape_r2c(shape, in_.len(), out.len())?;
        let plan = Plan32(validate_plan(excall!(ffi::fftwf_plan_dft_r2c(
            shape.len() as i32,
            shape.iter().map(|&x| x as i32).collect_vec().as_mut_ptr() as *mut _,
            in_.as_mut_ptr(),
            out.as_mut_ptr(),
            FFTW_ESTIMATE
        )))?);

        unsafe {
            ffi::fftwf_execute_dft_r2c(plan.0, in_.as_mut_ptr(), out.as_mut_ptr());
        };

        unsafe {
            ffi::fftwf_destroy_plan(plan.0);
        };

        Ok(())
    }

    fn c2r(in_: &mut [Complex<Self>], out: &mut [Self], shape: &[usize]) -> Result<(), FftError> {
        let info = validate_shape_c2r(shape, in_.len(), out.len())?;

        let plan = Plan32(validate_plan(excall!(ffi::fftwf_plan_dft_c2r(
            shape.len() as i32,
            shape.iter().map(|&x| x as i32).collect_vec().as_mut_ptr() as *mut _,
            in_.as_mut_ptr(),
            out.as_mut_ptr(),
            FFTW_ESTIMATE
        )))?);

        unsafe { ffi::fftwf_execute_dft_c2r(plan.0, in_.as_mut_ptr(), out.as_mut_ptr()) }

        // Normalise
        out.iter_mut()
            .for_each(|value| *value *= 1.0 / (info.n as f32));

        unsafe {
            ffi::fftwf_destroy_plan(plan.0);
        };
        Ok(())
    }
}

impl RealToComplexFft3D for f64 {
    fn r2c_batch_par(
        in_: &mut [Self],
        out: &mut [Complex<Self>],
        shape: &[usize],
    ) -> Result<(), FftError> {
        let info = validate_shape_r2c(shape, in_.len(), out.len())?;
        let plan = Plan64(validate_plan(excall!(ffi::fftw_plan_dft_r2c(
            shape.len() as i32,
            shape.iter().map(|&x| x as i32).collect_vec().as_mut_ptr() as *mut _,
            in_.as_mut_ptr(),
            out.as_mut_ptr(),
            FFTW_ESTIMATE,
        )))?);

        let it_in_ = in_.par_chunks_exact_mut(info.n);
        let it_out = out.par_chunks_exact_mut(info.n_sub);

        it_in_.zip(it_out).for_each(|(in_, out)| {
            let p = plan;
            unsafe { ffi::fftw_execute_dft_r2c(p.0, in_.as_mut_ptr(), out.as_mut_ptr()) };
        });

        unsafe {
            ffi::fftw_destroy_plan(plan.0);
        };

        Ok(())
    }

    fn c2r_batch_par(
        in_: &mut [Complex<Self>],
        out: &mut [Self],
        shape: &[usize],
    ) -> Result<(), FftError> {
        let info = validate_shape_c2r(shape, in_.len(), out.len())?;
        let plan = Plan64(validate_plan(excall!(ffi::fftw_plan_dft_c2r(
            shape.len() as i32,
            shape.iter().map(|&x| x as i32).collect_vec().as_mut_ptr() as *mut _,
            in_.as_mut_ptr(),
            out.as_mut_ptr(),
            FFTW_ESTIMATE,
        )))?);

        let it_in_ = in_.par_chunks_exact_mut(info.n_sub);
        let it_out = out.par_chunks_exact_mut(info.n);

        it_in_.zip(it_out).for_each(|(in_, out)| {
            let p = plan;
            unsafe { ffi::fftw_execute_dft_c2r(p.0, in_.as_mut_ptr(), out.as_mut_ptr()) }

            // Normalise output
            out.iter_mut()
                .for_each(|value| *value *= 1.0 / (info.n as f64));
        });

        unsafe {
            ffi::fftw_destroy_plan(plan.0);
        };

        Ok(())
    }

    fn r2c_batch(
        in_: &mut [Self],
        out: &mut [Complex<Self>],
        shape: &[usize],
    ) -> Result<(), FftError> {
        // let size: usize = shape.iter().product();
        // let size_d = shape.last().unwrap();
        // let size_real = (size / size_d) * (size_d / 2 + 1);

        let info = validate_shape_r2c(shape, in_.len(), out.len())?;

        let plan = Plan64(validate_plan(excall!(ffi::fftw_plan_dft_r2c(
            shape.len() as i32,
            shape.iter().map(|&x| x as i32).collect_vec().as_mut_ptr() as *mut _,
            in_.as_mut_ptr(),
            out.as_mut_ptr(),
            FFTW_ESTIMATE
        )))?);

        let it_in_ = in_.chunks_exact_mut(info.n);
        let it_out = out.chunks_exact_mut(info.n_sub);

        it_in_.zip(it_out).for_each(|(in_, out)| {
            unsafe { ffi::fftw_execute_dft_r2c(plan.0, in_.as_mut_ptr(), out.as_mut_ptr()) };
        });

        unsafe {
            ffi::fftw_destroy_plan(plan.0);
        };

        Ok(())
    }

    fn c2r_batch(
        in_: &mut [Complex<Self>],
        out: &mut [Self],
        shape: &[usize],
    ) -> Result<(), FftError> {
        let info = validate_shape_c2r(shape, in_.len(), out.len())?;
        let plan = Plan64(validate_plan(excall!(ffi::fftw_plan_dft_c2r(
            shape.len() as i32,
            shape.iter().map(|&x| x as i32).collect_vec().as_mut_ptr() as *mut _,
            in_.as_mut_ptr(),
            out.as_mut_ptr(),
            FFTW_ESTIMATE
        )))?);

        let it_in_ = in_.chunks_exact_mut(info.n_sub);
        let it_out = out.chunks_exact_mut(info.n);

        it_in_.zip(it_out).for_each(|(in_, out)| {
            unsafe { ffi::fftw_execute_dft_c2r(plan.0, in_.as_mut_ptr(), out.as_mut_ptr()) }

            // Normalise output
            out.iter_mut()
                .for_each(|value| *value *= 1.0 / (info.n as f64));
        });

        unsafe {
            ffi::fftw_destroy_plan(plan.0);
        };

        Ok(())
    }

    fn r2c(in_: &mut [Self], out: &mut [Complex<Self>], shape: &[usize]) -> Result<(), FftError> {
        let _info = validate_shape_r2c(shape, in_.len(), out.len())?;
        let plan = Plan64(validate_plan(excall!(ffi::fftw_plan_dft_r2c(
            shape.len() as i32,
            shape.iter().map(|&x| x as i32).collect_vec().as_mut_ptr() as *mut _,
            in_.as_mut_ptr(),
            out.as_mut_ptr(),
            FFTW_ESTIMATE
        )))?);

        unsafe {
            ffi::fftw_execute_dft_r2c(plan.0, in_.as_mut_ptr(), out.as_mut_ptr());
        };

        unsafe {
            ffi::fftw_destroy_plan(plan.0);
        };

        Ok(())
    }

    fn c2r(in_: &mut [Complex<Self>], out: &mut [Self], shape: &[usize]) -> Result<(), FftError> {
        let info = validate_shape_c2r(shape, in_.len(), out.len())?;
        let plan = Plan64(validate_plan(excall!(ffi::fftw_plan_dft_c2r(
            shape.len() as i32,
            shape.iter().map(|&x| x as i32).collect_vec().as_mut_ptr() as *mut _,
            in_.as_mut_ptr(),
            out.as_mut_ptr(),
            FFTW_ESTIMATE
        )))?);

        unsafe { ffi::fftw_execute_dft_c2r(plan.0, in_.as_mut_ptr(), out.as_mut_ptr()) }

        // Normalise
        out.iter_mut()
            .for_each(|value| *value *= 1.0 / (info.n as f64));

        unsafe {
            ffi::fftw_destroy_plan(plan.0);
        };
        Ok(())
    }
}
