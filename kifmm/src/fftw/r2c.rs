//! # Real to Complex Transform
use fftw_sys as ffi;

use itertools::Itertools;
use num_complex::Complex;

use rayon::prelude::*;

use super::helpers::validate_plan;
use super::types::{FftError, Plan32, Plan64, ShapeInfo};
use crate::traits::fftw::RealToComplexFft3D;

use crate::excall;
use crate::fftw::types::{FFTW_ESTIMATE, FFTW_MUTEX};

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

    println!("{:?} {:?} {:?}", n, n_d, n_sub);

    let valid = in_shape.len() == 3 && in_len % n == 0 && out_len % n_sub == 0;
    if valid {
        Ok(ShapeInfo {
            n_input: n,
            n_output: n_sub,
        })
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
    println!("{:?} {:?} {:?}", in_len, out_len, 42);

    let valid = in_shape.len() == 3 && in_len % n_sub == 0 && out_len % n == 0;
    if valid {
        Ok(ShapeInfo {
            n_input: n,
            n_output: n_sub,
        })
    } else {
        Err(FftError::InvalidDimensionError)
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

        let it_in_ = in_.par_chunks_exact_mut(info.n_input);
        let it_out = out.par_chunks_exact_mut(info.n_output);

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

        let it_in_ = in_.par_chunks_exact_mut(info.n_output);
        let it_out = out.par_chunks_exact_mut(info.n_input);

        it_in_.zip(it_out).for_each(|(in_, out)| {
            let p = plan;
            unsafe { ffi::fftwf_execute_dft_c2r(p.0, in_.as_mut_ptr(), out.as_mut_ptr()) }

            // Normalise output
            out.iter_mut()
                .for_each(|value| *value *= 1.0 / (info.n_input as f32));
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

        let it_in_ = in_.chunks_exact_mut(info.n_input);
        let it_out = out.chunks_exact_mut(info.n_output);

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

        let it_in_ = in_.chunks_exact_mut(info.n_output);
        let it_out = out.chunks_exact_mut(info.n_input);

        it_in_.zip(it_out).for_each(|(in_, out)| {
            unsafe { ffi::fftwf_execute_dft_c2r(plan.0, in_.as_mut_ptr(), out.as_mut_ptr()) }

            // Normalise output
            out.iter_mut()
                .for_each(|value| *value *= 1.0 / (info.n_input as f32));
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
            .for_each(|value| *value *= 1.0 / (info.n_input as f32));

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

        let it_in_ = in_.par_chunks_exact_mut(info.n_input);
        let it_out = out.par_chunks_exact_mut(info.n_output);

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

        let it_in_ = in_.par_chunks_exact_mut(info.n_output);
        let it_out = out.par_chunks_exact_mut(info.n_input);

        it_in_.zip(it_out).for_each(|(in_, out)| {
            let p = plan;
            unsafe { ffi::fftw_execute_dft_c2r(p.0, in_.as_mut_ptr(), out.as_mut_ptr()) }

            // Normalise output
            out.iter_mut()
                .for_each(|value| *value *= 1.0 / (info.n_input as f64));
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

        let it_in_ = in_.chunks_exact_mut(info.n_input);
        let it_out = out.chunks_exact_mut(info.n_output);

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

        let it_in_ = in_.chunks_exact_mut(info.n_output);
        let it_out = out.chunks_exact_mut(info.n_input);

        it_in_.zip(it_out).for_each(|(in_, out)| {
            unsafe { ffi::fftw_execute_dft_c2r(plan.0, in_.as_mut_ptr(), out.as_mut_ptr()) }

            // Normalise output
            out.iter_mut()
                .for_each(|value| *value *= 1.0 / (info.n_input as f64));
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
            .for_each(|value| *value *= 1.0 / (info.n_input as f64));

        unsafe {
            ffi::fftw_destroy_plan(plan.0);
        };
        Ok(())
    }
}

#[cfg(test)]
mod test {

    use super::RealToComplexFft3D;
    use num::traits::Zero;
    use rlst::c64;

    #[test]
    fn test_r2c2r_identity() {
        let nd = 3;
        let n = nd * nd * nd;
        let n_sub = nd * nd * (nd / 2 + 1);
        let mut a = vec![0.0; n];
        let mut b = vec![c64::zero(); n_sub];

        for (i, a_i) in a.iter_mut().enumerate().take(n) {
            *a_i = i as f64;
        }

        f64::r2c(&mut a, &mut b, &[nd, nd, nd]).unwrap();
        f64::c2r(&mut b, &mut a, &[nd, nd, nd]).unwrap();

        for (i, &v) in a.iter().enumerate() {
            let expected = i as f64;
            let dif = (v - expected).abs();
            if dif > 1e-7 {
                panic!("Large difference: v={}, dif={}", v, dif);
            }
        }
    }
}
