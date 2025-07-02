//! # Real to Complex Transform
use kifmm_fftw_sys as ffi;

use num_complex::Complex;
use rayon::prelude::*;

use crate::{
    fftw::types::{FftError, ShapeInfo},
    traits::fftw::{Dft, RealToComplexFft3D},
};

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
        plan: &<Self as Dft>::Plan,
    ) -> Result<(), FftError> {
        let info = validate_shape_r2c(shape, in_.len(), out.len())?;

        let it_in_ = in_.par_chunks_exact_mut(info.n_input);
        let it_out = out.par_chunks_exact_mut(info.n_output);

        it_in_.zip(it_out).for_each(|(in_, out)| {
            unsafe { ffi::fftwf_execute_dft_r2c(plan.plan.0, in_.as_mut_ptr(), out.as_mut_ptr()) };
        });

        Ok(())
    }

    fn c2r_batch_par(
        in_: &mut [Complex<Self>],
        out: &mut [Self],
        shape: &[usize],
        plan: &<Self as Dft>::Plan,
    ) -> Result<(), FftError> {
        let info = validate_shape_c2r(shape, in_.len(), out.len())?;

        let it_in_ = in_.par_chunks_exact_mut(info.n_output);
        let it_out = out.par_chunks_exact_mut(info.n_input);

        it_in_.zip(it_out).for_each(|(in_, out)| {
            unsafe { ffi::fftwf_execute_dft_c2r(plan.plan.0, in_.as_mut_ptr(), out.as_mut_ptr()) }

            // Normalise output
            out.iter_mut()
                .for_each(|value| *value *= 1.0 / (info.n_input as f32));
        });

        Ok(())
    }

    fn r2c_batch(
        in_: &mut [Self],
        out: &mut [Complex<Self>],
        shape: &[usize],
        plan: &<Self as Dft>::Plan,
    ) -> Result<(), FftError> {
        let info = validate_shape_r2c(shape, in_.len(), out.len())?;

        let it_in_ = in_.chunks_exact_mut(info.n_input);
        let it_out = out.chunks_exact_mut(info.n_output);

        it_in_.zip(it_out).for_each(|(in_, out)| {
            unsafe { ffi::fftwf_execute_dft_r2c(plan.plan.0, in_.as_mut_ptr(), out.as_mut_ptr()) };
        });

        Ok(())
    }

    fn c2r_batch(
        in_: &mut [Complex<Self>],
        out: &mut [Self],
        shape: &[usize],
        plan: &<Self as Dft>::Plan,
    ) -> Result<(), FftError> {
        let info = validate_shape_c2r(shape, in_.len(), out.len())?;

        let it_in_ = in_.chunks_exact_mut(info.n_output);
        let it_out = out.chunks_exact_mut(info.n_input);

        it_in_.zip(it_out).for_each(|(in_, out)| {
            unsafe { ffi::fftwf_execute_dft_c2r(plan.plan.0, in_.as_mut_ptr(), out.as_mut_ptr()) }

            // Normalise output
            out.iter_mut()
                .for_each(|value| *value *= 1.0 / (info.n_input as f32));
        });

        Ok(())
    }

    fn r2c(
        in_: &mut [Self],
        out: &mut [Complex<Self>],
        shape: &[usize],
        plan: &<Self as Dft>::Plan,
    ) -> Result<(), FftError> {
        let _info = validate_shape_r2c(shape, in_.len(), out.len())?;

        unsafe {
            ffi::fftwf_execute_dft_r2c(plan.plan.0, in_.as_mut_ptr(), out.as_mut_ptr());
        };

        Ok(())
    }

    fn c2r(
        in_: &mut [Complex<Self>],
        out: &mut [Self],
        shape: &[usize],
        plan: &<Self as Dft>::Plan,
    ) -> Result<(), FftError> {
        let info = validate_shape_c2r(shape, in_.len(), out.len())?;

        unsafe { ffi::fftwf_execute_dft_c2r(plan.plan.0, in_.as_mut_ptr(), out.as_mut_ptr()) }

        // Normalise
        out.iter_mut()
            .for_each(|value| *value *= 1.0 / (info.n_input as f32));

        Ok(())
    }
}

impl RealToComplexFft3D for f64 {
    fn r2c_batch_par(
        in_: &mut [Self],
        out: &mut [Complex<Self>],
        shape: &[usize],
        plan: &<Self as Dft>::Plan,
    ) -> Result<(), FftError> {
        let info = validate_shape_r2c(shape, in_.len(), out.len())?;

        let it_in_ = in_.par_chunks_exact_mut(info.n_input);
        let it_out = out.par_chunks_exact_mut(info.n_output);

        it_in_.zip(it_out).for_each(|(in_, out)| {
            unsafe { ffi::fftw_execute_dft_r2c(plan.plan.0, in_.as_mut_ptr(), out.as_mut_ptr()) };
        });

        Ok(())
    }

    fn c2r_batch_par(
        in_: &mut [Complex<Self>],
        out: &mut [Self],
        shape: &[usize],
        plan: &<Self as Dft>::Plan,
    ) -> Result<(), FftError> {
        let info = validate_shape_c2r(shape, in_.len(), out.len())?;

        let it_in_ = in_.par_chunks_exact_mut(info.n_output);
        let it_out = out.par_chunks_exact_mut(info.n_input);

        it_in_.zip(it_out).for_each(|(in_, out)| {
            unsafe { ffi::fftw_execute_dft_c2r(plan.plan.0, in_.as_mut_ptr(), out.as_mut_ptr()) }

            // Normalise output
            out.iter_mut()
                .for_each(|value| *value *= 1.0 / (info.n_input as f64));
        });

        Ok(())
    }

    fn r2c_batch(
        in_: &mut [Self],
        out: &mut [Complex<Self>],
        shape: &[usize],
        plan: &<Self as Dft>::Plan,
    ) -> Result<(), FftError> {
        let info = validate_shape_r2c(shape, in_.len(), out.len())?;

        let it_in_ = in_.chunks_exact_mut(info.n_input);
        let it_out = out.chunks_exact_mut(info.n_output);

        it_in_.zip(it_out).for_each(|(in_, out)| {
            unsafe { ffi::fftw_execute_dft_r2c(plan.plan.0, in_.as_mut_ptr(), out.as_mut_ptr()) };
        });

        Ok(())
    }

    fn c2r_batch(
        in_: &mut [Complex<Self>],
        out: &mut [Self],
        shape: &[usize],
        plan: &<Self as Dft>::Plan,
    ) -> Result<(), FftError> {
        let info = validate_shape_c2r(shape, in_.len(), out.len())?;

        let it_in_ = in_.chunks_exact_mut(info.n_output);
        let it_out = out.chunks_exact_mut(info.n_input);

        it_in_.zip(it_out).for_each(|(in_, out)| {
            unsafe { ffi::fftw_execute_dft_c2r(plan.plan.0, in_.as_mut_ptr(), out.as_mut_ptr()) }

            // Normalise output
            out.iter_mut()
                .for_each(|value| *value *= 1.0 / (info.n_input as f64));
        });

        Ok(())
    }

    fn r2c(
        in_: &mut [Self],
        out: &mut [Complex<Self>],
        shape: &[usize],
        plan: &<Self as Dft>::Plan,
    ) -> Result<(), FftError> {
        let _info = validate_shape_r2c(shape, in_.len(), out.len())?;

        unsafe {
            ffi::fftw_execute_dft_r2c(plan.plan.0, in_.as_mut_ptr(), out.as_mut_ptr());
        };

        Ok(())
    }

    fn c2r(
        in_: &mut [Complex<Self>],
        out: &mut [Self],
        shape: &[usize],
        plan: &<Self as Dft>::Plan,
    ) -> Result<(), FftError> {
        let info = validate_shape_c2r(shape, in_.len(), out.len())?;

        unsafe { ffi::fftw_execute_dft_c2r(plan.plan.0, in_.as_mut_ptr(), out.as_mut_ptr()) }

        // Normalise
        out.iter_mut()
            .for_each(|value| *value *= 1.0 / (info.n_input as f64));

        Ok(())
    }
}

#[cfg(test)]
mod test {

    use super::{Dft, RealToComplexFft3D};
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

        let plan = f64::plan_forward(&mut a, &mut b, &[nd, nd, nd], None).unwrap();
        f64::r2c(&mut a, &mut b, &[nd, nd, nd], &plan).unwrap();

        let plan = f64::plan_backward(&mut b, &mut a, &[nd, nd, nd], None).unwrap();
        f64::c2r(&mut b, &mut a, &[nd, nd, nd], &plan).unwrap();

        for (i, &v) in a.iter().enumerate() {
            let expected = i as f64;
            let dif = (v - expected).abs();
            if dif > 1e-7 {
                panic!("Large difference: v={v}, dif={dif}");
            }
        }
    }
}
