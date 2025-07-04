//! # Complex to Complex Transform
use kifmm_fftw_sys as ffi;

use rayon::prelude::*;
use rlst::{c32, c64, RlstScalar};

use crate::{
    fftw::types::{FftError, ShapeInfo, Sign},
    traits::fftw::{ComplexToComplexFft3D, Dft},
};

/// Validate the dimensions of the (batch) input and output sequences in complex-to-complex DFTs
///
/// # Arguments
/// * `shape` - Shape of the input sequences.
/// * `in_len` - Length of the input sequence
/// * `out_len` - Length of the output sequence
fn validate_shape_c2c(
    in_shape: &[usize],
    in_len: usize,
    out_len: usize,
) -> Result<ShapeInfo, FftError> {
    let n_input = in_shape.iter().product();
    let n_output = n_input;

    let valid = in_shape.len() == 3 && in_len % n_input == 0 && out_len % n_input == 0;
    if valid {
        Ok(ShapeInfo { n_input, n_output })
    } else {
        Err(FftError::InvalidDimensionError)
    }
}

impl ComplexToComplexFft3D for c64 {
    fn c2c(
        in_: &mut [Self],
        out: &mut [Self],
        shape: &[usize],
        sign: Sign,
        plan: &<Self as Dft>::Plan,
    ) -> Result<(), FftError> {
        let info = validate_shape_c2c(shape, in_.len(), out.len())?;

        match sign {
            Sign::Forward => {
                unsafe { ffi::fftw_execute_dft(plan.plan.0, in_.as_mut_ptr(), out.as_mut_ptr()) };
            }
            Sign::Backward => {
                unsafe { ffi::fftw_execute_dft(plan.plan.0, in_.as_mut_ptr(), out.as_mut_ptr()) };

                out.iter_mut()
                    .for_each(|value| *value = value.mul_real(1.0 / info.n_input as f64))
            }
        }

        Ok(())
    }

    fn c2c_batch(
        in_: &mut [Self],
        out: &mut [Self],
        shape: &[usize],
        sign: Sign,
        plan: &<Self as Dft>::Plan,
    ) -> Result<(), FftError> {
        let info = validate_shape_c2c(shape, in_.len(), out.len())?;

        let it_in_ = in_.chunks_exact_mut(info.n_input);
        let it_out = out.chunks_exact_mut(info.n_output);

        match sign {
            Sign::Forward => {
                it_in_.zip(it_out).for_each(|(in_, out)| unsafe {
                    ffi::fftw_execute_dft(plan.plan.0, in_.as_mut_ptr(), out.as_mut_ptr())
                });
            }
            Sign::Backward => {
                it_in_.zip(it_out).for_each(|(in_, out)| unsafe {
                    ffi::fftw_execute_dft(plan.plan.0, in_.as_mut_ptr(), out.as_mut_ptr())
                });
                out.iter_mut()
                    .for_each(|value| *value = value.mul_real(1.0 / (info.n_input as f64)))
            }
        }

        Ok(())
    }

    fn c2c_batch_par(
        in_: &mut [Self],
        out: &mut [Self],
        shape: &[usize],
        sign: Sign,
        plan: &<Self as Dft>::Plan,
    ) -> Result<(), FftError> {
        let info = validate_shape_c2c(shape, in_.len(), out.len())?;

        let it_in_ = in_.par_chunks_exact_mut(info.n_input);
        let it_out = out.par_chunks_exact_mut(info.n_output);

        match sign {
            Sign::Forward => {
                it_in_.zip(it_out).for_each(|(in_, out)| unsafe {
                    ffi::fftw_execute_dft(plan.plan.0, in_.as_mut_ptr(), out.as_mut_ptr())
                });
            }
            Sign::Backward => {
                it_in_.zip(it_out).for_each(|(in_, out)| unsafe {
                    ffi::fftw_execute_dft(plan.plan.0, in_.as_mut_ptr(), out.as_mut_ptr())
                });
                out.iter_mut()
                    .for_each(|value| *value = value.mul_real(1.0 / (info.n_input as f64)))
            }
        }

        Ok(())
    }
}

impl ComplexToComplexFft3D for c32 {
    fn c2c(
        in_: &mut [Self],
        out: &mut [Self],
        shape: &[usize],
        sign: Sign,
        plan: &<Self as Dft>::Plan,
    ) -> Result<(), FftError> {
        let info = validate_shape_c2c(shape, in_.len(), out.len())?;

        match sign {
            Sign::Forward => {
                unsafe { ffi::fftwf_execute_dft(plan.plan.0, in_.as_mut_ptr(), out.as_mut_ptr()) };
            }
            Sign::Backward => {
                unsafe { ffi::fftwf_execute_dft(plan.plan.0, in_.as_mut_ptr(), out.as_mut_ptr()) };

                out.iter_mut()
                    .for_each(|value| *value = value.mul_real(1.0 / info.n_input as f32))
            }
        }
        Ok(())
    }

    fn c2c_batch(
        in_: &mut [Self],
        out: &mut [Self],
        shape: &[usize],
        sign: Sign,
        plan: &<Self as Dft>::Plan,
    ) -> Result<(), FftError> {
        let info = validate_shape_c2c(shape, in_.len(), out.len())?;

        let it_in_ = in_.chunks_exact_mut(info.n_input);
        let it_out = out.chunks_exact_mut(info.n_output);

        match sign {
            Sign::Forward => {
                it_in_.zip(it_out).for_each(|(in_, out)| unsafe {
                    ffi::fftwf_execute_dft(plan.plan.0, in_.as_mut_ptr(), out.as_mut_ptr())
                });
            }
            Sign::Backward => {
                it_in_.zip(it_out).for_each(|(in_, out)| unsafe {
                    ffi::fftwf_execute_dft(plan.plan.0, in_.as_mut_ptr(), out.as_mut_ptr())
                });
                out.iter_mut()
                    .for_each(|value| *value = value.mul_real(1.0 / (info.n_input as f32)))
            }
        }

        Ok(())
    }

    fn c2c_batch_par(
        in_: &mut [Self],
        out: &mut [Self],
        shape: &[usize],
        sign: Sign,
        plan: &<Self as Dft>::Plan,
    ) -> Result<(), FftError> {
        let info = validate_shape_c2c(shape, in_.len(), out.len())?;

        let it_in_ = in_.par_chunks_exact_mut(info.n_input);
        let it_out = out.par_chunks_exact_mut(info.n_output);

        match sign {
            Sign::Forward => {
                it_in_.zip(it_out).for_each(|(in_, out)| unsafe {
                    ffi::fftwf_execute_dft(plan.plan.0, in_.as_mut_ptr(), out.as_mut_ptr())
                });
            }
            Sign::Backward => {
                it_in_.zip(it_out).for_each(|(in_, out)| unsafe {
                    ffi::fftwf_execute_dft(plan.plan.0, in_.as_mut_ptr(), out.as_mut_ptr())
                });
                out.iter_mut()
                    .for_each(|value| *value = value.mul_real(1.0 / (info.n_input as f32)))
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod test {

    use crate::traits::fftw::Dft;

    use super::ComplexToComplexFft3D;
    use num::traits::Zero;
    use rlst::c64;

    #[test]
    fn test_c2c2c_identity() {
        let nd = 3;
        let n = nd * nd * nd;
        let mut a = vec![c64::zero(); n];
        let mut b = vec![c64::zero(); n];

        for (i, a_i) in a.iter_mut().enumerate().take(n) {
            *a_i = c64::new(i as f64, 0.);
        }

        let plan = c64::plan_forward(&mut a, &mut b, &[nd, nd, nd], None).unwrap();

        c64::c2c(
            &mut a,
            &mut b,
            &[nd, nd, nd],
            crate::fftw::types::Sign::Forward,
            &plan,
        )
        .unwrap();

        let plan = c64::plan_backward(&mut a, &mut b, &[nd, nd, nd], None).unwrap();
        c64::c2c(
            &mut b,
            &mut a,
            &[nd, nd, nd],
            crate::fftw::types::Sign::Backward,
            &plan,
        )
        .unwrap();

        for (i, &v) in a.iter().enumerate() {
            let expected = c64::new(i as f64, 0.);
            let dif = (v - expected).norm();
            if dif > 1e-7 {
                panic!("Large difference: v={v}, dif={dif}");
            }
        }
    }
}
