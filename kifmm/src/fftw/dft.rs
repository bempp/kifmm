use itertools::Itertools;
use rlst::{c32, c64};

use kifmm_fftw_sys as ffi;

use crate::{
    excall,
    fftw::{
        helpers::validate_plan,
        types::{BatchSize, FftError, Plan, Plan32, Plan64, Sign, FFTW_ESTIMATE},
    },
    traits::fftw::{ComplexToComplexFft3D, Dft, DftType, RealToComplexFft3D},
};

impl Dft for f32 {
    type Plan = Plan<Self, Plan32>;

    fn plan_forward(
        in_: &mut [<Self as DftType>::InputType],
        out: &mut [<Self as DftType>::OutputType],
        shape: &[usize],
        batch: Option<super::types::BatchSize>,
    ) -> Result<Self::Plan, FftError> {
        let plan = if let Some(BatchSize(howmany)) = batch {
            let size_in: usize = shape.iter().product();
            let shape_out = [shape[0], shape[1], shape[2] / 2];
            let size_out: usize = shape_out.iter().product();
            let inembed = std::ptr::null();
            let istride = 1i32;
            let idist = size_in as i32;
            let onembed = std::ptr::null();
            let ostride = 1i32;
            let odist = size_out as i32;

            Plan32(validate_plan(excall!(ffi::fftwf_plan_many_dft_r2c(
                shape.len() as i32,
                shape.iter().map(|&x| x as i32).collect_vec().as_mut_ptr() as *mut _,
                howmany as i32,
                in_.as_mut_ptr(),
                inembed,
                istride,
                idist,
                out.as_mut_ptr(),
                onembed,
                ostride,
                odist,
                FFTW_ESTIMATE
            )))?)
        } else {
            Plan32(validate_plan(excall!(ffi::fftwf_plan_dft_r2c(
                shape.len() as i32,
                shape.iter().map(|&x| x as i32).collect_vec().as_mut_ptr() as *mut _,
                in_.as_mut_ptr(),
                out.as_mut_ptr(),
                FFTW_ESTIMATE
            )))?)
        };

        Ok(Plan::new(plan))
    }

    fn plan_backward(
        in_: &mut [<Self as DftType>::OutputType],
        out: &mut [<Self as DftType>::InputType],
        shape: &[usize],
        batch: Option<super::types::BatchSize>,
    ) -> Result<Self::Plan, FftError> {
        let plan = if let Some(BatchSize(howmany)) = batch {
            let size_in: usize = shape.iter().product();
            let shape_out = [shape[0], shape[1], shape[2] / 2];
            let size_out: usize = shape_out.iter().product();
            let inembed = std::ptr::null();
            let istride = 1i32;
            let idist = size_in as i32;
            let onembed = std::ptr::null();
            let ostride = 1i32;
            let odist = size_out as i32;

            Plan32(validate_plan(excall!(ffi::fftwf_plan_many_dft_c2r(
                shape.len() as i32,
                shape.iter().map(|&x| x as i32).collect_vec().as_mut_ptr() as *mut _,
                howmany as i32,
                in_.as_mut_ptr(),
                inembed,
                istride,
                idist,
                out.as_mut_ptr(),
                onembed,
                ostride,
                odist,
                FFTW_ESTIMATE
            )))?)
        } else {
            Plan32(validate_plan(excall!(ffi::fftwf_plan_dft_c2r(
                shape.len() as i32,
                shape.iter().map(|&x| x as i32).collect_vec().as_mut_ptr() as *mut _,
                in_.as_mut_ptr(),
                out.as_mut_ptr(),
                FFTW_ESTIMATE
            )))?)
        };
        Ok(Plan::new(plan))
    }

    fn shape_in(expansion_order: usize) -> [usize; 3] {
        let m = 2 * expansion_order - 1; // Size of each dimension of 3D kernel/signal
        let pad_size = 1;
        let p = m + pad_size; // Size of each dimension of padded 3D kernel/signal
        [p, p, p]
    }

    fn size_in(expansion_order: usize) -> usize {
        Self::shape_in(expansion_order).iter().product()
    }

    fn shape_out(expansion_order: usize) -> [usize; 3] {
        let m = 2 * expansion_order - 1; // Size of each dimension of 3D kernel/signal
        let pad_size = 1;
        let p = m + pad_size; // Size of each dimension of padded 3D kernel/signal
        [p, p, p / 2 + 1]
    }

    fn size_out(expansion_order: usize) -> usize {
        Self::shape_out(expansion_order).iter().product()
    }

    fn forward_dft(
        in_: &mut [<Self as DftType>::InputType],
        out: &mut [<Self as DftType>::OutputType],
        shape: &[usize],
        plan: &Self::Plan,
    ) -> Result<(), FftError> {
        f32::r2c(in_, out, shape, plan)?;
        Ok(())
    }

    fn forward_dft_batch(
        in_: &mut [<Self as DftType>::InputType],
        out: &mut [<Self as DftType>::OutputType],
        shape: &[usize],
        plan: &Self::Plan,
    ) -> Result<(), FftError> {
        f32::r2c_batch(in_, out, shape, plan)?;
        Ok(())
    }

    fn forward_dft_batch_par(
        in_: &mut [<Self as DftType>::InputType],
        out: &mut [<Self as DftType>::OutputType],
        shape: &[usize],
        plan: &Self::Plan,
    ) -> Result<(), FftError> {
        f32::r2c_batch_par(in_, out, shape, plan)?;
        Ok(())
    }

    fn backward_dft(
        in_: &mut [<Self as DftType>::OutputType],
        out: &mut [<Self as DftType>::InputType],
        shape: &[usize],
        plan: &Self::Plan,
    ) -> Result<(), FftError> {
        f32::c2r(in_, out, shape, plan)?;
        Ok(())
    }

    fn backward_dft_batch(
        in_: &mut [<Self as DftType>::OutputType],
        out: &mut [<Self as DftType>::InputType],
        shape: &[usize],
        plan: &Self::Plan,
    ) -> Result<(), FftError> {
        f32::c2r_batch(in_, out, shape, plan)?;
        Ok(())
    }

    fn backward_dft_batch_par(
        in_: &mut [<Self as DftType>::OutputType],
        out: &mut [<Self as DftType>::InputType],
        shape: &[usize],
        plan: &Self::Plan,
    ) -> Result<(), FftError> {
        f32::c2r_batch_par(in_, out, shape, plan)?;
        Ok(())
    }
}

impl Dft for f64 {
    type Plan = Plan<Self, Plan64>;

    fn plan_forward(
        in_: &mut [<Self as DftType>::InputType],
        out: &mut [<Self as DftType>::OutputType],
        shape: &[usize],
        batch: Option<BatchSize>,
    ) -> Result<Self::Plan, FftError> {
        let plan = if let Some(BatchSize(howmany)) = batch {
            let size_in: usize = shape.iter().product();
            let shape_out = [shape[0], shape[1], shape[2] / 2];
            let size_out: usize = shape_out.iter().product();
            let inembed = std::ptr::null();
            let istride = 1i32;
            let idist = size_in as i32;
            let onembed = std::ptr::null();
            let ostride = 1i32;
            let odist = size_out as i32;

            Plan64(
                validate_plan(excall!(ffi::fftw_plan_many_dft_r2c(
                    shape.len() as i32,
                    shape.iter().map(|&x| x as i32).collect_vec().as_mut_ptr() as *mut _,
                    howmany as i32,
                    in_.as_mut_ptr(),
                    inembed,
                    istride,
                    idist,
                    out.as_mut_ptr(),
                    onembed,
                    ostride,
                    odist,
                    FFTW_ESTIMATE
                )))
                .unwrap(),
            )
        } else {
            Plan64(
                validate_plan(excall!(ffi::fftw_plan_dft_r2c(
                    shape.len() as i32,
                    shape.iter().map(|&x| x as i32).collect_vec().as_mut_ptr() as *mut _,
                    in_.as_mut_ptr(),
                    out.as_mut_ptr(),
                    FFTW_ESTIMATE
                )))
                .unwrap(),
            )
        };

        Ok(Plan::new(plan))
    }

    fn plan_backward(
        in_: &mut [<Self as DftType>::OutputType],
        out: &mut [<Self as DftType>::InputType],
        shape: &[usize],
        batch: Option<BatchSize>,
    ) -> Result<Self::Plan, FftError> {
        let plan = if let Some(BatchSize(howmany)) = batch {
            let size_in: usize = shape.iter().product();
            let shape_out = [shape[0], shape[1], shape[2] / 2];
            let size_out: usize = shape_out.iter().product();
            let inembed = std::ptr::null();
            let istride = 1i32;
            let idist = size_in as i32;
            let onembed = std::ptr::null();
            let ostride = 1i32;
            let odist = (out.len() / size_out) as i32;

            Plan64(
                validate_plan(excall!(ffi::fftw_plan_many_dft_c2r(
                    shape.len() as i32,
                    shape.iter().map(|&x| x as i32).collect_vec().as_mut_ptr() as *mut _,
                    howmany as i32,
                    in_.as_mut_ptr(),
                    inembed,
                    istride,
                    idist,
                    out.as_mut_ptr(),
                    onembed,
                    ostride,
                    odist,
                    FFTW_ESTIMATE
                )))
                .unwrap(),
            )
        } else {
            Plan64(
                validate_plan(excall!(ffi::fftw_plan_dft_c2r(
                    shape.len() as i32,
                    shape.iter().map(|&x| x as i32).collect_vec().as_mut_ptr() as *mut _,
                    in_.as_mut_ptr(),
                    out.as_mut_ptr(),
                    FFTW_ESTIMATE
                )))
                .unwrap(),
            )
        };

        Ok(Plan::new(plan))
    }

    fn size_in(expansion_order: usize) -> usize {
        Self::shape_in(expansion_order).iter().product()
    }

    fn shape_in(expansion_order: usize) -> [usize; 3] {
        let m = 2 * expansion_order - 1; // Size of each dimension of 3D kernel/signal
        let pad_size = 1;
        let p = m + pad_size; // Size of each dimension of padded 3D kernel/signal
        [p, p, p]
    }

    fn shape_out(expansion_order: usize) -> [usize; 3] {
        let m = 2 * expansion_order - 1; // Size of each dimension of 3D kernel/signal
        let pad_size = 1;
        let p = m + pad_size; // Size of each dimension of padded 3D kernel/signal
        [p, p, p / 2 + 1]
    }

    fn size_out(expansion_order: usize) -> usize {
        Self::shape_out(expansion_order).iter().product()
    }

    fn forward_dft(
        in_: &mut [<Self as DftType>::InputType],
        out: &mut [<Self as DftType>::OutputType],
        shape: &[usize],
        plan: &Self::Plan,
    ) -> Result<(), FftError> {
        f64::r2c(in_, out, shape, plan)?;
        Ok(())
    }

    fn forward_dft_batch(
        in_: &mut [<Self as DftType>::InputType],
        out: &mut [<Self as DftType>::OutputType],
        shape: &[usize],
        plan: &Self::Plan,
    ) -> Result<(), FftError> {
        f64::r2c_batch(in_, out, shape, plan)?;
        Ok(())
    }

    fn forward_dft_batch_par(
        in_: &mut [<Self as DftType>::InputType],
        out: &mut [<Self as DftType>::OutputType],
        shape: &[usize],
        plan: &Self::Plan,
    ) -> Result<(), FftError> {
        f64::r2c_batch_par(in_, out, shape, plan)?;
        Ok(())
    }

    fn backward_dft(
        in_: &mut [<Self as DftType>::OutputType],
        out: &mut [<Self as DftType>::InputType],
        shape: &[usize],
        plan: &Self::Plan,
    ) -> Result<(), FftError> {
        f64::c2r(in_, out, shape, plan)?;
        Ok(())
    }

    fn backward_dft_batch(
        in_: &mut [<Self as DftType>::OutputType],
        out: &mut [<Self as DftType>::InputType],
        shape: &[usize],
        plan: &Self::Plan,
    ) -> Result<(), FftError> {
        f64::c2r_batch(in_, out, shape, plan)?;
        Ok(())
    }

    fn backward_dft_batch_par(
        in_: &mut [<Self as DftType>::OutputType],
        out: &mut [<Self as DftType>::InputType],
        shape: &[usize],
        plan: &Self::Plan,
    ) -> Result<(), FftError> {
        f64::c2r_batch_par(in_, out, shape, plan)?;
        Ok(())
    }
}

impl Dft for c32 {
    type Plan = Plan<Self, Plan32>;

    fn plan_forward(
        in_: &mut [<Self as DftType>::InputType],
        out: &mut [<Self as DftType>::OutputType],
        shape: &[usize],
        batch: Option<BatchSize>,
    ) -> Result<Self::Plan, FftError> {
        let plan = if let Some(BatchSize(howmany)) = batch {
            let size_in: usize = shape.iter().product();
            let shape_out = [shape[0], shape[1], shape[2] / 2];
            let size_out: usize = shape_out.iter().product();
            let inembed = std::ptr::null();
            let istride = 1i32;
            let idist = size_in as i32;
            let onembed = std::ptr::null();
            let ostride = 1i32;
            let odist = size_out as i32;

            Plan32(validate_plan(excall!(ffi::fftwf_plan_many_dft(
                shape.len() as i32,
                shape.iter().map(|&x| x as i32).collect_vec().as_mut_ptr() as *mut _,
                howmany as i32,
                in_.as_mut_ptr(),
                inembed,
                istride,
                idist,
                out.as_mut_ptr(),
                onembed,
                ostride,
                odist,
                Sign::Forward as i32,
                FFTW_ESTIMATE
            )))?)
        } else {
            Plan32(validate_plan(excall!(ffi::fftwf_plan_dft(
                shape.len() as i32,
                shape.iter().map(|&x| x as i32).collect_vec().as_mut_ptr() as *mut _,
                in_.as_mut_ptr(),
                out.as_mut_ptr(),
                Sign::Forward as i32,
                FFTW_ESTIMATE
            )))?)
        };

        Ok(Plan::new(plan))
    }

    fn plan_backward(
        in_: &mut [<Self as DftType>::OutputType],
        out: &mut [<Self as DftType>::InputType],
        shape: &[usize],
        batch: Option<BatchSize>,
    ) -> Result<Self::Plan, FftError> {
        let plan = if let Some(BatchSize(howmany)) = batch {
            let size_in: usize = shape.iter().product();
            let shape_out = [shape[0], shape[1], shape[2] / 2];
            let size_out: usize = shape_out.iter().product();
            let inembed = std::ptr::null();
            let istride = 1i32;
            let idist = size_in as i32;
            let onembed = std::ptr::null();
            let ostride = 1i32;
            let odist = size_out as i32;

            Plan32(validate_plan(excall!(ffi::fftwf_plan_many_dft(
                shape.len() as i32,
                shape.iter().map(|&x| x as i32).collect_vec().as_mut_ptr() as *mut _,
                howmany as i32,
                in_.as_mut_ptr(),
                inembed,
                istride,
                idist,
                out.as_mut_ptr(),
                onembed,
                ostride,
                odist,
                Sign::Backward as i32,
                FFTW_ESTIMATE
            )))?)
        } else {
            Plan32(validate_plan(excall!(ffi::fftwf_plan_dft(
                shape.len() as i32,
                shape.iter().map(|&x| x as i32).collect_vec().as_mut_ptr() as *mut _,
                in_.as_mut_ptr(),
                out.as_mut_ptr(),
                Sign::Backward as i32,
                FFTW_ESTIMATE
            )))?)
        };
        Ok(Plan::new(plan))
    }

    fn size_in(expansion_order: usize) -> usize {
        Self::shape_in(expansion_order).iter().product()
    }

    fn shape_in(expansion_order: usize) -> [usize; 3] {
        let m = 2 * expansion_order - 1; // Size of each dimension of 3D kernel/signal
        let pad_size = 1;
        let p = m + pad_size; // Size of each dimension of padded 3D kernel/signal
        [p, p, p]
    }

    fn shape_out(expansion_order: usize) -> [usize; 3] {
        let m = 2 * expansion_order - 1; // Size of each dimension of 3D kernel/signal
        let pad_size = 1;
        let p = m + pad_size; // Size of each dimension of padded 3D kernel/signal
        [p, p, p]
    }

    fn size_out(expansion_order: usize) -> usize {
        Self::shape_out(expansion_order).iter().product()
    }

    fn forward_dft(
        in_: &mut [<Self as DftType>::InputType],
        out: &mut [<Self as DftType>::OutputType],
        shape: &[usize],
        plan: &Self::Plan,
    ) -> Result<(), FftError> {
        c32::c2c(in_, out, shape, Sign::Forward, plan)?;
        Ok(())
    }

    fn forward_dft_batch(
        in_: &mut [<Self as DftType>::InputType],
        out: &mut [<Self as DftType>::OutputType],
        shape: &[usize],
        plan: &Self::Plan,
    ) -> Result<(), FftError> {
        c32::c2c_batch(in_, out, shape, Sign::Forward, plan)?;
        Ok(())
    }

    fn forward_dft_batch_par(
        in_: &mut [<Self as DftType>::InputType],
        out: &mut [<Self as DftType>::OutputType],
        shape: &[usize],
        plan: &Self::Plan,
    ) -> Result<(), FftError> {
        c32::c2c_batch_par(in_, out, shape, Sign::Forward, plan)?;
        Ok(())
    }

    fn backward_dft(
        in_: &mut [<Self as DftType>::OutputType],
        out: &mut [<Self as DftType>::InputType],
        shape: &[usize],
        plan: &Self::Plan,
    ) -> Result<(), FftError> {
        c32::c2c(in_, out, shape, Sign::Backward, plan)?;
        Ok(())
    }

    fn backward_dft_batch(
        in_: &mut [<Self as DftType>::OutputType],
        out: &mut [<Self as DftType>::InputType],
        shape: &[usize],
        plan: &Self::Plan,
    ) -> Result<(), FftError> {
        c32::c2c_batch(in_, out, shape, Sign::Backward, plan)?;
        Ok(())
    }

    fn backward_dft_batch_par(
        in_: &mut [<Self as DftType>::OutputType],
        out: &mut [<Self as DftType>::InputType],
        shape: &[usize],
        plan: &Self::Plan,
    ) -> Result<(), FftError> {
        c32::c2c_batch_par(in_, out, shape, Sign::Backward, plan)?;
        Ok(())
    }
}

impl Dft for c64 {
    type Plan = Plan<Self, Plan64>;

    fn plan_forward(
        in_: &mut [<Self as DftType>::InputType],
        out: &mut [<Self as DftType>::OutputType],
        shape: &[usize],
        batch: Option<BatchSize>,
    ) -> Result<Self::Plan, FftError> {
        let plan = if let Some(BatchSize(howmany)) = batch {
            let size_in: usize = shape.iter().product();
            let shape_out = [shape[0], shape[1], shape[2] / 2];
            let size_out: usize = shape_out.iter().product();
            let inembed = std::ptr::null();
            let istride = 1i32;
            let idist = size_in as i32;
            let onembed = std::ptr::null();
            let ostride = 1i32;
            let odist = size_out as i32;

            Plan64(validate_plan(excall!(ffi::fftw_plan_many_dft(
                shape.len() as i32,
                shape.iter().map(|&x| x as i32).collect_vec().as_mut_ptr() as *mut _,
                howmany as i32,
                in_.as_mut_ptr(),
                inembed,
                istride,
                idist,
                out.as_mut_ptr(),
                onembed,
                ostride,
                odist,
                Sign::Forward as i32,
                FFTW_ESTIMATE
            )))?)
        } else {
            Plan64(validate_plan(excall!(ffi::fftw_plan_dft(
                shape.len() as i32,
                shape.iter().map(|&x| x as i32).collect_vec().as_mut_ptr() as *mut _,
                in_.as_mut_ptr(),
                out.as_mut_ptr(),
                Sign::Forward as i32,
                FFTW_ESTIMATE
            )))?)
        };

        Ok(Plan::new(plan))
    }

    fn plan_backward(
        in_: &mut [<Self as DftType>::OutputType],
        out: &mut [<Self as DftType>::InputType],
        shape: &[usize],
        batch: Option<BatchSize>,
    ) -> Result<Self::Plan, FftError> {
        let plan = if let Some(BatchSize(howmany)) = batch {
            let size_in: usize = shape.iter().product();
            let shape_out = [shape[0], shape[1], shape[2] / 2];
            let size_out: usize = shape_out.iter().product();
            let inembed = std::ptr::null();
            let istride = 1i32;
            let idist = size_in as i32;
            let onembed = std::ptr::null();
            let ostride = 1i32;
            let odist = size_out as i32;

            Plan64(validate_plan(excall!(ffi::fftw_plan_many_dft(
                shape.len() as i32,
                shape.iter().map(|&x| x as i32).collect_vec().as_mut_ptr() as *mut _,
                howmany as i32,
                in_.as_mut_ptr(),
                inembed,
                istride,
                idist,
                out.as_mut_ptr(),
                onembed,
                ostride,
                odist,
                Sign::Backward as i32,
                FFTW_ESTIMATE
            )))?)
        } else {
            Plan64(validate_plan(excall!(ffi::fftw_plan_dft(
                shape.len() as i32,
                shape.iter().map(|&x| x as i32).collect_vec().as_mut_ptr() as *mut _,
                in_.as_mut_ptr(),
                out.as_mut_ptr(),
                Sign::Backward as i32,
                FFTW_ESTIMATE
            )))?)
        };
        Ok(Plan::new(plan))
    }

    fn size_in(expansion_order: usize) -> usize {
        Self::shape_in(expansion_order).iter().product()
    }

    fn shape_in(expansion_order: usize) -> [usize; 3] {
        let m = 2 * expansion_order - 1; // Size of each dimension of 3D kernel/signal
        let pad_size = 1;
        let p = m + pad_size; // Size of each dimension of padded 3D kernel/signal
        [p, p, p]
    }

    fn shape_out(expansion_order: usize) -> [usize; 3] {
        let m = 2 * expansion_order - 1; // Size of each dimension of 3D kernel/signal
        let pad_size = 1;
        let p = m + pad_size; // Size of each dimension of padded 3D kernel/signal
        [p, p, p]
    }

    fn size_out(expansion_order: usize) -> usize {
        Self::shape_out(expansion_order).iter().product()
    }

    fn forward_dft(
        in_: &mut [<Self as DftType>::InputType],
        out: &mut [<Self as DftType>::OutputType],
        shape: &[usize],
        plan: &Self::Plan,
    ) -> Result<(), FftError> {
        c64::c2c(in_, out, shape, Sign::Forward, plan)?;
        Ok(())
    }

    fn forward_dft_batch(
        in_: &mut [<Self as DftType>::InputType],
        out: &mut [<Self as DftType>::OutputType],
        shape: &[usize],
        plan: &Self::Plan,
    ) -> Result<(), FftError> {
        c64::c2c_batch(in_, out, shape, Sign::Forward, plan)?;
        Ok(())
    }

    fn forward_dft_batch_par(
        in_: &mut [<Self as DftType>::InputType],
        out: &mut [<Self as DftType>::OutputType],
        shape: &[usize],
        plan: &Self::Plan,
    ) -> Result<(), FftError> {
        c64::c2c_batch_par(in_, out, shape, Sign::Forward, plan)?;
        Ok(())
    }

    fn backward_dft(
        in_: &mut [<Self as DftType>::OutputType],
        out: &mut [<Self as DftType>::InputType],
        shape: &[usize],
        plan: &Self::Plan,
    ) -> Result<(), FftError> {
        c64::c2c(in_, out, shape, Sign::Backward, plan)?;
        Ok(())
    }

    fn backward_dft_batch(
        in_: &mut [<Self as DftType>::OutputType],
        out: &mut [<Self as DftType>::InputType],
        shape: &[usize],
        plan: &Self::Plan,
    ) -> Result<(), FftError> {
        c64::c2c_batch(in_, out, shape, Sign::Backward, plan)?;
        Ok(())
    }

    fn backward_dft_batch_par(
        in_: &mut [<Self as DftType>::OutputType],
        out: &mut [<Self as DftType>::InputType],
        shape: &[usize],
        plan: &Self::Plan,
    ) -> Result<(), FftError> {
        c64::c2c_batch_par(in_, out, shape, Sign::Backward, plan)?;
        Ok(())
    }
}

impl DftType for f64 {
    type InputType = Self;
    type OutputType = c64;
}

impl DftType for f32 {
    type InputType = Self;
    type OutputType = c32;
}

impl DftType for c32 {
    type InputType = Self;
    type OutputType = Self;
}

impl DftType for c64 {
    type InputType = Self;
    type OutputType = Self;
}
