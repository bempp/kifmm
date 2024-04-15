use rlst::{c32, c64};

use crate::{
    fftw::types::{FftError, Sign},
    traits::fftw::{ComplexToComplexFft3D, Dft, DftType, RealToComplexFft3D},
};

impl Dft for f32 {
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
    ) -> Result<(), FftError> {
        f32::r2c(in_, out, shape)?;
        Ok(())
    }

    fn forward_dft_batch(
        in_: &mut [<Self as DftType>::InputType],
        out: &mut [<Self as DftType>::OutputType],
        shape: &[usize],
    ) -> Result<(), FftError> {
        f32::r2c_batch(in_, out, shape)?;
        Ok(())
    }

    fn forward_dft_batch_par(
        in_: &mut [<Self as DftType>::InputType],
        out: &mut [<Self as DftType>::OutputType],
        shape: &[usize],
    ) -> Result<(), FftError> {
        f32::r2c_batch_par(in_, out, shape)?;
        Ok(())
    }

    fn backward_dft(
        in_: &mut [<Self as DftType>::OutputType],
        out: &mut [<Self as DftType>::InputType],
        shape: &[usize],
    ) -> Result<(), FftError> {
        f32::c2r(in_, out, shape)?;
        Ok(())
    }

    fn backward_dft_batch(
        in_: &mut [<Self as DftType>::OutputType],
        out: &mut [<Self as DftType>::InputType],
        shape: &[usize],
    ) -> Result<(), FftError> {
        f32::c2r_batch(in_, out, shape)?;
        Ok(())
    }

    fn backward_dft_batch_par(
        in_: &mut [<Self as DftType>::OutputType],
        out: &mut [<Self as DftType>::InputType],
        shape: &[usize],
    ) -> Result<(), FftError> {
        f32::c2r_batch_par(in_, out, shape)?;
        Ok(())
    }
}

impl Dft for f64 {
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
    ) -> Result<(), FftError> {
        f64::r2c(in_, out, shape)?;
        Ok(())
    }

    fn forward_dft_batch(
        in_: &mut [<Self as DftType>::InputType],
        out: &mut [<Self as DftType>::OutputType],
        shape: &[usize],
    ) -> Result<(), FftError> {
        f64::r2c_batch(in_, out, shape)?;
        Ok(())
    }

    fn forward_dft_batch_par(
        in_: &mut [<Self as DftType>::InputType],
        out: &mut [<Self as DftType>::OutputType],
        shape: &[usize],
    ) -> Result<(), FftError> {
        f64::r2c_batch_par(in_, out, shape)?;
        Ok(())
    }

    fn backward_dft(
        in_: &mut [<Self as DftType>::OutputType],
        out: &mut [<Self as DftType>::InputType],
        shape: &[usize],
    ) -> Result<(), FftError> {
        f64::c2r(in_, out, shape)?;
        Ok(())
    }

    fn backward_dft_batch(
        in_: &mut [<Self as DftType>::OutputType],
        out: &mut [<Self as DftType>::InputType],
        shape: &[usize],
    ) -> Result<(), FftError> {
        f64::c2r_batch(in_, out, shape)?;
        Ok(())
    }

    fn backward_dft_batch_par(
        in_: &mut [<Self as DftType>::OutputType],
        out: &mut [<Self as DftType>::InputType],
        shape: &[usize],
    ) -> Result<(), FftError> {
        f64::c2r_batch_par(in_, out, shape)?;
        Ok(())
    }
}

impl Dft for c32 {
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
    ) -> Result<(), FftError> {
        c32::c2c(in_, out, shape, Sign::Forward)?;
        Ok(())
    }

    fn forward_dft_batch(
        in_: &mut [<Self as DftType>::InputType],
        out: &mut [<Self as DftType>::OutputType],
        shape: &[usize],
    ) -> Result<(), FftError> {
        c32::c2c_batch(in_, out, shape, Sign::Forward)?;
        Ok(())
    }

    fn forward_dft_batch_par(
        in_: &mut [<Self as DftType>::InputType],
        out: &mut [<Self as DftType>::OutputType],
        shape: &[usize],
    ) -> Result<(), FftError> {
        c32::c2c_batch_par(in_, out, shape, Sign::Forward)?;
        Ok(())
    }

    fn backward_dft(
        in_: &mut [<Self as DftType>::OutputType],
        out: &mut [<Self as DftType>::InputType],
        shape: &[usize],
    ) -> Result<(), FftError> {
        c32::c2c(in_, out, shape, Sign::Backward)?;
        Ok(())
    }

    fn backward_dft_batch(
        in_: &mut [<Self as DftType>::OutputType],
        out: &mut [<Self as DftType>::InputType],
        shape: &[usize],
    ) -> Result<(), FftError> {
        c32::c2c_batch(in_, out, shape, Sign::Backward)?;
        Ok(())
    }

    fn backward_dft_batch_par(
        in_: &mut [<Self as DftType>::OutputType],
        out: &mut [<Self as DftType>::InputType],
        shape: &[usize],
    ) -> Result<(), FftError> {
        c32::c2c_batch_par(in_, out, shape, Sign::Backward)?;
        Ok(())
    }
}

impl Dft for c64 {
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
    ) -> Result<(), FftError> {
        c64::c2c(in_, out, shape, Sign::Forward)?;
        Ok(())
    }

    fn forward_dft_batch(
        in_: &mut [<Self as DftType>::InputType],
        out: &mut [<Self as DftType>::OutputType],
        shape: &[usize],
    ) -> Result<(), FftError> {
        c64::c2c_batch(in_, out, shape, Sign::Forward)?;
        Ok(())
    }

    fn forward_dft_batch_par(
        in_: &mut [<Self as DftType>::InputType],
        out: &mut [<Self as DftType>::OutputType],
        shape: &[usize],
    ) -> Result<(), FftError> {
        c64::c2c_batch_par(in_, out, shape, Sign::Forward)?;
        Ok(())
    }

    fn backward_dft(
        in_: &mut [<Self as DftType>::OutputType],
        out: &mut [<Self as DftType>::InputType],
        shape: &[usize],
    ) -> Result<(), FftError> {
        c64::c2c(in_, out, shape, Sign::Backward)?;
        Ok(())
    }

    fn backward_dft_batch(
        in_: &mut [<Self as DftType>::OutputType],
        out: &mut [<Self as DftType>::InputType],
        shape: &[usize],
    ) -> Result<(), FftError> {
        c64::c2c_batch(in_, out, shape, Sign::Backward)?;
        Ok(())
    }

    fn backward_dft_batch_par(
        in_: &mut [<Self as DftType>::OutputType],
        out: &mut [<Self as DftType>::InputType],
        shape: &[usize],
    ) -> Result<(), FftError> {
        c64::c2c_batch_par(in_, out, shape, Sign::Backward)?;
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
