//! FFTW Traits
use num::Float;
use num_complex::{Complex, ComplexFloat};
use rlst::RlstScalar;

use crate::fftw::types::{FftError, Sign};

/// Interface for Real-to_Complex DFT computed with FFT library for 3D real data.
///
/// # Example usage
///
/// ```rust
/// use kifmm::traits::fftw::RealToComplexFft3D;
/// use num_complex::Complex;
/// use num::Zero;
/// use rlst::{rlst_dynamic_array3, RawAccessMut, c64};
///
///
/// // Single R2C FFT
/// {
///     let shape_in = [3, 3, 3];
///     let shape_out = [3, 3, 3/2 + 1];
///     let mut in_ = rlst_dynamic_array3!(f64, shape_in);
///     let mut rng = rand::thread_rng();
///     in_.fill_from_standard_normal(&mut rng); // fill with random data
///     let mut out = rlst_dynamic_array3!(c64, shape_out);
///
///     let _ = f64::r2c(in_.data_mut(), out.data_mut(), &shape_in).unwrap();
/// }
///
/// // (parallel) Batch operation
/// {
///     let n_batch = 10; // Number of DFTs in batch
///
///     let shape_in = [3, 3, 3];
///     let shape_out = [3, 3, 3/2 + 1];
///     let shape_in_batch = [3, 3, 3 * n_batch];
///     let shape_out_batch = [3, 3, (3/2 + 1) * n_batch];
///
///     let mut in_ = rlst_dynamic_array3!(f64, shape_in_batch);
///     let mut rng = rand::thread_rng();
///     in_.fill_from_standard_normal(&mut rng); // fill with random data
///     let mut out = rlst_dynamic_array3!(c64, shape_out_batch);
///
///     let _ = f64::r2c_batch(in_.data_mut(), out.data_mut(), &shape_in).unwrap();
///
///     // Optionally parallel
///     let _ = f64::r2c_batch_par(in_.data_mut(), out.data_mut(), &shape_in).unwrap();
/// }
///
/// ```
pub trait RealToComplexFft3D
where
    Self: Sized + RlstScalar + Float,
    Complex<Self>: RlstScalar,
{
    /// Compute in parallel real-to-complex DFT over batches of input sequences of dimension `shape`, where shape is of the form `[n1, n2, n3]` representing the
    /// 3D input sequence. Returns a batched output, where each output sequence of shape [n1, n2, n3/2 + 1]. The input and output sequences are expected in column major order.
    ///
    /// # Arguments
    /// * `input` - Batches of input sequences, each of shape [n1, n2, n3]. Where the total length of `input` is a multiple of `n1*n2*n3`. Expected in column major order.
    /// * `output` - Batches of output sequences, corresponding to the real-to-complex DFT of each input sequence. The length of `output` is a multiple of `n1 * n2 * n3/2 + 1`. Expected in column major order.
    /// * `shape` - Shape of each input sequence.
    fn r2c_batch_par(
        in_: &mut [Self],
        out: &mut [Complex<Self>],
        shape: &[usize],
    ) -> Result<(), FftError>;

    /// Compute real-to-complex DFT over batches of input sequences of dimension `shape`, where shape is of the form `[n1, n2, n3]` representing the
    /// 3D input sequence. Returns a batched output, where each output sequence of shape [n1, n2, n3/2 + 1]. The input and output sequences are expected in column major order.
    ///
    /// # Arguments
    /// * `input` - Batches of input sequences, each of shape [n1, n2, n3]. Where the total length of `input` is a multiple of `n1*n2*n3`. Expected in column major order.
    /// * `output` - Batches of output sequences, corresponding to the real-to-complex DFT of each input sequence. The length of `output` is a multiple of `n1 * n2 * n3/2 + 1`. Expected in column major order.
    /// * `shape` - Shape of each input sequence.
    fn r2c_batch(
        in_: &mut [Self],
        out: &mut [Complex<Self>],
        shape: &[usize],
    ) -> Result<(), FftError>;

    /// Compute a parallel complex-to-real DFT over batches of input sequences of dimension `shape`, where shape is of the form `[n1, n2, n3/2 + 1]` representing the
    /// 3D output sequence of a real-to-complex DFT. Returns a batched output, where each output sequence of shape [n1, n2, n3]. The input and output sequences are expected in column major order.
    ///
    /// # Arguments
    /// * `input` - Batches of output sequences, corresponding to the real-to-complex DFT of each input sequence. The length of `output` is a multiple of `n1 * n2 * n3/2 + 1`. Expected in column major order.
    /// * `output` - Batches of input sequences, each of shape [n1, n2, n3]. Where the total length of `input` is a multiple of `n1*n2*n3`. Expected in column major order.
    /// * `shape` - Shape of each input sequence.
    fn c2r_batch_par(
        in_: &mut [Complex<Self>],
        out: &mut [Self],
        shape: &[usize],
    ) -> Result<(), FftError>;

    /// Compute a complex-to-real DFT over batches of input sequences of dimension `shape`, where shape is of the form `[n1, n2, n3/2 + 1]` representing the
    /// 3D output sequence of a real-to-complex DFT. Returns a batched output, where each output sequence of shape [n1, n2, n3]. The input and output sequences are expected in column major order.
    ///
    /// # Arguments
    /// * `input` - Batches of output sequences, corresponding to the real-to-complex DFT of each input sequence. The length of `output` is a multiple of `n1 * n2 * n3/2 + 1`. Expected in column major order.
    /// * `output` - Batches of input sequences, each of shape [n1, n2, n3]. Where the total length of `input` is a multiple of `n1*n2*n3`. Expected in column major order.
    /// * `shape` - Shape of each input sequence.
    fn c2r_batch(
        in_: &mut [Complex<Self>],
        out: &mut [Self],
        shape: &[usize],
    ) -> Result<(), FftError>;

    /// Compute a real-to-complex DFT over an input sequences of dimension `shape`, where shape is of the form `[n1, n2, n3]`. Returns an output sequence of shape [n1, n2, n3/2 + 1].
    /// The input and output sequences are expected in column major order.
    ///
    /// # Arguments
    /// * `input` - 3D input sequence, Expected in column major order.
    /// * `output` - 3D output sequence. Expected in column major order.
    /// * `shape` - Shape of each input sequence.
    fn r2c(in_: &mut [Self], out: &mut [Complex<Self>], shape: &[usize]) -> Result<(), FftError>;

    /// Compute a complex-to-real DFT over an input sequences of dimension `shape`, where shape is of the form `[n1, n2, n3/2 + 1]` representing the 3D output sequence of a real-to-complex DFT.
    /// Returns an output sequence of shape [n1, n2, n3]. The input and output sequences are expected in column major order.
    /// The input and output sequences are expected in column major order.
    ///
    /// # Arguments
    /// * `input` - 3D input sequence, Expected in column major order.
    /// * `output` - 3D output sequence. Expected in column major order.
    /// * `shape` - Shape of each input sequence.
    fn c2r(in_: &mut [Complex<Self>], out: &mut [Self], shape: &[usize]) -> Result<(), FftError>;
}

/// Interface for Complex-to_Complex DFT computed with FFT library for 3D real data.
///
/// # Example usage
///
/// ```rust
/// use kifmm::traits::fftw::ComplexToComplexFft3D;
/// use kifmm::fftw::types::Sign;
/// use num_complex::Complex;
/// use num::Zero;
/// use rlst::{rlst_dynamic_array3, RawAccessMut, c64};
///
///
/// // Single forward C2C FFT
/// {
///     let shape = [3, 3, 3];
///     let mut in_ = rlst_dynamic_array3!(c64, shape);
///     let mut rng = rand::thread_rng();
///     in_.fill_from_standard_normal(&mut rng); // fill with random data
///     let mut out = rlst_dynamic_array3!(c64, shape);
///
///     let _ = c64::c2c(in_.data_mut(), out.data_mut(), &shape, Sign::Forward).unwrap();
/// }
///
/// // (parallel) Batch operation
/// {
///     let n_batch = 10; // Number of DFTs in batch
///
///     let shape = [3, 3, 3];
///     let shape_batch = [3, 3, 3 * n_batch];
///
///     let mut in_ = rlst_dynamic_array3!(c64, shape_batch);
///     let mut rng = rand::thread_rng();
///     in_.fill_from_standard_normal(&mut rng); // fill with random data
///     let mut out = rlst_dynamic_array3!(c64, shape_batch);
///
///     let _ = c64::c2c_batch(in_.data_mut(), out.data_mut(), &shape, Sign::Forward).unwrap();
///
///     // Optionally parallel
///     let _ = c64::c2c_batch_par(in_.data_mut(), out.data_mut(), &shape, Sign::Forward).unwrap();
/// }
///
/// ```
pub trait ComplexToComplexFft3D
where
    Self: Sized + RlstScalar + ComplexFloat,
{
    /// Compute in parallel complex-to-complex DFT over batches of input sequences of dimension `shape`, where shape is of the form `[n1, n2, n3]` representing the
    /// 3D input sequence. Returns a batched output, where each output sequence of shape [n1, n2, n3]. The input and output sequences are expected in column major order.
    ///
    /// # Arguments
    /// * `input` - Batches of input sequences, each of shape [n1, n2, n3]. Where the total length of `input` is a multiple of `n1*n2*n3`. Expected in column major order.
    /// * `output` - Batches of output sequences, corresponding to the complex-to-complex DFT of each input sequence. The length of `output` is a multiple of `n1 * n2 * n3`. Expected in column major order.
    /// * `shape` - Shape of each input sequence.
    /// * `sign` - Direction of transform
    fn c2c_batch_par(
        in_: &mut [Self],
        out: &mut [Self],
        shape: &[usize],
        sign: Sign,
    ) -> Result<(), FftError>;

    /// Compute complex-to-complex DFT over batches of input sequences of dimension `shape`, where shape is of the form `[n1, n2, n3]` representing the
    /// 3D input sequence. Returns a batched output, where each output sequence of shape [n1, n2, n3]. The input and output sequences are expected in column major order.
    ///
    /// # Arguments
    /// * `input` - Batches of input sequences, each of shape [n1, n2, n3]. Where the total length of `input` is a multiple of `n1*n2*n3`. Expected in column major order.
    /// * `output` - Batches of output sequences, corresponding to the complex-to-complex DFT of each input sequence. The length of `output` is a multiple of `n1 * n2 * n3`. Expected in column major order.
    /// * `shape` - Shape of each input sequence.
    /// * `sign` - Direction of transform
    fn c2c_batch(
        in_: &mut [Self],
        out: &mut [Self],
        shape: &[usize],
        sign: Sign,
    ) -> Result<(), FftError>;

    /// Compute a complex-to-complex DFT over an input sequences of dimension `shape`, where shape is of the form `[n1, n2, n3]`. Returns an output sequence of shape [n1, n2, n3].
    /// The input and output sequences are expected in column major order.
    ///
    /// # Arguments
    /// * `input` - 3D input sequence, Expected in column major order.
    /// * `output` - 3D output sequence. Expected in column major order.
    /// * `shape` - Shape of each input sequence.
    /// * `sign` - Direction of transform
    fn c2c(in_: &mut [Self], out: &mut [Self], shape: &[usize], sign: Sign)
        -> Result<(), FftError>;
}

/// Input and output types for which DFT trait is defined
pub trait DftType {
    /// Input type
    type InputType: RlstScalar + Default + Sized;

    /// Output type
    type OutputType: RlstScalar + Default + Sized;
}

/// Interface for DFT for use from FMM
pub trait Dft
where
    Self: Sized + RlstScalar + DftType,
{
    /// Input shape, shape of convolution grid
    fn shape_in(expansion_order: usize) -> [usize; 3];

    /// Input size, number of points on convolution grid
    fn size_in(expansion_order: usize) -> usize;

    /// Transformed shape in Fourier space, sequence shortened for real to complex transforms
    fn shape_out(expansion_order: usize) -> [usize; 3];

    /// Transformed size in Fourier space, sequence shortened for real to complex transforms
    fn size_out(expansion_order: usize) -> usize;

    /// Forward transform, if real data is used as input computes R2C dft
    fn forward_dft(
        in_: &mut [<Self as DftType>::InputType],
        out: &mut [<Self as DftType>::OutputType],
        shape: &[usize],
    ) -> Result<(), FftError>;

    /// Forward transform, if real data is used as input computes R2C dft, batched over multiple datasets
    fn forward_dft_batch(
        in_: &mut [<Self as DftType>::InputType],
        out: &mut [<Self as DftType>::OutputType],
        shape: &[usize],
    ) -> Result<(), FftError>;

    /// Forward transform, if real data is used as input computes R2C dft, batched in parallel over multiple datasets
    fn forward_dft_batch_par(
        in_: &mut [<Self as DftType>::InputType],
        out: &mut [<Self as DftType>::OutputType],
        shape: &[usize],
    ) -> Result<(), FftError>;

    /// Backward transform, if real data is used as input computes C2R dft
    fn backward_dft(
        in_: &mut [<Self as DftType>::OutputType],
        out: &mut [<Self as DftType>::InputType],
        shape: &[usize],
    ) -> Result<(), FftError>;

    /// Backward transform, if real data is used as input computes C2R dft, batched over multiple datasets
    fn backward_dft_batch(
        in_: &mut [<Self as DftType>::OutputType],
        out: &mut [<Self as DftType>::InputType],
        shape: &[usize],
    ) -> Result<(), FftError>;

    /// Backward transform, if real data is used as input computes C2R dft, batched in parallel over multiple datasets
    fn backward_dft_batch_par(
        in_: &mut [<Self as DftType>::OutputType],
        out: &mut [<Self as DftType>::InputType],
        shape: &[usize],
    ) -> Result<(), FftError>;
}
