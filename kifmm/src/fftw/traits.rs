//! FFTW traits
use crate::fftw::fftw::FftError;
use num_complex::Complex;

/// Interface for Real-to_Complex DFT computed with FFT library for 3D real data.
///
/// # Example usage
///
/// ```rust
/// use kifmm::fftw::traits::RealToComplexFft3D;
/// use num_complex::Complex;
/// use num::Zero;
/// use rlst::{rlst_dynamic_array3, RawAccessMut};
///
///
/// // Single R2C FFT
/// {
///     let shape_in = [3, 3, 3];
///     let shape_out = [3, 3, 3/2 + 1];
///     let mut in_ = rlst_dynamic_array3!(f64, shape_in);
///     let mut rng = rand::thread_rng();
///     in_.fill_from_standard_normal(&mut rng); // fill with random data
///     let mut out = rlst_dynamic_array3!(f64, shape_out)
///
///     let _success = f64::r2c(in_.data_mut(), out.data_mut(), &shape_in).unwrap();
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
///     let mut out = rlst_dynamic_array3!(f64, shape_out_batch)
///
///     let _success_batch = f64::r2c_batch(&mut in_, &mut out, &shape_in).unwrap();
///
///     // Optionally parallel
///     let _success_batch_par = f64::r2c_batch_par(&mut in_, &mut out, &shape_in).unwrap();
/// }
///
/// ```
pub trait RealToComplexFft3D
where
    Self: Sized,
{
    /// Compute in parallel real-to-complex DFT over batches of input sequences of dimension `shape`, where shape is of the form `[n1, n2, n3]` representing the
    /// 3D input sequence. Returns a batched output, where each output sequence of shape [n1, n2, n3/2 + 1]. The input and output sequences are expected in column major order.
    ///
    /// # Arguments
    /// * `input` - Batches of input sequences, each of shape [n1, n2, n3]. Where the total length of `input` is a multiple of `n1*n2*n3`. Expected in column major order.
    /// * `output` - Batches of output sequences, corresponding to the real-to-complex DFT of each input sequence. The length of `output` is a multiple of `n1 * n2 * n3/2 + 1`. Expected in column major order.
    /// * `shape` - Shape of each input sequence.
    fn r2c_batch_par(
        input: &mut [Self],
        output: &mut [Complex<Self>],
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
        input: &mut [Self],
        output: &mut [Complex<Self>],
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
        input: &mut [Complex<Self>],
        output: &mut [Self],
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
        input: &mut [Complex<Self>],
        output: &mut [Self],
        shape: &[usize],
    ) -> Result<(), FftError>;

    /// Compute a real-to-complex DFT over an input sequences of dimension `shape`, where shape is of the form `[n1, n2, n3]`. Returns an output sequence of shape [n1, n2, n3/2 + 1].
    /// The input and output sequences are expected in column major order.
    ///
    /// # Arguments
    /// * `input` - 3D input sequence, Expected in column major order.
    /// * `output` - 3D output sequence. Expected in column major order.
    /// * `shape` - Shape of each input sequence.
    fn r2c(
        input: &mut [Self],
        output: &mut [Complex<Self>],
        shape: &[usize],
    ) -> Result<(), FftError>;

    /// Compute a complex-to-real DFT over an input sequences of dimension `shape`, where shape is of the form `[n1, n2, n3/2 + 1]` representing the 3D output sequence of a real-to-complex DFT.
    /// Returns an output sequence of shape [n1, n2, n3]. The input and output sequences are expected in column major order.
    /// The input and output sequences are expected in column major order.
    ///
    /// # Arguments
    /// * `input` - 3D input sequence, Expected in column major order.
    /// * `output` - 3D output sequence. Expected in column major order.
    /// * `shape` - Shape of each input sequence.
    fn c2r(
        input: &mut [Complex<Self>],
        output: &mut [Self],
        shape: &[usize],
    ) -> Result<(), FftError>;
}
