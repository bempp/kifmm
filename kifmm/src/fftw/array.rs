//! Cache aligned arrays
use std::{
    ops::{Deref, DerefMut},
    os::raw::c_void,
    slice::{from_raw_parts, from_raw_parts_mut},
};

use kifmm_fftw_sys as ffi;
use num::Zero;
use rlst::{c32, c64};

use crate::excall;

/// A RAII-wrapper of `fftw_alloc` and `fftw_free` with the [SIMD alignment].
///
/// [SIMD alignment]: http://www.fftw.org/fftw3_doc/SIMD-alignment-and-fftw_005fmalloc.html
#[derive(Debug)]
pub struct AlignedVec<T> {
    n: usize,
    data: *mut T,
}

/// Allocate SIMD-aligned memory of Real/Complex type
pub trait AlignedAllocable: Zero + Clone + Copy + Sized {
    /// Allocate SIMD-aligned memory
    ///
    /// # Safety
    unsafe fn alloc(n: usize) -> *mut Self;
}

impl AlignedAllocable for f64 {
    unsafe fn alloc(n: usize) -> *mut Self {
        ffi::fftw_alloc_real(n)
    }
}

impl AlignedAllocable for f32 {
    unsafe fn alloc(n: usize) -> *mut Self {
        ffi::fftwf_alloc_real(n)
    }
}

impl AlignedAllocable for c64 {
    unsafe fn alloc(n: usize) -> *mut Self {
        ffi::fftw_alloc_complex(n)
    }
}

impl AlignedAllocable for c32 {
    unsafe fn alloc(n: usize) -> *mut Self {
        ffi::fftwf_alloc_complex(n)
    }
}

impl<T> AlignedVec<T> {
    /// Return slice
    pub fn as_slice(&self) -> &[T] {
        unsafe { from_raw_parts(self.data, self.n) }
    }

    /// Return mutable slice
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        unsafe { from_raw_parts_mut(self.data, self.n) }
    }
}

impl<T> Deref for AlignedVec<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> DerefMut for AlignedVec<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_slice_mut()
    }
}

impl<T> AlignedVec<T>
where
    T: AlignedAllocable,
{
    /// Create array with `fftw_malloc` (`fftw_free` will be automatically called by `Drop` trait)
    pub fn new(n: usize) -> Self {
        let ptr = excall! { T::alloc(n) };
        let mut vec = AlignedVec { n, data: ptr };
        for v in vec.iter_mut() {
            *v = T::zero();
        }
        vec
    }
}

impl<T> Drop for AlignedVec<T> {
    fn drop(&mut self) {
        excall! { ffi::fftw_free(self.data as *mut c_void) };
    }
}

impl<T> Clone for AlignedVec<T>
where
    T: AlignedAllocable,
{
    fn clone(&self) -> Self {
        let mut new_vec = Self::new(self.n);
        new_vec.copy_from_slice(self);
        new_vec
    }
}

unsafe impl<T: Send> Send for AlignedVec<T> {}
unsafe impl<T: Sync> Sync for AlignedVec<T> {}

/// Cache alignment value as a Rust type
pub type Alignment = i32;

/// Check the alignment of slice
///
/// ```
/// # use kifmm::fftw::array::{alignment_of, AlignedVec};
///
/// let a = AlignedVec::<f32>::new(123);
/// assert_eq!(alignment_of(&a), 0);  // aligned
/// ```
pub fn alignment_of<T>(a: &[T]) -> Alignment {
    unsafe { ffi::fftw_alignment_of(a.as_ptr() as *mut _) }
}
