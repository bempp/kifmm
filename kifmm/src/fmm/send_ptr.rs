//! Implementation of traits for threadsafe pointers
use crate::fmm::types::{SendPtr, SendPtrMut};
use rlst::RlstScalar;

unsafe impl<T: RlstScalar> Sync for SendPtrMut<T> {}
unsafe impl<T: RlstScalar> Send for SendPtrMut<T> {}
unsafe impl<T: RlstScalar> Sync for SendPtr<T> {}
unsafe impl<T: RlstScalar> Send for SendPtr<T> {}

impl<T> Default for SendPtrMut<T> {
    fn default() -> Self {
        SendPtrMut {
            raw: std::ptr::null_mut(),
        }
    }
}

impl<T> Default for SendPtr<T> {
    fn default() -> Self {
        SendPtr {
            raw: std::ptr::null(),
        }
    }
}
