//! Implementation of traits for threadsafe pointers
use crate::fmm::types::{SendPtr, SendPtrMut};

unsafe impl<T> Sync for SendPtrMut<T> {}
unsafe impl<T> Send for SendPtrMut<T> {}
unsafe impl<T> Sync for SendPtr<T> {}
unsafe impl<T> Send for SendPtr<T> {}

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
