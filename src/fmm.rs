//! A a general framework for implementing Fast Multipole Methods.
pub mod builder;
pub mod constants;
#[allow(clippy::module_inception)]
pub mod fmm;
pub mod helpers;
pub mod pinv;
pub mod send_ptr;
pub mod tree;
pub mod types;

mod field_translation {
    pub mod matmul;
    pub mod source;
    pub mod source_to_target {
        pub mod blas;
        pub mod fft;
    }
    pub mod target;
}

#[cfg(test)]
mod test {
    extern crate blas_src;
    extern crate lapack_src;
}
