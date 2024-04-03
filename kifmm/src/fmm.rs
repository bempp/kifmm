//! A 3D Kernel Independent Fast Multipole Method
pub mod builder;
pub mod constants;
#[allow(clippy::module_inception)]
pub mod fmm;
pub mod pinv;
pub mod send_ptr;
pub mod tree;
pub mod types;

pub mod field_translation {
    pub mod source;
    pub mod source_to_target {
        pub mod blas;
        pub mod fft;
        pub mod matmul;
        pub mod array;
        pub mod field;
        pub mod types;
        pub mod transfer_vector;
    }
    pub mod target;
}

#[cfg(test)]
mod test {
    extern crate blas_src;
    extern crate lapack_src;
}
