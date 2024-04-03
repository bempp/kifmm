//! A 3D Kernel Independent Fast Multipole Method
mod builder;
pub mod constants;
#[allow(clippy::module_inception)]
mod fmm;
pub mod pinv;
mod send_ptr;
pub mod tree;
pub mod types;

pub mod field_translation;

#[cfg(test)]
mod test {
    extern crate blas_src;
    extern crate lapack_src;
}
