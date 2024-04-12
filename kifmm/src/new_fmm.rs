//! TODO DOCS
pub mod builder;
pub mod constants;
pub mod helpers;
mod multi_node;
pub mod pinv;
mod send_ptr;
mod single_node;
mod tree;
pub mod types;

mod field_translation;

pub use types::KiFmm;

#[cfg(test)]
mod test {
    extern crate blas_src;
    extern crate lapack_src;
}
