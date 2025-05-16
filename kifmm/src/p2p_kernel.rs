//! Kernels for AMD,NVIDIA and CPU architectures
pub mod cpu;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "amd")]
pub mod hip;


