#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use libc::FILE;
pub use num_complex::Complex32 as fftwf_complex;
pub use num_complex::Complex64 as fftw_complex;

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
