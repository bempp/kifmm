//! Useful traits
use num::Complex;
use rlst::{c32, c64, RlstScalar};

use crate::fmm::types::Isa;

/// Returns the maximum machine precision corresponding to a type
pub trait Epsilon
where
    Self: RlstScalar,
{
    /// Returns epsilon, a small positive value.
    fn epsilon() -> Self::Real;
}

macro_rules! impl_epsilon {
    // Implementations for floating-point types directly returning their EPSILON
    ($t:ty) => {
        impl Epsilon for $t {
            fn epsilon() -> Self {
                <$t>::EPSILON
            }
        }
    };
    // Implementations for complex types, returning the EPSILON of their real component type
    ($t:ty, $real_t:ty) => {
        impl Epsilon for $t {
            fn epsilon() -> Self::Real {
                <$real_t>::EPSILON
            }
        }
    };
}

impl_epsilon!(f32);
impl_epsilon!(f64);
impl_epsilon!(c32, f32);
impl_epsilon!(c64, f64);

/// Returns a complex representation of this type
pub trait AsComplex {
    /// The associated complex type, implements RlstScalar
    type ComplexType: RlstScalar + Default;
}

macro_rules! impl_as_complex {
    ($t:ty, $complex_t:ty) => {
        impl AsComplex for $t {
            type ComplexType = $complex_t;
        }
    };
}

impl_as_complex!(f32, Complex<f32>);
impl_as_complex!(f64, Complex<f64>);
impl_as_complex!(c32, c32);
impl_as_complex!(c64, c64);

/// Implement vectorised 8x8 Hadamard operation,
pub trait Hadamard8x8 {
    /// Associated scalar type
    type Scalar: RlstScalar;

    /// 8x8 Hadamard product
    fn hadamard8x8(
        isa: Isa,
        matrix: &[Self::Scalar; 64],
        vector: &[Self::Scalar; 8],
        result: &mut [Self::Scalar; 8],
        scale: Self::Scalar,
    );
}
