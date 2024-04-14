//! Traits that are useful across modules
use num::Complex;
use rlst::RlstScalar;
use rlst::{c32, c64};

/// A trait that provides the maximum machine precision corresponding to a type
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
