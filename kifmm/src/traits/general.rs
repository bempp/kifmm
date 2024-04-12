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

impl Epsilon for f32 {
    fn epsilon() -> Self {
        f32::EPSILON
    }
}

impl Epsilon for f64 {
    fn epsilon() -> Self {
        f64::EPSILON
    }
}

impl Epsilon for c32 {
    fn epsilon() -> Self::Real {
        f32::EPSILON
    }
}

impl Epsilon for c64 {
    fn epsilon() -> Self::Real {
        f64::EPSILON
    }
}

/// Returns a complex representation of this type
pub trait AsComplex {
    /// The associated complex type, implements RlstScalar
    type ComplexType: RlstScalar + Default;
}

impl AsComplex for f32 {
    type ComplexType = Complex<f32>;
}

impl AsComplex for f64 {
    type ComplexType = Complex<f64>;
}

impl AsComplex for c32 {
    type ComplexType = c32;
}

impl AsComplex for c64 {
    type ComplexType = c64;
}
