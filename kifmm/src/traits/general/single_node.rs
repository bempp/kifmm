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

/// Defines higher precision scalar type corresponding to Self
pub trait Upcast
where
    Self: Sized,
{
    /// The higher precision type (e.g. f32->f64, c32->c64)
    type Higher: RlstScalar + Cast<Self>;
}

impl Upcast for f32 {
    type Higher = f64;
}

impl Upcast for f64 {
    type Higher = f64;
}

impl Upcast for c32 {
    type Higher = c64;
}

impl Upcast for c64 {
    type Higher = c64;
}

/// Cast between scalar types Self and T
#[allow(dead_code)]
pub trait Cast<T> {
    /// Return a new scalar, cast to T
    fn cast(&self) -> T;
}

impl Cast<f32> for f64 {
    fn cast(&self) -> f32 {
        *self as f32
    }
}

impl Cast<f32> for f32 {
    fn cast(&self) -> f32 {
        *self
    }
}

impl Cast<f64> for f64 {
    fn cast(&self) -> f64 {
        *self
    }
}

impl Cast<f64> for f32 {
    fn cast(&self) -> f64 {
        *self as f64
    }
}

impl Cast<c32> for c64 {
    fn cast(&self) -> c32 {
        c32::new(self.re() as f32, self.im() as f32)
    }
}

impl Cast<c32> for c32 {
    fn cast(&self) -> c32 {
        *self
    }
}

impl Cast<c64> for c64 {
    fn cast(&self) -> c64 {
        *self
    }
}

impl Cast<c64> for c32 {
    fn cast(&self) -> c64 {
        c64::new(self.re() as f64, self.im() as f64)
    }
}

/// Trait that abstracts over real/complex numbers for their magnitude
pub trait ArgmaxValue<Scalar: RlstScalar> {
    /// Return the magnitude
    fn argmax_value(&self) -> <Scalar as RlstScalar>::Real;
}

macro_rules! impl_argmax_value {
    // For real numbers, argmax defined by value
    ($t:ty) => {
        impl ArgmaxValue<$t> for $t {
            fn argmax_value(&self) -> $t {
                *self
            }
        }
    };

    // For complex numbers, argmax defined by magnitude
    ($t:ty, $r:ty) => {
        impl ArgmaxValue<$t> for $t {
            fn argmax_value(&self) -> $r {
                self.abs()
            }
        }
    };
}

impl_argmax_value!(f32);
impl_argmax_value!(f64);
impl_argmax_value!(c32, f32);
impl_argmax_value!(c64, f64);
