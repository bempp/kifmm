//! Implementation of traits for field translations via the FFT and BLAS.
// use super::transfer_vector::compute_transfer_vectors;

// use crate::new_fmm::helpers::{flip3, ncoeffs_kifmm};
use crate::new_fmm::types::{BlasFieldTranslation, BlasMetadata, FftFieldTranslation, FftMetadata};
use crate::traits::field::ConfigureSourceToTargetData;
use crate::traits::{fftw::RealToComplexFft3D, field::SourceToTargetData, tree::FmmTreeNode};
use crate::tree::{
    constants::{
        ALPHA_INNER, NCORNERS, NHALO, NSIBLINGS, NSIBLINGS_SQUARED, NTRANSFER_VECTORS_KIFMM,
    },
    helpers::find_corners,
    types::{Domain, MortonKey},
};
use crate::RlstScalarComplexFloat;
use green_kernels::{traits::Kernel, types::EvalType};
use itertools::Itertools;
use num::{Complex, Float, Zero};
use num_complex::ComplexFloat;
use rlst::{
    empty_array, rlst_array_from_slice2, rlst_dynamic_array2, rlst_dynamic_array3, Array,
    BaseArray, Gemm, MatrixSvd, MultIntoResize, RawAccess, RawAccessMut, RlstScalar, Shape,
    SvdMode, UnsafeRandomAccessByRef, UnsafeRandomAccessMut, VectorContainer,
};
use std::collections::HashSet;

impl<Scalar, Kern> SourceToTargetData for FftFieldTranslation<Scalar, Kern>
where
    Scalar: RlstScalar,
    Kern: Kernel<T = Scalar> + Default,
{
    type Metadata = FftMetadata<Scalar>;
}

impl<Scalar, Kern> ConfigureSourceToTargetData for FftFieldTranslation<Scalar, Kern>
where
    Scalar: RlstScalar,
    Kern: Kernel<T = Scalar> + Default,
{
    type Scalar = Scalar;
    type Domain = Domain<Scalar::Real>;
    type Kernel = Kern;

    fn expansion_order(&mut self, expansion_order: usize) {}

    fn kernel(&mut self, kernel: Self::Kernel) {}

    fn operator_data(&mut self, expansion_order: usize, domain: Self::Domain) {}
}

impl<Scalar, Kern> SourceToTargetData for BlasFieldTranslation<Scalar, Kern>
where
    Scalar: RlstScalar,
    Kern: Kernel<T = Scalar> + Default,
{
    type Metadata = FftMetadata<Scalar>;
}

impl<Scalar, Kern> ConfigureSourceToTargetData for BlasFieldTranslation<Scalar, Kern>
where
    Scalar: RlstScalar,
    Kern: Kernel<T = Scalar> + Default,
{
    type Scalar = Scalar;
    type Domain = Domain<Scalar::Real>;
    type Kernel = Kern;

    fn expansion_order(&mut self, expansion_order: usize) {}

    fn kernel(&mut self, kernel: Self::Kernel) {}

    fn operator_data(&mut self, expansion_order: usize, domain: Self::Domain) {}
}
