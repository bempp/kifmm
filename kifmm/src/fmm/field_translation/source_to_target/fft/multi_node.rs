//! Multipole to local field translation trait implementation using FFT.
use std::collections::HashSet;

use itertools::Itertools;
use mpi::traits::Equivalence;
use num::{Float, One, Zero};

use rayon::prelude::*;
use rlst::{
    empty_array, rlst_dynamic_array2, MultIntoResize, RandomAccessMut, RawAccess, RlstScalar,
};

use green_kernels::traits::Kernel as KernelTrait;

use crate::{
    fftw::array::{AlignedAllocable, AlignedVec},
    fmm::{
        helpers::single_node::{chunk_size, homogenous_kernel_scale, m2l_scale},
        types::{FmmEvalType, KiFmmMulti, SendPtrMut},
        KiFmm,
    },
    traits::{
        fftw::Dft,
        fmm::{FmmOperatorData, HomogenousKernel, SourceToTargetTranslation},
        general::{AsComplex, Hadamard8x8},
        tree::{SingleFmmTree, SingleTree},
        types::FmmError,
    },
    tree::{
        constants::{NHALO, NSIBLINGS, NSIBLINGS_SQUARED},
        types::MortonKey,
    },
    FftFieldTranslation, SingleFmm,
};

impl<Scalar, Kernel> SourceToTargetTranslation
    for KiFmmMulti<Scalar, Kernel, FftFieldTranslation<f32>>
where
    Scalar: RlstScalar
        + AsComplex
        + Dft<InputType = Scalar, OutputType = <Scalar as AsComplex>::ComplexType>
        + Default
        + AlignedAllocable
        + Equivalence
        + Float,
    <Scalar as AsComplex>::ComplexType:
        Hadamard8x8<Scalar = <Scalar as AsComplex>::ComplexType> + AlignedAllocable,
    Kernel: KernelTrait<T = Scalar> + HomogenousKernel + Default + Send + Sync,
    <Scalar as RlstScalar>::Real: Default + Equivalence + Float,
    <Scalar as Dft>::Plan: Sync,
    // Self: FmmOperatorData,
{
    fn m2l(&self, level: u64) -> Result<(), FmmError> {
        Ok(())
    }

    fn p2l(&self, level: u64) -> Result<(), FmmError> {
        Ok(())
    }
}
