//! Multipole to local field translation trait implementation using FFT.
use mpi::traits::Equivalence;
use num::Float;

use rlst::RlstScalar;

use green_kernels::traits::Kernel as KernelTrait;

use crate::{
    fftw::array::AlignedAllocable,
    fmm::types::KiFmmMulti,
    traits::{
        fftw::Dft,
        fmm::{FmmDataAccessMulti, HomogenousKernel, SourceToTargetTranslation},
        general::single_node::{AsComplex, Hadamard8x8},
        types::FmmError,
    },
    FftFieldTranslation,
};

impl<Scalar, Kernel> SourceToTargetTranslation
    for KiFmmMulti<Scalar, Kernel, FftFieldTranslation<Scalar>>
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
    Self: FmmDataAccessMulti<Scalar = Scalar>,
{
    fn m2l(&self, level: u64) -> Result<(), FmmError> {
        let _multipoles = self.multipoles(level);
        Ok(())
    }

    fn p2l(&self, _level: u64) -> Result<(), FmmError> {
        Ok(())
    }
}
