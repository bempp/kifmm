//! Multipole to local field translation trait implementation using BLAS.

use std::sync::Mutex;

use itertools::Itertools;
use mpi::traits::Equivalence;
use num::Float;
use rayon::prelude::*;
use rlst::{
    empty_array, rlst_array_from_slice2, rlst_dynamic_array2, MultIntoResize, RawAccess,
    RawAccessMut, RlstScalar,
};

use green_kernels::traits::Kernel as KernelTrait;

use crate::{
    fmm::{
        helpers::single_node::{homogenous_kernel_scale, m2l_scale},
        types::{BlasFieldTranslationIa, FmmEvalType, KiFmmMulti, SendPtrMut},
        KiFmm,
    },
    traits::{
        fmm::{FmmOperatorData, HomogenousKernel, SourceToTargetTranslation},
        tree::{SingleFmmTree, SingleTree},
        types::FmmError,
    },
    tree::constants::NTRANSFER_VECTORS_KIFMM,
    BlasFieldTranslationSaRcmp, SingleFmm,
};

impl<Scalar, Kernel> SourceToTargetTranslation
    for KiFmmMulti<Scalar, Kernel, BlasFieldTranslationSaRcmp<Scalar>>
where
    Scalar: RlstScalar + Default + Equivalence + Float,
    <Scalar as RlstScalar>::Real: Default + Equivalence + Float,
    Kernel: KernelTrait<T = Scalar> + HomogenousKernel + Default + Send + Sync,
{
    fn m2l(&self, level: u64) -> Result<(), FmmError> {
        Ok(())
    }

    fn p2l(&self, level: u64) -> Result<(), FmmError> {
        Ok(())
    }
}
