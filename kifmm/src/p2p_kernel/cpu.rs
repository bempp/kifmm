//! CPU based P2P kernel using parallel multithreading
use rlst::RlstScalar;
use green_kernels::traits::Kernel as KernelTrait;

use crate::{traits::{field::FieldTranslation as FieldTranslationTrait, fmm::{HomogenousKernel, MetadataAccess}, p2p_kernel::P2PKernel}, DataAccess, KiFmm};


impl <Scalar, Kernel, FieldTranslation> P2PKernel for KiFmm<Scalar, Kernel, FieldTranslation>
where
    Scalar: RlstScalar,
    Kernel: KernelTrait<T = Scalar> + HomogenousKernel + Send + Sync,
    FieldTranslation: FieldTranslationTrait + Send + Sync,
    <Scalar as RlstScalar>::Real: Default,
    Self: MetadataAccess + DataAccess<Scalar = Scalar, Kernel = Kernel>,
{
    type Scalar = Scalar;

    fn p2p_kernel(
            targets: &[<Self::Scalar as rlst::RlstScalar>::Real],
            targets_counts: &[usize],
            targets_displacements: &[usize],
            sources: &[<Self::Scalar as rlst::RlstScalar>::Real],
            sources_counts: &[usize],
            sources_displacements: &[usize],
            charges: &[Self::Scalar],
            result: &mut [Self::Scalar]
        ) {

            

    }
}
