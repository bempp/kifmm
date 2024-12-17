use green_kernels::traits::Kernel as KernelTrait;
use mpi::traits::Equivalence;
use num::Float;
use rlst::RlstScalar;

use crate::{
    fmm::types::FmmEvalType,
    traits::{
        field::FieldTranslation as FieldTranslationTrait,
        fmm::{ChargeHandler, HomogenousKernel},
        tree::{MultiFmmTree, MultiTree},
        types::FmmError,
    },
    KiFmmMulti,
};

impl<Scalar, Kernel, FieldTranslation> ChargeHandler
    for KiFmmMulti<Scalar, Kernel, FieldTranslation>
where
    Scalar: RlstScalar + Default + Equivalence + Float,
    <Scalar as RlstScalar>::Real: Default + Equivalence + Float,
    Kernel: KernelTrait<T = Scalar> + HomogenousKernel + Default + Send + Sync,
    FieldTranslation: FieldTranslationTrait + Send + Sync,
{
    type Scalar = Scalar;

    fn clear(&mut self) -> Result<(), FmmError> {
        for m in self.multipoles.iter_mut() {
            *m = Scalar::zero();
        }

        for l in self.locals.iter_mut() {
            *l = Scalar::zero();
        }

        for p in self.potentials.iter_mut() {
            *p = Scalar::zero();
        }

        for c in self.charges.iter_mut() {
            *c = Scalar::zero();
        }

        Ok(())
    }

    fn attach_charges_ordered(
        &mut self,
        charges: &[<Self as ChargeHandler>::Scalar],
    ) -> Result<(), FmmError> {
        let n_matvecs = match self.fmm_eval_type {
            FmmEvalType::Vector => 1,
            FmmEvalType::Matrix(n) => n,
        };

        let n_source_points = self.tree.source_tree().n_coordinates_tot().unwrap();
        let n_matvecs_input = charges.len() / n_source_points;

        if n_matvecs == n_matvecs_input {
            self.charges = charges.to_vec();
            Ok(())
        } else {
            Err(FmmError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "Expected {} right hand side vectors, found {}",
                    n_matvecs, n_matvecs_input
                ),
            )))
        }
    }

    fn attach_charges_unordered(
        &mut self,
        _charges: &[<Self as ChargeHandler>::Scalar],
    ) -> Result<(), FmmError> {
        Err(FmmError::Unimplemented(
            "Attaching unordered charges currently unimplemented.".to_string(),
        ))
    }
}
