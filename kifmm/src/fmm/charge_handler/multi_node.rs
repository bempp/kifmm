//! Charge handling in multi-node setting
use green_kernels::traits::Kernel as KernelTrait;
use itertools::izip;
use mpi::{
    datatype::{Partition, PartitionMut},
    traits::Equivalence,
};
use num::Float;
use rlst::RlstScalar;

use crate::{
    fmm::types::FmmEvalType,
    traits::{
        field::FieldTranslation as FieldTranslationTrait,
        fmm::{ChargeHandler, HomogenousKernel},
        general::multi_node::GhostExchange,
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
    Self: GhostExchange,
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

        for m in self.ghost_fmm_u.multipoles.iter_mut() {
            *m = Scalar::zero();
        }

        for l in self.ghost_fmm_u.locals.iter_mut() {
            *l = Scalar::zero();
        }

        for p in self.ghost_fmm_u.potentials.iter_mut() {
            *p = Scalar::zero();
        }

        for c in self.ghost_fmm_u.charges.iter_mut() {
            *c = Scalar::zero();
        }

        for m in self.ghost_fmm_v.multipoles.iter_mut() {
            *m = Scalar::zero();
        }

        for l in self.ghost_fmm_v.locals.iter_mut() {
            *l = Scalar::zero();
        }

        for p in self.ghost_fmm_v.potentials.iter_mut() {
            *p = Scalar::zero();
        }

        for c in self.ghost_fmm_v.charges.iter_mut() {
            *c = Scalar::zero();
        }

        for m in self.global_fmm.multipoles.iter_mut() {
            *m = Scalar::zero();
        }

        for l in self.global_fmm.locals.iter_mut() {
            *l = Scalar::zero();
        }

        for p in self.global_fmm.potentials.iter_mut() {
            *p = Scalar::zero();
        }

        for c in self.global_fmm.charges.iter_mut() {
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
        let n_source_points_input = charges.len() / n_matvecs;

        if n_source_points == n_source_points_input {
            self.clear().unwrap();
            self.charges = charges.to_vec();
            // Setup data dependencies for U list
            self.u_list_exchange();
            Ok(())
        } else {
            Err(FmmError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "Expected {} charges at rank {}, found {}",
                    n_source_points, self.rank, n_source_points_input
                ),
            )))
        }
    }

    fn attach_charges_unordered(
        &mut self,
        charges: &[<Self as ChargeHandler>::Scalar],
    ) -> Result<(), FmmError> {
        if (charges.len() as u64) == (self.local_count_charges) {
            self.clear().unwrap();

            let mut available_queries = Vec::new();

            for (&count, &displacement) in izip!(
                &self.ghost_received_queries_charge_counts,
                &self.ghost_received_queries_charge_displacements
            ) {
                let l = displacement as usize;
                let r = l + (count as usize);

                // Received queries from this rank
                let received_queries_rank = &self.ghost_received_queries_charge[l..r];

                let mut available_queries_rank = Vec::new();

                for &query in received_queries_rank {
                    available_queries_rank
                        .push(charges[(query - self.local_displacement_charges) as usize]);
                }

                available_queries.extend(available_queries_rank);
            }

            let mut requested_queries = vec![Scalar::default(); self.charges.len()];
            // Communicate ghost charges
            {
                let partition_send = Partition::new(
                    &available_queries,
                    &self.charge_send_queries_counts[..],
                    &self.charge_send_queries_displacements[..],
                );

                let mut partition_receive = PartitionMut::new(
                    &mut requested_queries,
                    &self.charge_receive_queries_counts[..],
                    &self.charge_receive_queries_displacements[..],
                );

                self.neighbourhood_communicator_charge
                    .all_to_all_varcount_into(&partition_send, &mut partition_receive);
            }

            self.charges = requested_queries;

            // Setup data dependencies for U list
            self.u_list_exchange();

            Ok(())
        } else {
            Err(FmmError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "Expected {} charges at rank {}, found {}",
                    self.local_count_charges,
                    self.rank,
                    charges.len()
                ),
            )))
        }
    }
}
