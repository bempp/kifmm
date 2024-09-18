//! Multipole to Multipole field translation

use std::sync::Mutex;

use itertools::Itertools;
use mpi::traits::Equivalence;
use num::Float;
use rayon::prelude::*;
use rlst::{
    empty_array, rlst_array_from_slice2, rlst_dynamic_array2, MultIntoResize, RawAccess,
    RawAccessMut, RlstScalar,
};

use green_kernels::{traits::Kernel as KernelTrait, types::EvalType};

use crate::{
    fmm::{
        constants::P2M_MAX_BLOCK_SIZE,
        helpers::single_node::chunk_size,
        types::{FmmEvalType, KiFmmMulti},
    },
    traits::{
        field::SourceToTargetData as SourceToTargetDataTrait,
        fmm::{HomogenousKernel, SourceTranslation},
        tree::{MultiFmmTree, MultiTree},
    },
};

impl<Scalar, Kernel, SourceToTargetData> SourceTranslation
    for KiFmmMulti<Scalar, Kernel, SourceToTargetData>
where
    Scalar: RlstScalar + Default + Equivalence + Float,
    <Scalar as RlstScalar>::Real: Default + Equivalence + Float,
    SourceToTargetData: SourceToTargetDataTrait,
    Kernel: KernelTrait<T = Scalar> + HomogenousKernel + Default + Send + Sync,
{
    fn p2m(&self) -> Result<(), crate::traits::types::FmmError> {
        if let Some(leaves) = self.tree.source_tree().all_leaves() {
            let n_coeffs_equivalent_surface = self.n_coeffs_equivalent_surface;
            let n_coeffs_check_surface = self.n_coeffs_check_surface;
            let n_leaves = self.tree.source_tree().n_leaves().unwrap();
            let check_surface_size = n_coeffs_check_surface * self.dim;

            let coordinates = self.tree.source_tree().all_coordinates().unwrap();
            let n_coordinates = self.tree.source_tree().n_coordinates_tot().unwrap();

            let all_charges = &self.charges;
            let kernel = &self.kernel;
            let dim = self.dim;

            match self.fmm_eval_type {
                FmmEvalType::Vector => {
                    let mut check_potentials =
                        rlst_dynamic_array2!(Scalar, [n_leaves * n_coeffs_check_surface, 1]);

                    // Compute check potential for each box
                    check_potentials
                        .data_mut()
                        .par_chunks_exact_mut(n_coeffs_check_surface)
                        .zip(
                            self.leaf_upward_check_surfaces_sources
                                .par_chunks_exact(check_surface_size),
                        )
                        .zip(&self.charge_index_pointer_sources)
                        .for_each(
                            |((check_potential, upward_check_surface), charge_index_pointer)| {
                                let charges =
                                    &all_charges[charge_index_pointer.0..charge_index_pointer.1];

                                let coordinates_row_major = &coordinates
                                    [charge_index_pointer.0 * dim..charge_index_pointer.1 * dim];
                                let nsources = coordinates_row_major.len() / dim;

                                if nsources > 0 {
                                    kernel.evaluate_st(
                                        EvalType::Value,
                                        coordinates_row_major,
                                        upward_check_surface,
                                        charges,
                                        check_potential,
                                    );
                                }
                            },
                        );

                    let chunk_size = chunk_size(n_leaves, P2M_MAX_BLOCK_SIZE);
                }

                FmmEvalType::Matrix(_n) => {
                    panic!("unimplemented for matrix input")
                }
            }
        }

        Ok(())
    }

    fn m2m(&self, level: u64) -> Result<(), crate::traits::types::FmmError> {
        Ok(())
    }
}
