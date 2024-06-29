//! Multipole to local field translation trait implementation using FFT.
use std::{collections::HashSet, sync::RwLock, time::Instant};

use itertools::Itertools;
use num::{One, Zero};

use crate::{fftw::{array::{AlignedAllocable, AlignedVec}, types::BatchSize}, fmm::types::SendPtr, tree::constants::NSIBLINGS_SQUARED};
use pulp::Scalar;
use rayon::prelude::*;
use rlst::{
    empty_array, rlst_dynamic_array2, MultIntoResize, RandomAccessMut, RawAccess, RlstScalar,
};

use green_kernels::traits::Kernel as KernelTrait;

use crate::{
    fmm::{
        helpers::{chunk_size, homogenous_kernel_scale, m2l_scale},
        types::{FmmEvalType, SendPtrMut},
        KiFmm,
    },
    traits::{
        fftw::Dft,
        fmm::{FmmOperatorData, HomogenousKernel, SourceToTargetTranslation},
        general::{AsComplex, Gemv8x8},
        tree::{FmmTree, Tree},
        types::FmmError,
    },
    tree::{
        constants::{NHALO, NSIBLINGS},
        types::MortonKey,
    },
    FftFieldTranslation, Fmm,
};

impl<Scalar, Kernel> SourceToTargetTranslation
    for KiFmm<Scalar, Kernel, FftFieldTranslation<Scalar>>
where
    Scalar: RlstScalar
        + AsComplex
        + Dft<InputType = Scalar, OutputType = <Scalar as AsComplex>::ComplexType>
        + Default,
    <Scalar as AsComplex>::ComplexType: Gemv8x8<Scalar = <Scalar as AsComplex>::ComplexType>,
    Kernel: KernelTrait<T = Scalar> + HomogenousKernel + Default + Send + Sync,
    <Scalar as RlstScalar>::Real: Default,
    Self: FmmOperatorData,
    <Scalar as Dft>::Plan: Send + Sync,
    <Scalar as AsComplex>::ComplexType: AlignedAllocable,
    Scalar: AlignedAllocable
{
    fn m2l(&self, level: u64) -> Result<(), FmmError> {
        match self.fmm_eval_type {
            FmmEvalType::Vector => {
                let Some(targets) = self.tree().target_tree().keys(level) else {
                    return Err(FmmError::Failed(format!(
                        "M2L failed at level {:?}, no targets found",
                        level
                    )));
                };
                let Some(sources) = self.tree().source_tree().keys(level) else {
                    return Err(FmmError::Failed(format!(
                        "M2L failed at level {:?}, no sources found",
                        level
                    )));
                };

                let m2l_operator_index = self.m2l_operator_index(level);
                let fft_map_index = self.fft_map_index(level);
                let c2e_operator_index = self.c2e_operator_index(level);
                let displacement_index = self.displacement_index(level);
                let ncoeffs_equivalent_surface = self.ncoeffs_equivalent_surface(level);

                // Number of target and source boxes at this level
                let ntargets = targets.len();
                let nsources = sources.len();

                // Find parents of targets
                let targets_parents: HashSet<MortonKey<_>> =
                    targets.iter().map(|target| target.parent()).collect();

                let mut targets_parents = targets_parents.into_iter().collect_vec();
                targets_parents.sort();
                let ntargets_parents = targets_parents.len();

                let sources_parents: HashSet<MortonKey<_>> =
                    sources.iter().map(|source| source.parent()).collect();
                let nsources_parents = sources_parents.len();

                // Size of input FFT sequence
                let shape_in = <Scalar as Dft>::shape_in(self.equivalent_surface_order(level));
                let size_in: usize = <Scalar as Dft>::size_in(self.equivalent_surface_order(level));

                // Size of transformed FFT sequence
                let size_out = <Scalar as Dft>::size_out(self.equivalent_surface_order(level));

                // Calculate displacements of multipole data with respect to source tree
                let all_displacements = &self.source_to_target.displacements[displacement_index];

                // Lookup multipole data from source tree
                let multipoles = self.multipoles(level).unwrap();

                let nzeros = 8; // pad amount

                // Buffer to store FFT of multipole data in frequency order
                let mut signals_hat_f = AlignedVec::new(size_out * nsources);

                // Pre processing chunk size, in terms of number of source parents
                let max_chunk_size;
                if level == 2 {
                    max_chunk_size = 8
                } else if level == 3 {
                    max_chunk_size = 64
                } else {
                    max_chunk_size = self.source_to_target.block_size
                }

                let chunk_size_kernel = chunk_size(ntargets_parents, max_chunk_size);
                let mut check_potentials_hat_f = AlignedVec::new(size_out * ntargets);

                // // Amount to scale the application of the kernel by
                let scale = if self.kernel.is_homogenous() {
                    m2l_scale::<<Scalar as AsComplex>::ComplexType>(level).unwrap()
                        * homogenous_kernel_scale(level)
                } else {
                    <<Scalar as AsComplex>::ComplexType>::one()
                };

                // Lookup all of the precomputed Green's function evaluations' FFT sequences
                let kernel_data_ft =
                    &self.source_to_target.metadata[m2l_operator_index].kernel_data_f;

                // Allocate buffer to store the check potentials in frequency order
                let mut check_potential_hat_c = AlignedVec::new(size_out * ntargets);

                let mut in_ = AlignedVec::new(size_in);
                let mut out = AlignedVec::new(size_out);

                let plan = Scalar::plan_forward(
                    &mut in_,
                    &mut out,
                    &shape_in,
                    None
                ).unwrap();

                let s = Instant::now();
                {
                    multipoles
                        .par_chunks_exact(
                            ncoeffs_equivalent_surface * NSIBLINGS,
                        )
                        .zip(signals_hat_f.par_chunks_exact_mut(NSIBLINGS * size_out))
                        .enumerate()
                        .for_each(|(i, (multipole_chunk, signals_hat_f_chunk))| {
                            // let mut signal_chunk =
                            // vec![Scalar::zero(); size_in * NSIBLINGS];
                            let mut signal_chunk = AlignedVec::new(size_in * NSIBLINGS);

                        for i in 0..NSIBLINGS {
                            let multipole = &multipole_chunk[i * ncoeffs_equivalent_surface
                                ..(i + 1) * ncoeffs_equivalent_surface];
                            let signal = &mut signal_chunk[i * size_in..(i + 1) * size_in];
                            for (surf_idx, &conv_idx) in self.source_to_target.surf_to_conv_map
                                [fft_map_index]
                                .iter()
                                .enumerate()
                            {
                                signal[conv_idx] = multipole[surf_idx]
                            }
                        }

                        // // Temporary buffer to hold results of FFT
                        let mut signal_hat_chunk_c = AlignedVec::new(size_out * NSIBLINGS);

                        let _ = Scalar::forward_dft_batch(
                            &mut signal_chunk,
                            &mut signal_hat_chunk_c,
                            &shape_in,
                            &plan
                        );

                        for i in 0..size_out {
                            for j in 0..NSIBLINGS {
                                signals_hat_f_chunk[NSIBLINGS * i + j] =
                                    signal_hat_chunk_c[size_out * j + i]
                            }
                        }

                    });
                }
                println!("M2L OP 1 level  {:?} time {:?}", level, s.elapsed());

                // Form interaction list
                let zeros = vec![<Scalar as AsComplex>::ComplexType::zero(); nzeros];
                let mut signals_hat_f_ptrs = Vec::new();
                // Loop over sibling sets which exist, and single sibling set of zeros
                for freq in 0..size_out {
                    let freq_displacement = freq*NSIBLINGS;
                    let mut tmp = Vec::new();
                    for sibling_index in 0..nsources_parents {
                        let sibling_displacement = size_out * NSIBLINGS * sibling_index;
                        let ptr = unsafe { SendPtr{raw: signals_hat_f.as_ptr().add(sibling_displacement+freq_displacement)} };
                        tmp.push(ptr)
                    }
                    tmp.push(SendPtr{raw: zeros.as_ptr()});
                    signals_hat_f_ptrs.append(&mut tmp);
                }

                // 2. Compute the Hadamard product
                {
                    (0..size_out)
                        .into_par_iter()
                        .zip(signals_hat_f_ptrs.par_chunks_exact(nsources_parents + 1))
                        .zip(check_potentials_hat_f.par_chunks_exact_mut(ntargets))
                        .for_each(|((freq, signal_hat_f), check_potential_hat_f)| {
                            (0..ntargets_parents).step_by(chunk_size_kernel).for_each(
                                |chunk_start| {
                                    let chunk_end = std::cmp::min(
                                        chunk_start + chunk_size_kernel,
                                        ntargets_parents,
                                    );

                                    let save_locations = &mut check_potential_hat_f
                                        [chunk_start * NSIBLINGS..chunk_end * NSIBLINGS];

                                    for i in 0..NHALO {
                                        let frequency_offset = freq * NHALO;
                                        let k_f = &kernel_data_ft[i + frequency_offset];

                                        let k_f_slice = unsafe {
                                            &*(k_f.as_slice().as_ptr()
                                                as *const [<Scalar as AsComplex>::ComplexType; NSIBLINGS_SQUARED])
                                        };

                                        // Lookup signals
                                        let displacements = &all_displacements[i].read().unwrap()[chunk_start..chunk_end];

                                        for j in 0..(chunk_end - chunk_start) {
                                            let displacement = displacements[j];
                                            let s_f = unsafe { std::slice::from_raw_parts(signal_hat_f[displacement / 8].raw, 8) };
                                            let s_f_slice = unsafe {
                                                &*(s_f.as_ptr()
                                                    as *const [<Scalar as AsComplex>::ComplexType;
                                                        8])
                                            };

                                            let save_locations = &mut save_locations
                                                [j * NSIBLINGS..(j + 1) * NSIBLINGS];
                                            let save_locations_slice = unsafe {
                                                &mut *(save_locations.as_ptr()
                                                    as *mut [<Scalar as AsComplex>::ComplexType; 8])
                                            };

                                            <Scalar as AsComplex>::ComplexType::gemv8x8(
                                                self.isa,
                                                k_f_slice,
                                                s_f_slice,
                                                save_locations_slice,
                                                scale,
                                            );
                                        }
                                    }
                                },
                            );
                        });
                }

                // 3. Post process to find local expansions at target boxes
                {
                    let mut out= AlignedVec::new(size_in);
                    let mut in_ = AlignedVec::new(size_out);

                    let plan = Scalar::plan_backward(
                        &mut in_,
                        &mut out,
                        &shape_in,
                        None
                    ).unwrap();

                    check_potential_hat_c
                        .par_chunks_exact_mut(size_out * NSIBLINGS)
                        .zip(check_potentials_hat_f.par_chunks_exact(size_out * NSIBLINGS))
                        .zip(self.level_locals[level as usize].par_chunks_exact(NSIBLINGS))
                        .enumerate()
                        .for_each(|(i, ((check_potential_hat_chunk, check_potentials_hat_f_chunk), local_ptrs))| {

                            // Lookup all frequencies for this target box
                            for j in 0..size_out {
                                for k in 0..NSIBLINGS {
                                    check_potential_hat_chunk[size_out * k + j] = check_potentials_hat_f_chunk[NSIBLINGS * j + k]
                                }
                            }

                            let mut check_potential_chunk = AlignedVec::new(size_in * NSIBLINGS);

                            // Compute inverse FFT
                            let _ = Scalar::backward_dft_batch(
                                check_potential_hat_chunk,
                                &mut check_potential_chunk,
                                &shape_in,
                                &plan
                            );

                            // Map to surface grid
                            let mut potential_chunk = rlst_dynamic_array2!(
                                Scalar,
                                [ncoeffs_equivalent_surface, NSIBLINGS]
                            );

                            for i in 0..NSIBLINGS {
                                for (surf_idx, &conv_idx) in self.source_to_target.conv_to_surf_map
                                    [fft_map_index]
                                    .iter()
                                    .enumerate()
                                {
                                    *potential_chunk.get_mut([surf_idx, i]).unwrap() =
                                        check_potential_chunk[i * size_in + conv_idx];
                                }
                            }

                            // Can now find local expansion coefficients
                            let local_chunk = empty_array::<Scalar, 2>().simple_mult_into_resize(
                                self.dc2e_inv_1[c2e_operator_index].view(),
                                empty_array::<Scalar, 2>().simple_mult_into_resize(
                                    self.dc2e_inv_2[c2e_operator_index].view(),
                                    potential_chunk,
                                ),
                            );

                            local_chunk
                                .data()
                                .chunks_exact(ncoeffs_equivalent_surface)
                                .zip(local_ptrs)
                                .for_each(|(result, local)| {
                                    let local = unsafe {
                                        std::slice::from_raw_parts_mut(
                                            local[0].raw,
                                            ncoeffs_equivalent_surface,
                                        )
                                    };
                                    local.iter_mut().zip(result).for_each(|(l, r)| *l += *r);
                                });

                        });

                }

                Ok(())
            }
            FmmEvalType::Matrix(_nmatvecs) => Err(FmmError::Unimplemented(
                "M2L unimplemented for matrix input with FFT field translations".to_string(),
            )),
        }
    }

    fn p2l(&self, _level: u64) -> Result<(), FmmError> {
        Err(FmmError::Unimplemented("P2L unimplemented".to_string()))
    }
}
