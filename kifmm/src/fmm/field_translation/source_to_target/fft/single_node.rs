//! Multipole to local field translation trait implementation using FFT.
use std::collections::HashSet;

use itertools::Itertools;
use num::{One, Zero};

use rayon::prelude::*;
use rlst::{
    empty_array, rlst_dynamic_array2, MultIntoResize, RandomAccessMut, RawAccess, RlstScalar,
};

use green_kernels::traits::Kernel as KernelTrait;

use crate::{
    fftw::array::{AlignedAllocable, AlignedVec},
    fmm::{
        helpers::single_node::{chunk_size, homogenous_kernel_scale, m2l_scale},
        types::{FmmEvalType, SendPtrMut},
        KiFmm,
    },
    traits::{
        fftw::Dft,
        field::SourceToTargetTranslation,
        fmm::{DataAccess, HomogenousKernel, MetadataAccess},
        general::single_node::{AsComplex, Hadamard8x8},
        tree::{SingleFmmTree, SingleTree},
        types::FmmError,
    },
    tree::{
        constants::{NHALO, NSIBLINGS, NSIBLINGS_SQUARED},
        types::MortonKey,
    },
    FftFieldTranslation, SingleNodeFmmTree,
};

impl<Scalar, Kernel> SourceToTargetTranslation
    for KiFmm<Scalar, Kernel, FftFieldTranslation<Scalar>>
where
    Scalar: RlstScalar
        + AsComplex
        + Dft<InputType = Scalar, OutputType = <Scalar as AsComplex>::ComplexType>
        + Default
        + AlignedAllocable,
    <Scalar as AsComplex>::ComplexType:
        Hadamard8x8<Scalar = <Scalar as AsComplex>::ComplexType> + AlignedAllocable,
    Kernel: KernelTrait<T = Scalar> + HomogenousKernel + Default + Send + Sync,
    <Scalar as RlstScalar>::Real: Default,
    Self: MetadataAccess
        + DataAccess<Scalar = Scalar, Kernel = Kernel, Tree = SingleNodeFmmTree<Scalar::Real>>,
    <Scalar as Dft>::Plan: Sync,
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
                let n_coeffs_equivalent_surface = self.n_coeffs_equivalent_surface(level);

                // Number of target and source boxes at this level
                let n_targets = targets.len();
                let n_sources = sources.len();

                // Find parents of targets
                let targets_parents: HashSet<MortonKey<_>> =
                    targets.iter().map(|target| target.parent()).collect();

                let targets_parents = targets_parents.into_iter().collect_vec();
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

                // Buffer to store FFT of multipole data in frequency order
                let nzeros = 8; // pad amount
                let mut signals_hat_f: AlignedVec<<Scalar as AsComplex>::ComplexType> =
                    AlignedVec::new(size_out * (n_sources + nzeros));

                // A thread safe mutable pointer for saving to this vector
                let signals_hat_f_ptr = SendPtrMut {
                    raw: signals_hat_f.as_mut_ptr(),
                };

                // Pre processing chunk size, in terms of number of source parents
                let max_chunk_size;
                if level == 2 {
                    max_chunk_size = 8
                } else if level == 3 {
                    max_chunk_size = 64
                } else {
                    max_chunk_size = self.source_to_target.block_size
                }

                let chunk_size_pre_proc = chunk_size(nsources_parents, max_chunk_size);
                let chunk_size_kernel = chunk_size(ntargets_parents, max_chunk_size);

                // Amount to scale the application of the kernel by
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
                let mut check_potentials_hat_f = AlignedVec::new(size_out * n_targets);

                // Allocate buffer to store the check potentials in box order
                let mut check_potential_hat_c = AlignedVec::new(size_out * n_targets);
                let mut check_potential = AlignedVec::new(size_in * n_targets);

                // 1. Compute FFT of all multipoles in source boxes at this level
                {
                    let mut in_ = AlignedVec::new(size_in);
                    let mut out = AlignedVec::new(size_out);
                    let plan = Scalar::plan_forward(&mut in_, &mut out, &shape_in, None).unwrap();

                    multipoles
                        .par_chunks_exact(
                            n_coeffs_equivalent_surface * NSIBLINGS * chunk_size_pre_proc,
                        )
                        .enumerate()
                        .for_each(|(i, multipole_chunk)| {
                            // Place Signal on convolution grid
                            let mut signal_chunk =
                                AlignedVec::new(size_in * NSIBLINGS * chunk_size_pre_proc);

                            for i in 0..NSIBLINGS * chunk_size_pre_proc {
                                let multipole = &multipole_chunk[i * n_coeffs_equivalent_surface
                                    ..(i + 1) * n_coeffs_equivalent_surface];
                                let signal = &mut signal_chunk[i * size_in..(i + 1) * size_in];
                                for (surf_idx, &conv_idx) in self.source_to_target.surf_to_conv_map
                                    [fft_map_index]
                                    .iter()
                                    .enumerate()
                                {
                                    signal[conv_idx] = multipole[surf_idx]
                                }
                            }

                            // Temporary buffer to hold results of FFT
                            let mut signal_hat_chunk_c =
                                AlignedVec::new(size_out * NSIBLINGS * chunk_size_pre_proc);

                            let _ = Scalar::forward_dft_batch(
                                &mut signal_chunk,
                                &mut signal_hat_chunk_c,
                                &shape_in,
                                &plan,
                            );

                            // Re-order the temporary buffer into frequency order before flushing to main memory
                            let signal_hat_chunk_f_buffer =
                                vec![
                                    Scalar::Real::zero();
                                    size_out * NSIBLINGS * chunk_size_pre_proc * 2
                                ];
                            let signal_hat_chunk_f_c;
                            unsafe {
                                let ptr = signal_hat_chunk_f_buffer.as_ptr()
                                    as *mut <Scalar as AsComplex>::ComplexType;
                                signal_hat_chunk_f_c = std::slice::from_raw_parts_mut(
                                    ptr,
                                    size_out * NSIBLINGS * chunk_size_pre_proc,
                                );
                            }

                            for i in 0..size_out {
                                for j in 0..NSIBLINGS * chunk_size_pre_proc {
                                    signal_hat_chunk_f_c[NSIBLINGS * chunk_size_pre_proc * i + j] =
                                        signal_hat_chunk_c[size_out * j + i]
                                }
                            }

                            // Storing the results of the FFT in frequency order
                            unsafe {
                                let sibling_offset = i * NSIBLINGS * chunk_size_pre_proc;

                                // Pointer to storage buffer for frequency ordered FFT of signals
                                let ptr = signals_hat_f_ptr;

                                for i in 0..size_out {
                                    let frequency_offset = i * (n_sources + nzeros);

                                    // Head of buffer for each frequency
                                    let head = ptr.raw.add(frequency_offset).add(sibling_offset);

                                    let signal_hat_f_chunk = std::slice::from_raw_parts_mut(
                                        head,
                                        NSIBLINGS * chunk_size_pre_proc,
                                    );

                                    // Store results for this frequency for this sibling set chunk
                                    let results_i =
                                        &signal_hat_chunk_f_c[i * NSIBLINGS * chunk_size_pre_proc
                                            ..(i + 1) * NSIBLINGS * chunk_size_pre_proc];

                                    signal_hat_f_chunk
                                        .iter_mut()
                                        .zip(results_i)
                                        .for_each(|(c, r)| *c += *r);
                                }
                            }
                        });
                }

                // 2. Compute the Hadamard product
                {
                    (0..size_out)
                        .into_par_iter()
                        .zip(signals_hat_f.par_chunks_exact(n_sources + nzeros))
                        .zip(check_potentials_hat_f.par_chunks_exact_mut(n_targets))
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
                                                as *const [<Scalar as AsComplex>::ComplexType;
                                                    NSIBLINGS_SQUARED])
                                        };

                                        // Lookup signals
                                        let displacements = &all_displacements[i].read().unwrap()
                                            [chunk_start..chunk_end];

                                        for j in 0..(chunk_end - chunk_start) {
                                            let displacement = displacements[j];
                                            let s_f = &signal_hat_f
                                                [displacement..displacement + NSIBLINGS];
                                            let s_f_slice = unsafe {
                                                &*(s_f.as_ptr()
                                                    as *const [<Scalar as AsComplex>::ComplexType;
                                                        NSIBLINGS])
                                            };

                                            let save_locations = &mut save_locations
                                                [j * NSIBLINGS..(j + 1) * NSIBLINGS];
                                            let save_locations_slice = unsafe {
                                                &mut *(save_locations.as_ptr()
                                                    as *mut [<Scalar as AsComplex>::ComplexType;
                                                        NSIBLINGS])
                                            };

                                            <Scalar as AsComplex>::ComplexType::hadamard8x8(
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
                    check_potential_hat_c
                        .par_chunks_exact_mut(size_out)
                        .enumerate()
                        .for_each(|(i, check_potential_hat_chunk)| {
                            // Lookup all frequencies for this target box
                            for j in 0..size_out {
                                check_potential_hat_chunk[j] =
                                    check_potentials_hat_f[j * n_targets + i]
                            }
                        });

                    // Compute inverse FFT
                    let mut out = AlignedVec::new(size_in);
                    let mut in_ = AlignedVec::new(size_out);
                    let plan = Scalar::plan_backward(&mut in_, &mut out, &shape_in, None).unwrap();

                    let _ = Scalar::backward_dft_batch_par(
                        &mut check_potential_hat_c,
                        &mut check_potential,
                        &shape_in,
                        &plan,
                    );

                    check_potential
                        .par_chunks_exact(NSIBLINGS * size_in)
                        .zip(self.level_locals[level as usize].par_chunks_exact(NSIBLINGS))
                        .for_each(|(check_potential_chunk, local_ptrs)| {
                            // Map to surface grid
                            let mut potential_chunk = rlst_dynamic_array2!(
                                Scalar,
                                [n_coeffs_equivalent_surface, NSIBLINGS]
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
                                .chunks_exact(n_coeffs_equivalent_surface)
                                .zip(local_ptrs)
                                .for_each(|(result, local)| {
                                    let local = unsafe {
                                        std::slice::from_raw_parts_mut(
                                            local[0].raw,
                                            n_coeffs_equivalent_surface,
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
