//! Multipole to local field translation trait implementation using FFT.
use std::{collections::HashSet, sync::RwLock};

use itertools::Itertools;
use num::{One, Zero};

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
        general::AsComplex,
        tree::{FmmTree, Tree},
        types::FmmError,
    },
    tree::{
        constants::{NHALO, NSIBLINGS},
        types::MortonKey,
    },
    FftFieldTranslation, Fmm,
};

use super::gemv::Gemv8x8;

// impl<Scalar, Kernel> KiFmm<Scalar, Kernel, FftFieldTranslation<Scalar>>
// where
//     Scalar: RlstScalar
//         + AsComplex
//         + Dft<InputType = Scalar, OutputType = <Scalar as AsComplex>::ComplexType>
//         + Default,
//     <Scalar as AsComplex>::ComplexType: Gemv8x8<Scalar = <Scalar as AsComplex>::ComplexType>,
//     Kernel: KernelTrait<T = Scalar> + HomogenousKernel + Default + Send + Sync,
//     <Scalar as RlstScalar>::Real: Default,
//     Self: FmmOperatorData,
// {
//     /// Map between each transfer vector, at the level of a cluster (eight siblings together), of source cluster
//     /// to target cluster.
//     ///
//     /// Returns a vector of read-write locked index vectors, of length 26 - i.e. the number of unique halo positions
//     /// between a target cluster and its source clusters for homogenous kernels. Each element consists of a vector
//     /// of indices containing the index of the source cluster multipole coefficients being translated for each target
//     /// cluster index respectively.
//     fn displacements(&self, level: u64) -> Vec<RwLock<Vec<usize>>> {
//         let targets = self.tree().target_tree().keys(level).unwrap();

//         let targets_parents: HashSet<MortonKey<_>> =
//             targets.iter().map(|target| target.parent()).collect();
//         let mut targets_parents = targets_parents.into_iter().collect_vec();
//         targets_parents.sort();
//         let ntargets_parents = targets_parents.len();

//         let sources = self.tree().source_tree().keys(level).unwrap();

//         let sources_parents: HashSet<MortonKey<_>> =
//             sources.iter().map(|source| source.parent()).collect();
//         let mut sources_parents = sources_parents.into_iter().collect_vec();
//         sources_parents.sort();
//         let nsources_parents = sources_parents.len();

//         let result = vec![Vec::new(); NHALO];
//         let result = result.into_iter().map(RwLock::new).collect_vec();

//         let targets_parents_neighbors = targets_parents
//             .iter()
//             .map(|parent| parent.all_neighbors())
//             .collect_vec();

//         let zero_displacement = nsources_parents * NSIBLINGS;

//         (0..NHALO).into_par_iter().for_each(|i| {
//             let mut result_i = result[i].write().unwrap();
//             for all_neighbors in targets_parents_neighbors.iter().take(ntargets_parents) {
//                 // Check if neighbor exists in a valid tree
//                 if let Some(neighbor) = all_neighbors[i] {
//                     // If it does, check if first child exists in the source tree
//                     let first_child = neighbor.first_child();
//                     if let Some(neighbor_displacement) =
//                         self.level_index_pointer_multipoles[level as usize].get(&first_child)
//                     {
//                         result_i.push(*neighbor_displacement)
//                     } else {
//                         result_i.push(zero_displacement)
//                     }
//                 } else {
//                     result_i.push(zero_displacement)
//                 }
//             }
//         });

//         result
//     }
// }

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
                let c2e_operator_index = self.c2e_operator_index(level);
                let displacement_index = self.displacement_index(level);

                // Number of target and source boxes at this level
                let ntargets = targets.len();
                let nsources = sources.len();

                // Find parents of targets
                let targets_parents: HashSet<MortonKey<_>> =
                    targets.iter().map(|target| target.parent()).collect();

                let targets_parents = targets_parents.into_iter().collect_vec();
                let ntargets_parents = targets_parents.len();

                let sources_parents: HashSet<MortonKey<_>> =
                    sources.iter().map(|source| source.parent()).collect();
                let nsources_parents = sources_parents.len();

                // Size of input FFT sequence
                let shape_in = <Scalar as Dft>::shape_in(self.expansion_order);
                let size_in: usize = <Scalar as Dft>::size_in(self.expansion_order);

                // Size of transformed FFT sequence
                let size_out = <Scalar as Dft>::size_out(self.expansion_order);

                // Calculate displacements of multipole data with respect to source tree
                let all_displacements = &self.source_to_target.displacements[displacement_index];

                // Lookup multipole data from source tree
                let min = &sources[0];
                let max = &sources[nsources - 1];
                let min_idx = self.tree().source_tree().index(min).unwrap();
                let max_idx = self.tree().source_tree().index(max).unwrap();
                let multipoles =
                    &self.multipoles[min_idx * self.ncoeffs..(max_idx + 1) * self.ncoeffs];

                // Buffer to store FFT of multipole data in frequency order
                let nzeros = 8; // pad amount
                let mut signals_hat_f_buffer =
                    vec![Scalar::Real::zero(); size_out * (nsources + nzeros) * 2];
                let signals_hat_f: &mut [<Scalar as AsComplex>::ComplexType];
                unsafe {
                    let ptr = signals_hat_f_buffer.as_mut_ptr()
                        as *mut <Scalar as AsComplex>::ComplexType;
                    signals_hat_f =
                        std::slice::from_raw_parts_mut(ptr, size_out * (nsources + nzeros));
                }

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
                    max_chunk_size = 128
                }
                let chunk_size_pre_proc = chunk_size(nsources_parents, max_chunk_size);
                let chunk_size_kernel = chunk_size(ntargets_parents, max_chunk_size);

                let mut check_potentials_hat_f_buffer =
                    vec![Scalar::Real::zero(); 2 * size_out * ntargets];
                let check_potentials_hat_f: &mut [<Scalar as AsComplex>::ComplexType];
                unsafe {
                    let ptr = check_potentials_hat_f_buffer.as_mut_ptr()
                        as *mut <Scalar as AsComplex>::ComplexType;
                    check_potentials_hat_f =
                        std::slice::from_raw_parts_mut(ptr, size_out * ntargets);
                }

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
                let mut check_potential_hat = vec![Scalar::Real::zero(); size_out * ntargets * 2];

                // Allocate buffer to store the check potentials in box order
                let mut check_potential = vec![Scalar::zero(); size_in * ntargets];
                let check_potential_hat_c;
                unsafe {
                    let ptr =
                        check_potential_hat.as_mut_ptr() as *mut <Scalar as AsComplex>::ComplexType;
                    check_potential_hat_c = std::slice::from_raw_parts_mut(ptr, size_out * ntargets)
                }

                // 1. Compute FFT of all multipoles in source boxes at this level
                {
                    multipoles
                        .par_chunks_exact(self.ncoeffs * NSIBLINGS * chunk_size_pre_proc)
                        .enumerate()
                        .for_each(|(i, multipole_chunk)| {
                            // Place Signal on convolution grid
                            let mut signal_chunk =
                                vec![Scalar::zero(); size_in * NSIBLINGS * chunk_size_pre_proc];

                            for i in 0..NSIBLINGS * chunk_size_pre_proc {
                                let multipole =
                                    &multipole_chunk[i * self.ncoeffs..(i + 1) * self.ncoeffs];
                                let signal = &mut signal_chunk[i * size_in..(i + 1) * size_in];
                                for (surf_idx, &conv_idx) in
                                    self.source_to_target.surf_to_conv_map.iter().enumerate()
                                {
                                    signal[conv_idx] = multipole[surf_idx]
                                }
                            }

                            // Temporary buffer to hold results of FFT
                            let signal_hat_chunk_buffer =
                                vec![
                                    Scalar::Real::zero();
                                    size_out * NSIBLINGS * chunk_size_pre_proc * 2
                                ];
                            let signal_hat_chunk_c;
                            unsafe {
                                let ptr = signal_hat_chunk_buffer.as_ptr()
                                    as *mut <Scalar as AsComplex>::ComplexType;
                                signal_hat_chunk_c = std::slice::from_raw_parts_mut(
                                    ptr,
                                    size_out * NSIBLINGS * chunk_size_pre_proc,
                                );
                            }

                            let _ = Scalar::forward_dft_batch(
                                &mut signal_chunk,
                                signal_hat_chunk_c,
                                &shape_in,
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
                                    let frequency_offset = i * (nsources + nzeros);

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
                        .zip(signals_hat_f.par_chunks_exact(nsources + nzeros))
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
                                                as *const [<Scalar as AsComplex>::ComplexType; 64])
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
                    check_potential_hat_c
                        .par_chunks_exact_mut(size_out)
                        .enumerate()
                        .for_each(|(i, check_potential_hat_chunk)| {
                            // Lookup all frequencies for this target box
                            for j in 0..size_out {
                                check_potential_hat_chunk[j] =
                                    check_potentials_hat_f[j * ntargets + i]
                            }
                        });

                    // Compute inverse FFT
                    let _ = Scalar::backward_dft_batch_par(
                        check_potential_hat_c,
                        &mut check_potential,
                        &shape_in,
                    );

                    check_potential
                        .par_chunks_exact(NSIBLINGS * size_in)
                        .zip(self.level_locals[level as usize].par_chunks_exact(NSIBLINGS))
                        .for_each(|(check_potential_chunk, local_ptrs)| {
                            // Map to surface grid
                            let mut potential_chunk =
                                rlst_dynamic_array2!(Scalar, [self.ncoeffs, NSIBLINGS]);

                            for i in 0..NSIBLINGS {
                                for (surf_idx, &conv_idx) in
                                    self.source_to_target.conv_to_surf_map.iter().enumerate()
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
                                .chunks_exact(self.ncoeffs)
                                .zip(local_ptrs)
                                .for_each(|(result, local)| {
                                    let local = unsafe {
                                        std::slice::from_raw_parts_mut(local[0].raw, self.ncoeffs)
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
