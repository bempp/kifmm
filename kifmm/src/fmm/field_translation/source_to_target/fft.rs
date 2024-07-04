//! Multipole to local field translation trait implementation using FFT.
use std::{
    collections::HashSet,
    sync::{Mutex, RwLock},
    time::Instant,
};

use itertools::Itertools;
use num::{One, Zero};

use crate::{
    fftw::{
        array::{AlignedAllocable, AlignedVec},
        types::BatchSize,
    },
    fmm::types::SendPtr,
    tree::constants::NSIBLINGS_SQUARED,
};
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
    Scalar: AlignedAllocable,
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

                let mut in_ = AlignedVec::new(size_in);
                let mut out = AlignedVec::new(size_out);

                let plan = Scalar::plan_forward(&mut in_, &mut out, &shape_in, None).unwrap();

                let s = Instant::now();

                // 1. Compute FFT of all multipoles in source boxes at this level
                {
                    multipoles
                        .par_chunks_exact(ncoeffs_equivalent_surface * NSIBLINGS)
                        .zip(signals_hat_f.par_chunks_exact_mut(size_out * NSIBLINGS))
                        .enumerate()
                        .for_each(|(i, (multipole_chunk, signal_hat_chunk_f_c))| {
                            // Place Signal on convolution grid

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

                            let mut signal_hat_chunk_c = AlignedVec::new(size_out * NSIBLINGS);

                            let _ = Scalar::forward_dft_batch(
                                &mut signal_chunk,
                                &mut signal_hat_chunk_c,
                                &shape_in,
                                &plan,
                            );

                            // // Re-order the temporary buffer into frequency order before flushing to main memory
                            for i in 0..size_out {
                                for j in 0..NSIBLINGS {
                                    signal_hat_chunk_f_c[NSIBLINGS * i + j] =
                                        signal_hat_chunk_c[size_out * j + i]
                                }
                            }
                        });
                }

                let zeros = vec![<Scalar as AsComplex>::ComplexType::zero(); NSIBLINGS];

                let s = Instant::now();

                // nfreqs long, each of which is block size long
                let mut signals_hat_f_ = Vec::new(); // All signal pointers, in clusters, in frequency order
                let mut check_potentials_hat_f_ = Vec::new();
                // let nchunks = ntargets_parents;
                // I want to have, arranged in chunks of chunk_size, the source/target pointers

                for freq in 0..size_out {
                    let freq_displacement = freq * NSIBLINGS;

                    let mut tmp = Vec::new();
                    for chunk_index in 0..ntargets_parents {
                        let chunk_displacement = chunk_index * NSIBLINGS * size_out; // First, then second etc lot of sibling signal data
                        let check_potential_hat_f_chunk =
                            &mut check_potentials_hat_f[chunk_displacement + freq_displacement];
                        let ptr = SendPtrMut {
                            raw: check_potential_hat_f_chunk
                                as *mut <Scalar as AsComplex>::ComplexType,
                        };
                        tmp.push(ptr)
                    }

                    check_potentials_hat_f_.push(tmp)
                }

                // Need matching pointers for matching signals
                for freq in 0..size_out {
                    let freq_displacement = freq * NSIBLINGS;
                    let mut tmp = Vec::new();

                    for cluster_index in 0..ntargets_parents {
                        // Target clusters in this chunk, defined by parent Morton
                        let target_cluster = targets[cluster_index * NSIBLINGS].parent();

                        let source_clusters = target_cluster.all_neighbors();

                        let mut tmp2 = Vec::new();

                        for source_cluster in source_clusters.iter() {
                            if let Some(source_cluster) = source_cluster {
                                let first_child = source_cluster.first_child();
                                let &first_child_index = self.level_index_pointer_multipoles
                                    [level as usize]
                                    .get(&first_child)
                                    .unwrap();
                                let source_cluster_index = first_child_index / NSIBLINGS;
                                let ptr = SendPtr {
                                    raw: &signals_hat_f[source_cluster_index + freq_displacement],
                                };
                                tmp2.push(ptr);
                            } else {
                                let ptr = SendPtr { raw: &zeros[0] };
                                tmp2.push(ptr);
                            }
                        }

                        tmp.push(tmp2)
                    }

                    signals_hat_f_.push(tmp)
                }

                println!("level {:?} {:?}", level, s.elapsed());

                // frequency [], number of trgets [][] ntargets parents long, number of source clusters for each target 26 long
                // println!(
                //     "check potentials hat {:?} {:?}=={:?} {:?}=={:?} {:?}={:?}",
                //     level,
                //     check_potentials_hat_f_.len(),
                //     size_out,
                //     check_potentials_hat_f_[0].len(),
                //     nchunks,
                //     signals_hat_f_[0][0].len(),
                //     26
                // );
                // assert!(false);

                // 2. Compute the Hadamard product
                (0..size_out)
                    .into_par_iter()
                    .zip(signals_hat_f_.par_iter())
                    .zip(check_potentials_hat_f_.par_iter())
                    .for_each(|((freq, signal_hat_f), check_potential_hat_f)| {
                        let frequency_offset = freq * NHALO;

                        // Iterate over each chunk of target check potential, i.e. over all target clusters
                        check_potential_hat_f
                            .chunks_exact(chunk_size_kernel)
                            .zip(signal_hat_f.chunks_exact(chunk_size_kernel))
                            .enumerate()
                            .for_each(|(i, (c_f_chunk, s_f_chunk))| {

                                for (c_f, s_f) in c_f_chunk.iter().zip(s_f_chunk) {
                                    let c_f_slice = unsafe {
                                        &mut **(&mut {
                                            c_f.raw
                                                as *mut [<Scalar as AsComplex>::ComplexType; NSIBLINGS]
                                        })
                                    };

                                    for halo_index in 0..NHALO {

                                        let s_f_i = &s_f[halo_index];
                                        let s_f_slice = unsafe {
                                            &*(s_f_i.raw
                                                as *const [<Scalar as AsComplex>::ComplexType;
                                                    8])
                                        };

                                        let k_f = &kernel_data_ft[halo_index + frequency_offset];

                                        let k_f_slice = unsafe {
                                            &*(k_f.as_slice().as_ptr()
                                                as *const [<Scalar as AsComplex>::ComplexType; 64])
                                        };

                                        <Scalar as AsComplex>::ComplexType::gemv8x8(
                                            self.isa,
                                            k_f_slice,
                                            s_f_slice,
                                            c_f_slice,
                                            scale,
                                        );

                                    }
                                }

                            });

                    });


                // 3. Post process to find local expansions at target boxes
                {
                    let mut out = AlignedVec::new(size_in);
                    let mut in_ = AlignedVec::new(size_out);

                    let plan = Scalar::plan_backward(&mut in_, &mut out, &shape_in, None).unwrap();

                    check_potential_hat_c
                        .par_chunks_exact_mut(size_out)
                        .zip(check_potential.par_chunks_exact_mut(size_out))
                        .zip(self.level_locals[level as usize].par_iter())
                        .enumerate()
                        .for_each(|(i, ((check_potential_hat_chunk, check_potential_chunk), local_ptrs))| {

                            // Lookup all frequencies for this target box
                            for j in 0..size_out {
                                check_potential_hat_chunk[j] =
                                    check_potentials_hat_f[j * ntargets + i]
                            }

                            let _ = Scalar::backward_dft_batch(
                                check_potential_hat_chunk,
                                check_potential_chunk,
                                &shape_in,
                                &plan,
                            );

                            // Map to surface grid
                            let mut potential_chunk = rlst_dynamic_array2!(
                                Scalar,
                                [ncoeffs_equivalent_surface, 1]
                            );


                            for (surf_idx, &conv_idx) in self.source_to_target.conv_to_surf_map
                                [fft_map_index]
                                .iter()
                                .enumerate()
                            {
                                *potential_chunk.get_mut([surf_idx, 0]).unwrap() =
                                    check_potential_chunk[conv_idx];
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
                                            local.raw,
                                            ncoeffs_equivalent_surface,
                                        )
                                    };

                                    local.iter_mut().zip(result).for_each(|(l, r)| *l += *r);
                                });
                        });

                    // Compute inverse FFT

                    // check_potential
                    //     .par_chunks_exact(NSIBLINGS * size_in)
                    //     .zip(self.level_locals[level as usize].par_chunks_exact(NSIBLINGS))
                    //     .for_each(|(check_potential_chunk, local_ptrs)| {
                    //         // Map to surface grid
                    //         let mut potential_chunk = rlst_dynamic_array2!(
                    //             Scalar,
                    //             [ncoeffs_equivalent_surface, NSIBLINGS]
                    //         );

                    //         for i in 0..NSIBLINGS {
                    //             for (surf_idx, &conv_idx) in self.source_to_target.conv_to_surf_map
                    //                 [fft_map_index]
                    //                 .iter()
                    //                 .enumerate()
                    //             {
                    //                 *potential_chunk.get_mut([surf_idx, i]).unwrap() =
                    //                     check_potential_chunk[i * size_in + conv_idx];
                    //             }
                    //         }

                    //         // Can now find local expansion coefficients
                    //         let local_chunk = empty_array::<Scalar, 2>().simple_mult_into_resize(
                    //             self.dc2e_inv_1[c2e_operator_index].view(),
                    //             empty_array::<Scalar, 2>().simple_mult_into_resize(
                    //                 self.dc2e_inv_2[c2e_operator_index].view(),
                    //                 potential_chunk,
                    //             ),
                    //         );

                    //         local_chunk
                    //             .data()
                    //             .chunks_exact(ncoeffs_equivalent_surface)
                    //             .zip(local_ptrs)
                    //             .for_each(|(result, local)| {
                    //                 let local = unsafe {
                    //                     std::slice::from_raw_parts_mut(
                    //                         local[0].raw,
                    //                         ncoeffs_equivalent_surface,
                    //                     )
                    //                 };
                    //                 local.iter_mut().zip(result).for_each(|(l, r)| *l += *r);
                    //             });
                    //     });
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
