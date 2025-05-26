use std::{
    collections::{HashMap, HashSet},
    sync::{Mutex, RwLock},
};

use green_kernels::{
    helmholtz_3d::Helmholtz3dKernel, laplace_3d::Laplace3dKernel, traits::Kernel as KernelTrait,
    types::GreenKernelEvalType,
};
use itertools::{izip, Itertools};
use mpi::traits::{Communicator, Equivalence};
use num::{Float, Zero};
use pulp::Scalar;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use rlst::{
    empty_array, rlst_array_from_slice2, rlst_dynamic_array2, rlst_dynamic_array3, MultIntoResize,
    RawAccess, RawAccessMut, RlstScalar, Shape, SvdMode, UnsafeRandomAccessByRef,
    UnsafeRandomAccessMut,
};

use crate::{
    fmm::{
        field_translation::source_to_target::transfer_vector::{
            self, compute_transfer_vectors_at_level,
        },
        helpers::{
            multi_node::{
                all_gather_v_serialised, calculate_precomputation_load, deserialise_nested_array,
                deserialise_nested_vec, deserialise_vec, deserialise_vec_blas_metadata_sarcmp,
                deserialise_vec_fft_metadata, serialise_nested_array, serialise_nested_vec,
                serialise_vec, serialise_vec_blas_metadata_sarcmp, serialise_vec_fft_metadata,
            },
            single_node::{find_cutoff_rank, flip3, ncoeffs_kifmm},
        },
        types::{BlasMetadataIa, BlasMetadataSaRcmp, FftMetadata, KiFmmMulti},
    },
    linalg::rsvd::MatrixRsvd,
    traits::{
        fftw::{Dft, DftType},
        field::{FieldTranslation as FieldTranslationTrait, SourceToTargetTranslationMetadata},
        fmm::MetadataAccess,
        general::single_node::AsComplex,
        tree::{Domain, FmmTreeNode, MultiFmmTree, MultiTree},
    },
    tree::{
        constants::{ALPHA_INNER, NHALO, NSIBLINGS, NSIBLINGS_SQUARED, NTRANSFER_VECTORS_KIFMM},
        helpers::find_corners,
        types::MortonKey,
    },
    BlasFieldTranslationIa, FftFieldTranslation, FmmSvdMode,
};

impl<Scalar> SourceToTargetTranslationMetadata
    for KiFmmMulti<Scalar, Helmholtz3dKernel<Scalar>, FftFieldTranslation<Scalar>>
where
    Scalar: RlstScalar<Complex = Scalar>
        + Default
        + AsComplex
        + Dft<InputType = Scalar, OutputType = <Scalar as AsComplex>::ComplexType>
        + Equivalence,
    <Scalar as RlstScalar>::Real: RlstScalar + Default + Equivalence + Float,
{
    fn displacements(&mut self, start_level: Option<u64>) {
        let mut displacements = Vec::new();
        let start_level =
            start_level.unwrap_or_else(|| std::cmp::max(2, self.tree.source_tree().global_depth()));

        for level in start_level..=self.tree.source_tree.total_depth() {
            let targets = self.tree.target_tree().keys(level).unwrap_or_default();
            let targets_parents: HashSet<MortonKey<_>> =
                targets.iter().map(|target| target.parent()).collect();
            let mut targets_parents = targets_parents.into_iter().collect_vec();
            targets_parents.sort();
            let ntargets_parents = targets_parents.len();

            let sources = self.tree.source_tree().keys(level).unwrap_or_default();

            let sources_parents: HashSet<MortonKey<_>> =
                sources.iter().map(|source| source.parent()).collect();
            let mut sources_parents = sources_parents.into_iter().collect_vec();
            sources_parents.sort();
            let nsources_parents = sources_parents.len();

            let result = vec![Vec::new(); NHALO];
            let result = result.into_iter().map(RwLock::new).collect_vec();

            let targets_parents_neighbors = targets_parents
                .iter()
                .map(|parent| parent.all_neighbors())
                .collect_vec();

            let zero_displacement = nsources_parents * NSIBLINGS;

            (0..NHALO).into_par_iter().for_each(|i| {
                let mut result_i = result[i].write().unwrap();
                for all_neighbors in targets_parents_neighbors.iter().take(ntargets_parents) {
                    // Check if neighbor exists in a valid tree
                    if let Some(neighbor) = all_neighbors[i] {
                        // If it does, check if first child exists in the source tree
                        let first_child = neighbor.first_child();
                        if let Some(neighbor_displacement) =
                            self.level_index_pointer_multipoles[level as usize].get(&first_child)
                        {
                            result_i.push(*neighbor_displacement)
                        } else {
                            result_i.push(zero_displacement)
                        }
                    } else {
                        result_i.push(zero_displacement)
                    }
                }
            });

            displacements.push(result);
        }

        self.source_to_target.displacements = displacements;
    }

    fn source_to_target(&mut self) {}
}

impl<Scalar> SourceToTargetTranslationMetadata
    for KiFmmMulti<Scalar, Helmholtz3dKernel<Scalar>, BlasFieldTranslationIa<Scalar>>
where
    Scalar: RlstScalar<Complex = Scalar> + Default + MatrixRsvd + Equivalence,
    <Scalar as RlstScalar>::Real: Default + Equivalence + Float,
    Self: MetadataAccess,
{
    fn displacements(&mut self, start_level: Option<u64>) {
        let mut displacements = Vec::new();
        let start_level = start_level.unwrap_or(2).max(2);

        for level in start_level..=self.tree.source_tree().total_depth() {
            let sources = self.tree.source_tree().keys(level).unwrap();
            let n_sources = sources.len();
            let m2l_operator_index = self.m2l_operator_index(level);
            let sentinel = -1i32;

            let result = vec![vec![sentinel; n_sources]; 316];
            let result = result.into_iter().map(RwLock::new).collect_vec();

            let tmp = HashSet::new();
            let target_tree_keys_set = self.tree.target_tree().all_keys_set().unwrap_or(&tmp);

            sources
                .into_par_iter()
                .enumerate()
                .for_each(|(source_idx, source)| {
                    // Find interaction list of each source, as this defines scatter locations
                    let interaction_list = source
                        .parent()
                        .neighbors()
                        .iter()
                        .flat_map(|pn| pn.children())
                        .filter(|pnc| {
                            !source.is_adjacent(pnc) && target_tree_keys_set.contains(pnc)
                        })
                        .collect_vec();

                    let transfer_vectors = interaction_list
                        .iter()
                        .map(|target| source.find_transfer_vector(target).unwrap())
                        .collect_vec();

                    let mut transfer_vectors_map = HashMap::new();
                    for (i, v) in transfer_vectors.iter().enumerate() {
                        transfer_vectors_map.insert(v, i);
                    }

                    let transfer_vectors_set: HashSet<_> =
                        transfer_vectors.iter().cloned().collect();

                    // Mark items in interaction list for scattering
                    for (tv_idx, tv) in self.source_to_target.transfer_vectors[m2l_operator_index]
                        .iter()
                        .enumerate()
                    {
                        let mut all_displacements_lock = result[tv_idx].write().unwrap();
                        if transfer_vectors_set.contains(&tv.hash) {
                            // Look up scatter location in target tree
                            let target =
                                &interaction_list[*transfer_vectors_map.get(&tv.hash).unwrap()];
                            let &target_idx = self.level_index_pointer_locals[level as usize]
                                .get(target)
                                .unwrap();
                            all_displacements_lock[source_idx] = target_idx as i32;
                        }
                    }
                });

            displacements.push(result);
        }

        self.source_to_target.displacements = displacements;
    }

    fn source_to_target(&mut self) {
        let size = self.communicator.size();
        let rank = self.communicator.rank();
        let alpha = Scalar::real(ALPHA_INNER);
        let total_depth = self.tree.source_tree().total_depth();

        let iterator = if self.variable_expansion_order {
            (2..=total_depth)
                .zip(self.equivalent_surface_order.iter().skip(2).cloned())
                .zip(self.check_surface_order.iter().skip(2).cloned())
                .collect_vec()
        } else {
            (2..=total_depth)
                .zip(vec![
                    *self.equivalent_surface_order.last().unwrap();
                    (total_depth - 1) as usize
                ])
                .zip(vec![
                    *self.check_surface_order.last().unwrap();
                    (total_depth - 1) as usize
                ])
                .collect_vec()
        };

        // First need to enumerate tasks
        let mut tasks = Vec::new();
        for &((level, equivalent_surface_order), check_surface_order) in iterator.iter() {
            let transfer_vectors =
                compute_transfer_vectors_at_level::<Scalar::Real>(level).unwrap();
            for &t in transfer_vectors.iter() {
                tasks.push((level, equivalent_surface_order, check_surface_order, t));
            }
        }

        // Now distribute tasks more granularly, as doing independent compressions for each transfer vctor at each level
        let n_precomputations = tasks.len() as i32;

        let (load_counts, load_displacement) =
            calculate_precomputation_load(n_precomputations, size).unwrap();

        // Compute mandated local portion
        let local_load_count = load_counts[rank as usize];
        let local_load_displacement = load_displacement[rank as usize];

        let tasks_r = &tasks[(local_load_displacement as usize)
            ..((local_load_displacement + local_load_count) as usize)];

        let domain = self.tree.source_tree().domain();

        let mut u_r = Vec::new();
        let mut vt_r = Vec::new();
        let mut cutoff_ranks_r = Vec::new();
        let mut level_r = Vec::new();

        for &(level, equivalent_surface_order, check_surface_order, transfer_vector) in
            tasks_r.iter()
        {
            let source_equivalent_surface =
                transfer_vector
                    .source
                    .surface_grid(equivalent_surface_order, domain, alpha);
            let n_sources = ncoeffs_kifmm(equivalent_surface_order);

            let target_check_surface =
                transfer_vector
                    .target
                    .surface_grid(check_surface_order, domain, alpha);
            let n_targets = ncoeffs_kifmm(check_surface_order);

            let mut tmp_gram = rlst_dynamic_array2!(Scalar, [n_targets, n_sources]);

            self.kernel.assemble_st(
                GreenKernelEvalType::Value,
                &target_check_surface[..],
                &source_equivalent_surface[..],
                tmp_gram.data_mut(),
            );

            let mu = tmp_gram.shape()[0];
            let nvt = tmp_gram.shape()[1];
            let k = std::cmp::min(mu, nvt);

            let mut u = rlst_dynamic_array2!(Scalar, [mu, k]);
            let mut sigma = vec![Scalar::zero().re(); k];
            let mut vt = rlst_dynamic_array2!(Scalar, [k, nvt]);

            let target_rank;

            match &self.source_to_target.svd_mode {
                &FmmSvdMode::Random {
                    n_components,
                    normaliser,
                    n_oversamples,
                    random_state,
                } => {
                    // Estimate target rank if unspecified by user
                    if let Some(n_components) = n_components {
                        target_rank = n_components
                    } else {
                        let max_equivalent_surface_ncoeffs =
                            self.n_coeffs_equivalent_surface.iter().max().unwrap();
                        let max_check_surface_ncoeffs =
                            self.n_coeffs_check_surface.iter().max().unwrap();
                        target_rank =
                            *max_equivalent_surface_ncoeffs.max(max_check_surface_ncoeffs);
                    }

                    (sigma, u, vt) = Scalar::rsvd_fixed_rank(
                        &tmp_gram,
                        target_rank,
                        n_oversamples,
                        normaliser,
                        random_state,
                    )
                    .unwrap();
                }

                FmmSvdMode::Deterministic => {
                    tmp_gram
                        .into_svd_alloc(u.r_mut(), vt.r_mut(), &mut sigma[..], SvdMode::Reduced)
                        .unwrap();
                }
            }

            let mut sigma_mat = rlst_dynamic_array2!(Scalar, [k, k]);

            for (j, s) in sigma.iter().enumerate().take(k) {
                unsafe {
                    *sigma_mat.get_unchecked_mut([j, j]) = Scalar::from(*s).unwrap();
                }
            }

            let vt = empty_array::<Scalar, 2>().simple_mult_into_resize(sigma_mat.r(), vt.r());

            let cutoff_rank = find_cutoff_rank(&sigma, self.source_to_target.threshold, n_sources);

            let mut u_compressed = rlst_dynamic_array2!(Scalar, [mu, cutoff_rank]);
            let mut vt_compressed = rlst_dynamic_array2!(Scalar, [cutoff_rank, nvt]);

            u_compressed.fill_from(u.into_subview([0, 0], [mu, cutoff_rank]));
            vt_compressed.fill_from(vt.into_subview([0, 0], [cutoff_rank, nvt]));
            u_r.push(u_compressed);
            vt_r.push(vt_compressed);
            cutoff_ranks_r.push(cutoff_rank as i32);
            level_r.push(level);
        }

        // Communicate metadata
        let u_r_serialised = serialise_nested_array(&u_r);
        let global_u_serialised = all_gather_v_serialised(&u_r_serialised, &self.communicator);

        let vt_r_serialised = serialise_nested_array(&vt_r);
        let global_vt_serialised = all_gather_v_serialised(&vt_r_serialised, &self.communicator);

        let cutoff_rank_r_serialised = serialise_vec(&cutoff_ranks_r);
        let global_cutoff_rank_serialised =
            all_gather_v_serialised(&cutoff_rank_r_serialised, &self.communicator);

        let level_r_serialised = serialise_vec(&level_r);
        let global_level_serialised =
            all_gather_v_serialised(&level_r_serialised, &self.communicator);

        // Reconstruct metadata
        let (mut global_u, mut rest) = deserialise_nested_array::<Scalar>(&global_u_serialised);
        while !rest.is_empty() {
            let (mut t1, t2) = deserialise_nested_array::<Scalar>(rest);
            global_u.append(&mut t1);
            rest = t2;
        }

        let (mut global_vt, mut rest) = deserialise_nested_array::<Scalar>(&global_vt_serialised);
        while !rest.is_empty() {
            let (mut t1, t2) = deserialise_nested_array::<Scalar>(rest);
            global_vt.append(&mut t1);
            rest = t2;
        }

        let (mut global_cutoff_ranks, mut rest) =
            deserialise_nested_vec::<i32>(&global_cutoff_rank_serialised);
        while !rest.is_empty() {
            let (mut t1, t2) = deserialise_nested_vec::<i32>(rest);
            global_cutoff_ranks.append(&mut t1);
            rest = t2;
        }

        let (mut global_levels, mut rest) = deserialise_nested_vec::<u64>(&global_level_serialised);
        while !rest.is_empty() {
            let (mut t1, t2) = deserialise_nested_vec::<u64>(rest);
            global_levels.append(&mut t1);
            rest = t2;
        }

        // test that results are returned in level order
        println!("levels {:?}", global_levels)
    }
}
