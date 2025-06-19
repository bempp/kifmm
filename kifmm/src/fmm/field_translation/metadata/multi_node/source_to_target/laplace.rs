use std::{
    collections::{HashMap, HashSet},
    sync::{Mutex, RwLock},
};

use green_kernels::{
    laplace_3d::Laplace3dKernel, traits::Kernel as KernelTrait, types::GreenKernelEvalType,
};
use itertools::{izip, Itertools};
use mpi::traits::{Communicator, Equivalence};
use num::{Float, Zero};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use rlst::{
    empty_array, rlst_array_from_slice2, rlst_dynamic_array2, rlst_dynamic_array3, MultIntoResize,
    RawAccess, RawAccessMut, RlstScalar, Shape, SvdMode, UnsafeRandomAccessByRef,
    UnsafeRandomAccessMut,
};

use crate::{
    fmm::{
        field_translation::source_to_target::transfer_vector::compute_transfer_vectors_at_level,
        helpers::{
            multi_node::{
                all_gather_v_serialised, calculate_precomputation_load, deserialise_nested_vec,
                deserialise_vec, deserialise_vec_blas_metadata_sarcmp,
                deserialise_vec_fft_metadata, serialise_nested_vec, serialise_vec,
                serialise_vec_blas_metadata_sarcmp, serialise_vec_fft_metadata,
            },
            single_node::{find_cutoff_rank, flip3, ncoeffs_kifmm},
        },
        types::{BlasMetadataSaRcmp, FftMetadata, KiFmmMulti},
    },
    linalg::rsvd::MatrixRsvd,
    traits::{
        fftw::{Dft, DftType},
        field::{FieldTranslation as FieldTranslationTrait, SourceToTargetTranslationMetadata},
        general::single_node::AsComplex,
        tree::{Domain, FmmTreeNode, MultiFmmTree, MultiTree},
    },
    tree::{
        constants::{ALPHA_INNER, NHALO, NSIBLINGS, NSIBLINGS_SQUARED, NTRANSFER_VECTORS_KIFMM},
        helpers::find_corners,
        types::MortonKey,
    },
    BlasFieldTranslationSaRcmp, FftFieldTranslation, FmmSvdMode,
};

impl<Scalar> SourceToTargetTranslationMetadata
    for KiFmmMulti<Scalar, Laplace3dKernel<Scalar>, FftFieldTranslation<Scalar>>
where
    Scalar: RlstScalar
        + AsComplex
        + Default
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

    fn source_to_target(&mut self) {
        // Compute the field translation operators
        let size = self.communicator.size();
        let rank = self.communicator.rank();

        // Pick a point in the middle of the domain
        let two = Scalar::real(2.0);
        let midway = self
            .tree
            .domain()
            .side_length()
            .iter()
            .map(|d| *d / two)
            .collect_vec();

        let point = midway
            .iter()
            .zip(self.tree.domain().origin())
            .map(|(m, o)| *m + *o)
            .collect_vec();

        let point = [point[0], point[1], point[2]];

        // Encode point in centre of domain and compute halo of parent, and their resp. children
        let key = MortonKey::from_point(&point, self.tree.source_tree().domain(), 3);
        let siblings = key.siblings();
        let parent = key.parent();
        let halo = parent.neighbors();
        let halo_children = halo.iter().map(|h| h.children()).collect_vec();

        let iterator = if self.variable_expansion_order {
            self.equivalent_surface_order
                .iter()
                .skip(2)
                .cloned()
                .collect_vec()
        } else {
            self.equivalent_surface_order.clone()
        };

        let n_precomputations = if self.variable_expansion_order {
            iterator.len() as i32
        } else {
            1
        };

        let (load_counts, load_displacement) =
            calculate_precomputation_load(n_precomputations, size).unwrap();

        // Compute mandated local portion
        let local_load_count = load_counts[rank as usize];
        let local_load_displacement = load_displacement[rank as usize];

        let iterator_r = &iterator[(local_load_displacement as usize)
            ..((local_load_displacement + local_load_count) as usize)];

        let mut metadata_r: Vec<FftMetadata<<Scalar as AsComplex>::ComplexType>> = Vec::new();

        for &equivalent_surface_order in iterator_r.iter() {
            // The child boxes in the halo of the sibling set
            let mut sources = vec![];
            // The sibling set
            let mut targets = vec![];
            // The transfer vectors corresponding to source->target translations
            let mut transfer_vectors = vec![];
            // Green's function evaluations for each source, target pair interaction
            let mut kernel_data_vec = vec![];

            for _ in &halo_children {
                sources.push(vec![]);
                targets.push(vec![]);
                transfer_vectors.push(vec![]);
                kernel_data_vec.push(vec![]);
            }

            // Each set of 64 M2L operators will correspond to a point in the halo
            // Computing transfer of potential from sibling set to halo
            for (i, halo_child_set) in halo_children.iter().enumerate() {
                let mut tmp_transfer_vectors = vec![];
                let mut tmp_targets = vec![];
                let mut tmp_sources = vec![];

                // Consider all halo children for a given sibling at a time
                for sibling in siblings.iter() {
                    for halo_child in halo_child_set.iter() {
                        tmp_transfer_vectors.push(halo_child.find_transfer_vector(sibling));
                        tmp_targets.push(sibling);
                        tmp_sources.push(halo_child);
                    }
                }

                // From source to target
                transfer_vectors[i] = tmp_transfer_vectors;
                targets[i] = tmp_targets;
                sources[i] = tmp_sources;
            }

            let shape = <Scalar as Dft>::shape_in(equivalent_surface_order);
            let transform_shape = <Scalar as Dft>::shape_out(equivalent_surface_order);
            let transform_size = <Scalar as Dft>::size_out(equivalent_surface_order);
            let alpha = Scalar::real(ALPHA_INNER);

            // Iterate over each set of convolutions in the halo (26)
            for i in 0..transfer_vectors.len() {
                // Iterate over each unique convolution between sibling set, and halo siblings (64)
                for j in 0..transfer_vectors[i].len() {
                    // Translating from sibling set to boxes in its M2L halo
                    let target = targets[i][j];
                    let source = sources[i][j];

                    let source_equivalent_surface = source.surface_grid(
                        equivalent_surface_order,
                        self.tree.source_tree().domain(),
                        alpha,
                    );
                    let target_check_surface = target.surface_grid(
                        equivalent_surface_order,
                        self.tree.source_tree().domain(),
                        alpha,
                    );

                    let v_list: HashSet<MortonKey<_>> = target
                        .parent()
                        .neighbors()
                        .iter()
                        .flat_map(|pn| pn.children())
                        .filter(|pnc| !target.is_adjacent(pnc))
                        .collect();

                    if v_list.contains(source) {
                        // Compute convolution grid around the source box
                        let conv_point_corner_index = 7;
                        let corners = find_corners(&source_equivalent_surface[..]);
                        let conv_point_corner = [
                            corners[self.dim * conv_point_corner_index],
                            corners[self.dim * conv_point_corner_index + 1],
                            corners[self.dim * conv_point_corner_index + 2],
                        ];

                        let (conv_grid, _) = source.convolution_grid(
                            equivalent_surface_order,
                            self.tree.source_tree().domain(),
                            alpha,
                            &conv_point_corner,
                            conv_point_corner_index,
                        );

                        // Calculate Green's fct evaluations with respect to a 'kernel point' on the target box
                        let kernel_point_index = 0;
                        let kernel_point = [
                            target_check_surface[self.dim * kernel_point_index],
                            target_check_surface[self.dim * kernel_point_index + 1],
                            target_check_surface[self.dim * kernel_point_index + 2],
                        ];

                        // Compute Green's fct evaluations
                        let mut kernel = flip3(&self.evaluate_greens_fct_convolution_grid(
                            equivalent_surface_order,
                            &conv_grid,
                            kernel_point,
                        ));

                        // Compute FFT of padded kernel
                        let mut kernel_hat =
                            rlst_dynamic_array3!(<Scalar as DftType>::OutputType, transform_shape);

                        let plan = Scalar::plan_forward(
                            kernel.data_mut(),
                            kernel_hat.data_mut(),
                            &shape,
                            None,
                        )
                        .unwrap();
                        let _ = Scalar::forward_dft(
                            kernel.data_mut(),
                            kernel_hat.data_mut(),
                            &shape,
                            &plan,
                        );

                        kernel_data_vec[i].push(kernel_hat);
                    } else {
                        // Fill with zeros when interaction doesn't exist
                        let kernel_hat_zeros =
                            rlst_dynamic_array3!(<Scalar as DftType>::OutputType, transform_shape);
                        kernel_data_vec[i].push(kernel_hat_zeros);
                    }
                }
            }

            // Each element corresponds to all evaluations for each sibling (in order) at that halo position
            let mut kernel_data = vec![
                vec![
                    <Scalar as DftType>::OutputType::zero();
                    NSIBLINGS_SQUARED * transform_size
                ];
                halo_children.len()
            ];

            // For each halo position
            for i in 0..halo_children.len() {
                // For each unique interaction
                for j in 0..NSIBLINGS_SQUARED {
                    let offset = j * transform_size;
                    kernel_data[i][offset..offset + transform_size]
                        .copy_from_slice(kernel_data_vec[i][j].data())
                }
            }

            // We want to use this data by frequency in the implementation of FFT M2L
            // Rearrangement: Grouping by frequency, then halo child, then sibling
            let mut kernel_data_f = vec![];
            for _ in &halo_children {
                kernel_data_f.push(vec![]);
            }
            for i in 0..halo_children.len() {
                let current_vector = &kernel_data[i];
                for l in 0..transform_size {
                    // halo child
                    for k in 0..NSIBLINGS {
                        // sibling
                        for j in 0..NSIBLINGS {
                            let index = j * transform_size * 8 + k * transform_size + l;
                            kernel_data_f[i].push(current_vector[index]);
                        }
                    }
                }
            }

            // Re-order
            let mut kernel_data_ft = Vec::new();
            for freq in 0..transform_size {
                let frequency_offset = NSIBLINGS_SQUARED * freq;
                for kernel_f in kernel_data_f.iter().take(NHALO) {
                    let k_f = &kernel_f[frequency_offset..(frequency_offset + NSIBLINGS_SQUARED)]
                        .to_vec();
                    let k_f_ = rlst_array_from_slice2!(k_f.as_slice(), [NSIBLINGS, NSIBLINGS]);
                    let mut k_ft = rlst_dynamic_array2!(
                        <Scalar as DftType>::OutputType,
                        [NSIBLINGS, NSIBLINGS]
                    );
                    k_ft.fill_from(k_f_.r());
                    kernel_data_ft.push(k_ft.data().to_vec());
                }
            }

            let metadata = FftMetadata {
                kernel_data,
                kernel_data_f: kernel_data_ft,
            };

            metadata_r.push(metadata);
        }

        // Communicate metadata
        let metadata_r_serialised = serialise_vec_fft_metadata(&metadata_r);
        let global_metadata_serialised =
            all_gather_v_serialised(&metadata_r_serialised, &self.communicator);

        // Reconstruct metadata
        let (mut global_metadata, mut rest) =
            deserialise_vec_fft_metadata(&global_metadata_serialised);
        while !rest.is_empty() {
            let (mut t1, t2) =
                deserialise_vec_fft_metadata::<<Scalar as AsComplex>::ComplexType>(rest);
            global_metadata.append(&mut t1);
            rest = t2;
        }

        for metadata in global_metadata.iter() {
            // Copy for global FMM
            self.global_fmm
                .source_to_target
                .metadata
                .push(metadata.clone());

            // Set operator data
            self.source_to_target.metadata.push(metadata.clone());
        }

        // Compute and attach maps
        let mut tmp1 = Vec::new();
        let mut tmp2 = Vec::new();
        for &expansion_order in &iterator {
            let (surf_to_conv_map, conv_to_surf_map) =
                Self::compute_surf_to_conv_map(expansion_order);
            tmp1.push(surf_to_conv_map);
            tmp2.push(conv_to_surf_map)
        }
        self.source_to_target.surf_to_conv_map = tmp1;
        self.source_to_target.conv_to_surf_map = tmp2;

        // Copy for global FMM
        self.global_fmm.source_to_target.surf_to_conv_map =
            self.source_to_target.surf_to_conv_map.clone();
        self.global_fmm.source_to_target.conv_to_surf_map =
            self.source_to_target.conv_to_surf_map.clone();
    }
}

impl<Scalar> SourceToTargetTranslationMetadata
    for KiFmmMulti<Scalar, Laplace3dKernel<Scalar>, BlasFieldTranslationSaRcmp<Scalar>>
where
    Scalar: RlstScalar + Default + MatrixRsvd + Equivalence + Float,
    <Scalar as RlstScalar>::Real: Default + Equivalence + Float,
{
    fn displacements(&mut self, start_level: Option<u64>) {
        let mut displacements = Vec::new();

        let start_level =
            start_level.unwrap_or_else(|| std::cmp::max(2, self.tree.source_tree().global_depth()));

        for level in start_level..=self.tree.source_tree().total_depth() {
            let sources = self.tree.source_tree().keys(level).unwrap_or_default();
            let n_sources = sources.len();

            let sentinel = -1_i32;
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
                    for (i, &v) in transfer_vectors.iter().enumerate() {
                        transfer_vectors_map.insert(v, i);
                    }

                    let transfer_vectors_set: HashSet<_> = transfer_vectors.into_iter().collect();

                    // Mark items in interaction list for scattering
                    for (tv_idx, tv) in self.source_to_target.transfer_vectors.iter().enumerate() {
                        let mut result_lock = result[tv_idx].write().unwrap();
                        if transfer_vectors_set.contains(&tv.hash) {
                            // Look up scatter location in target tree
                            let target =
                                &interaction_list[*transfer_vectors_map.get(&tv.hash).unwrap()];
                            let &target_idx = self.level_index_pointer_locals[level as usize]
                                .get(target)
                                .unwrap();
                            result_lock[source_idx] = target_idx as i32;
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
        let total_depth = self.tree.source_tree().total_depth();
        let alpha = Scalar::real(ALPHA_INNER);

        // Distribute SVD by level
        // Number of pre-computations
        // Distribute the SVD by level
        let iterator = if self.variable_expansion_order {
            self.equivalent_surface_order
                .iter()
                .skip(2)
                .cloned()
                .zip(self.check_surface_order.iter().skip(2).cloned())
                .collect_vec()
        } else {
            vec![*self.equivalent_surface_order.last().unwrap(); (total_depth - 1) as usize]
                .iter()
                .cloned()
                .zip(vec![
                    *self.check_surface_order.last().unwrap();
                    (total_depth - 1) as usize
                ])
                .collect_vec()
        };

        let n_precomputations = if self.variable_expansion_order {
            iterator.len() as i32
        } else {
            1
        };

        let (load_counts, load_displacement) =
            calculate_precomputation_load(n_precomputations, size).unwrap();

        // Compute mandated local portion
        let local_load_count = load_counts[rank as usize];
        let local_load_displacement = load_displacement[rank as usize];

        let iterator_r = &iterator[(local_load_displacement as usize)
            ..((local_load_displacement + local_load_count) as usize)];

        let mut cutoff_rank_r = Vec::new();
        let mut directional_cutoff_ranks_r = Vec::new();
        let mut metadata_r = Vec::new();

        for &(equivalent_surface_order, check_surface_order) in iterator_r.iter() {
            let transfer_vectors = compute_transfer_vectors_at_level::<Scalar::Real>(3).unwrap();

            let n_rows = ncoeffs_kifmm(check_surface_order);
            let n_cols = ncoeffs_kifmm(equivalent_surface_order);

            let mut se2tc_fat =
                rlst_dynamic_array2!(Scalar, [n_rows, n_cols * NTRANSFER_VECTORS_KIFMM]);
            let mut se2tc_thin =
                rlst_dynamic_array2!(Scalar, [n_rows * NTRANSFER_VECTORS_KIFMM, n_cols]);

            transfer_vectors.iter().enumerate().for_each(|(i, t)| {
                let source_equivalent_surface = t.source.surface_grid(
                    equivalent_surface_order,
                    self.tree.source_tree().domain(),
                    alpha,
                );
                let target_check_surface = t.target.surface_grid(
                    check_surface_order,
                    self.tree.source_tree().domain(),
                    alpha,
                );

                let mut tmp_gram = rlst_dynamic_array2!(Scalar, [n_rows, n_cols]);

                self.kernel.assemble_st(
                    GreenKernelEvalType::Value,
                    &target_check_surface[..],
                    &source_equivalent_surface[..],
                    tmp_gram.data_mut(),
                );

                let mut block = se2tc_fat
                    .r_mut()
                    .into_subview([0, i * n_cols], [n_rows, n_cols]);
                block.fill_from(tmp_gram.r());

                let mut block_column = se2tc_thin
                    .r_mut()
                    .into_subview([i * n_rows, 0], [n_rows, n_cols]);
                block_column.fill_from(tmp_gram.r());
            });

            let mu = se2tc_fat.shape()[0];
            let nvt = se2tc_fat.shape()[1];
            let k = std::cmp::min(mu, nvt);

            let mut u_big = rlst_dynamic_array2!(Scalar, [mu, k]);
            let mut sigma = vec![Scalar::zero().re(); k];
            let mut vt_big = rlst_dynamic_array2!(Scalar, [k, nvt]);

            // Target rank defined by max dimension before cutoff
            let mut target_rank = k;

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
                            max_equivalent_surface_ncoeffs.max(max_check_surface_ncoeffs) / 2;
                    }

                    let mut se2tc_fat_transpose =
                        rlst_dynamic_array2!(Scalar, se2tc_fat.r().transpose().shape());
                    se2tc_fat_transpose
                        .r_mut()
                        .fill_from(se2tc_fat.r().transpose());

                    let (sigma_t, u_big_t, vt_big_t) = Scalar::rsvd_fixed_rank(
                        &se2tc_fat_transpose,
                        target_rank,
                        n_oversamples,
                        normaliser,
                        random_state,
                    )
                    .unwrap();
                    u_big = rlst_dynamic_array2!(Scalar, [mu, sigma_t.len()]);
                    vt_big = rlst_dynamic_array2!(Scalar, [sigma_t.len(), nvt]);

                    vt_big.fill_from(u_big_t.transpose());
                    u_big.fill_from(vt_big_t.transpose());
                    sigma = sigma_t;
                }
                FmmSvdMode::Deterministic => {
                    se2tc_fat
                        .into_svd_alloc(
                            u_big.r_mut(),
                            vt_big.r_mut(),
                            &mut sigma[..],
                            SvdMode::Reduced,
                        )
                        .unwrap();
                }
            }

            // Cutoff rank is the minimum of the target rank and the value found by user threshold
            let cutoff_rank =
                find_cutoff_rank(&sigma, self.source_to_target.threshold, n_cols).min(target_rank);

            let mut u = rlst_dynamic_array2!(Scalar, [mu, cutoff_rank]);
            let mut sigma_mat = rlst_dynamic_array2!(Scalar, [cutoff_rank, cutoff_rank]);
            let mut vt = rlst_dynamic_array2!(Scalar, [cutoff_rank, nvt]);

            // Store compressed M2L operators
            let thin_nrows = se2tc_thin.shape()[0];
            let nst = se2tc_thin.shape()[1];
            let k = std::cmp::min(thin_nrows, nst);
            let mut st;
            let mut _gamma;
            let mut _r;

            if self.source_to_target.surface_diff() == 0 {
                st = rlst_dynamic_array2!(Scalar, u_big.r().transpose().shape());
                st.fill_from(u_big.r().transpose())
            } else {
                match &self.source_to_target.svd_mode {
                    &FmmSvdMode::Random {
                        n_components,
                        normaliser,
                        n_oversamples,
                        random_state,
                    } => {
                        let target_rank;
                        if let Some(n_components) = n_components {
                            target_rank = n_components
                        } else {
                            // Estimate target rank
                            let max_equivalent_surface_ncoeffs =
                                self.n_coeffs_equivalent_surface.iter().max().unwrap();
                            let max_check_surface_ncoeffs =
                                self.n_coeffs_check_surface.iter().max().unwrap();
                            target_rank =
                                max_equivalent_surface_ncoeffs.max(max_check_surface_ncoeffs) / 2;
                        }

                        (_gamma, _r, st) = Scalar::rsvd_fixed_rank(
                            &se2tc_thin,
                            target_rank,
                            n_oversamples,
                            normaliser,
                            random_state,
                        )
                        .unwrap();
                    }
                    FmmSvdMode::Deterministic => {
                        _r = rlst_dynamic_array2!(Scalar, [thin_nrows, k]);
                        _gamma = vec![Scalar::zero().re(); k];
                        st = rlst_dynamic_array2!(Scalar, [k, nst]);
                        se2tc_thin
                            .into_svd_alloc(
                                _r.r_mut(),
                                st.r_mut(),
                                &mut _gamma[..],
                                SvdMode::Reduced,
                            )
                            .unwrap();
                    }
                }
            }

            u.fill_from(u_big.into_subview([0, 0], [mu, cutoff_rank]));
            vt.fill_from(vt_big.into_subview([0, 0], [cutoff_rank, nvt]));
            for (j, s) in sigma.iter().enumerate().take(cutoff_rank) {
                unsafe {
                    *sigma_mat.get_unchecked_mut([j, j]) = Scalar::from(*s).unwrap();
                }
            }

            let mut s_trunc = rlst_dynamic_array2!(Scalar, [nst, cutoff_rank]);
            for j in 0..cutoff_rank {
                for i in 0..nst {
                    unsafe { *s_trunc.get_unchecked_mut([i, j]) = *st.get_unchecked([j, i]) }
                }
            }

            let c_u = Mutex::new(Vec::new());
            let c_vt = Mutex::new(Vec::new());
            let directional_cutoff_ranks = Mutex::new(vec![0i32; NTRANSFER_VECTORS_KIFMM]);

            for _ in 0..NTRANSFER_VECTORS_KIFMM {
                c_u.lock()
                    .unwrap()
                    .push(rlst_dynamic_array2!(Scalar, [1, 1]));
                c_vt.lock()
                    .unwrap()
                    .push(rlst_dynamic_array2!(Scalar, [1, 1]));
            }

            (0..NTRANSFER_VECTORS_KIFMM).into_par_iter().for_each(|i| {
                let vt_block = vt.r().into_subview([0, i * n_cols], [cutoff_rank, n_cols]);

                let tmp = empty_array::<Scalar, 2>().simple_mult_into_resize(
                    sigma_mat.r(),
                    empty_array::<Scalar, 2>().simple_mult_into_resize(vt_block.r(), s_trunc.r()),
                );

                let mut u_i = rlst_dynamic_array2!(Scalar, [cutoff_rank, cutoff_rank]);
                let mut sigma_i = vec![Scalar::zero().re(); cutoff_rank];
                let mut vt_i = rlst_dynamic_array2!(Scalar, [cutoff_rank, cutoff_rank]);

                tmp.into_svd_alloc(u_i.r_mut(), vt_i.r_mut(), &mut sigma_i, SvdMode::Full)
                    .unwrap();

                let directional_cutoff_rank =
                    find_cutoff_rank(&sigma_i, self.source_to_target.threshold, cutoff_rank);

                let mut u_i_compressed =
                    rlst_dynamic_array2!(Scalar, [cutoff_rank, directional_cutoff_rank]);
                let mut vt_i_compressed_ =
                    rlst_dynamic_array2!(Scalar, [directional_cutoff_rank, cutoff_rank]);

                let mut sigma_mat_i_compressed = rlst_dynamic_array2!(
                    Scalar,
                    [directional_cutoff_rank, directional_cutoff_rank]
                );

                u_i_compressed
                    .fill_from(u_i.into_subview([0, 0], [cutoff_rank, directional_cutoff_rank]));
                vt_i_compressed_
                    .fill_from(vt_i.into_subview([0, 0], [directional_cutoff_rank, cutoff_rank]));

                for (j, s) in sigma_i.iter().enumerate().take(directional_cutoff_rank) {
                    unsafe {
                        *sigma_mat_i_compressed.get_unchecked_mut([j, j]) =
                            Scalar::from(*s).unwrap();
                    }
                }

                let vt_i_compressed = empty_array::<Scalar, 2>()
                    .simple_mult_into_resize(sigma_mat_i_compressed.r(), vt_i_compressed_.r());

                directional_cutoff_ranks.lock().unwrap()[i] = directional_cutoff_rank as i32;
                c_u.lock().unwrap()[i] = u_i_compressed;
                c_vt.lock().unwrap()[i] = vt_i_compressed;
            });

            let mut st_trunc = rlst_dynamic_array2!(Scalar, [cutoff_rank, nst]);
            st_trunc.fill_from(s_trunc.transpose());

            let c_vt = std::mem::take(&mut *c_vt.lock().unwrap());
            let c_u = std::mem::take(&mut *c_u.lock().unwrap());
            let directional_cutoff_ranks =
                std::mem::take(&mut *directional_cutoff_ranks.lock().unwrap());

            let result = BlasMetadataSaRcmp {
                u,
                st: st_trunc,
                c_u,
                c_vt,
            };

            metadata_r.push(result);
            cutoff_rank_r.push(cutoff_rank as i32);
            directional_cutoff_ranks_r.push(directional_cutoff_ranks);
        }

        // Communicate metadata
        let metadata_r_serialised = serialise_vec_blas_metadata_sarcmp(&metadata_r);
        let global_metadata_serialised =
            all_gather_v_serialised(&metadata_r_serialised, &self.communicator);

        let cutoff_rank_r_serialised = serialise_vec(&cutoff_rank_r);
        let global_cutoff_rank_serialised =
            all_gather_v_serialised(&cutoff_rank_r_serialised, &self.communicator);

        let directional_cutoff_rank_r_serialised =
            serialise_nested_vec(&directional_cutoff_ranks_r);
        let global_directional_cutoff_rank_serialised =
            all_gather_v_serialised(&directional_cutoff_rank_r_serialised, &self.communicator);

        // Reconstruct metadata
        let (mut global_metadata, mut rest) =
            deserialise_vec_blas_metadata_sarcmp::<Scalar>(&global_metadata_serialised);
        while !rest.is_empty() {
            let (mut t1, t2) = deserialise_vec_blas_metadata_sarcmp::<Scalar>(rest);
            global_metadata.append(&mut t1);
            rest = t2;
        }

        let (mut global_metadata_clone, mut rest) =
            deserialise_vec_blas_metadata_sarcmp::<Scalar>(&global_metadata_serialised);
        while !rest.is_empty() {
            let (mut t1, t2) = deserialise_vec_blas_metadata_sarcmp::<Scalar>(rest);
            global_metadata_clone.append(&mut t1);
            rest = t2;
        }

        let mut buffer = Vec::new();
        let (global_cutoff_ranks, mut rest) =
            deserialise_vec::<i32>(&global_cutoff_rank_serialised);
        buffer.extend_from_slice(global_cutoff_ranks);
        while !rest.is_empty() {
            let (t1, t2) = deserialise_vec::<i32>(rest);
            buffer.extend_from_slice(t1);
            rest = t2;
        }

        let (mut global_directional_cutoff_ranks, mut rest) =
            deserialise_nested_vec::<i32>(&global_directional_cutoff_rank_serialised);
        while !rest.is_empty() {
            let (mut t1, t2) = deserialise_nested_vec::<i32>(rest);
            global_directional_cutoff_ranks.append(&mut t1);
            rest = t2;
        }

        let global_cutoff_ranks = buffer.iter().map(|&x| x as usize).collect_vec();
        let global_directional_cutoff_ranks = global_directional_cutoff_ranks
            .iter()
            .map(|vec| vec.iter().map(|&x| x as usize).collect_vec())
            .collect_vec();

        for (metadata, metadata_clone) in izip!(global_metadata, global_metadata_clone) {
            self.global_fmm.source_to_target.metadata.push(metadata);
            self.source_to_target.metadata.push(metadata_clone);
        }

        self.source_to_target.cutoff_rank = global_cutoff_ranks.clone();
        self.source_to_target.directional_cutoff_ranks = global_directional_cutoff_ranks.clone();

        self.global_fmm.source_to_target.cutoff_rank = global_cutoff_ranks.clone();
        self.global_fmm.source_to_target.directional_cutoff_ranks =
            global_directional_cutoff_ranks.clone();
        self.global_fmm.source_to_target.transfer_vectors =
            self.source_to_target.transfer_vectors.clone();

        self.ghost_fmm_v.source_to_target.transfer_vectors =
            self.source_to_target.transfer_vectors.clone();
    }
}
