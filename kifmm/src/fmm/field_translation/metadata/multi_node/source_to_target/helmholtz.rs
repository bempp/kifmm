use std::{
    collections::{HashMap, HashSet},
    sync::RwLock,
};

use green_kernels::{
    helmholtz_3d::Helmholtz3dKernel, traits::Kernel as KernelTrait, types::GreenKernelEvalType,
};
use itertools::{izip, Itertools};
use mpi::traits::{Communicator, Equivalence};
use num::{Float, Zero};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use rlst::{
    empty_array, rlst_array_from_slice2, rlst_dynamic_array2, rlst_dynamic_array3, MultIntoResize,
    RawAccess, RawAccessMut, RlstScalar, Shape, SvdMode, UnsafeRandomAccessMut,
};

use crate::{
    fmm::{
        field_translation::source_to_target::transfer_vector::compute_transfer_vectors_at_level,
        helpers::{
            multi_node::{
                all_gather_v_serialised, calculate_precomputation_load,
                deserialise_nested_array_2x2, deserialise_nested_array_3x3, deserialise_vec,
                serialise_nested_array_2x2, serialise_nested_array_3x3, serialise_vec,
            },
            single_node::{find_cutoff_rank, flip3, ncoeffs_kifmm},
        },
        types::{BlasMetadataIa, FftMetadata, KiFmmMulti},
    },
    linalg::rsvd::MatrixRsvd,
    traits::{
        fftw::{Dft, DftType},
        field::SourceToTargetTranslationMetadata,
        fmm::MetadataAccess,
        general::single_node::AsComplex,
        tree::{Domain, FmmTreeNode, MultiFmmTree, MultiTree},
    },
    tree::{
        constants::{ALPHA_INNER, NHALO, NSIBLINGS, NSIBLINGS_SQUARED},
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

    fn source_to_target(&mut self) {
        // Compute the field translation operators
        let size = self.communicator.size();
        let rank = self.communicator.rank();
        let alpha = Scalar::real(ALPHA_INNER);

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
        let domain = self.tree.source_tree().domain();

        let total_depth = self.tree.source_tree().total_depth();

        let mut metadata_r: Vec<FftMetadata<<Scalar as AsComplex>::ComplexType>> = Vec::new();

        if total_depth >= 2 {
            // Encode point in centre of domain and compute halo of parent, and their resp. children
            // Find unique transfer vectors in correct order at level 3
            let key = MortonKey::from_point(&point, domain, 3);
            let siblings = key.siblings();
            let parent = key.parent();

            let halo = parent.neighbors();
            let halo_children = halo.iter().map(|h| h.children()).collect_vec();

            let mut transfer_vector_index = vec![vec![0usize; NSIBLINGS_SQUARED]; NHALO];

            for (i, halo_child_set) in halo_children.iter().enumerate() {
                let outer_displacement = i;

                for (j, sibling) in siblings.iter().enumerate() {
                    for (k, halo_child) in halo_child_set.iter().enumerate() {
                        let tv = halo_child.find_transfer_vector(sibling).unwrap();

                        let inner_displacement = NSIBLINGS * j + k;
                        transfer_vector_index[outer_displacement][inner_displacement] = tv;
                    }
                }
            }

            // Compute data for level 2 separately, only on root rank
            {
                let equivalent_surface_order = if self.variable_expansion_order {
                    self.equivalent_surface_order[2]
                } else {
                    *self.equivalent_surface_order.last().unwrap()
                };

                let shape = <Scalar as Dft>::shape_in(equivalent_surface_order);
                let transform_shape = <Scalar as Dft>::shape_out(equivalent_surface_order);
                let transform_size = <Scalar as Dft>::size_out(equivalent_surface_order);

                // Need to find valid source/target pairs at this level with matching transfer vectors;
                let all_keys = MortonKey::<Scalar::Real>::root().descendants(2).unwrap();

                // The child boxes in the halo of the sibling set
                let mut sources = vec![];
                // The sibling set
                let mut targets = vec![];

                // Green's function evaluations for each source, target pair interaction
                let mut kernel_data_vec_r = vec![];

                for _ in 0..NHALO {
                    sources.push(vec![
                        MortonKey::<Scalar::Real>::default();
                        NSIBLINGS_SQUARED
                    ]);
                    targets.push(vec![
                        MortonKey::<Scalar::Real>::default();
                        NSIBLINGS_SQUARED
                    ]);
                    kernel_data_vec_r.push(vec![]);
                }

                let mut tv_source_target_pair_map = HashMap::new();
                for source in all_keys.iter() {
                    for target in all_keys.iter() {
                        let transfer_vector = source.find_transfer_vector(target).unwrap();

                        if !tv_source_target_pair_map.keys().contains(&transfer_vector) {
                            tv_source_target_pair_map.insert(transfer_vector, (source, target));
                        }
                    }
                }

                let mut tasks = Vec::new();

                // Identify tasks (FFTs) at this level to distribute over available resources
                // Iterate over each set of convolutions in the halo (26)
                for (i, tv_i) in transfer_vector_index.iter().enumerate().take(NHALO) {
                    // Iterate over each unique convolution between sibling set, and halo siblings (64)
                    for (j, tv) in tv_i.iter().enumerate().take(NSIBLINGS_SQUARED) {
                        let (source, target) = tv_source_target_pair_map.get(tv).unwrap();

                        let v_list: HashSet<MortonKey<_>> = target
                            .parent()
                            .neighbors()
                            .iter()
                            .flat_map(|pn| pn.children())
                            .filter(|pnc| !target.is_adjacent(pnc))
                            .collect();

                        if v_list.contains(source) {
                            tasks.push((i, j, true));
                        } else {
                            tasks.push((i, j, false));
                        }
                    }
                }

                // Partition tasks over available resources
                let n_precomputations = tasks.len() as i32;

                let (load_counts, load_displacement) =
                    calculate_precomputation_load(n_precomputations, size).unwrap();

                // Compute mandated local portion
                let local_load_count = load_counts[rank as usize];
                let local_load_displacement = load_displacement[rank as usize];

                let tasks_r = &tasks[(local_load_displacement as usize)
                    ..((local_load_displacement + local_load_count) as usize)];

                // Compute locally mandated portion of tasks
                for &(i, j, contains) in tasks_r {
                    let tv = transfer_vector_index[i][j];
                    let (source, target) = tv_source_target_pair_map.get(&tv).unwrap();

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

                    if contains {
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

                        kernel_data_vec_r[i].push(kernel_hat);
                    } else {
                        // Fill with zeros when interaction doesn't exist
                        let kernel_hat_zeros =
                            rlst_dynamic_array3!(<Scalar as DftType>::OutputType, transform_shape);
                        kernel_data_vec_r[i].push(kernel_hat_zeros);
                    }
                }

                // Serialise kernel data
                let mut kernel_data_vec_r_serialised = Vec::new();
                for vec in kernel_data_vec_r.iter() {
                    kernel_data_vec_r_serialised.push(serialise_nested_array_3x3(vec))
                }

                // Communicate kernel data
                let mut global_kernel_data_vec_serialised = vec![Vec::new(); NHALO];
                for (i, save_data) in global_kernel_data_vec_serialised
                    .iter_mut()
                    .take(NHALO)
                    .enumerate()
                {
                    let mut global_kernel_data_serialised_i = all_gather_v_serialised(
                        &kernel_data_vec_r_serialised[i],
                        &self.communicator,
                    );
                    save_data.append(&mut global_kernel_data_serialised_i);
                }

                // Deserialise kernel data
                let mut global_kernel_data_vec = Vec::new();
                for _ in 0..NHALO {
                    global_kernel_data_vec.push(vec![])
                }

                for (i, kernel_data_serialised) in
                    global_kernel_data_vec_serialised.iter().enumerate()
                {
                    let (mut kernel_data_i, mut rest) =
                        deserialise_nested_array_3x3::<<Scalar as AsComplex>::ComplexType>(
                            kernel_data_serialised,
                        );
                    while !rest.is_empty() {
                        let (mut t1, t2) = deserialise_nested_array_3x3::<
                            <Scalar as AsComplex>::ComplexType,
                        >(rest);
                        kernel_data_i.append(&mut t1);
                        rest = t2;
                    }
                    global_kernel_data_vec[i] = kernel_data_i;
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
                            .copy_from_slice(global_kernel_data_vec[i][j].data())
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

                // TODO: Get rid of this transpose
                // Transpose results for better cache locality in application
                let mut kernel_data_ft = Vec::new();
                for freq in 0..transform_size {
                    let frequency_offset = NSIBLINGS_SQUARED * freq;
                    for kernel_f in kernel_data_f.iter().take(NHALO) {
                        let k_f = &kernel_f
                            [frequency_offset..(frequency_offset + NSIBLINGS_SQUARED)]
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

                metadata_r.push(FftMetadata {
                    kernel_data,
                    kernel_data_f: kernel_data_ft,
                });
            }

            // Run on remaining levels
            {
                let iterator = if self.variable_expansion_order {
                    (3..=total_depth)
                        .zip(self.equivalent_surface_order.iter().cloned().skip(3))
                        .collect_vec()
                } else {
                    (3..=total_depth)
                        .zip(vec![
                            *self.equivalent_surface_order.last().unwrap();
                            (total_depth - 2) as usize
                        ])
                        .collect_vec()
                };

                for &(level, equivalent_surface_order) in &iterator {
                    let shape = <Scalar as Dft>::shape_in(equivalent_surface_order);
                    let transform_shape = <Scalar as Dft>::shape_out(equivalent_surface_order);
                    let transform_size = <Scalar as Dft>::size_out(equivalent_surface_order);

                    // Encode point in centre of domain and compute halo of parent, and their resp. children
                    let key = MortonKey::from_point(&point, domain, level);
                    let siblings = key.siblings();
                    let parent = key.parent();

                    let halo = parent.neighbors();
                    let halo_children = halo.iter().map(|h| h.children()).collect_vec();

                    // The child boxes in the halo of the sibling set
                    let mut sources = vec![];
                    // The sibling set
                    let mut targets = vec![];
                    // The transfer vectors corresponding to source->target translations
                    let mut transfer_vectors = vec![];
                    // Green's function evaluations for each source, target pair interaction
                    let mut kernel_data_vec_r = vec![];

                    for _ in &halo_children {
                        sources.push(vec![]);
                        targets.push(vec![]);
                        transfer_vectors.push(vec![]);
                        kernel_data_vec_r.push(vec![]);
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

                    let mut tasks = Vec::new();

                    // Iterate over each set of convolutions in the halo (26)
                    for i in 0..transfer_vectors.len() {
                        // Iterate over each unique convolution between sibling set, and halo siblings (64)
                        for j in 0..transfer_vectors[i].len() {
                            // Translating from sibling set to boxes in its M2L halo
                            let target = targets[i][j];
                            let source = sources[i][j];
                            let v_list: HashSet<MortonKey<_>> = target
                                .parent()
                                .neighbors()
                                .iter()
                                .flat_map(|pn| pn.children())
                                .filter(|pnc| !target.is_adjacent(pnc))
                                .collect();

                            if v_list.contains(source) {
                                tasks.push((i, j, true));
                            } else {
                                tasks.push((i, j, false));
                            }
                        }
                    }

                    let n_precomputations = tasks.len() as i32;

                    let (load_counts, load_displacement) =
                        calculate_precomputation_load(n_precomputations, size).unwrap();

                    // Compute mandated local portion
                    let local_load_count = load_counts[rank as usize];
                    let local_load_displacement = load_displacement[rank as usize];

                    let tasks_r = &tasks[(local_load_displacement as usize)
                        ..((local_load_displacement + local_load_count) as usize)];

                    for &(i, j, contains) in tasks_r {
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

                        if contains {
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
                            let mut kernel_hat = rlst_dynamic_array3!(
                                <Scalar as DftType>::OutputType,
                                transform_shape
                            );

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

                            kernel_data_vec_r[i].push(kernel_hat);
                        } else {
                            // Fill with zeros when interaction doesn't exist
                            let kernel_hat_zeros = rlst_dynamic_array3!(
                                <Scalar as DftType>::OutputType,
                                transform_shape
                            );
                            kernel_data_vec_r[i].push(kernel_hat_zeros);
                        }
                    }

                    // Serialise kernel data at this level
                    let mut kernel_data_vec_r_serialised = Vec::new();
                    for vec in kernel_data_vec_r.iter() {
                        kernel_data_vec_r_serialised.push(serialise_nested_array_3x3(vec))
                    }

                    // Communicate kernel data at this level
                    let mut global_kernel_data_vec_serialised = vec![Vec::new(); NHALO];
                    for (i, save_data) in global_kernel_data_vec_serialised
                        .iter_mut()
                        .take(NHALO)
                        .enumerate()
                    {
                        let mut global_kernel_data_serialised_i = all_gather_v_serialised(
                            &kernel_data_vec_r_serialised[i],
                            &self.communicator,
                        );
                        save_data.append(&mut global_kernel_data_serialised_i);
                    }

                    // Deserialise kernel data at this level
                    let mut global_kernel_data_vec = Vec::new();
                    for _ in 0..NHALO {
                        global_kernel_data_vec.push(vec![])
                    }

                    for (i, kernel_data_serialised) in
                        global_kernel_data_vec_serialised.iter().enumerate()
                    {
                        let (mut kernel_data_i, mut rest) =
                            deserialise_nested_array_3x3::<<Scalar as AsComplex>::ComplexType>(
                                kernel_data_serialised,
                            );
                        while !rest.is_empty() {
                            let (mut t1, t2) = deserialise_nested_array_3x3::<
                                <Scalar as AsComplex>::ComplexType,
                            >(rest);
                            kernel_data_i.append(&mut t1);
                            rest = t2;
                        }
                        global_kernel_data_vec[i] = kernel_data_i;
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
                                .copy_from_slice(global_kernel_data_vec[i][j].data())
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

                    // TODO: Get rid of this transpose
                    // Transpose results for better cache locality in application
                    let mut kernel_data_ft = Vec::new();
                    for freq in 0..transform_size {
                        let frequency_offset = NSIBLINGS_SQUARED * freq;
                        for kernel_f in kernel_data_f.iter().take(NHALO) {
                            let k_f = &kernel_f
                                [frequency_offset..(frequency_offset + NSIBLINGS_SQUARED)]
                                .to_vec();
                            let k_f_ =
                                rlst_array_from_slice2!(k_f.as_slice(), [NSIBLINGS, NSIBLINGS]);
                            let mut k_ft = rlst_dynamic_array2!(
                                <Scalar as DftType>::OutputType,
                                [NSIBLINGS, NSIBLINGS]
                            );
                            k_ft.fill_from(k_f_.r());
                            kernel_data_ft.push(k_ft.data().to_vec());
                        }
                    }

                    metadata_r.push(FftMetadata {
                        kernel_data,
                        kernel_data_f: kernel_data_ft,
                    });
                }
            }
        }

        // Set operator data
        self.source_to_target.metadata = metadata_r.clone();
        self.global_fmm.source_to_target.metadata = metadata_r.clone();

        // Compute and attach maps
        let iterator = if self.variable_expansion_order {
            self.equivalent_surface_order
                .iter()
                .skip(2)
                .cloned()
                .collect_vec()
        } else {
            self.equivalent_surface_order.clone()
        };

        let mut tmp1 = Vec::new();
        let mut tmp2 = Vec::new();
        for equivalent_surface_order in iterator {
            let (surf_to_conv_map, conv_to_surf_map) =
                Self::compute_surf_to_conv_map(equivalent_surface_order);
            tmp1.push(surf_to_conv_map);
            tmp2.push(conv_to_surf_map)
        }
        self.source_to_target.surf_to_conv_map = tmp1.clone();
        self.source_to_target.conv_to_surf_map = tmp2.clone();

        // Copy for global FMM
        self.global_fmm.source_to_target.surf_to_conv_map =
            self.source_to_target.surf_to_conv_map.clone();
        self.global_fmm.source_to_target.conv_to_surf_map =
            self.source_to_target.conv_to_surf_map.clone();
    }
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
        let start_level =
            start_level.unwrap_or_else(|| std::cmp::max(2, self.tree.source_tree().global_depth()));

        let tmp = HashSet::new();
        let target_tree_keys_set = self.tree.target_tree().all_keys_set().unwrap_or(&tmp);

        for level in start_level..=self.tree.source_tree().total_depth() {
            let sources = self.tree.source_tree().keys(level).unwrap();
            let n_sources = sources.len();
            let m2l_operator_index = self.m2l_operator_index(level);
            let sentinel = -1i32;

            let result = vec![vec![sentinel; n_sources]; 316];
            let result = result.into_iter().map(RwLock::new).collect_vec();

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
        let u_r_serialised = serialise_nested_array_2x2(&u_r);
        let global_u_serialised = all_gather_v_serialised(&u_r_serialised, &self.communicator);

        let vt_r_serialised = serialise_nested_array_2x2(&vt_r);
        let global_vt_serialised = all_gather_v_serialised(&vt_r_serialised, &self.communicator);

        let cutoff_rank_r_serialised = serialise_vec(&cutoff_ranks_r);
        let global_cutoff_rank_serialised =
            all_gather_v_serialised(&cutoff_rank_r_serialised, &self.communicator);

        let level_r_serialised = serialise_vec(&level_r);
        let global_level_serialised =
            all_gather_v_serialised(&level_r_serialised, &self.communicator);

        // Reconstruct metadata
        let (mut global_u, mut rest) = deserialise_nested_array_2x2::<Scalar>(&global_u_serialised);
        while !rest.is_empty() {
            let (mut t1, t2) = deserialise_nested_array_2x2::<Scalar>(rest);
            global_u.append(&mut t1);
            rest = t2;
        }

        let (mut global_vt, mut rest) =
            deserialise_nested_array_2x2::<Scalar>(&global_vt_serialised);
        while !rest.is_empty() {
            let (mut t1, t2) = deserialise_nested_array_2x2::<Scalar>(rest);
            global_vt.append(&mut t1);
            rest = t2;
        }

        let (global_cutoff_ranks, mut rest) =
            deserialise_vec::<i32>(&global_cutoff_rank_serialised);
        let mut global_cutoff_ranks = global_cutoff_ranks.to_vec();
        while !rest.is_empty() {
            let (t1, t2) = deserialise_vec::<i32>(rest);
            global_cutoff_ranks.append(&mut t1.to_vec());
            rest = t2;
        }

        let (global_levels, mut rest) = deserialise_vec::<u64>(&global_level_serialised);
        let mut global_levels = global_levels.to_vec();
        while !rest.is_empty() {
            let (t1, t2) = deserialise_vec::<u64>(rest);
            global_levels.append(&mut t1.to_vec());
            rest = t2;
        }

        // Reconstruct metadata
        let mut curr_idx = 0;
        let mut curr_level = global_levels[curr_idx];
        let mut metadata = vec![BlasMetadataIa::<Scalar>::default(); (total_depth - 1) as usize];
        let mut cutoff_ranks = vec![Vec::new(); (total_depth - 1) as usize];
        for (level, cutoff_rank, u, vt) in
            izip!(global_levels, global_cutoff_ranks, global_u, global_vt)
        {
            if level == curr_level {
                metadata[curr_idx].u.push(u);
                metadata[curr_idx].vt.push(vt);
                cutoff_ranks[curr_idx].push(cutoff_rank as usize);
            } else {
                curr_idx += 1;
                curr_level += 1;
                metadata[curr_idx].u.push(u);
                metadata[curr_idx].vt.push(vt);
                cutoff_ranks[curr_idx].push(cutoff_rank as usize);
            }
        }

        // Update metadata on global/local FMMs
        self.source_to_target.metadata = metadata.clone();
        self.global_fmm.source_to_target.metadata = metadata.clone();

        self.source_to_target.cutoff_ranks = cutoff_ranks.clone();
        self.global_fmm.source_to_target.cutoff_ranks = cutoff_ranks.clone();

        for level in 2..=total_depth {
            let transfer_vectors = compute_transfer_vectors_at_level(level).unwrap();
            self.source_to_target
                .transfer_vectors
                .push(transfer_vectors.clone());
            self.global_fmm
                .source_to_target
                .transfer_vectors
                .push(transfer_vectors.clone());
            self.ghost_fmm_v
                .source_to_target
                .transfer_vectors
                .push(transfer_vectors.clone());
        }
    }
}
