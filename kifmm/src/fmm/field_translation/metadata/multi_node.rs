use std::{collections::HashSet, sync::RwLock};

use green_kernels::{laplace_3d::Laplace3dKernel, traits::Kernel as KernelTrait, types::EvalType};
use itertools::Itertools;
use mpi::{
    topology::SimpleCommunicator,
    traits::{Collection, Communicator, Equivalence},
};
use num::{Float, Zero};
use pulp::Scalar;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rlst::{
    empty_array, rlst_array_from_slice2, rlst_dynamic_array2, rlst_dynamic_array3, Array,
    BaseArray, MatrixSvd, MultIntoResize, RawAccess, RawAccessMut, RlstScalar, VectorContainer,
};

use crate::{
    fmm::{
        helpers::{
            coordinate_index_pointer_multinode, flip3, homogenous_kernel_scale,
            leaf_expansion_pointers_multinode, leaf_surfaces, level_expansion_pointers,
            level_expansion_pointers_multinode, level_index_pointer, level_index_pointer_multinode,
            potential_pointers_multinode,
        },
        types::{FftFieldTranslationMultiNode, FftMetadata, NeighbourhoodCommunicator},
    },
    linalg::pinv::pinv,
    traits::{
        fftw::{Dft, DftType},
        field::SourceToTargetTranslationMetadata,
        fmm::{FmmMetadata, GhostExchange, MultiNodeFmm},
        general::AsComplex,
        tree::MultiNodeFmmTreeTrait,
    },
    tree::{
        constants::{NHALO, NSIBLINGS, NSIBLINGS_SQUARED},
        helpers::find_corners,
    },
    FftFieldTranslation,
};

use crate::{
    bindings::MortonKeys,
    fmm::{helpers::ncoeffs_kifmm, types::KiFmmMultiNode, KiFmm},
    linalg::pinv,
    traits::{
        field::{
            SourceAndTargetTranslationMetadata, SourceToTargetData as SourceToTargetDataTrait,
            SourceToTargetTranslationMultiNode,
        },
        fmm::{FmmOperatorData, HomogenousKernel, SourceToTargetTranslation},
        general::Epsilon,
        tree::{FmmTreeNode, SingleNodeFmmTreeTrait, SingleNodeTreeTrait},
    },
    tree::{
        constants::{ALPHA_INNER, ALPHA_OUTER},
        types::MortonKey,
    },
    MultiNodeFmmTree,
};

impl<Scalar, Kernel, SourceToTargetData> SourceToTargetTranslationMultiNode
    for KiFmmMultiNode<Scalar, Kernel, SourceToTargetData>
where
    Scalar: RlstScalar + Default + Equivalence + Float,
    Kernel: KernelTrait<T = Scalar> + HomogenousKernel + Default + Send + Sync,
    SourceToTargetData: SourceToTargetDataTrait + Send + Sync,
    <Scalar as RlstScalar>::Real: Default + Equivalence + Float,
    Self: SourceToTargetTranslation,
    MultiNodeFmmTree<Scalar, SimpleCommunicator>: Default,
{
    fn ranges(&mut self) {
        // All ranges for FMMs at this processor
        let mut ranges = Vec::new();

        for source_tree in self.tree.source_tree.trees.iter() {
            // Union of interaction lists for each FMM at this proc
            let mut interaction_lists = HashSet::new();

            for level in (2..=source_tree.depth) {
                let sources = source_tree.keys(level).unwrap();

                for source in sources.iter() {
                    let interaction_list = source
                        .parent()
                        .neighbors()
                        .iter()
                        .flat_map(|pn| pn.children())
                        .filter(|pnc| !source.is_adjacent(pnc))
                        .collect_vec();

                    for source in interaction_list.iter() {
                        interaction_lists.insert(*source);
                    }
                }
            }

            let mut interaction_lists = interaction_lists.into_iter().collect_vec();
            interaction_lists.sort();

            let range = (
                interaction_lists
                    .iter()
                    .min()
                    .unwrap()
                    .finest_first_child()
                    .morton,
                interaction_lists
                    .iter()
                    .max()
                    .unwrap()
                    .finest_last_child()
                    .morton,
            );
            ranges.push(range);
        }

        // self.ranges = ranges;
    }
}

impl<Scalar, SourceToTargetData> SourceAndTargetTranslationMetadata
    for KiFmmMultiNode<Scalar, Laplace3dKernel<Scalar>, SourceToTargetData>
where
    Scalar: RlstScalar + Default + Epsilon + MatrixSvd + Equivalence + Float,
    SourceToTargetData: SourceToTargetDataTrait + Send + Sync,
    <Scalar as RlstScalar>::Real: Default + Equivalence + Float,
    Self: SourceToTargetTranslation + MultiNodeFmm,
{
    fn source(&mut self) {
        let root = MortonKey::<Scalar::Real>::root(None);
        let alpha_outer = Scalar::from(ALPHA_OUTER).unwrap().re();
        let alpha_inner = Scalar::from(ALPHA_INNER).unwrap().re();
        let domain = self.tree.domain;

        let mut uc2e_inv_1 = Vec::new();
        let mut uc2e_inv_2 = Vec::new();

        let upward_equivalent_surface = root.surface_grid(
            self.equivalent_surface_order,
            &self.tree.domain,
            alpha_inner,
        );
        let upward_check_surface =
            root.surface_grid(self.check_surface_order, &self.tree.domain, alpha_outer);

        let nequiv_surface = ncoeffs_kifmm(self.equivalent_surface_order);
        let ncheck_surface = ncoeffs_kifmm(self.check_surface_order);

        let mut uc2e = rlst_dynamic_array2!(Scalar, [ncheck_surface, nequiv_surface]);
        self.kernel.assemble_st(
            EvalType::Value,
            &upward_check_surface[..],
            &upward_equivalent_surface[..],
            uc2e.data_mut(),
        );

        let (s, ut, v) = pinv(&uc2e, None, None).unwrap();

        let mut mat_s = rlst_dynamic_array2!(Scalar, [s.len(), s.len()]);
        for i in 0..s.len() {
            mat_s[[i, i]] = Scalar::from_real(s[i]);
        }

        uc2e_inv_1.push(empty_array::<Scalar, 2>().simple_mult_into_resize(v.view(), mat_s.view()));
        uc2e_inv_2.push(ut);

        let check_surface_order_parent = self.check_surface_order(0);
        let equivalent_surface_order_parent = self.equivalent_surface_order(0 as u64);
        let equivalent_surface_order_child = self.equivalent_surface_order(1 as u64);

        let parent_upward_check_surface =
            root.surface_grid(check_surface_order_parent, &domain, alpha_outer);

        let children = root.children();
        let ncheck_surface_parent = ncoeffs_kifmm(check_surface_order_parent);
        let nequiv_surface_child = ncoeffs_kifmm(equivalent_surface_order_child);
        let nequiv_surface_parent = ncoeffs_kifmm(equivalent_surface_order_parent);

        let mut m2m =
            rlst_dynamic_array2!(Scalar, [nequiv_surface_parent, 8 * nequiv_surface_child]);
        let mut m2m_vec = Vec::new();

        for (i, child) in children.iter().enumerate() {
            let child_upward_equivalent_surface =
                child.surface_grid(equivalent_surface_order_child, &domain, alpha_inner);

            let mut ce2pc =
                rlst_dynamic_array2!(Scalar, [ncheck_surface_parent, nequiv_surface_child]);

            self.kernel.assemble_st(
                EvalType::Value,
                &parent_upward_check_surface,
                &child_upward_equivalent_surface,
                ce2pc.data_mut(),
            );

            let tmp = empty_array::<Scalar, 2>().simple_mult_into_resize(
                uc2e_inv_1[0].view(),
                empty_array::<Scalar, 2>()
                    .simple_mult_into_resize(uc2e_inv_2[0].view(), ce2pc.view()),
            );

            let l = i * nequiv_surface_child * nequiv_surface_parent;
            let r = l + nequiv_surface_child * nequiv_surface_parent;

            m2m.data_mut()[l..r].copy_from_slice(tmp.data());
            m2m_vec.push(tmp);
        }

        self.source = vec![m2m];
        self.source_vec = vec![m2m_vec];
        self.uc2e_inv_1 = uc2e_inv_1;
        self.uc2e_inv_2 = uc2e_inv_2;
    }

    fn target(&mut self) {
        let root = MortonKey::<Scalar::Real>::root(None);
        let alpha_outer = Scalar::from(ALPHA_OUTER).unwrap().re();
        let alpha_inner = Scalar::from(ALPHA_INNER).unwrap().re();
        let domain = self.tree.domain;

        let mut l2l = Vec::new();
        let mut dc2e_inv_1 = Vec::new();
        let mut dc2e_inv_2 = Vec::new();

        let equivalent_surface_order = self.equivalent_surface_order;
        let check_surface_order = self.check_surface_order;

        // Compute required surfaces
        let downward_equivalent_surface =
            root.surface_grid(equivalent_surface_order, &domain, alpha_outer);
        let downward_check_surface = root.surface_grid(check_surface_order, &domain, alpha_inner);

        let nequiv_surface = ncoeffs_kifmm(equivalent_surface_order);
        let ncheck_surface = ncoeffs_kifmm(check_surface_order);

        // Assemble matrix of kernel evaluations between upward check to equivalent, and downward check to equivalent matrices
        // As well as estimating their inverses using SVD
        let mut dc2e = rlst_dynamic_array2!(Scalar, [ncheck_surface, nequiv_surface]);
        self.kernel.assemble_st(
            EvalType::Value,
            &downward_check_surface[..],
            &downward_equivalent_surface[..],
            dc2e.data_mut(),
        );

        let (s, ut, v) = pinv::<Scalar>(&dc2e, None, None).unwrap();

        let mut mat_s = rlst_dynamic_array2!(Scalar, [s.len(), s.len()]);
        for i in 0..s.len() {
            mat_s[[i, i]] = Scalar::from_real(s[i]);
        }

        dc2e_inv_1.push(empty_array::<Scalar, 2>().simple_mult_into_resize(v.view(), mat_s.view()));
        dc2e_inv_2.push(ut);

        let equivalent_surface_order_parent = self.equivalent_surface_order(0);
        let check_surface_order_child = self.check_surface_order(1);

        let parent_downward_equivalent_surface =
            root.surface_grid(equivalent_surface_order_parent, &domain, alpha_outer);

        let children = root.children();
        let ncheck_surface_child = ncoeffs_kifmm(check_surface_order_child);
        let nequiv_surface_parent = ncoeffs_kifmm(equivalent_surface_order_parent);

        for child in children.iter() {
            let child_downward_check_surface =
                child.surface_grid(check_surface_order_child, &domain, alpha_inner);

            let mut pe2cc =
                rlst_dynamic_array2!(Scalar, [ncheck_surface_child, nequiv_surface_parent]);

            self.kernel.assemble_st(
                EvalType::Value,
                &child_downward_check_surface,
                &parent_downward_equivalent_surface,
                pe2cc.data_mut(),
            );

            let mut tmp = empty_array::<Scalar, 2>().simple_mult_into_resize(
                dc2e_inv_1[0].view(),
                empty_array::<Scalar, 2>()
                    .simple_mult_into_resize(dc2e_inv_2[0].view(), pe2cc.view()),
            );

            tmp.data_mut()
                .iter_mut()
                .for_each(|d| *d *= homogenous_kernel_scale(child.level()));

            l2l.push(tmp);
        }

        self.target_vec = vec![l2l];
        self.dc2e_inv_1 = dc2e_inv_1;
        self.dc2e_inv_2 = dc2e_inv_2;
    }
}

impl<Scalar> SourceToTargetTranslationMetadata
    for KiFmmMultiNode<Scalar, Laplace3dKernel<Scalar>, FftFieldTranslationMultiNode<Scalar>>
where
    Scalar: RlstScalar
        + Equivalence
        + AsComplex
        + Default
        + Float
        + Dft<InputType = Scalar, OutputType = <Scalar as AsComplex>::ComplexType>,
    <Scalar as RlstScalar>::Real: RlstScalar + Default + Equivalence + Float,
{
    // Must be run AFTER multipole exchange.
    fn displacements(&mut self) {
        // let mut displacements = Vec::new();
        // let depth = self.tree.source_tree.local_depth + self.tree.target_tree.global_depth;

        // for fmm_index in 0..self.nfmms {
        //     let mut tmp = Vec::new();

        //     for level in 2..=depth {
        //         let targets = self.tree.target_tree.trees[fmm_index].keys(level).unwrap();
        //         let targets_parents: HashSet<MortonKey<_>> =
        //             targets.iter().map(|target| target.parent()).collect();
        //         let mut targets_parents = targets_parents.into_iter().collect_vec();
        //         targets_parents.sort();
        //         let ntargets_parents = targets_parents.len();

        //         let sources = self.tree.source_tree.trees[fmm_index].keys(level).unwrap();

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
        //                     if let Some(neighbor_displacement) = self.level_index_pointer_multipoles
        //                         [fmm_index][level as usize]
        //                         .get(&first_child)
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

        //         tmp.push(result);
        //     }
        //     displacements.push(tmp);
        // }
        // self.source_to_target.displacements = displacements;
    }

    fn source_to_target(&mut self) {
        let dim = 3;
        // Pick a point in the middle of the domain
        let two = Scalar::real(2.0);
        let midway = &self
            .tree
            .domain
            .side_length
            .iter()
            .map(|d| *d / two)
            .collect_vec();

        let point = midway
            .iter()
            .zip(&self.tree.domain.origin)
            .map(|(m, o)| *m + *o)
            .collect_vec();

        let point = [point[0], point[1], point[2]];

        // Encode point in centre of domain and compute halo of parent, and their resp. children
        let key = MortonKey::from_point(&point, &self.tree.domain, 3, None);
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
        let mut kernel_data_vec = vec![];

        for _ in &halo_children {
            sources.push(vec![]);
            targets.push(vec![]);
            transfer_vectors.push(vec![]);
            kernel_data_vec.push(vec![]);
        }

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

        let equivalent_surface_order = self.equivalent_surface_order;
        let shape = <Scalar as Dft>::shape_in(equivalent_surface_order);
        let transform_shape = <Scalar as Dft>::shape_out(equivalent_surface_order);
        let transform_size = <Scalar as Dft>::size_out(equivalent_surface_order);
        let alpha = Scalar::real(ALPHA_INNER);

        for i in 0..transfer_vectors.len() {
            for j in 0..transfer_vectors[i].len() {
                let target = targets[i][j];
                let source = sources[i][j];

                let source_equivalent_surface =
                    source.surface_grid(equivalent_surface_order, &self.tree.domain, alpha);

                let target_check_surface =
                    target.surface_grid(equivalent_surface_order, &self.tree.domain, alpha);

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
                        corners[dim * conv_point_corner_index],
                        corners[dim * conv_point_corner_index + 1],
                        corners[dim * conv_point_corner_index + 2],
                    ];

                    let (conv_grid, _) = source.convolution_grid(
                        equivalent_surface_order,
                        &self.tree.domain,
                        alpha,
                        &conv_point_corner,
                        conv_point_corner_index,
                    );

                    let kernel_point_index = 0;
                    let kernel_point = [
                        target_check_surface[dim * kernel_point_index],
                        target_check_surface[dim * kernel_point_index + 1],
                        target_check_surface[dim * kernel_point_index + 2],
                    ];

                    let mut kernel = flip3(&self.evaluate_greens_fct_convolution_grid(
                        equivalent_surface_order,
                        &conv_grid,
                        kernel_point,
                    ));

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
        let mut kernel_data =
            vec![
                vec![<Scalar as DftType>::OutputType::zero(); NSIBLINGS_SQUARED * transform_size];
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

        let mut kernel_data_ft = Vec::new();
        for freq in 0..transform_size {
            let frequency_offset = NSIBLINGS_SQUARED * freq;
            for kernel_f in kernel_data_f.iter().take(NHALO) {
                let k_f =
                    &kernel_f[frequency_offset..(frequency_offset + NSIBLINGS_SQUARED)].to_vec();
                let k_f_ = rlst_array_from_slice2!(k_f.as_slice(), [NSIBLINGS, NSIBLINGS]);
                let mut k_ft =
                    rlst_dynamic_array2!(<Scalar as DftType>::OutputType, [NSIBLINGS, NSIBLINGS]);
                k_ft.fill_from(k_f_.view());
                kernel_data_ft.push(k_ft.data().to_vec());
            }
        }

        let metadata = FftMetadata {
            kernel_data,
            kernel_data_f: kernel_data_ft,
        };

        self.source_to_target.metadata.push(metadata);

        // Set required maps
        let (surf_to_conv_map, conv_to_surf_map) =
            Self::compute_surf_to_conv_map(equivalent_surface_order);
        self.source_to_target.surf_to_conv_map = vec![surf_to_conv_map];
        self.source_to_target.conv_to_surf_map = vec![conv_to_surf_map];
    }
}

impl<Scalar> KiFmmMultiNode<Scalar, Laplace3dKernel<Scalar>, FftFieldTranslationMultiNode<Scalar>>
where
    Scalar: RlstScalar + AsComplex + Default + Dft + Equivalence + Float,
    <Scalar as RlstScalar>::Real: Default + Equivalence,
{
    pub fn evaluate_greens_fct_convolution_grid(
        &self,
        expansion_order: usize,
        convolution_grid: &[Scalar::Real],
        target_pt: [Scalar::Real; 3],
    ) -> Array<Scalar, BaseArray<Scalar, VectorContainer<Scalar>, 3>, 3> {
        let n = 2 * expansion_order - 1; // size of convolution grid
        let npad = n + 1; // padded size
        let nconv = n.pow(3); // length of buffer storing values on convolution grid

        let mut result = rlst_dynamic_array3!(Scalar, [npad, npad, npad]);

        let mut kernel_evals = vec![Scalar::zero(); nconv];
        self.kernel.assemble_st(
            EvalType::Value,
            convolution_grid,
            &target_pt,
            &mut kernel_evals[..],
        );

        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let conv_idx = i + j * n + k * n * n;
                    let save_idx = i + j * npad + k * npad * npad;
                    result.data_mut()[save_idx..(save_idx + 1)]
                        .copy_from_slice(&kernel_evals[(conv_idx)..(conv_idx + 1)]);
                }
            }
        }

        result
    }

    /// Place charge data on the convolution grid.
    ///
    /// # Arguments
    /// * `expansion_order` - The expansion order of the FMM
    /// * `charges` - A vector of charges.
    pub fn evaluate_charges_convolution_grid(
        &self,
        expansion_order: usize,
        expansion_order_index: usize,
        charges: &[Scalar],
    ) -> Array<Scalar, BaseArray<Scalar, VectorContainer<Scalar>, 3>, 3> {
        let n = 2 * expansion_order - 1;
        let npad = n + 1;
        let mut result = rlst_dynamic_array3!(Scalar, [npad, npad, npad]);
        for (i, &j) in self.source_to_target.surf_to_conv_map[expansion_order_index]
            .iter()
            .enumerate()
        {
            result.data_mut()[j] = charges[i];
        }

        result
    }

    /// Compute map between convolution grid indices and surface indices, return mapping and inverse mapping.
    ///
    /// # Arguments
    /// * `expansion_order` - The expansion order of the FMM
    pub fn compute_surf_to_conv_map(expansion_order: usize) -> (Vec<usize>, Vec<usize>) {
        // Number of points along each axis of convolution grid
        let n = 2 * expansion_order - 1;
        let npad = n + 1;

        let nsurf_grid = 6 * (expansion_order - 1).pow(2) + 2;

        // Index maps between surface and convolution grids
        let mut surf_to_conv = vec![0usize; nsurf_grid];
        let mut conv_to_surf = vec![0usize; nsurf_grid];

        // Initialise surface grid index
        let mut surf_index = 0;

        // The boundaries of the surface grid when embedded within the convolution grid
        let lower = expansion_order;
        let upper = 2 * expansion_order - 1;

        for k in 0..npad {
            for j in 0..npad {
                for i in 0..npad {
                    let conv_index = i + npad * j + npad * npad * k;
                    if (i >= lower && j >= lower && (k == lower || k == upper))
                        || (j >= lower && k >= lower && (i == lower || i == upper))
                        || (k >= lower && i >= lower && (j == lower || j == upper))
                    {
                        surf_to_conv[surf_index] = conv_index;
                        surf_index += 1;
                    }
                }
            }
        }

        let lower = 0;
        let upper = expansion_order - 1;
        let mut surf_index = 0;

        for k in 0..npad {
            for j in 0..npad {
                for i in 0..npad {
                    let conv_index = i + npad * j + npad * npad * k;
                    if (i <= upper && j <= upper && (k == lower || k == upper))
                        || (j <= upper && k <= upper && (i == lower || i == upper))
                        || (k <= upper && i <= upper && (j == lower || j == upper))
                    {
                        conv_to_surf[surf_index] = conv_index;
                        surf_index += 1;
                    }
                }
            }
        }

        (surf_to_conv, conv_to_surf)
    }
}

impl<Scalar, SourceToTargetData> FmmOperatorData
    for KiFmmMultiNode<Scalar, Laplace3dKernel<Scalar>, SourceToTargetData>
where
    Scalar: RlstScalar + Default + Equivalence + Float,
    SourceToTargetData: SourceToTargetDataTrait + Send + Sync,
    <Scalar as RlstScalar>::Real: Default + Float + Equivalence,
{
    fn fft_map_index(&self, level: u64) -> usize {
        0
    }

    fn expansion_index(&self, level: u64) -> usize {
        0
    }

    fn c2e_operator_index(&self, level: u64) -> usize {
        0
    }

    fn m2m_operator_index(&self, level: u64) -> usize {
        0
    }

    fn l2l_operator_index(&self, level: u64) -> usize {
        0
    }

    fn m2l_operator_index(&self, level: u64) -> usize {
        0
    }

    fn displacement_index(&self, level: u64) -> usize {
        0
    }
}

impl<Scalar, Kernel, SourceToTargetData> FmmMetadata
    for KiFmmMultiNode<Scalar, Kernel, SourceToTargetData>
where
    Scalar: RlstScalar + Default + Float + Equivalence,
    Kernel: KernelTrait<T = Scalar> + HomogenousKernel + Default + Send + Sync,
    SourceToTargetData: SourceToTargetDataTrait + Send + Sync,
    <Scalar as RlstScalar>::Real: Default + Float + Equivalence,
    Self: GhostExchange,
{
    type Scalar = Scalar;
    type Charges = Vec<Self::Scalar>;

    fn metadata<'a>(&mut self, eval_type: EvalType, charges: &'a [Self::Charges]) {
        // In a multinode setting this method sets the required metdata for the local upward passes, before ghost exchange.

        let alpha_outer = Scalar::real(ALPHA_OUTER);
        let alpha_inner = Scalar::real(ALPHA_INNER);

        // Check if computing potentials, or potentials and derivatives
        match eval_type {
            EvalType::Value => {}
            EvalType::ValueDeriv => {
                panic!("Only potential computation supported for now")
            }
        }
        let eval_size = match eval_type {
            EvalType::Value => 1,
            EvalType::ValueDeriv => 4,
        };

        let nsource_trees = self.nsource_trees;
        let ntarget_trees = self.ntarget_trees;
        let mut ntarget_points = 0;
        let mut nsource_points = 0;
        let mut nsource_keys = 0;
        let mut ntarget_keys = 0;
        let mut ntarget_leaves = 0;
        let mut nsource_leaves = 0;

        // Allocate buffers to store multipole and local data
        let mut multipoles = Vec::new();
        let mut locals = Vec::new();

        // Allocate buffer to store potential data at target points
        let mut potentials = Vec::new();

        for fmm_idx in 0..nsource_trees {
            let nsource_keys = self.tree.source_tree.trees[fmm_idx].n_keys_tot().unwrap();
            multipoles.push(vec![
                Scalar::default();
                nsource_keys * self.ncoeffs_equivalent_surface
            ]);
        }

        for fmm_idx in 0..ntarget_trees {
            let ntarget_keys = self.tree.target_tree.trees[fmm_idx].n_keys_tot().unwrap();
            let ntarget_points = self.tree.target_tree.trees[fmm_idx]
                .n_coordinates_tot()
                .unwrap();
            potentials.push(vec![Scalar::default(); ntarget_points * eval_size]);
            locals.push(vec![
                Scalar::default();
                ntarget_keys * self.ncoeffs_equivalent_surface
            ]);
        }

        // Index pointer of multipole and local data, indexed by fmm index, then by level
        let mut level_index_pointer_multipoles = level_index_pointer_multinode(
            &self.tree.source_tree.trees,
            self.tree.source_tree.local_depth,
            self.tree.source_tree.global_depth,
        );
        let mut level_index_pointer_locals = level_index_pointer_multinode(
            &self.tree.target_tree.trees,
            self.tree.target_tree.local_depth,
            self.tree.source_tree.global_depth,
        );

        let mut leaf_upward_equivalent_surfaces_sources = Vec::new();
        let mut leaf_upward_check_surfaces_sources = Vec::new();
        let mut leaf_downward_equivalent_surfaces_targets = Vec::new();

        // Precompute surfaces
        for fmm_idx in 0..nsource_trees {
            let source_tree = &self.tree.source_tree.trees[fmm_idx];

            let leaf_upward_equivalent_surfaces_sources_i = leaf_surfaces(
                source_tree,
                self.ncoeffs_equivalent_surface,
                alpha_inner,
                self.equivalent_surface_order,
            );

            let leaf_upward_check_surfaces_sources_i = leaf_surfaces(
                source_tree,
                self.ncoeffs_check_surface,
                alpha_outer,
                self.check_surface_order,
            );

            leaf_upward_equivalent_surfaces_sources.push(leaf_upward_equivalent_surfaces_sources_i);
            leaf_upward_check_surfaces_sources.push(leaf_upward_check_surfaces_sources_i);
        }

        for fmm_idx in 0..ntarget_trees {
            let target_tree = &self.tree.target_tree.trees[fmm_idx];
            let leaf_downward_equivalent_surfaces_targets_i = leaf_surfaces(
                target_tree,
                self.ncoeffs_equivalent_surface,
                alpha_outer,
                self.equivalent_surface_order,
            );

            leaf_downward_equivalent_surfaces_targets
                .push(leaf_downward_equivalent_surfaces_targets_i);
        }

        // Create mutalbe pointers to multipole and local data, indexed by fmm index, then by level.
        let level_multipoles = level_expansion_pointers_multinode(
            &self.tree.source_tree.trees,
            self.ncoeffs_equivalent_surface,
            &multipoles,
            self.tree.source_tree.local_depth,
            self.tree.source_tree.global_depth,
        );

        let level_locals = level_expansion_pointers_multinode(
            &self.tree.target_tree.trees,
            self.ncoeffs_equivalent_surface,
            &locals,
            self.tree.target_tree.local_depth,
            self.tree.target_tree.global_depth,
        );

        // Create mutable pointers to multipole and local data only at leaf level
        let leaf_multipoles = leaf_expansion_pointers_multinode(
            &self.tree.source_tree.trees,
            self.ncoeffs_equivalent_surface,
            &multipoles,
            self.tree.source_tree.local_depth,
            self.tree.source_tree.global_depth,
        );

        let leaf_locals = leaf_expansion_pointers_multinode(
            &self.tree.target_tree.trees,
            self.ncoeffs_equivalent_surface,
            &locals,
            self.tree.target_tree.local_depth,
            self.tree.target_tree.global_depth,
        );

        // Mutable pointers to potential data at each target leaf
        let potentials_send_pointers = potential_pointers_multinode(
            &self.tree.target_tree.trees,
            self.kernel_eval_size,
            &potentials,
        );

        // TODO: Replace with real charge distribution
        let mut charges = Vec::new();

        for fmm_idx in 0..nsource_trees {
            let nsource_points = self.tree.source_tree.trees[fmm_idx]
                .n_coordinates_tot()
                .unwrap();
            charges.push(vec![Scalar::one(); nsource_points]);
        }

        // TODO: real coordinate index pointers
        let charge_index_pointer_targets =
            coordinate_index_pointer_multinode(&self.tree.target_tree.trees);
        let charge_index_pointer_sources =
            coordinate_index_pointer_multinode(&self.tree.source_tree.trees);

        // New: Need to figure out which multipole data needs to be queried for and isn't contained in source
        // trees locally, local trees ideally need a tree ID, which associates them with a local and global rank.

        let mut locally_owned_domains = HashSet::new();
        for tree in self.tree.source_tree.trees.iter() {
            locally_owned_domains.insert(tree.root);
        }

        // Compute ranges on all processors
        self.set_layout();

        // Defines all non-locally contained multipole data, as well as established possibility of existence
        //  as defined by local roots,
        // Don't necessarily know if they contain data and therefore exist.
        let mut v_list_queries = HashSet::new();

        for target_tree in self.tree.target_tree.trees.iter() {
            for level in self.tree.target_tree.global_depth
                ..=(self.tree.target_tree.local_depth + self.tree.target_tree.global_depth)
            {
                if let Some(keys) = target_tree.keys(level) {
                    for key in keys.iter() {
                        // Compute interaction list
                        let interaction_list = key
                            .parent()
                            .neighbors()
                            .iter()
                            .flat_map(|pn| pn.children())
                            .filter(|pnc| !key.is_adjacent(pnc))
                            .collect_vec();

                        // Filter for those contained on foreign ranks
                        let interaction_list = interaction_list
                            .iter()
                            .filter_map(|key| {
                                // Try to get the rank from the key
                                if let Some(rank) = self.layout.rank_from_key(key) {
                                    // Filter out if the rank is equal to my_rank
                                    if rank != &self.rank {
                                        return Some(key);
                                    }
                                }
                                None
                            })
                            .collect_vec();

                        for key in interaction_list {
                            v_list_queries.insert(*key);
                        }
                    }
                }
            }
        }

        let v_list_queries = v_list_queries.into_iter().collect_vec();
        let mut v_list_ranks = Vec::new();
        let mut v_list_send_counts = vec![0i32; self.communicator.size() as usize];
        let mut v_list_to_send = vec![0i32; self.communicator.size() as usize];

        for query in v_list_queries.iter() {
            let rank = self.layout.rank_from_key(query).unwrap();
            v_list_ranks.push(*rank);
            v_list_send_counts[*rank as usize] += 1;
        }

        for (i, &value) in v_list_send_counts.iter().enumerate() {
            if value > 0 {
                v_list_to_send[i] = 1;
            }
        }

        // Sort queries by rank
        let v_list_queries = {
            let mut indices = (0..v_list_queries.len()).collect_vec();
            indices.sort_by_key(|&i| v_list_ranks[i]);

            let mut sorted_v_list_queries_t = Vec::with_capacity(v_list_queries.len());
            for i in indices {
                sorted_v_list_queries_t.push(v_list_queries[i].morton)
            }

            sorted_v_list_queries_t
        };

        self.neighbourhood_communicator_v =
            NeighbourhoodCommunicator::new(&self.communicator, &v_list_to_send);

        self.v_list_queries = v_list_queries;
        self.v_list_ranks = v_list_ranks;
        self.v_list_send_counts = v_list_send_counts;
        self.v_list_to_send = v_list_to_send;

        let mut u_list_queries = HashSet::new();
        for target_tree in self.tree.target_tree.trees.iter() {
            for leaf in target_tree.leaves.iter() {
                // Compute interaction list
                let interaction_list = leaf.neighbors();

                // Filter for those contained on foreign ranks
                let interaction_list = interaction_list
                    .iter()
                    .filter_map(|key| {
                        // Try to get the rank from the key
                        if let Some(rank) = self.layout.rank_from_key(key) {
                            // Filter out if the rank is equal to my_rank
                            if rank != &self.rank {
                                return Some(key);
                            }
                        }
                        None
                    })
                    .collect_vec();

                for key in interaction_list {
                    u_list_queries.insert(*key);
                }
            }
        }

        let u_list_queries = u_list_queries.into_iter().collect_vec();
        let mut u_list_ranks = Vec::new();
        let mut u_list_send_counts = vec![0i32; self.communicator.size() as usize];
        let mut u_list_to_send = vec![0i32; self.communicator.size() as usize];

        for query in u_list_queries.iter() {
            let rank = self.layout.rank_from_key(query).unwrap();
            u_list_ranks.push(*rank);
            u_list_send_counts[*rank as usize] += 1;
        }
        for (i, &value) in u_list_send_counts.iter().enumerate() {
            if value > 0 {
                u_list_to_send[i] = 1;
            }
        }
        // Sort queries by rank
        let u_list_queries = {
            let mut indices = (0..u_list_queries.len()).collect_vec();
            indices.sort_by_key(|&i| u_list_ranks[i]);

            let mut sorted_u_list_queries_t = Vec::with_capacity(u_list_queries.len());
            for i in indices {
                sorted_u_list_queries_t.push(u_list_queries[i].morton)
            }

            sorted_u_list_queries_t
        };

        self.neighbourhood_communicator_u =
            NeighbourhoodCommunicator::new(&self.communicator, &u_list_to_send);
        self.u_list_queries = u_list_queries;
        self.u_list_ranks = u_list_ranks;
        self.u_list_send_counts = u_list_send_counts;
        self.u_list_to_send = u_list_to_send;
        self.multipoles = multipoles;
        self.locals = locals;
        self.leaf_multipoles = leaf_multipoles;
        self.level_multipoles = level_multipoles;
        self.leaf_locals = leaf_locals;
        self.level_locals = level_locals;
        self.level_index_pointer_locals = level_index_pointer_locals;
        self.level_index_pointer_multipoles = level_index_pointer_multipoles;
        self.potentials = potentials;
        self.potentials_send_pointers = potentials_send_pointers;
        self.leaf_upward_equivalent_surfaces_sources = leaf_upward_equivalent_surfaces_sources;
        self.leaf_upward_check_surfaces_sources = leaf_upward_check_surfaces_sources;
        self.leaf_downward_equivalent_surfaces_targets = leaf_downward_equivalent_surfaces_targets;
        self.charges = charges.to_vec();
        self.charge_index_pointers_sources = charge_index_pointer_targets;
        self.charge_index_pointers_sources = charge_index_pointer_sources;
        self.kernel_eval_size = eval_size;

        // At this point I can exchange charge data for particle query packet
        // self.u_list_exchange();
    }
}
