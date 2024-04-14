//! Implementation of traits to compute metadata for field translation operations.
use std::collections::HashSet;

use green_kernels::{traits::Kernel as KernelTrait, types::EvalType};
use itertools::Itertools;
use num::{Float, Zero};
use rlst::{
    empty_array, rlst_array_from_slice2, rlst_dynamic_array2, rlst_dynamic_array3, Array,
    BaseArray, Gemm, MatrixSvd, MultIntoResize, RawAccess, RawAccessMut, RlstScalar, Shape,
    SvdMode, UnsafeRandomAccessByRef, UnsafeRandomAccessMut, VectorContainer,
};

use crate::{
    new_fmm::{
        field_translation::source_to_target::transfer_vector::compute_transfer_vectors,
        helpers::{flip3, ncoeffs_kifmm},
        types::{BlasFieldTranslation, BlasMetadata, FftFieldTranslation, FftMetadata},
    },
    traits::{
        fftw::{Dft, DftType},
        field::{ConfigureSourceToTargetData, SourceToTargetData},
        general::{AsComplex, Epsilon},
        tree::FmmTreeNode,
    },
    tree::{
        constants::{
            ALPHA_INNER, NCORNERS, NHALO, NSIBLINGS, NSIBLINGS_SQUARED, NTRANSFER_VECTORS_KIFMM,
        },
        helpers::find_corners,
        types::{Domain, MortonKey},
    },
};

fn find_cutoff_rank<T: Float + RlstScalar + Gemm>(singular_values: &[T], threshold: T) -> usize {
    for (i, &s) in singular_values.iter().enumerate() {
        if s <= threshold {
            return i;
        }
    }

    singular_values.len() - 1
}

impl<Scalar, Kernel> FftFieldTranslation<Scalar, Kernel>
where
    Scalar: RlstScalar + AsComplex + Dft + Default,
    <Scalar as RlstScalar>::Real: RlstScalar + Default,
    Kernel: KernelTrait<T = Scalar> + Default + Send + Sync,
{
    /// Constructor for FFT based field translations
    pub fn new() -> Self {
        Self {
            transfer_vectors: compute_transfer_vectors(),
            ..Default::default()
        }
    }

    /// Computes the unique kernel evaluations and places them on a convolution grid on the source box wrt to a given target point on the target box surface grid.
    ///
    /// # Arguments
    /// * `expansion_order` - The expansion order of the FMM
    /// * `convolution_grid` - Cartesian coordinates of points on the convolution grid at a source box, expected in column major order.
    /// * `target_pt` - The point on the target box's surface grid, with which kernels are being evaluated with respect to.
    pub fn compute_kernel(
        &self,
        expansion_order: usize,
        convolution_grid: &[Scalar::Real],
        target_pt: [Scalar::Real; 3],
    ) -> Array<Scalar, BaseArray<Scalar, VectorContainer<Scalar>, 3>, 3> {
        let n = 2 * expansion_order - 1;
        let npad = n + 1;

        let mut result = rlst_dynamic_array3!(Scalar, [npad, npad, npad]);

        let nconv = n.pow(3);
        let mut kernel_evals = vec![Scalar::zero(); nconv];
        self.kernel.assemble_st(
            EvalType::Value,
            convolution_grid,
            &target_pt[..],
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
    pub fn compute_signal(
        &self,
        expansion_order: usize,
        charges: &[Scalar],
    ) -> Array<Scalar, BaseArray<Scalar, VectorContainer<Scalar>, 3>, 3> {
        let n = 2 * expansion_order - 1;
        let npad = n + 1;

        let mut result = rlst_dynamic_array3!(Scalar, [npad, npad, npad]);

        for (i, &j) in self.surf_to_conv_map.iter().enumerate() {
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

impl<Scalar, Kernel> SourceToTargetData for FftFieldTranslation<Scalar, Kernel>
where
    Scalar: RlstScalar + AsComplex + Default + Dft,
    <Scalar as RlstScalar>::Real: RlstScalar + Default,
    Kernel: KernelTrait<T = Scalar> + Default + Send + Sync,
{
    type Metadata = FftMetadata<<Scalar as AsComplex>::ComplexType>;
}

impl<Scalar, Kernel> ConfigureSourceToTargetData for FftFieldTranslation<Scalar, Kernel>
where
    Scalar: RlstScalar
        + AsComplex
        + Default
        + Dft<InputType = Scalar, OutputType = <Scalar as AsComplex>::ComplexType>,
    <Scalar as RlstScalar>::Real: RlstScalar + Default,
    Kernel: KernelTrait<T = Scalar> + Default + Send + Sync,
{
    type Scalar = Scalar;
    type Domain = Domain<Scalar::Real>;
    type Kernel = Kernel;

    fn expansion_order(&mut self, expansion_order: usize) {
        self.expansion_order = expansion_order
    }

    fn kernel(&mut self, kernel: Self::Kernel) {
        self.kernel = kernel
    }

    fn operator_data(&mut self, expansion_order: usize, domain: Self::Domain) {
        // Parameters related to the FFT and Tree
        // let m = 2 * expansion_order - 1; // Size of each dimension of 3D kernel/signal
        // let pad_size = 1;
        // let p = m + pad_size; // Size of each dimension of padded 3D kernel/signal
        // let transform_size = p * p * (p / 2 + 1); // Number of Fourier coefficients when working with real data
        let shape = <Self::Scalar as Dft>::shape_in(expansion_order);
        let transform_shape = <Self::Scalar as Dft>::shape_out(expansion_order);
        let transform_size = <Self::Scalar as Dft>::size_out(expansion_order);

        // Pick a point in the middle of the domain
        let two = Scalar::real(2.0);
        let midway = domain.side_length.iter().map(|d| *d / two).collect_vec();
        let point = midway
            .iter()
            .zip(domain.origin)
            .map(|(m, o)| *m + o)
            .collect_vec();
        let point = [point[0], point[1], point[2]];

        // Encode point in centre of domain and compute halo of parent, and their resp. children
        let key = MortonKey::from_point(&point, &domain, 3);
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

        let n_source_equivalent_surface = 6 * (expansion_order - 1).pow(2) + 2;
        let n_target_check_surface = n_source_equivalent_surface;
        let alpha = Scalar::real(ALPHA_INNER);

        // Iterate over each set of convolutions in the halo (26)
        for i in 0..transfer_vectors.len() {
            // Iterate over each unique convolution between sibling set, and halo siblings (64)
            for j in 0..transfer_vectors[i].len() {
                // Translating from sibling set to boxes in its M2L halo
                let target = targets[i][j];
                let source = sources[i][j];

                let source_equivalent_surface =
                    source.surface_grid(expansion_order, &domain, alpha);
                let target_check_surface = target.surface_grid(expansion_order, &domain, alpha);

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
                        corners[conv_point_corner_index],
                        corners[NCORNERS + conv_point_corner_index],
                        corners[2 * NCORNERS + conv_point_corner_index],
                    ];

                    let (conv_grid, _) = source.convolution_grid(
                        expansion_order,
                        &domain,
                        alpha,
                        &conv_point_corner,
                        conv_point_corner_index,
                    );

                    // Calculate Green's fct evaluations with respect to a 'kernel point' on the target box
                    let kernel_point_index = 0;
                    let kernel_point = [
                        target_check_surface[kernel_point_index],
                        target_check_surface[n_target_check_surface + kernel_point_index],
                        target_check_surface[2 * n_target_check_surface + kernel_point_index],
                    ];

                    // Compute Green's fct evaluations
                    let mut kernel =
                        flip3(&self.compute_kernel(expansion_order, &conv_grid, kernel_point));

                    // Compute FFT of padded kernel
                    let mut kernel_hat =
                        rlst_dynamic_array3!(<Scalar as DftType>::OutputType, transform_shape);

                    // TODO: is kernel_hat the transpose of what it used to be?
                    let _ = Scalar::forward_dft(kernel.data_mut(), kernel_hat.data_mut(), &shape);

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

        // Transpose results for better cache locality in application
        let mut kernel_data_ft = Vec::new();
        for freq in 0..transform_size {
            let frequency_offset = NSIBLINGS_SQUARED * freq;
            for kernel_f in kernel_data_f.iter().take(NHALO) {
                let k_f =
                    &kernel_f[frequency_offset..(frequency_offset + NSIBLINGS_SQUARED)].to_vec();
                let k_f_ = rlst_array_from_slice2!(k_f.as_slice(), [NSIBLINGS, NSIBLINGS]);
                let mut k_ft =
                    rlst_dynamic_array2!(<Scalar as DftType>::OutputType, [NSIBLINGS, NSIBLINGS]);
                k_ft.fill_from(k_f_.view().transpose());
                kernel_data_ft.push(k_ft.data().to_vec());
            }
        }

        let result = FftMetadata {
            kernel_data,
            kernel_data_f: kernel_data_ft,
        };

        // Set operator data
        self.metadata = result;

        // Set required maps, TODO: Should be a part of operator data
        (self.surf_to_conv_map, self.conv_to_surf_map) =
            FftFieldTranslation::<Scalar, Kernel>::compute_surf_to_conv_map(self.expansion_order);
    }
}

impl<Scalar, Kernel> BlasFieldTranslation<Scalar, Kernel>
where
    Scalar: RlstScalar + Epsilon + Default,
    <Scalar as RlstScalar>::Real: Default,
    Kernel: KernelTrait<T = Scalar> + Default,
{
    /// Constructor for BLAS based field translations, specify a compression threshold for the SVD compressed operators
    /// TODO: More docs
    pub fn new(threshold: Option<Scalar::Real>) -> Self {
        let tmp = Scalar::real(4) * Scalar::epsilon().re();

        Self {
            threshold: threshold.unwrap_or(tmp),
            transfer_vectors: compute_transfer_vectors(),
            ..Default::default()
        }
    }
}

impl<Scalar, Kernel> SourceToTargetData for BlasFieldTranslation<Scalar, Kernel>
where
    Scalar: RlstScalar,
    Kernel: KernelTrait<T = Scalar> + Default,
{
    type Metadata = BlasMetadata<Scalar>;
}

impl<Scalar, Kern> ConfigureSourceToTargetData for BlasFieldTranslation<Scalar, Kern>
where
    Scalar: RlstScalar,
    Kern: KernelTrait<T = Scalar> + Default,
    Array<Scalar, BaseArray<Scalar, VectorContainer<Scalar>, 2>, 2>: MatrixSvd<Item = Scalar>,
{
    type Scalar = Scalar;
    type Domain = Domain<Scalar::Real>;
    type Kernel = Kern;

    fn expansion_order(&mut self, expansion_order: usize) {
        self.expansion_order = expansion_order;
    }

    fn kernel(&mut self, kernel: Self::Kernel) {
        self.kernel = kernel;
    }

    fn operator_data(&mut self, expansion_order: usize, domain: Self::Domain) {
        // Compute unique M2L interactions at Level 3 (smallest choice with all vectors)
        // Compute interaction matrices between source and unique targets, defined by unique transfer vectors
        let nrows = ncoeffs_kifmm(expansion_order);
        let ncols = ncoeffs_kifmm(expansion_order);

        let mut se2tc_fat = rlst_dynamic_array2!(Scalar, [nrows, ncols * NTRANSFER_VECTORS_KIFMM]);
        let mut se2tc_thin = rlst_dynamic_array2!(Scalar, [nrows * NTRANSFER_VECTORS_KIFMM, ncols]);

        let alpha = Scalar::from(ALPHA_INNER).unwrap().re();

        for (i, t) in self.transfer_vectors.iter().enumerate() {
            let source_equivalent_surface = t.source.surface_grid(expansion_order, &domain, alpha);
            let nsources = source_equivalent_surface.len() / self.kernel.space_dimension();

            let target_check_surface = t.target.surface_grid(expansion_order, &domain, alpha);
            let ntargets = target_check_surface.len() / self.kernel.space_dimension();

            let mut tmp_gram_t = rlst_dynamic_array2!(Scalar, [ntargets, nsources]);

            self.kernel.assemble_st(
                EvalType::Value,
                &source_equivalent_surface[..],
                &target_check_surface[..],
                tmp_gram_t.data_mut(),
            );

            // Need to transpose so that rows correspond to targets, and columns to sources
            let mut tmp_gram = rlst_dynamic_array2!(Scalar, [nsources, ntargets]);
            tmp_gram.fill_from(tmp_gram_t.transpose());

            let mut block = se2tc_fat
                .view_mut()
                .into_subview([0, i * ncols], [nrows, ncols]);
            block.fill_from(tmp_gram.view());

            let mut block_column = se2tc_thin
                .view_mut()
                .into_subview([i * nrows, 0], [nrows, ncols]);
            block_column.fill_from(tmp_gram.view());
        }

        let mu = se2tc_fat.shape()[0];
        let nvt = se2tc_fat.shape()[1];
        let k = std::cmp::min(mu, nvt);

        let mut u_big = rlst_dynamic_array2!(Scalar, [mu, k]);
        let mut sigma = vec![Scalar::zero().re(); k];
        let mut vt_big = rlst_dynamic_array2!(Scalar, [k, nvt]);

        se2tc_fat
            .into_svd_alloc(
                u_big.view_mut(),
                vt_big.view_mut(),
                &mut sigma[..],
                SvdMode::Reduced,
            )
            .unwrap();
        let cutoff_rank = find_cutoff_rank(&sigma, self.threshold);
        let mut u = rlst_dynamic_array2!(Scalar, [mu, cutoff_rank]);
        let mut sigma_mat = rlst_dynamic_array2!(Scalar, [cutoff_rank, cutoff_rank]);
        let mut vt = rlst_dynamic_array2!(Scalar, [cutoff_rank, nvt]);

        u.fill_from(u_big.into_subview([0, 0], [mu, cutoff_rank]));
        vt.fill_from(vt_big.into_subview([0, 0], [cutoff_rank, nvt]));
        for (j, s) in sigma.iter().enumerate().take(cutoff_rank) {
            unsafe {
                *sigma_mat.get_unchecked_mut([j, j]) = Scalar::from(*s).unwrap();
            }
        }

        // Store compressed M2L operators
        let thin_nrows = se2tc_thin.shape()[0];
        let nst = se2tc_thin.shape()[1];
        let k = std::cmp::min(thin_nrows, nst);
        let mut _gamma = rlst_dynamic_array2!(Scalar, [thin_nrows, k]);
        let mut _r = vec![Scalar::zero().re(); k];
        let mut st = rlst_dynamic_array2!(Scalar, [k, nst]);

        se2tc_thin
            .into_svd_alloc(
                _gamma.view_mut(),
                st.view_mut(),
                &mut _r[..],
                SvdMode::Reduced,
            )
            .unwrap();

        let mut s_trunc = rlst_dynamic_array2!(Scalar, [nst, cutoff_rank]);
        for j in 0..cutoff_rank {
            for i in 0..nst {
                unsafe { *s_trunc.get_unchecked_mut([i, j]) = *st.get_unchecked([j, i]) }
            }
        }

        let mut c_u = Vec::new();
        let mut c_vt = Vec::new();

        for i in 0..self.transfer_vectors.len() {
            let vt_block = vt.view().into_subview([0, i * ncols], [cutoff_rank, ncols]);

            let tmp = empty_array::<Scalar, 2>().simple_mult_into_resize(
                sigma_mat.view(),
                empty_array::<Scalar, 2>().simple_mult_into_resize(vt_block.view(), s_trunc.view()),
            );

            let mut u_i = rlst_dynamic_array2!(Scalar, [cutoff_rank, cutoff_rank]);
            let mut sigma_i = vec![Scalar::zero().re(); cutoff_rank];
            let mut vt_i = rlst_dynamic_array2!(Scalar, [cutoff_rank, cutoff_rank]);

            tmp.into_svd_alloc(u_i.view_mut(), vt_i.view_mut(), &mut sigma_i, SvdMode::Full)
                .unwrap();

            let directional_cutoff_rank = find_cutoff_rank(&sigma_i, self.threshold);

            let mut u_i_compressed =
                rlst_dynamic_array2!(Scalar, [cutoff_rank, directional_cutoff_rank]);
            let mut vt_i_compressed_ =
                rlst_dynamic_array2!(Scalar, [directional_cutoff_rank, cutoff_rank]);

            let mut sigma_mat_i_compressed =
                rlst_dynamic_array2!(Scalar, [directional_cutoff_rank, directional_cutoff_rank]);

            u_i_compressed
                .fill_from(u_i.into_subview([0, 0], [cutoff_rank, directional_cutoff_rank]));
            vt_i_compressed_
                .fill_from(vt_i.into_subview([0, 0], [directional_cutoff_rank, cutoff_rank]));

            for (j, s) in sigma_i.iter().enumerate().take(directional_cutoff_rank) {
                unsafe {
                    *sigma_mat_i_compressed.get_unchecked_mut([j, j]) = Scalar::from(*s).unwrap();
                }
            }

            let vt_i_compressed = empty_array::<Scalar, 2>()
                .simple_mult_into_resize(sigma_mat_i_compressed.view(), vt_i_compressed_.view());

            c_u.push(u_i_compressed);
            c_vt.push(vt_i_compressed);
        }

        let mut st_trunc = rlst_dynamic_array2!(Scalar, [cutoff_rank, nst]);
        st_trunc.fill_from(s_trunc.transpose());

        let result = BlasMetadata {
            u,
            st: st_trunc,
            c_u,
            c_vt,
        };
        self.metadata = result;
        self.cutoff_rank = cutoff_rank;
    }
}
