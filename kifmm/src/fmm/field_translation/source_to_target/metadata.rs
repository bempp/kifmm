//! Implementation of traits for field translations via the FFT and BLAS.
use super::{array::flip3, transfer_vector::compute_transfer_vectors};

use crate::fmm::helpers::ncoeffs_kifmm;
use crate::fmm::types::{BlasFieldTranslation, BlasMetadata, FftFieldTranslation, FftMetadata};
use crate::traits::{fftw::RealToComplexFft3D, field::SourceToTargetData};
use crate::tree::{
    constants::{
        ALPHA_INNER, NCORNERS, NHALO, NSIBLINGS, NSIBLINGS_SQUARED, NTRANSFER_VECTORS_KIFMM,
    },
    helpers::find_corners,
    types::{Domain, MortonKey},
};
use green_kernels::{traits::Kernel, types::EvalType};
use itertools::Itertools;
use num::{Complex, Float, Zero};
use num_complex::ComplexFloat;
use rlst::{
    empty_array, rlst_array_from_slice2, rlst_dynamic_array2, rlst_dynamic_array3, Array,
    BaseArray, Gemm, MatrixSvd, MultIntoResize, RawAccess, RawAccessMut, RlstScalar, Shape,
    SvdMode, UnsafeRandomAccessByRef, UnsafeRandomAccessMut, VectorContainer,
};
use std::collections::HashSet;

fn find_cutoff_rank<T: Float + Default + RlstScalar<Real = T> + Gemm>(
    singular_values: &[T],
    threshold: T,
) -> usize {
    for (i, &s) in singular_values.iter().enumerate() {
        if s <= threshold {
            return i;
        }
    }

    singular_values.len() - 1
}

impl<T, U> SourceToTargetData<U> for BlasFieldTranslation<T, U>
where
    T: Float + Default,
    T: RlstScalar<Real = T> + Gemm,
    U: Kernel<T = T> + Default,
    Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>: MatrixSvd<Item = T>,
{
    type OperatorData = BlasMetadata<T>;
    type Domain = Domain<T>;

    fn operator_data<'a>(&mut self, expansion_order: usize, domain: Self::Domain) {
        // Compute unique M2L interactions at Level 3 (smallest choice with all vectors)

        // Compute interaction matrices between source and unique targets, defined by unique transfer vectors
        let nrows = ncoeffs_kifmm(expansion_order);
        let ncols = ncoeffs_kifmm(expansion_order);

        let mut se2tc_fat = rlst_dynamic_array2!(T, [nrows, ncols * NTRANSFER_VECTORS_KIFMM]);
        let mut se2tc_thin = rlst_dynamic_array2!(T, [nrows * NTRANSFER_VECTORS_KIFMM, ncols]);

        let alpha = T::from(ALPHA_INNER).unwrap();

        for (i, t) in self.transfer_vectors.iter().enumerate() {
            let source_equivalent_surface =
                t.source
                    .compute_kifmm_surface(&domain, expansion_order, alpha);
            let nsources = source_equivalent_surface.len() / self.kernel.space_dimension();

            let target_check_surface =
                t.target
                    .compute_kifmm_surface(&domain, expansion_order, alpha);
            let ntargets = target_check_surface.len() / self.kernel.space_dimension();

            let mut tmp_gram_t = rlst_dynamic_array2!(T, [ntargets, nsources]);

            self.kernel.assemble_st(
                EvalType::Value,
                &source_equivalent_surface[..],
                &target_check_surface[..],
                tmp_gram_t.data_mut(),
            );

            // Need to transpose so that rows correspond to targets, and columns to sources
            let mut tmp_gram = rlst_dynamic_array2!(T, [nsources, ntargets]);
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

        let mut u_big = rlst_dynamic_array2!(T, [mu, k]);
        let mut sigma = vec![T::zero(); k];
        let mut vt_big = rlst_dynamic_array2!(T, [k, nvt]);

        se2tc_fat
            .into_svd_alloc(
                u_big.view_mut(),
                vt_big.view_mut(),
                &mut sigma[..],
                SvdMode::Reduced,
            )
            .unwrap();
        let cutoff_rank = find_cutoff_rank(&sigma, self.threshold);
        let mut u = rlst_dynamic_array2!(T, [mu, cutoff_rank]);
        let mut sigma_mat = rlst_dynamic_array2!(T, [cutoff_rank, cutoff_rank]);
        let mut vt = rlst_dynamic_array2!(T, [cutoff_rank, nvt]);

        u.fill_from(u_big.into_subview([0, 0], [mu, cutoff_rank]));
        vt.fill_from(vt_big.into_subview([0, 0], [cutoff_rank, nvt]));
        for (j, s) in sigma.iter().enumerate().take(cutoff_rank) {
            unsafe {
                *sigma_mat.get_unchecked_mut([j, j]) = T::from(*s).unwrap();
            }
        }

        // Store compressed M2L operators
        let thin_nrows = se2tc_thin.shape()[0];
        let nst = se2tc_thin.shape()[1];
        let k = std::cmp::min(thin_nrows, nst);
        let mut _gamma = rlst_dynamic_array2!(T, [thin_nrows, k]);
        let mut _r = vec![T::zero(); k];
        let mut st = rlst_dynamic_array2!(T, [k, nst]);

        se2tc_thin
            .into_svd_alloc(
                _gamma.view_mut(),
                st.view_mut(),
                &mut _r[..],
                SvdMode::Reduced,
            )
            .unwrap();

        let mut s_trunc = rlst_dynamic_array2!(T, [nst, cutoff_rank]);
        for j in 0..cutoff_rank {
            for i in 0..nst {
                unsafe { *s_trunc.get_unchecked_mut([i, j]) = *st.get_unchecked([j, i]) }
            }
        }

        let mut c_u = Vec::new();
        let mut c_vt = Vec::new();

        for i in 0..self.transfer_vectors.len() {
            let vt_block = vt.view().into_subview([0, i * ncols], [cutoff_rank, ncols]);

            let tmp = empty_array::<T, 2>().simple_mult_into_resize(
                sigma_mat.view(),
                empty_array::<T, 2>().simple_mult_into_resize(vt_block.view(), s_trunc.view()),
            );

            let mut u_i = rlst_dynamic_array2!(T, [cutoff_rank, cutoff_rank]);
            let mut sigma_i = vec![T::zero(); cutoff_rank];
            let mut vt_i = rlst_dynamic_array2!(T, [cutoff_rank, cutoff_rank]);

            tmp.into_svd_alloc(u_i.view_mut(), vt_i.view_mut(), &mut sigma_i, SvdMode::Full)
                .unwrap();

            let directional_cutoff_rank = find_cutoff_rank(&sigma_i, self.threshold);

            let mut u_i_compressed =
                rlst_dynamic_array2!(T, [cutoff_rank, directional_cutoff_rank]);
            let mut vt_i_compressed_ =
                rlst_dynamic_array2!(T, [directional_cutoff_rank, cutoff_rank]);

            let mut sigma_mat_i_compressed =
                rlst_dynamic_array2!(T, [directional_cutoff_rank, directional_cutoff_rank]);

            u_i_compressed
                .fill_from(u_i.into_subview([0, 0], [cutoff_rank, directional_cutoff_rank]));
            vt_i_compressed_
                .fill_from(vt_i.into_subview([0, 0], [directional_cutoff_rank, cutoff_rank]));

            for (j, s) in sigma_i.iter().enumerate().take(directional_cutoff_rank) {
                unsafe {
                    *sigma_mat_i_compressed.get_unchecked_mut([j, j]) = T::from(*s).unwrap();
                }
            }

            let vt_i_compressed = empty_array::<T, 2>()
                .simple_mult_into_resize(sigma_mat_i_compressed.view(), vt_i_compressed_.view());

            c_u.push(u_i_compressed);
            c_vt.push(vt_i_compressed);
        }

        let mut st_trunc = rlst_dynamic_array2!(T, [cutoff_rank, nst]);
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

    fn expansion_order(&mut self, expansion_order: usize) {
        self.expansion_order = expansion_order;
    }

    fn kernel(&mut self, kernel: U) {
        self.kernel = kernel;
    }
}

impl<T, U> BlasFieldTranslation<T, U>
where
    T: Float + Default,
    T: RlstScalar<Real = T>,
    U: Kernel<T = T> + Default,
    Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>: MatrixSvd<Item = T>,
{
    /// Create new
    pub fn new(threshold: Option<T>) -> Self {
        let tmp = T::from(4).unwrap() * T::epsilon();
        BlasFieldTranslation {
            threshold: threshold.unwrap_or(tmp),
            transfer_vectors: compute_transfer_vectors(),
            ..Default::default()
        }
    }
}

impl<T, U> SourceToTargetData<U> for FftFieldTranslation<T, U>
where
    T: RlstScalar<Real = T> + Float + Default + RealToComplexFft3D,
    Complex<T>: RlstScalar + ComplexFloat,
    U: Kernel<T = T> + Default,
{
    type Domain = Domain<T>;

    type OperatorData = FftMetadata<Complex<T>>;

    fn operator_data(&mut self, expansion_order: usize, domain: Self::Domain) {
        // Parameters related to the FFT and Tree
        let m = 2 * expansion_order - 1; // Size of each dimension of 3D kernel/signal
        let pad_size = 1;
        let p = m + pad_size; // Size of each dimension of padded 3D kernel/signal
        let size_real = p * p * (p / 2 + 1); // Number of Fourier coefficients when working with real data

        // Pick a point in the middle of the domain
        let two = T::from(2.0).unwrap();
        let midway = domain.diameter.iter().map(|d| *d / two).collect_vec();
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
        let alpha = T::from(ALPHA_INNER).unwrap();

        // Iterate over each set of convolutions in the halo (26)
        for i in 0..transfer_vectors.len() {
            // Iterate over each unique convolution between sibling set, and halo siblings (64)
            for j in 0..transfer_vectors[i].len() {
                // Translating from sibling set to boxes in its M2L halo
                let target = targets[i][j];
                let source = sources[i][j];

                let source_equivalent_surface =
                    source.compute_kifmm_surface(&domain, expansion_order, alpha);
                let target_check_surface =
                    target.compute_kifmm_surface(&domain, expansion_order, alpha);

                let v_list: HashSet<MortonKey> = target
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

                    let (conv_grid, _) = source.kifmm_convolution_grid(
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
                    let kernel = self.compute_kernel(expansion_order, &conv_grid, kernel_point);

                    let mut kernel = flip3(&kernel);

                    // Compute FFT of padded kernel
                    let mut kernel_hat = rlst_dynamic_array3!(Complex<T>, [p, p, p / 2 + 1]);

                    // TODO: is kernel_hat the transpose of what it used to be?
                    let _ = T::r2c(kernel.data_mut(), kernel_hat.data_mut(), &[p, p, p]);

                    kernel_data_vec[i].push(kernel_hat);
                } else {
                    // Fill with zeros when interaction doesn't exist
                    let n = 2 * expansion_order - 1;
                    let p = n + 1;
                    let kernel_hat_zeros = rlst_dynamic_array3!(Complex<T>, [p, p, p / 2 + 1]);
                    kernel_data_vec[i].push(kernel_hat_zeros);
                }
            }
        }

        // Each element corresponds to all evaluations for each sibling (in order) at that halo position
        let mut kernel_data =
            vec![vec![Complex::<T>::zero(); NSIBLINGS_SQUARED * size_real]; halo_children.len()];

        // For each halo position
        for i in 0..halo_children.len() {
            // For each unique interaction
            for j in 0..NSIBLINGS_SQUARED {
                let offset = j * size_real;
                kernel_data[i][offset..offset + size_real]
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
            for l in 0..size_real {
                // halo child
                for k in 0..NSIBLINGS {
                    // sibling
                    for j in 0..NSIBLINGS {
                        let index = j * size_real * 8 + k * size_real + l;
                        kernel_data_f[i].push(current_vector[index]);
                    }
                }
            }
        }

        // Transpose results for better cache locality in application
        let mut kernel_data_ft = Vec::new();
        for freq in 0..size_real {
            let frequency_offset = NSIBLINGS_SQUARED * freq;
            for kernel_f in kernel_data_f.iter().take(NHALO) {
                let k_f =
                    &kernel_f[frequency_offset..(frequency_offset + NSIBLINGS_SQUARED)].to_vec();
                let k_f_ =
                    rlst_array_from_slice2!(Complex<T>, k_f.as_slice(), [NSIBLINGS, NSIBLINGS]);
                let mut k_ft = rlst_dynamic_array2!(Complex<T>, [NSIBLINGS, NSIBLINGS]);
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
            FftFieldTranslation::<T, U>::compute_surf_to_conv_map(self.expansion_order);
    }

    fn expansion_order(&mut self, expansion_order: usize) {
        self.expansion_order = expansion_order;
    }

    fn kernel(&mut self, kernel: U) {
        self.kernel = kernel;
    }
}

impl<T, U> FftFieldTranslation<T, U>
where
    T: Float + RlstScalar<Real = T> + Default + RealToComplexFft3D,
    Complex<T>: RlstScalar + ComplexFloat,
    U: Kernel<T = T> + Default,
{
    /// Create new
    pub fn new() -> Self {
        FftFieldTranslation {
            transfer_vectors: compute_transfer_vectors(),
            ..Default::default()
        }
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

    /// Computes the unique kernel evaluations and places them on a convolution grid on the source box wrt to a given target point on the target box surface grid.
    ///
    /// # Arguments
    /// * `expansion_order` - The expansion order of the FMM
    /// * `convolution_grid` - Cartesian coordinates of points on the convolution grid at a source box, expected in column major order.
    /// * `target_pt` - The point on the target box's surface grid, with which kernels are being evaluated with respect to.
    pub fn compute_kernel(
        &self,
        expansion_order: usize,
        convolution_grid: &[T],
        target_pt: [T; 3],
    ) -> Array<T, BaseArray<T, VectorContainer<T>, 3>, 3> {
        let n = 2 * expansion_order - 1;
        let npad = n + 1;

        let mut result = rlst_dynamic_array3!(T, [npad, npad, npad]);

        let nconv = n.pow(3);
        let mut kernel_evals = vec![T::zero(); nconv];
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
        charges: &[T],
    ) -> Array<T, BaseArray<T, VectorContainer<T>, 3>, 3> {
        let n = 2 * expansion_order - 1;
        let npad = n + 1;

        let mut result = rlst_dynamic_array3!(T, [npad, npad, npad]);

        for (i, &j) in self.surf_to_conv_map.iter().enumerate() {
            result.data_mut()[j] = charges[i];
        }

        result
    }
}

#[cfg(test)]
mod test {
    use green_kernels::laplace_3d::Laplace3dKernel;
    use rlst::RandomAccessByRef;
    use rlst::RandomAccessMut;

    use super::*;

    #[test]
    fn test_blas_field_translation() {
        let kernel = Laplace3dKernel::new();
        let expansion_order = 6;

        let domain = Domain {
            origin: [0., 0., 0.],
            diameter: [1., 1., 1.],
        };
        let alpha = 1.05;
        let threshold = 1e-5;
        let cutoff_rank = 1000;

        // Some expansion data
        let ncoeffs = 6 * (expansion_order - 1usize).pow(2) + 2;
        let mut multipole = rlst_dynamic_array2!(f64, [ncoeffs, 1]);

        for i in 0..ncoeffs {
            *multipole.get_mut([i, 0]).unwrap() = i as f64;
        }
        let transfer_vectors = compute_transfer_vectors();

        // Create field translation object
        let mut blas = BlasFieldTranslation {
            kernel,
            threshold,
            transfer_vectors,
            expansion_order,
            cutoff_rank,
            ..Default::default()
        };

        blas.operator_data(expansion_order, domain);

        let idx = 123;

        let transfer_vectors = compute_transfer_vectors();
        let transfer_vector = &transfer_vectors[idx];

        // Lookup correct components of SVD compressed M2L operator matrix
        let c_idx = blas
            .transfer_vectors
            .iter()
            .position(|x| x.hash == transfer_vector.hash)
            .unwrap();

        let c_u = &blas.metadata.c_u[c_idx];
        let c_vt = &blas.metadata.c_vt[c_idx];

        let compressed_multipole = empty_array::<f64, 2>()
            .simple_mult_into_resize(blas.metadata.st.view(), multipole.view());

        let compressed_check_potential = empty_array::<f64, 2>().simple_mult_into_resize(
            c_u.view(),
            empty_array::<f64, 2>()
                .simple_mult_into_resize(c_vt.view(), compressed_multipole.view()),
        );

        // Post process to find check potential
        let check_potential = empty_array::<f64, 2>()
            .simple_mult_into_resize(blas.metadata.u.view(), compressed_check_potential.view());

        let sources = transfer_vector
            .source
            .compute_kifmm_surface(&domain, expansion_order, alpha);
        let targets = transfer_vector
            .target
            .compute_kifmm_surface(&domain, expansion_order, alpha);
        let mut direct = vec![0f64; ncoeffs];
        blas.kernel.evaluate_st(
            EvalType::Value,
            &sources[..],
            &targets[..],
            multipole.data(),
            &mut direct[..],
        );

        let abs_error: f64 = check_potential
            .data()
            .iter()
            .zip(direct.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        let rel_error: f64 = abs_error / (direct.iter().sum::<f64>());

        assert!(rel_error < 1e-5);
    }

    fn m2l_scale(level: u64) -> f64 {
        if level < 2 {
            panic!("M2L only performed on level 2 and below")
        }
        if level == 2 {
            1. / 2.
        } else {
            2_f64.powf((level - 3) as f64)
        }
    }

    #[test]
    fn test_fft_operator_data_kernels() {
        let kernel = Laplace3dKernel::new();
        let expansion_order: usize = 2;

        let domain = Domain {
            origin: [0., 0., 0.],
            diameter: [1., 1., 1.],
        };

        // Some expansion data998
        let ncoeffs = ncoeffs_kifmm(expansion_order);
        let mut multipole = rlst_dynamic_array2!(f64, [ncoeffs, 1]);

        for i in 0..ncoeffs {
            *multipole.get_mut([i, 0]).unwrap() = i as f64;
        }

        let level = 2;

        let transfer_vectors = compute_transfer_vectors();

        // Create field translation object
        let mut fft = FftFieldTranslation {
            kernel,
            expansion_order,
            transfer_vectors,
            ..Default::default()
        };

        fft.operator_data(expansion_order, domain);

        let kernels = &fft.metadata.kernel_data;

        let key = MortonKey::from_point(&[0.5, 0.5, 0.5], &domain, level);

        let parent_neighbours = key.parent().neighbors();
        let mut v_list_structured = vec![];
        for _ in 0..26 {
            v_list_structured.push(vec![]);
        }
        for (i, pn) in parent_neighbours.iter().enumerate() {
            for child in pn.children() {
                if !key.is_adjacent(&child) {
                    v_list_structured[i].push(Some(child));
                } else {
                    v_list_structured[i].push(None)
                }
            }
        }

        // pick a halo position
        let halo_idx = 0;
        // pick a halo child position
        let halo_child_idx = 2;
        let n = 2 * expansion_order - 1;
        let p = n + 1;
        let size_real = p * p * (p / 2 + 1);

        // Find kernel from precomputation;
        let kernel_hat =
            &kernels[halo_idx][halo_child_idx * size_real..(halo_child_idx + 1) * size_real];

        // Apply scaling
        let scale = m2l_scale(level);
        let kernel_hat = kernel_hat.iter().map(|k| *k * scale).collect_vec();

        let target = key;
        let source = v_list_structured[halo_idx][halo_child_idx].unwrap();
        let source_equivalent_surface =
            source.compute_kifmm_surface(&domain, expansion_order, ALPHA_INNER);
        let target_check_surface =
            target.compute_kifmm_surface(&domain, expansion_order, ALPHA_INNER);
        let ntargets = target_check_surface.len() / 3;

        // Compute conv grid
        let conv_point_corner_index = 7;
        let corners = find_corners(&source_equivalent_surface[..]);
        let conv_point_corner = [
            corners[conv_point_corner_index],
            corners[8 + conv_point_corner_index],
            corners[16 + conv_point_corner_index],
        ];

        let (conv_grid, _) = source.kifmm_convolution_grid(
            expansion_order,
            &domain,
            ALPHA_INNER,
            &conv_point_corner,
            conv_point_corner_index,
        );

        let kernel_point_index = 0;
        let kernel_point = [
            target_check_surface[kernel_point_index],
            target_check_surface[ntargets + kernel_point_index],
            target_check_surface[2 * ntargets + kernel_point_index],
        ];

        // Compute kernel from source/target pair
        let test_kernel = fft.compute_kernel(expansion_order, &conv_grid, kernel_point);
        let [m, n, o] = test_kernel.shape();

        let mut test_kernel = flip3(&test_kernel);

        // Compute FFT of padded kernel
        let mut test_kernel_hat = rlst_dynamic_array3!(Complex<f64>, [m, n, o / 2 + 1]);
        f64::r2c(
            test_kernel.data_mut(),
            test_kernel_hat.data_mut(),
            &[m, n, o],
        )
        .unwrap();

        for (p, t) in test_kernel_hat.data().iter().zip(kernel_hat.iter()) {
            assert!((p - t).norm() < 1e-6)
        }
    }

    #[test]
    fn test_kernel_rearrangement() {
        // Dummy data mirroring unrearranged kernels
        // here each '1000' corresponds to a sibling index
        // each '100' to a child in a given halo element
        // and each '1' to a frequency
        let mut kernel_data_mat = vec![];
        for _ in 0..26 {
            kernel_data_mat.push(vec![]);
        }
        let size_real = 10;

        for elem in kernel_data_mat.iter_mut().take(26) {
            // sibling index
            for j in 0..8 {
                // halo child index
                for k in 0..8 {
                    // frequency
                    for l in 0..size_real {
                        elem.push(Complex::new((1000 * j + 100 * k + l) as f64, 0.))
                    }
                }
            }
        }

        // We want to use this data by frequency in the implementation of FFT M2L
        // Rearrangement: Grouping by frequency, then halo child, then sibling
        let mut rearranged = vec![];
        for _ in 0..26 {
            rearranged.push(vec![]);
        }
        for i in 0..26 {
            let current_vector = &kernel_data_mat[i];
            for l in 0..size_real {
                // halo child
                for k in 0..8 {
                    // sibling
                    for j in 0..8 {
                        let index = j * size_real * 8 + k * size_real + l;
                        rearranged[i].push(current_vector[index]);
                    }
                }
            }
        }

        // We expect the first 64 elements to correspond to the first frequency components of all
        // siblings with all elements in a given halo position
        let freq = 4;
        let offset = freq * 64;
        let result = &rearranged[0][offset..offset + 64];

        // For each halo child
        for i in 0..8 {
            // for each sibling
            for j in 0..8 {
                let expected = (i * 100 + j * 1000 + freq) as f64;
                assert!(expected == result[i * 8 + j].re)
            }
        }
    }

    #[test]
    fn test_fft_field_translation() {
        let kernel = Laplace3dKernel::new();
        let expansion_order: usize = 2;

        let domain = Domain {
            origin: [0., 0., 0.],
            diameter: [5., 5., 5.],
        };

        let transfer_vectors = compute_transfer_vectors();

        // Some expansion data
        let ncoeffs = ncoeffs_kifmm(expansion_order);
        let mut multipole = rlst_dynamic_array2!(f64, [ncoeffs, 1]);

        for i in 0..ncoeffs {
            *multipole.get_mut([i, 0]).unwrap() = i as f64;
        }

        let mut fft = FftFieldTranslation {
            kernel,
            expansion_order,
            transfer_vectors,
            ..Default::default()
        };

        fft.operator_data(expansion_order, domain);

        // Compute all M2L operators
        // Pick a random source/target pair
        let idx = 123;
        let all_transfer_vectors = compute_transfer_vectors();

        let transfer_vector = &all_transfer_vectors[idx];

        // Compute FFT of the representative signal
        let mut signal = fft.compute_signal(expansion_order, multipole.data());
        let [m, n, o] = signal.shape();
        let mut signal_hat = rlst_dynamic_array3!(Complex<f64>, [m, n, o / 2 + 1]);

        let _ = f64::r2c(signal.data_mut(), signal_hat.data_mut(), &[m, n, o]);

        let source_equivalent_surface =
            transfer_vector
                .source
                .compute_kifmm_surface(&domain, expansion_order, ALPHA_INNER);
        let target_check_surface =
            transfer_vector
                .target
                .compute_kifmm_surface(&domain, expansion_order, ALPHA_INNER);
        let ntargets = target_check_surface.len() / 3;

        // Compute conv grid
        let conv_point_corner_index = 7;
        let corners = find_corners(&source_equivalent_surface[..]);
        let conv_point_corner = [
            corners[conv_point_corner_index],
            corners[8 + conv_point_corner_index],
            corners[16 + conv_point_corner_index],
        ];

        let (conv_grid, _) = transfer_vector.source.kifmm_convolution_grid(
            expansion_order,
            &domain,
            ALPHA_INNER,
            &conv_point_corner,
            conv_point_corner_index,
        );

        let kernel_point_index = 0;
        let kernel_point = [
            target_check_surface[kernel_point_index],
            target_check_surface[ntargets + kernel_point_index],
            target_check_surface[2 * ntargets + kernel_point_index],
        ];

        // Compute kernel
        let kernel = fft.compute_kernel(expansion_order, &conv_grid, kernel_point);
        let [m, n, o] = kernel.shape();

        let mut kernel = flip3(&kernel);

        // Compute FFT of padded kernel
        let mut kernel_hat = rlst_dynamic_array3!(Complex<f64>, [m, n, o / 2 + 1]);
        let _ = f64::r2c(kernel.data_mut(), kernel_hat.data_mut(), &[m, n, o]);

        let mut hadamard_product = rlst_dynamic_array3!(Complex<f64>, [m, n, o / 2 + 1]);
        for k in 0..o / 2 + 1 {
            for j in 0..n {
                for i in 0..m {
                    *hadamard_product.get_mut([i, j, k]).unwrap() =
                        kernel_hat.get([i, j, k]).unwrap() * signal_hat.get([i, j, k]).unwrap();
                }
            }
        }
        let mut potentials = rlst_dynamic_array3!(f64, [m, n, o]);

        let _ = f64::c2r(
            hadamard_product.data_mut(),
            potentials.data_mut(),
            &[m, n, o],
        );

        let mut result = vec![0f64; ntargets];
        for (i, &idx) in fft.conv_to_surf_map.iter().enumerate() {
            result[i] = potentials.data()[idx];
        }

        // Get direct evaluations for testing
        let mut direct = vec![0f64; ncoeffs];
        fft.kernel.evaluate_st(
            EvalType::Value,
            &source_equivalent_surface[..],
            &target_check_surface[..],
            multipole.data(),
            &mut direct[..],
        );

        println!("r {:?} \n d {:?}", result, direct);

        let abs_error: f64 = result
            .iter()
            .zip(direct.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        let rel_error: f64 = abs_error / (direct.iter().sum::<f64>());

        assert!(rel_error < 1e-15);
    }
}