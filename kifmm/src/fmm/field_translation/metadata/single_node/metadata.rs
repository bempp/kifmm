//! Implementation of traits to compute metadata for field translation operations.
use green_kernels::{
    helmholtz_3d::Helmholtz3dKernel, laplace_3d::Laplace3dKernel, traits::Kernel as KernelTrait,
    types::GreenKernelEvalType,
};
use rlst::{rlst_dynamic_array3, Array, BaseArray, RawAccessMut, RlstScalar, VectorContainer};

use crate::{
    fmm::{
        constants::DEFAULT_M2L_FFT_BLOCK_SIZE,
        field_translation::source_to_target::transfer_vector::compute_transfer_vectors_at_level,
        helpers::single_node::{
            coordinate_index_pointer_single_node, leaf_expansion_pointers_single_node,
            leaf_surfaces_single_node, level_expansion_pointers_single_node,
            level_index_pointer_single_node, potential_pointers_single_node,
        },
        types::{
            BlasFieldTranslationIa, BlasFieldTranslationSaRcmp, BlasMetadataSaRcmp,
            FftFieldTranslation, FftMetadata, FmmSvdMode,
        },
    },
    traits::{
        fftw::Dft,
        field::FieldTranslation as FieldTranslationTrait,
        fmm::{HomogenousKernel, Metadata, MetadataAccess},
        general::single_node::{AsComplex, Epsilon},
        tree::{SingleFmmTree, SingleTree},
    },
    tree::constants::{ALPHA_INNER, ALPHA_OUTER},
    KiFmm,
};

impl<Scalar, Kernel> KiFmm<Scalar, Kernel, FftFieldTranslation<Scalar>>
where
    Scalar: RlstScalar + AsComplex + Default + Dft,
    Kernel: KernelTrait<T = Scalar> + HomogenousKernel + Default + Send + Sync,
    <Scalar as RlstScalar>::Real: Default,
{
    /// Computes the unique Green's function evaluations and places them on a convolution grid on the source box wrt to a given
    /// target point on the target box surface grid.
    ///
    /// # Arguments
    /// * `expansion_order` - The expansion order of the FMM
    /// * `convolution_grid` - Cartesian coordinates of points on the convolution grid at a source box, expected in row major order.
    /// * `target_pt` - The point on the target box's surface grid, with which kernels are being evaluated with respect to.
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
            GreenKernelEvalType::Value,
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

impl<Scalar, FieldTranslation> MetadataAccess
    for KiFmm<Scalar, Laplace3dKernel<Scalar>, FieldTranslation>
where
    Scalar: RlstScalar + Default,
    FieldTranslation: FieldTranslationTrait + Send + Sync,
    <Scalar as RlstScalar>::Real: Default,
{
    fn fft_map_index(&self, level: u64) -> usize {
        if self.variable_expansion_order {
            (level - 2) as usize
        } else {
            0
        }
    }

    fn expansion_index(&self, level: u64) -> usize {
        if self.variable_expansion_order {
            level as usize
        } else {
            0
        }
    }

    fn c2e_operator_index(&self, level: u64) -> usize {
        if self.variable_expansion_order {
            level as usize
        } else {
            0
        }
    }

    fn m2m_operator_index(&self, level: u64) -> usize {
        if self.variable_expansion_order {
            (level - 1) as usize
        } else {
            0
        }
    }

    fn l2l_operator_index(&self, level: u64) -> usize {
        if self.variable_expansion_order {
            (level - 1) as usize
        } else {
            0
        }
    }

    fn m2l_operator_index(&self, level: u64) -> usize {
        if self.variable_expansion_order {
            (level - 2) as usize
        } else {
            0
        }
    }

    fn displacement_index(&self, level: u64) -> usize {
        (level - 2) as usize
    }
}

impl<Scalar, FieldTranslation> MetadataAccess
    for KiFmm<Scalar, Helmholtz3dKernel<Scalar>, FieldTranslation>
where
    Scalar: RlstScalar<Complex = Scalar> + Default,
    FieldTranslation: FieldTranslationTrait + Send + Sync,
    <Scalar as RlstScalar>::Real: Default,
{
    fn fft_map_index(&self, level: u64) -> usize {
        if self.variable_expansion_order {
            (level - 2) as usize
        } else {
            0
        }
    }

    fn expansion_index(&self, level: u64) -> usize {
        if self.variable_expansion_order {
            level as usize
        } else {
            0
        }
    }

    fn c2e_operator_index(&self, level: u64) -> usize {
        level as usize
    }

    fn m2m_operator_index(&self, level: u64) -> usize {
        (level - 1) as usize
    }

    fn l2l_operator_index(&self, level: u64) -> usize {
        (level - 1) as usize
    }

    fn m2l_operator_index(&self, level: u64) -> usize {
        (level - 2) as usize
    }

    fn displacement_index(&self, level: u64) -> usize {
        (level - 2) as usize
    }
}

impl<Scalar, Kernel, FieldTranslation> Metadata for KiFmm<Scalar, Kernel, FieldTranslation>
where
    Scalar: RlstScalar + Default,
    Kernel: KernelTrait<T = Scalar> + HomogenousKernel + Default + Send + Sync,
    FieldTranslation: FieldTranslationTrait + Send + Sync,
    <Scalar as RlstScalar>::Real: Default,
{
    type Scalar = Scalar;

    fn metadata(&mut self, eval_type: GreenKernelEvalType, charges: &[Self::Scalar]) {
        let alpha_outer = Scalar::real(ALPHA_OUTER);
        let alpha_inner = Scalar::real(ALPHA_INNER);

        // Check if computing potentials, or potentials and derivatives
        let kernel_eval_size = match eval_type {
            GreenKernelEvalType::Value => 1,
            GreenKernelEvalType::ValueDeriv => self.dim + 1,
        };

        let n_target_points = self.tree.target_tree.n_coordinates_tot().unwrap();
        let n_source_points = self.tree.source_tree.n_coordinates_tot().unwrap();
        let n_matvecs = charges.len() / n_source_points;
        let n_source_keys = self.tree.source_tree.n_keys_tot().unwrap();
        let n_target_keys = self.tree.target_tree.n_keys_tot().unwrap();

        // Buffers to store all multipole and local data
        let n_multipole_coeffs;
        let n_local_coeffs;
        if self.equivalent_surface_order.len() > 1 {
            n_multipole_coeffs = (0..=self.tree.source_tree().depth())
                .zip(self.n_coeffs_equivalent_surface.iter())
                .fold(0usize, |acc, (level, &ncoeffs)| {
                    acc + self.tree.source_tree().n_keys(level).unwrap() * ncoeffs
                });

            n_local_coeffs = (0..=self.tree.target_tree().depth())
                .zip(self.n_coeffs_equivalent_surface.iter())
                .fold(0usize, |acc, (level, &ncoeffs)| {
                    acc + self.tree.target_tree().n_keys(level).unwrap() * ncoeffs
                })
        } else {
            n_multipole_coeffs = n_source_keys * self.n_coeffs_equivalent_surface.last().unwrap();
            n_local_coeffs = n_target_keys * self.n_coeffs_equivalent_surface.last().unwrap();
        }

        let multipoles = vec![Scalar::default(); n_multipole_coeffs * n_matvecs];
        let locals = vec![Scalar::default(); n_local_coeffs * n_matvecs];

        // Index pointers of multipole and local data, indexed by level
        let level_index_pointer_multipoles =
            level_index_pointer_single_node(&self.tree.source_tree);
        let level_index_pointer_locals = level_index_pointer_single_node(&self.tree.target_tree);

        // Buffer to store evaluated potentials and/or gradients at target points
        let potentials = vec![Scalar::default(); n_target_points * kernel_eval_size * n_matvecs];

        // Pre compute check surfaces
        let leaf_upward_equivalent_surfaces_sources = leaf_surfaces_single_node(
            &self.tree.source_tree,
            *self.n_coeffs_equivalent_surface.last().unwrap(),
            alpha_inner,
            *self.equivalent_surface_order.last().unwrap(),
        );

        let leaf_upward_check_surfaces_sources = leaf_surfaces_single_node(
            &self.tree.source_tree,
            *self.n_coeffs_check_surface.last().unwrap(),
            alpha_outer,
            *self.check_surface_order.last().unwrap(),
        );

        let leaf_downward_equivalent_surfaces_targets = leaf_surfaces_single_node(
            &self.tree.target_tree,
            *self.n_coeffs_equivalent_surface.last().unwrap(),
            alpha_outer,
            *self.equivalent_surface_order.last().unwrap(),
        );

        // Mutable pointers to multipole and local data, indexed by level
        let level_multipoles = level_expansion_pointers_single_node(
            &self.tree.source_tree,
            &self.n_coeffs_equivalent_surface,
            n_matvecs,
            &multipoles,
        );

        let level_locals = level_expansion_pointers_single_node(
            &self.tree.source_tree,
            &self.n_coeffs_equivalent_surface,
            n_matvecs,
            &locals,
        );

        // Mutable pointers to multipole and local data only at leaf level
        let leaf_multipoles = leaf_expansion_pointers_single_node(
            &self.tree.source_tree,
            &self.n_coeffs_equivalent_surface,
            n_matvecs,
            &multipoles,
        );

        let leaf_locals = leaf_expansion_pointers_single_node(
            &self.tree.target_tree,
            &self.n_coeffs_equivalent_surface,
            n_matvecs,
            &locals,
        );

        // Mutable pointers to potential data at each target leaf
        let potentials_send_pointers = potential_pointers_single_node(
            &self.tree.target_tree,
            n_matvecs,
            kernel_eval_size,
            &potentials,
        );

        // Index pointer of charge data at each target leaf
        let charge_index_pointer_targets =
            coordinate_index_pointer_single_node(&self.tree.target_tree);
        let charge_index_pointer_sources =
            coordinate_index_pointer_single_node(&self.tree.source_tree);

        // Set data
        self.multipoles = multipoles;
        self.leaf_multipoles = leaf_multipoles;
        self.level_multipoles = level_multipoles;
        self.locals = locals;
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
        self.charge_index_pointer_targets = charge_index_pointer_targets;
        self.charge_index_pointer_sources = charge_index_pointer_sources;
        self.kernel_eval_size = kernel_eval_size;
    }
}

impl<Scalar> FftFieldTranslation<Scalar>
where
    Scalar: RlstScalar + AsComplex + Dft + Default,
    <Scalar as RlstScalar>::Real: RlstScalar + Default,
{
    /// Constructor for FFT based field translations
    pub fn new(block_size: Option<usize>) -> Self {
        Self {
            transfer_vectors: compute_transfer_vectors_at_level::<Scalar::Real>(3).unwrap(),
            block_size: block_size.unwrap_or(DEFAULT_M2L_FFT_BLOCK_SIZE),
            ..Default::default()
        }
    }
}

impl<Scalar> FieldTranslationTrait for FftFieldTranslation<Scalar>
where
    Scalar: RlstScalar + AsComplex + Default + Dft,
    <Scalar as RlstScalar>::Real: RlstScalar + Default,
{
    type Metadata = FftMetadata<<Scalar as AsComplex>::ComplexType>;

    fn overdetermined(&self) -> bool {
        false
    }

    fn surface_diff(&self) -> usize {
        0
    }
}

impl<Scalar> BlasFieldTranslationSaRcmp<Scalar>
where
    Scalar: RlstScalar + Epsilon + Default,
    <Scalar as RlstScalar>::Real: Default,
{
    /// Constructor for BLAS based field translations, specify a compression threshold for the SVD compressed operators
    /// TODO: More docs
    pub fn new(
        threshold: Option<Scalar::Real>,
        surface_diff: Option<usize>,
        svd_mode: FmmSvdMode,
    ) -> Self {
        let tmp = Scalar::epsilon().re();

        Self {
            threshold: threshold.unwrap_or(tmp),
            transfer_vectors: compute_transfer_vectors_at_level::<Scalar::Real>(3).unwrap(),
            surface_diff: surface_diff.unwrap_or_default(),
            svd_mode,
            ..Default::default()
        }
    }
}

impl<Scalar> FieldTranslationTrait for BlasFieldTranslationSaRcmp<Scalar>
where
    Scalar: RlstScalar,
{
    type Metadata = BlasMetadataSaRcmp<Scalar>;

    fn overdetermined(&self) -> bool {
        true
    }

    fn surface_diff(&self) -> usize {
        self.surface_diff
    }
}

impl<Scalar> BlasFieldTranslationIa<Scalar>
where
    Scalar: RlstScalar + Epsilon + Default,
    <Scalar as RlstScalar>::Real: Default,
{
    /// Constructor for BLAS based field translations, specify a compression threshold for the SVD compressed operators
    /// TODO: More docs
    pub fn new(
        threshold: Option<Scalar::Real>,
        surface_diff: Option<usize>,
        svd_mode: FmmSvdMode,
    ) -> Self {
        let tmp = Scalar::epsilon().re();

        Self {
            threshold: threshold.unwrap_or(tmp),
            surface_diff: surface_diff.unwrap_or_default(),
            svd_mode,
            ..Default::default()
        }
    }
}

impl<Scalar> FieldTranslationTrait for BlasFieldTranslationIa<Scalar>
where
    Scalar: RlstScalar,
{
    type Metadata = BlasFieldTranslationIa<Scalar>;

    fn overdetermined(&self) -> bool {
        true
    }

    fn surface_diff(&self) -> usize {
        self.surface_diff
    }
}

#[cfg(test)]
mod test {
    #![allow(clippy::type_complexity)]

    use std::collections::HashSet;

    use itertools::*;
    use num::{Complex, Zero};
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use rlst::{
        c64, empty_array, rlst_dynamic_array2, rlst_dynamic_array3, MultIntoResize,
        RandomAccessByRef, RandomAccessMut, RawAccess, RawAccessMut, RlstScalar, Shape,
    };

    use green_kernels::{
        helmholtz_3d::Helmholtz3dKernel, laplace_3d::Laplace3dKernel,
        traits::Kernel as KernelTrait, types::GreenKernelEvalType,
    };

    use crate::{
        fmm::{
            field_translation::source_to_target::transfer_vector::compute_transfer_vectors_at_level,
            helpers::single_node::flip3,
        },
        traits::{
            fftw::Dft,
            fmm::{DataAccess, MetadataAccess},
            tree::{FmmTreeNode, SingleFmmTree, SingleTree},
        },
        tree::{
            constants::ALPHA_INNER,
            helpers::{find_corners, points_fixture},
            types::MortonKey,
        },
        BlasFieldTranslationIa, BlasFieldTranslationSaRcmp, FftFieldTranslation, FmmSvdMode,
        SingleNodeBuilder,
    };

    #[test]
    fn test_blas_field_translation_laplace() {
        // Setup random sources and targets
        let n_sources = 10000;
        let n_targets = 10000;
        let sources = points_fixture::<f64>(n_sources, None, None, Some(1));
        let targets = points_fixture::<f64>(n_targets, None, None, Some(1));

        // FMM parameters
        let n_crit = Some(100);
        let depth = None;
        let expansion_order = [6];
        let prune_empty = true;

        // Charge data
        let nvecs = 1;
        let mut rng = StdRng::seed_from_u64(0);
        let mut charges = rlst_dynamic_array2!(f64, [n_sources, nvecs]);
        charges.data_mut().iter_mut().for_each(|c| *c = rng.gen());

        let fmm = SingleNodeBuilder::new(false)
            .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
            .unwrap()
            .parameters(
                charges.data(),
                &expansion_order,
                Laplace3dKernel::new(),
                GreenKernelEvalType::Value,
                BlasFieldTranslationSaRcmp::new(Some(1e-5), None, FmmSvdMode::Deterministic),
            )
            .unwrap()
            .build()
            .unwrap();

        let idx = 123;
        let level = 3;
        let transfer_vectors = compute_transfer_vectors_at_level::<f64>(level).unwrap();
        let transfer_vector = &transfer_vectors[idx];

        // Lookup correct components of SVD compressed M2L operator matrix
        let c_idx = fmm
            .source_to_target
            .transfer_vectors
            .iter()
            .position(|x| x.hash == transfer_vector.hash)
            .unwrap();

        let c_u = &fmm.source_to_target.metadata[0].c_u[c_idx];
        let c_vt = &fmm.source_to_target.metadata[0].c_vt[c_idx];

        let mut multipole = rlst_dynamic_array2!(f64, [fmm.n_coeffs_equivalent_surface(level), 1]);
        for i in 0..fmm.n_coeffs_equivalent_surface(level) {
            *multipole.get_mut([i, 0]).unwrap() = i as f64;
        }

        let compressed_multipole = empty_array::<f64, 2>()
            .simple_mult_into_resize(fmm.source_to_target.metadata[0].st.r(), multipole.r());

        let compressed_check_potential = empty_array::<f64, 2>().simple_mult_into_resize(
            c_u.r(),
            empty_array::<f64, 2>().simple_mult_into_resize(c_vt.r(), compressed_multipole.r()),
        );

        // Post process to find check potential
        let check_potential = empty_array::<f64, 2>().simple_mult_into_resize(
            fmm.source_to_target.metadata[0].u.r(),
            compressed_check_potential.r(),
        );

        let alpha = ALPHA_INNER;

        let sources = transfer_vector.source.surface_grid(
            fmm.equivalent_surface_order(level),
            &fmm.tree.domain,
            alpha,
        );

        let targets = transfer_vector.target.surface_grid(
            fmm.equivalent_surface_order(level),
            &fmm.tree.domain,
            alpha,
        );

        let mut direct = vec![0f64; fmm.n_coeffs_check_surface(level)];

        fmm.kernel.evaluate_st(
            GreenKernelEvalType::Value,
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

    #[test]
    fn test_blas_field_translation_helmholtz() {
        // Setup random sources and targets
        let n_sources = 10000;
        let n_targets = 10000;
        let sources = points_fixture::<f64>(n_sources, None, None, Some(1));
        let targets = points_fixture::<f64>(n_targets, None, None, Some(1));

        // FMM parameters
        let n_crit = Some(100);
        let depth = None;
        let expansion_order = [6];
        let prune_empty = true;
        let wavenumber = 2.5;

        // Charge data
        let nvecs = 1;
        let mut rng = StdRng::seed_from_u64(0);
        let mut charges = rlst_dynamic_array2!(c64, [n_sources, nvecs]);
        charges.data_mut().iter_mut().for_each(|c| *c = rng.gen());

        let fmm = SingleNodeBuilder::new(false)
            .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
            .unwrap()
            .parameters(
                charges.data(),
                &expansion_order,
                Helmholtz3dKernel::new(wavenumber),
                GreenKernelEvalType::Value,
                BlasFieldTranslationIa::new(None, None, FmmSvdMode::Deterministic),
            )
            .unwrap()
            .build()
            .unwrap();

        let level = 2;

        let target = fmm.tree.target_tree().keys(level).unwrap()[0];

        let interaction_list = target
            .parent()
            .neighbors()
            .iter()
            .flat_map(|pn| pn.children())
            .filter(|pnc| {
                !target.is_adjacent(pnc)
                    && fmm.tree.source_tree().all_keys_set().unwrap().contains(pnc)
            })
            .collect_vec();

        let source = interaction_list[0];
        let transfer_vector = source.find_transfer_vector(&target).unwrap();

        let transfer_vectors = compute_transfer_vectors_at_level::<f64>(level).unwrap();

        let m2l_operator_index = fmm.m2l_operator_index(level);

        // Lookup correct components of SVD compressed M2L operator matrix
        let c_idx = transfer_vectors
            .iter()
            .position(|x| x.hash == transfer_vector)
            .unwrap();

        let u = &fmm.source_to_target.metadata[m2l_operator_index].u[c_idx];
        let vt = &fmm.source_to_target.metadata[m2l_operator_index].vt[c_idx];

        let mut multipole = rlst_dynamic_array2!(c64, [fmm.n_coeffs_equivalent_surface(level), 1]);
        for i in 0..fmm.n_coeffs_equivalent_surface(level) {
            *multipole.get_mut([i, 0]).unwrap() = c64::from(i as f64);
        }

        let check_potential = empty_array::<c64, 2>().simple_mult_into_resize(
            u.r(),
            empty_array::<c64, 2>().simple_mult_into_resize(vt.r(), multipole.r()),
        );

        let alpha = ALPHA_INNER;

        let sources =
            source.surface_grid(fmm.equivalent_surface_order(level), &fmm.tree.domain, alpha);
        let targets = target.surface_grid(fmm.check_surface_order(level), &fmm.tree.domain, alpha);

        let mut direct = vec![c64::zero(); fmm.n_coeffs_check_surface(level)];

        fmm.kernel.evaluate_st(
            GreenKernelEvalType::Value,
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
        let rel_error: f64 = abs_error / (direct.iter().sum::<c64>().abs());

        println!("abs {:?} rel {:?}", abs_error, rel_error);
        assert!(rel_error < 1e-5);
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
    fn test_fft_field_translation_laplace() {
        // Setup random sources and targets
        let n_sources = 10000;
        let n_targets = 10000;
        let sources = points_fixture::<f64>(n_sources, None, None, Some(1));
        let targets = points_fixture::<f64>(n_targets, None, None, Some(1));

        // FMM parameters
        let n_crit = Some(100);
        let depth = None;
        let expansion_order = [6];
        let prune_empty = true;

        // Charge data
        let nvecs = 1;
        let mut rng = StdRng::seed_from_u64(0);
        let mut charges = rlst_dynamic_array2!(f64, [n_sources, nvecs]);
        charges.data_mut().iter_mut().for_each(|c| *c = rng.gen());

        let fmm = SingleNodeBuilder::new(false)
            .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
            .unwrap()
            .parameters(
                charges.data(),
                &expansion_order,
                Laplace3dKernel::new(),
                GreenKernelEvalType::Value,
                FftFieldTranslation::new(None),
            )
            .unwrap()
            .build()
            .unwrap();

        let level = 3;
        let coeff_idx = fmm.expansion_index(level);

        let mut multipole = rlst_dynamic_array2!(f64, [fmm.n_coeffs_equivalent_surface(level), 1]);

        for i in 0..fmm.n_coeffs_equivalent_surface(level) {
            *multipole.get_mut([i, 0]).unwrap() = i as f64;
        }

        // Compute all M2L operators
        // Pick a random source/target pair
        let idx = 123;
        let all_transfer_vectors = compute_transfer_vectors_at_level::<f64>(level).unwrap();

        let transfer_vector = &all_transfer_vectors[idx];

        // Compute FFT of the representative signal
        let mut signal = fmm.evaluate_charges_convolution_grid(
            expansion_order[coeff_idx],
            coeff_idx,
            multipole.data(),
        );
        let [m, n, o] = signal.shape();
        let mut signal_hat = rlst_dynamic_array3!(Complex<f64>, [m, n, o / 2 + 1]);

        let plan =
            f64::plan_forward(signal.data_mut(), signal_hat.data_mut(), &[m, n, o], None).unwrap();
        let _ = f64::forward_dft(signal.data_mut(), signal_hat.data_mut(), &[m, n, o], &plan);

        let source_equivalent_surface = transfer_vector.source.surface_grid(
            expansion_order[coeff_idx],
            &fmm.tree.source_tree.domain,
            ALPHA_INNER,
        );
        let target_check_surface = transfer_vector.target.surface_grid(
            expansion_order[coeff_idx],
            &fmm.tree.source_tree.domain,
            ALPHA_INNER,
        );
        let n_targets = target_check_surface.len() / 3;

        // Compute conv grid
        let conv_point_corner_index = 7;
        let corners = find_corners(&source_equivalent_surface[..]);
        let conv_point_corner = [
            corners[fmm.dim * conv_point_corner_index],
            corners[fmm.dim * conv_point_corner_index + 1],
            corners[fmm.dim * conv_point_corner_index + 2],
        ];

        let (conv_grid, _) = transfer_vector.source.convolution_grid(
            expansion_order[coeff_idx],
            &fmm.tree.source_tree.domain,
            ALPHA_INNER,
            &conv_point_corner,
            conv_point_corner_index,
        );

        let kernel_point_index = 0;
        let kernel_point = [
            target_check_surface[fmm.dim() * kernel_point_index],
            target_check_surface[fmm.dim() * kernel_point_index + 1],
            target_check_surface[fmm.dim() * kernel_point_index + 2],
        ];

        // Compute kernel
        let kernel = fmm.evaluate_greens_fct_convolution_grid(
            expansion_order[coeff_idx],
            &conv_grid,
            kernel_point,
        );
        let [m, n, o] = kernel.shape();

        let mut kernel = flip3(&kernel);

        // Compute FFT of padded kernel
        let mut kernel_hat = rlst_dynamic_array3!(Complex<f64>, [m, n, o / 2 + 1]);
        let plan =
            f64::plan_forward(kernel.data_mut(), kernel_hat.data_mut(), &[m, n, o], None).unwrap();
        let _ = f64::forward_dft(kernel.data_mut(), kernel_hat.data_mut(), &[m, n, o], &plan);

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

        let plan = f64::plan_backward(
            hadamard_product.data_mut(),
            potentials.data_mut(),
            &[m, n, o],
            None,
        )
        .unwrap();
        let _ = f64::backward_dft(
            hadamard_product.data_mut(),
            potentials.data_mut(),
            &[m, n, o],
            &plan,
        );

        let mut result = vec![0f64; n_targets];
        for (i, &idx) in fmm.source_to_target.conv_to_surf_map[coeff_idx]
            .iter()
            .enumerate()
        {
            result[i] = potentials.data()[idx];
        }

        // Get direct evaluations for testing
        let mut direct = vec![0f64; fmm.n_coeffs_check_surface(level)];
        fmm.kernel.evaluate_st(
            GreenKernelEvalType::Value,
            &source_equivalent_surface[..],
            &target_check_surface[..],
            multipole.data(),
            &mut direct[..],
        );

        let abs_error: f64 = result
            .iter()
            .zip(direct.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        let rel_error: f64 = abs_error / (direct.iter().sum::<f64>());

        assert!(rel_error < 1e-15);
    }

    #[test]
    fn test_fft_field_translation_helmholtz() {
        // Setup random sources and targets
        let n_sources = 10000;
        let n_targets = 10000;
        let sources = points_fixture::<f64>(n_sources, None, None, Some(1));
        let targets = points_fixture::<f64>(n_targets, None, None, Some(1));

        // FMM parameters
        let n_crit = Some(100);
        let depth = None;
        let expansion_order = [6];
        let prune_empty = true;
        let wavenumber = 1.0;

        // Charge data
        let nvecs = 1;
        let mut rng = StdRng::seed_from_u64(0);
        let mut charges = rlst_dynamic_array2!(c64, [n_sources, nvecs]);
        charges.data_mut().iter_mut().for_each(|c| *c = rng.gen());

        let fmm = SingleNodeBuilder::new(false)
            .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
            .unwrap()
            .parameters(
                charges.data(),
                &expansion_order,
                Helmholtz3dKernel::new(wavenumber),
                GreenKernelEvalType::Value,
                FftFieldTranslation::new(None),
            )
            .unwrap()
            .build()
            .unwrap();

        let level = 2;
        let coeff_idx = fmm.expansion_index(level);
        let mut multipole = rlst_dynamic_array2!(c64, [fmm.n_coeffs_equivalent_surface(level), 1]);

        for i in 0..fmm.n_coeffs_equivalent_surface(level) {
            *multipole.get_mut([i, 0]).unwrap() = c64::from(i as f64);
        }

        let source = fmm.tree().source_tree().keys(level).unwrap()[0];

        let v_list: HashSet<MortonKey<_>> = source
            .parent()
            .neighbors()
            .iter()
            .flat_map(|pn| pn.children())
            .filter(|pnc| {
                !source.is_adjacent(pnc) && fmm.tree().source_tree().keys_set.contains(pnc)
            })
            .collect();

        let v_list = v_list.into_iter().collect_vec();
        let target = v_list[0];

        // Compute FFT of the representative signal
        let mut signal = fmm.evaluate_charges_convolution_grid(
            expansion_order[coeff_idx],
            coeff_idx,
            multipole.data(),
        );
        let [m, n, o] = signal.shape();
        let mut signal_hat = rlst_dynamic_array3!(Complex<f64>, [m, n, o]);

        let plan =
            c64::plan_forward(signal.data_mut(), signal_hat.data_mut(), &[m, n, o], None).unwrap();
        let _ = c64::forward_dft(signal.data_mut(), signal_hat.data_mut(), &[m, n, o], &plan);

        let source_equivalent_surface = source.surface_grid(
            expansion_order[coeff_idx],
            &fmm.tree.source_tree.domain,
            ALPHA_INNER,
        );
        let target_check_surface = target.surface_grid(
            expansion_order[coeff_idx],
            &fmm.tree.source_tree.domain,
            ALPHA_INNER,
        );
        let n_targets = target_check_surface.len() / 3;

        // Compute conv grid
        let conv_point_corner_index = 7;
        let corners = find_corners(&source_equivalent_surface[..]);
        let conv_point_corner = [
            corners[fmm.dim * conv_point_corner_index],
            corners[fmm.dim * conv_point_corner_index + 1],
            corners[fmm.dim * conv_point_corner_index + 2],
        ];

        let (conv_grid, _) = source.convolution_grid(
            expansion_order[coeff_idx],
            &fmm.tree.source_tree.domain,
            ALPHA_INNER,
            &conv_point_corner,
            conv_point_corner_index,
        );

        let kernel_point_index = 0;
        let kernel_point = [
            target_check_surface[fmm.dim() * kernel_point_index],
            target_check_surface[fmm.dim() * kernel_point_index + 1],
            target_check_surface[fmm.dim() * kernel_point_index + 2],
        ];

        // Compute kernel
        let kernel = fmm.evaluate_greens_fct_convolution_grid(
            expansion_order[coeff_idx],
            &conv_grid,
            kernel_point,
        );
        let [m, n, o] = kernel.shape();

        let mut kernel = flip3(&kernel);

        // Compute FFT of padded kernel
        let mut kernel_hat = rlst_dynamic_array3!(Complex<f64>, [m, n, o]);
        let plan =
            c64::plan_forward(kernel.data_mut(), kernel_hat.data_mut(), &[m, n, o], None).unwrap();
        let _ = c64::forward_dft(kernel.data_mut(), kernel_hat.data_mut(), &[m, n, o], &plan);

        let mut hadamard_product = rlst_dynamic_array3!(Complex<f64>, [m, n, o]);
        for k in 0..o {
            for j in 0..n {
                for i in 0..m {
                    *hadamard_product.get_mut([i, j, k]).unwrap() =
                        kernel_hat.get([i, j, k]).unwrap() * signal_hat.get([i, j, k]).unwrap();
                }
            }
        }
        let mut potentials = rlst_dynamic_array3!(c64, [m, n, o]);

        let plan = c64::plan_backward(
            hadamard_product.data_mut(),
            potentials.data_mut(),
            &[m, n, o],
            None,
        )
        .unwrap();
        let _ = c64::backward_dft(
            hadamard_product.data_mut(),
            potentials.data_mut(),
            &[m, n, o],
            &plan,
        );

        let mut result = vec![c64::zero(); n_targets];
        for (i, &idx) in fmm.source_to_target.conv_to_surf_map[coeff_idx]
            .iter()
            .enumerate()
        {
            result[i] = potentials.data()[idx];
        }

        // Get direct evaluations for testing
        let mut direct = vec![c64::zero(); fmm.n_coeffs_check_surface(level)];
        fmm.kernel.evaluate_st(
            GreenKernelEvalType::Value,
            &source_equivalent_surface[..],
            &target_check_surface[..],
            multipole.data(),
            &mut direct[..],
        );

        let abs_error: f64 = result
            .iter()
            .zip(direct.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        let rel_error: f64 = abs_error / (direct.iter().sum::<c64>().abs());

        assert!(rel_error < 1e-15);
    }
}
