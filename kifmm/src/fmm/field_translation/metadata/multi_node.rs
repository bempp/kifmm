use green_kernels::laplace_3d::Laplace3dKernel;
use green_kernels::traits::Kernel as KernelTrait;
use green_kernels::types::EvalType;
use mpi::topology::SimpleCommunicator;
use mpi::traits::Equivalence;
use num::Float;
use rlst::{
    empty_array, rlst_dynamic_array2, rlst_dynamic_array3, Array, BaseArray, MatrixSvd,
    MultIntoResize, RawAccess, RawAccessMut, RlstScalar, VectorContainer,
};
use std::collections::HashMap;

use crate::fmm::helpers::multi_node::{
    coordinate_index_pointer_multi_node, leaf_expansion_pointers_multi_node,
    leaf_scales_multi_node, leaf_surfaces_multi_node, level_expansion_pointers_multi_node,
    level_index_pointer_multi_node, potential_pointers_multi_node,
};
use crate::fmm::helpers::single_node::homogenous_kernel_scale;
use crate::linalg::pinv::pinv;
use crate::traits::fftw::Dft;
use crate::traits::fmm::{FmmMetadata, MultiFmm};
use crate::traits::general::AsComplex;
use crate::traits::parallel::GhostExchange;
use crate::traits::tree::{FmmTreeNode, MultiFmmTree, MultiTree};
use crate::tree::types::MortonKey;
use crate::FftFieldTranslation;
use crate::{
    fmm::types::KiFmmMulti,
    linalg::rsvd::MatrixRsvd,
    traits::{
        field::{
            SourceAndTargetTranslationMetadata, SourceToTargetData as SourceToTargetDataTrait,
            SourceToTargetTranslationMetadata,
        },
        fmm::{HomogenousKernel, SourceToTargetTranslation},
        general::Epsilon,
    },
    tree::constants::{ALPHA_INNER, ALPHA_OUTER},
    BlasFieldTranslationSaRcmp,
};

impl<Scalar, SourceToTargetData> SourceAndTargetTranslationMetadata
    for KiFmmMulti<Scalar, Laplace3dKernel<Scalar>, SourceToTargetData>
where
    Scalar: RlstScalar + Default + Epsilon + MatrixSvd + Equivalence + Float,
    <Scalar as RlstScalar>::Real: Default + Equivalence + Float,
    SourceToTargetData: SourceToTargetDataTrait + Send + Sync,
{
    fn source(&mut self) {
        let root = MortonKey::<Scalar::Real>::root();
        let equivalent_surface_order = self.equivalent_surface_order;
        let check_surface_order = self.check_surface_order;
        let n_coeffs_equivalent_surface = self.n_coeffs_equivalent_surface;
        let n_coeffs_check_surface = self.n_coeffs_check_surface;

        // Cast surface parameters
        let alpha_outer = Scalar::from(ALPHA_OUTER).unwrap().re();
        let alpha_inner = Scalar::from(ALPHA_INNER).unwrap().re();
        let domain = self.tree.domain();

        let mut m2m = rlst_dynamic_array2!(
            Scalar,
            [n_coeffs_equivalent_surface, 8 * n_coeffs_equivalent_surface]
        );
        let mut m2m_vec = Vec::new();

        // Compute required surfaces
        let upward_equivalent_surface =
            root.surface_grid(equivalent_surface_order, domain, alpha_inner);

        let upward_check_surface = root.surface_grid(check_surface_order, domain, alpha_outer);

        // Assemble matrix of kernel evaluations between upward check to equivalent, and downward check to equivalent matrices
        // As well as estimating their inverses using SVD
        let mut uc2e = rlst_dynamic_array2!(
            Scalar,
            [n_coeffs_check_surface, n_coeffs_equivalent_surface]
        );

        self.kernel.assemble_st(
            EvalType::Value,
            &upward_check_surface,
            &upward_equivalent_surface,
            uc2e.data_mut(),
        );

        let (s, ut, v) = pinv(&uc2e, None, None).unwrap();

        let mut mat_s = rlst_dynamic_array2!(Scalar, [s.len(), s.len()]);
        for i in 0..s.len() {
            mat_s[[i, i]] = Scalar::from_real(s[i]);
        }

        let uc2e_inv_1 =
            vec![empty_array::<Scalar, 2>().simple_mult_into_resize(v.view(), mat_s.view())];

        let uc2e_inv_2 = vec![ut];

        let parent_upward_check_surface =
            root.surface_grid(check_surface_order, domain, alpha_outer);
        let children = root.children();

        for (i, child) in children.iter().enumerate() {
            let child_upward_equivalent_surface =
                child.surface_grid(equivalent_surface_order, domain, alpha_inner);

            let mut ce2pc = rlst_dynamic_array2!(
                Scalar,
                [n_coeffs_check_surface, n_coeffs_equivalent_surface]
            );

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

            let l = i * n_coeffs_equivalent_surface * n_coeffs_equivalent_surface;
            let r = l + n_coeffs_equivalent_surface * n_coeffs_equivalent_surface;

            m2m.data_mut()[l..r].copy_from_slice(tmp.data());
            m2m_vec.push(tmp);
        }

        self.source = m2m;
        self.source_vec = m2m_vec;
        self.uc2e_inv_1 = uc2e_inv_1;
        self.uc2e_inv_2 = uc2e_inv_2;
    }

    fn target(&mut self) {
        let root = MortonKey::<Scalar::Real>::root();
        let equivalent_surface_order = self.equivalent_surface_order;
        let check_surface_order = self.check_surface_order;
        let n_coeffs_equivalent_surface = self.n_coeffs_equivalent_surface;
        let n_coeffs_check_surface = self.n_coeffs_check_surface;

        // Cast surface parameters
        let alpha_outer = Scalar::from(ALPHA_OUTER).unwrap().re();
        let alpha_inner = Scalar::from(ALPHA_INNER).unwrap().re();
        let domain = self.tree.domain();

        let mut l2l = Vec::new();

        let downward_equivalent_surface =
            root.surface_grid(equivalent_surface_order, domain, alpha_outer);
        let downward_check_surface = root.surface_grid(check_surface_order, domain, alpha_inner);

        let mut dc2e = rlst_dynamic_array2!(
            Scalar,
            [n_coeffs_check_surface, n_coeffs_equivalent_surface]
        );
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

        let dc2e_inv_1 =
            vec![empty_array::<Scalar, 2>().simple_mult_into_resize(v.view(), mat_s.view())];
        let dc2e_inv_2 = vec![ut];

        let parent_downward_equivalent_surface =
            root.surface_grid(equivalent_surface_order, domain, alpha_outer);

        let children = root.children();

        for child in children.iter() {
            let child_downward_check_surface =
                child.surface_grid(check_surface_order, domain, alpha_inner);

            // Note, this way around due to calling convention of kernel, source/targets are 'swapped'
            let mut pe2cc = rlst_dynamic_array2!(
                Scalar,
                [n_coeffs_check_surface, n_coeffs_equivalent_surface]
            );
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

        self.target_vec = l2l;
        self.dc2e_inv_1 = dc2e_inv_1;
        self.dc2e_inv_2 = dc2e_inv_2;
    }
}

impl<Scalar> SourceToTargetTranslationMetadata
    for KiFmmMulti<Scalar, Laplace3dKernel<Scalar>, BlasFieldTranslationSaRcmp<Scalar>>
where
    Scalar: RlstScalar + Default + MatrixRsvd + Equivalence + Float,
    <Scalar as RlstScalar>::Real: Default + Equivalence + Float,
{
    fn displacements(&mut self) {}

    fn source_to_target(&mut self) {}
}

impl<Scalar> SourceToTargetTranslationMetadata
    for KiFmmMulti<Scalar, Laplace3dKernel<Scalar>, FftFieldTranslation<Scalar>>
where
    Scalar: RlstScalar
        + AsComplex
        + Default
        + Dft<InputType = Scalar, OutputType = <Scalar as AsComplex>::ComplexType>
        + Equivalence
        + Float,
    <Scalar as RlstScalar>::Real: RlstScalar + Default + Equivalence + Float,
{
    fn displacements(&mut self) {}

    fn source_to_target(&mut self) {}
}

impl<Scalar, Kernel, SourceToTargetData> FmmMetadata
    for KiFmmMulti<Scalar, Kernel, SourceToTargetData>
where
    Scalar: RlstScalar + Default + Float + Equivalence,
    <Scalar as RlstScalar>::Real: Default + Float + Equivalence,
    Kernel: KernelTrait<T = Scalar> + HomogenousKernel + Default + Send + Sync,
    SourceToTargetData: SourceToTargetDataTrait + Send + Sync,
    Self: MultiFmm + GhostExchange,
{
    type Scalar = Scalar;

    fn metadata(&mut self, eval_type: EvalType, _charges: &[Self::Scalar]) {
        // Check if computing potentials, or potentials and derivatives
        match eval_type {
            EvalType::Value => {}
            EvalType::ValueDeriv => {
                panic!("Only potential computation supported for now")
            }
        }
        let kernel_eval_size = match eval_type {
            EvalType::Value => 1,
            EvalType::ValueDeriv => 4,
        };

        let alpha_outer = Scalar::from(ALPHA_OUTER).unwrap().re();
        let n_target_points = self.tree.target_tree.n_coordinates_tot().unwrap();
        let n_source_points = self.tree.source_tree.n_coordinates_tot().unwrap();
        let n_source_keys = self.tree.source_tree().n_keys_tot().unwrap();
        let n_target_keys = self.tree.target_tree().n_keys_tot().unwrap();

        let equivalent_surface_order = self.equivalent_surface_order;
        let check_surface_order = self.check_surface_order;
        let n_coeffs_equivalent_surface = self.n_coeffs_equivalent_surface;
        let n_coeffs_check_surface = self.n_coeffs_check_surface;

        // Allocate multipole and local buffers for all locally owned source/target octants
        let multipoles = vec![Scalar::default(); n_source_keys * n_coeffs_equivalent_surface];
        let locals = vec![Scalar::default(); n_target_keys * n_coeffs_equivalent_surface];

        // Index pointers of multipole and local data, indexed by level
        let level_index_pointer_multipoles = level_index_pointer_multi_node(&self.tree.source_tree);
        let level_index_pointer_locals = level_index_pointer_multi_node(&self.tree.target_tree);

        // Allocate buffers for local potential data
        let potentials = vec![Scalar::default(); n_target_points * kernel_eval_size];

        // Kernel scale at each target and source leaf
        let leaf_scales_sources =
            leaf_scales_multi_node::<Scalar>(&self.tree.source_tree, n_coeffs_equivalent_surface);

        // Pre compute check surfaces
        let leaf_upward_equivalent_surfaces_sources = leaf_surfaces_multi_node(
            &self.tree.source_tree,
            n_coeffs_equivalent_surface,
            alpha_outer,
            check_surface_order,
        );

        let leaf_upward_check_surfaces_sources = leaf_surfaces_multi_node(
            &self.tree.source_tree,
            n_coeffs_equivalent_surface,
            alpha_outer,
            check_surface_order,
        );

        let leaf_downward_equivalent_surfaces_targets = leaf_surfaces_multi_node(
            &self.tree.target_tree,
            n_coeffs_equivalent_surface,
            alpha_outer,
            equivalent_surface_order,
        );

        // Mutable pointers to multipole and local data, indexed by level
        let level_multipoles = level_expansion_pointers_multi_node(
            &self.tree.source_tree,
            n_coeffs_equivalent_surface,
            &multipoles,
        );

        let level_locals = level_expansion_pointers_multi_node(
            &self.tree.target_tree,
            n_coeffs_equivalent_surface,
            &locals,
        );

        // Mutable pointers to multipole and local data only at leaf level, for utility
        let leaf_multipoles = leaf_expansion_pointers_multi_node(
            &self.tree.source_tree,
            n_coeffs_equivalent_surface,
            &multipoles,
        );

        let leaf_locals = leaf_expansion_pointers_multi_node(
            &self.tree.target_tree,
            n_coeffs_equivalent_surface,
            &locals,
        );

        // Mutable pointers to potential data at each target leaf
        let potential_send_pointers =
            potential_pointers_multi_node(&self.tree.target_tree, kernel_eval_size, &potentials);

        // TODO: Add functionality for charges at some point
        let charges = vec![Scalar::one(); self.tree.source_tree().n_coordinates_tot().unwrap()];
        let charge_index_pointer_targets =
            coordinate_index_pointer_multi_node(&self.tree.target_tree);
        let charge_index_pointer_sources =
            coordinate_index_pointer_multi_node(&self.tree.source_tree);

        // Set metadata
        self.multipoles = multipoles;
        self.leaf_multipoles = leaf_multipoles;
        self.level_multipoles = level_multipoles;
        self.level_index_pointer_multipoles = level_index_pointer_multipoles;
        self.locals = locals;
        self.leaf_locals = leaf_locals;
        self.level_locals = level_locals;
        self.level_index_pointer_locals = level_index_pointer_locals;
        self.potentials = potentials;
        self.potentials_send_pointers = potential_send_pointers;
        self.leaf_upward_equivalent_surfaces_sources = leaf_upward_equivalent_surfaces_sources;
        self.leaf_upward_check_surfaces_sources = leaf_upward_check_surfaces_sources;
        self.leaf_downward_equivalent_surfaces_targets = leaf_downward_equivalent_surfaces_targets;
        self.charges = charges;
        self.charge_index_pointer_sources = charge_index_pointer_sources;
        self.charge_index_pointer_targets = charge_index_pointer_targets;
        self.leaf_scales_sources = leaf_scales_sources;
        self.kernel_eval_size = kernel_eval_size;
    }
}

impl<Scalar, Kernel> KiFmmMulti<Scalar, Kernel, FftFieldTranslation<Scalar>>
where
    Scalar: RlstScalar + AsComplex + Default + Dft + Float + Equivalence,
    Kernel: KernelTrait<T = Scalar> + HomogenousKernel + Default + Send + Sync,
    <Scalar as RlstScalar>::Real: Default + Float + Equivalence,
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
