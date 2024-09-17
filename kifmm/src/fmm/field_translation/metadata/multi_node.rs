use green_kernels::laplace_3d::Laplace3dKernel;
use green_kernels::traits::Kernel as KernelTrait;
use green_kernels::types::EvalType;
use mpi::traits::Equivalence;
use num::Float;
use rlst::{
    rlst_dynamic_array3, Array, BaseArray, MatrixSvd, RawAccessMut, RlstScalar, VectorContainer,
};
use std::collections::HashMap;

use crate::traits::fftw::Dft;
use crate::traits::fmm::{FmmMetadata, MultiFmm};
use crate::traits::general::AsComplex;
use crate::traits::parallel::GhostExchange;
use crate::traits::tree::{MultiFmmTree, MultiTree, SingleTree};
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
    fn source(&mut self) {}

    fn target(&mut self) {}
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

    fn metadata(&mut self, eval_type: EvalType, charges: &[Self::Scalar]) {
        let dim = self.dim();
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

        let source_trees = self.tree.source_tree().trees();
        let target_trees = self.tree.target_tree().trees();
        let global_depth = self.tree.source_tree().global_depth();
        let local_depth = self.tree.source_tree().local_depth();
        let total_depth = self.tree.source_tree().local_depth();

        // Allocate buffers to store locally available multipole, local and potential data
        // multipole and local data is arranged by level

        let mut source_counts = Vec::new(); // arranged by level, then by tree
        let mut source_displacement = Vec::new(); // arranged by level, then by tree
        let mut n_sources = 0; // total number of source boxes
        let mut level_displacement = 0;

        let mut source_to_index = HashMap::new();

        for level in global_depth..=total_depth {
            let mut tree_displacement = Vec::new();
            let mut tree_counts = Vec::new();
            let mut tree_displacement_ = level_displacement;

            for tree in source_trees.iter() {
                let keys = tree.keys(level).unwrap();
                let n_keys = keys.len();
                tree_displacement.push(tree_displacement_);
                tree_counts.push(n_keys);

                for (key_idx, k) in keys.iter().enumerate() {
                    source_to_index.insert(*k, key_idx + tree_displacement_);
                }

                tree_displacement_ += n_keys;
                n_sources += n_keys;
            }

            level_displacement += tree_displacement_;
            source_counts.push(tree_counts);
            source_displacement.push(tree_displacement);
        }

        let mut tree_displacement = 0;
        let mut source_to_leaf_index = HashMap::new();
        for tree in source_trees.iter() {
            for (leaf_idx, leaf) in tree.all_leaves().unwrap().iter().enumerate() {
                source_to_leaf_index.insert(*leaf, leaf_idx + tree_displacement);
            }

            tree_displacement += tree.n_leaves().unwrap();
        }

        let mut multipoles = vec![Scalar::default(); n_sources * self.ncoeffs_equivalent_surface];

        // Set layout
        self.set_source_layout();

        // Set metadata
        self.multipoles = multipoles;
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
