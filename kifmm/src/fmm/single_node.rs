//! Implementation of Fmm Trait.
use crate::fmm::helpers::{
    leaf_expansion_pointers, level_expansion_pointers, map_charges, potential_pointers,
};
use crate::fmm::types::{Charges, FmmEvalType, KiFmm};
use crate::traits::{
    field::SourceToTargetData,
    fmm::{Fmm, SourceToTargetTranslation, SourceTranslation, TargetTranslation},
    tree::{FmmTree, Tree},
};
use crate::tree::types::{MortonKey, SingleNodeTree};
use green_kernels::traits::Kernel;
use green_kernels::types::EvalType;
use num::Float;
use rlst::{rlst_dynamic_array2, RawAccess, RlstScalar, Shape};

impl<T, U, V, W> Fmm for KiFmm<T, U, V, W>
where
    T: FmmTree<Tree = SingleNodeTree<W>, Node = MortonKey> + Send + Sync,
    U: SourceToTargetData + Send + Sync,
    V: Kernel<T = W> + Send + Sync,
    W: RlstScalar<Real = W> + Float + Default,
    Self: SourceToTargetTranslation,
{
    type Node = T::Node;
    type Scalar = W;
    type Kernel = V;
    type Tree = T;

    fn dim(&self) -> usize {
        self.dim
    }

    fn multipole(&self, key: &Self::Node) -> Option<&[Self::Scalar]> {
        if let Some(index) = self.tree.source_tree().index(key) {
            match self.fmm_eval_type {
                FmmEvalType::Vector => {
                    Some(&self.multipoles[index * self.ncoeffs..(index + 1) * self.ncoeffs])
                }
                FmmEvalType::Matrix(nmatvecs) => Some(
                    &self.multipoles
                        [index * self.ncoeffs * nmatvecs..(index + 1) * self.ncoeffs * nmatvecs],
                ),
            }
        } else {
            None
        }
    }

    fn local(&self, key: &Self::Node) -> Option<&[Self::Scalar]> {
        if let Some(index) = self.tree.target_tree().index(key) {
            match self.fmm_eval_type {
                FmmEvalType::Vector => {
                    Some(&self.locals[index * self.ncoeffs..(index + 1) * self.ncoeffs])
                }
                FmmEvalType::Matrix(nmatvecs) => Some(
                    &self.locals
                        [index * self.ncoeffs * nmatvecs..(index + 1) * self.ncoeffs * nmatvecs],
                ),
            }
        } else {
            None
        }
    }

    fn potential(&self, leaf: &Self::Node) -> Option<Vec<&[Self::Scalar]>> {
        if let Some(&leaf_idx) = self.tree.target_tree().leaf_index(leaf) {
            let (l, r) = self.charge_index_pointer_targets[leaf_idx];
            let ntargets = r - l;

            match self.fmm_eval_type {
                FmmEvalType::Vector => Some(vec![
                    &self.potentials[l * self.kernel_eval_size..r * self.kernel_eval_size],
                ]),
                FmmEvalType::Matrix(nmatvecs) => {
                    let nleaves = self.tree.target_tree().nleaves().unwrap();
                    let mut slices = Vec::new();
                    for eval_idx in 0..nmatvecs {
                        let potentials_pointer =
                            self.potentials_send_pointers[eval_idx * nleaves + leaf_idx].raw;
                        slices.push(unsafe {
                            std::slice::from_raw_parts(
                                potentials_pointer,
                                ntargets * self.kernel_eval_size,
                            )
                        });
                    }
                    Some(slices)
                }
            }
        } else {
            None
        }
    }

    fn expansion_order(&self) -> usize {
        self.expansion_order
    }

    fn kernel(&self) -> &Self::Kernel {
        &self.kernel
    }

    fn tree(&self) -> &Self::Tree {
        &self.tree
    }

    fn evaluate(&self) {
        // Upward pass
        {
            self.p2m();
            for level in (1..=self.tree.source_tree().depth()).rev() {
                self.m2m(level);
            }
        }

        // Downward pass
        {
            for level in 2..=self.tree.target_tree().depth() {
                if level > 2 {
                    self.l2l(level);
                }
                self.m2l(level);
                self.p2l(level)
            }

            // Leaf level computations
            self.m2p();
            self.p2p();
            self.l2p();
        }
    }

    fn clear(&mut self, charges: &Charges<W>) {
        let [_ncharges, nmatvecs] = charges.shape();
        let ntarget_points = self.tree().target_tree().ncoordinates_tot().unwrap();
        let nsource_leaves = self.tree().source_tree().nleaves().unwrap();
        let ntarget_leaves = self.tree().target_tree().nleaves().unwrap();

        // Clear buffers and set new buffers
        self.multipoles = vec![W::default(); self.multipoles.len()];
        self.locals = vec![W::default(); self.locals.len()];
        self.potentials = vec![W::default(); self.potentials.len()];
        self.charges = vec![W::default(); self.charges.len()];

        // Recreate mutable pointers for new buffers
        let potentials_send_pointers = potential_pointers(
            self.tree.target_tree(),
            nmatvecs,
            ntarget_leaves,
            ntarget_points,
            self.kernel_eval_size,
            &self.potentials,
        );

        let leaf_multipoles = leaf_expansion_pointers(
            self.tree().source_tree(),
            self.ncoeffs,
            nmatvecs,
            nsource_leaves,
            &self.multipoles,
        );

        let level_multipoles = level_expansion_pointers(
            self.tree().source_tree(),
            self.ncoeffs,
            nmatvecs,
            &self.multipoles,
        );

        let level_locals = level_expansion_pointers(
            self.tree().target_tree(),
            self.ncoeffs,
            nmatvecs,
            &self.locals,
        );

        let leaf_locals = leaf_expansion_pointers(
            self.tree().target_tree(),
            self.ncoeffs,
            nmatvecs,
            ntarget_leaves,
            &self.locals,
        );

        // Set mutable pointers
        self.level_locals = level_locals;
        self.level_multipoles = level_multipoles;
        self.leaf_locals = leaf_locals;
        self.leaf_multipoles = leaf_multipoles;
        self.potentials_send_pointers = potentials_send_pointers;

        // Set new charges
        self.charges = map_charges(
            self.tree.source_tree().all_global_indices().unwrap(),
            charges,
        )
        .data()
        .to_vec();
    }
}

impl<T, U, V, W> Default for KiFmm<T, U, V, W>
where
    T: FmmTree<Tree = SingleNodeTree<W>> + Default,
    U: SourceToTargetData + Default,
    V: Kernel + Default,
    W: RlstScalar<Real = W> + Float + Default,
{
    fn default() -> Self {
        let uc2e_inv_1 = rlst_dynamic_array2!(W, [1, 1]);
        let uc2e_inv_2 = rlst_dynamic_array2!(W, [1, 1]);
        let dc2e_inv_1 = rlst_dynamic_array2!(W, [1, 1]);
        let dc2e_inv_2 = rlst_dynamic_array2!(W, [1, 1]);
        let source = rlst_dynamic_array2!(W, [1, 1]);

        KiFmm {
            tree: T::default(),
            source_to_target: U::default(),
            kernel: V::default(),
            expansion_order: 0,
            fmm_eval_type: FmmEvalType::Vector,
            kernel_eval_type: EvalType::Value,
            kernel_eval_size: 0,
            dim: 0,
            ncoeffs: 0,
            uc2e_inv_1,
            uc2e_inv_2,
            dc2e_inv_1,
            dc2e_inv_2,
            source,
            source_vec: Vec::default(),
            target_vec: Vec::default(),
            multipoles: Vec::default(),
            locals: Vec::default(),
            leaf_multipoles: Vec::default(),
            level_multipoles: Vec::default(),
            leaf_locals: Vec::default(),
            level_locals: Vec::default(),
            level_index_pointer_locals: Vec::default(),
            level_index_pointer_multipoles: Vec::default(),
            potentials: Vec::default(),
            potentials_send_pointers: Vec::default(),
            leaf_upward_surfaces_sources: Vec::default(),
            leaf_upward_surfaces_targets: Vec::default(),
            charges: Vec::default(),
            charge_index_pointer_sources: Vec::default(),
            charge_index_pointer_targets: Vec::default(),
            leaf_scales_sources: Vec::default(),
        }
    }
}

#[cfg(test)]
mod test {

    use super::*;
    use crate::tree::constants::{ALPHA_INNER, ROOT};
    use crate::tree::helpers::points_fixture;
    use crate::{BlasFieldTranslation, FftFieldTranslation, SingleNodeBuilder, SingleNodeFmmTree};
    use green_kernels::laplace_3d::Laplace3dKernel;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use rlst::rlst_array_from_slice2;
    use rlst::Array;
    use rlst::BaseArray;
    use rlst::RawAccessMut;
    use rlst::VectorContainer;

    fn test_single_node_fmm_vector_helper<T: RlstScalar<Real = T> + Float + Default>(
        fmm: Box<
            dyn Fmm<
                Scalar = T,
                Node = MortonKey,
                Kernel = Laplace3dKernel<T>,
                Tree = SingleNodeFmmTree<T>,
            >,
        >,
        eval_type: EvalType,
        sources: &Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>,
        charges: &Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>,
        threshold: T,
    ) {
        let eval_size = match eval_type {
            EvalType::Value => 1,
            EvalType::ValueDeriv => 4,
        };

        let leaf_idx = 0;
        let leaf: MortonKey = fmm.tree().target_tree().all_leaves().unwrap()[leaf_idx];
        let potential = fmm.potential(&leaf).unwrap()[0];

        let leaf_targets = fmm.tree().target_tree().coordinates(&leaf).unwrap();

        let ntargets = leaf_targets.len() / fmm.dim();
        let mut direct = vec![T::zero(); ntargets * eval_size];

        let leaf_coordinates_row_major =
            rlst_array_from_slice2!(T, leaf_targets, [ntargets, fmm.dim()], [fmm.dim(), 1]);
        let mut leaf_coordinates_col_major = rlst_dynamic_array2!(T, [ntargets, fmm.dim()]);
        leaf_coordinates_col_major.fill_from(leaf_coordinates_row_major.view());

        fmm.kernel().evaluate_st(
            eval_type,
            sources.data(),
            leaf_coordinates_col_major.data(),
            charges.data(),
            &mut direct,
        );

        direct.iter().zip(potential).for_each(|(&d, &p)| {
            let abs_error = num::Float::abs(d - p);
            let rel_error = abs_error / p;
            println!("err {:?} \nd {:?} \np {:?}", rel_error, direct, potential);
            assert!(rel_error <= threshold)
        });
    }

    fn test_single_node_fmm_matrix_helper<T: RlstScalar<Real = T> + Float + Default>(
        fmm: Box<
            dyn Fmm<
                Scalar = T,
                Node = MortonKey,
                Kernel = Laplace3dKernel<T>,
                Tree = SingleNodeFmmTree<T>,
            >,
        >,
        eval_type: EvalType,
        sources: &Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>,
        charges: &Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>,
        threshold: T,
    ) {
        let eval_size = match eval_type {
            EvalType::Value => 1,
            EvalType::ValueDeriv => 4,
        };

        let leaf_idx = 0;
        let leaf: MortonKey = fmm.tree().target_tree().all_leaves().unwrap()[leaf_idx];

        let leaf_targets = fmm.tree().target_tree().coordinates(&leaf).unwrap();

        let ntargets = leaf_targets.len() / fmm.dim();

        let leaf_coordinates_row_major =
            rlst_array_from_slice2!(T, leaf_targets, [ntargets, fmm.dim()], [fmm.dim(), 1]);

        let mut leaf_coordinates_col_major = rlst_dynamic_array2!(T, [ntargets, fmm.dim()]);
        leaf_coordinates_col_major.fill_from(leaf_coordinates_row_major.view());

        let [nsources, nmatvecs] = charges.shape();

        for i in 0..nmatvecs {
            let potential_i = fmm.potential(&leaf).unwrap()[i];
            let charges_i = &charges.data()[nsources * i..nsources * (i + 1)];
            let mut direct_i = vec![T::zero(); ntargets * eval_size];
            fmm.kernel().evaluate_st(
                eval_type,
                sources.data(),
                leaf_coordinates_col_major.data(),
                charges_i,
                &mut direct_i,
            );

            println!(
                "i {:?} \n direct_i {:?}\n potential_i {:?}",
                i, direct_i, potential_i
            );
            direct_i.iter().zip(potential_i).for_each(|(&d, &p)| {
                let abs_error = num::Float::abs(d - p);
                let rel_error = abs_error / p;
                assert!(rel_error <= threshold)
            })
        }
    }

    #[test]
    fn test_fmm_api() {
        // Setup random sources and targets
        let nsources = 9000;
        let ntargets = 10000;

        let min = None;
        let max = None;
        let sources = points_fixture::<f64>(nsources, min, max, Some(0));
        let targets = points_fixture::<f64>(ntargets, min, max, Some(1));

        // FMM parameters
        let n_crit = Some(100);
        let expansion_order = 6;
        let sparse = true;
        let threshold_pot = 1e-5;

        // Set charge data and evaluate an FMM
        let nvecs = 1;
        let mut rng = StdRng::seed_from_u64(0);
        let mut charges = rlst_dynamic_array2!(f64, [nsources, nvecs]);
        charges.data_mut().iter_mut().for_each(|c| *c = rng.gen());

        let mut fmm = SingleNodeBuilder::new()
            .tree(&sources, &targets, n_crit, sparse)
            .unwrap()
            .parameters(
                &charges,
                expansion_order,
                Laplace3dKernel::new(),
                EvalType::Value,
                FftFieldTranslation::new(),
            )
            .unwrap()
            .build()
            .unwrap();
        fmm.evaluate();

        // Reset Charge data and re-evaluate potential
        let mut rng = StdRng::seed_from_u64(1);
        charges.data_mut().iter_mut().for_each(|c| *c = rng.gen());

        fmm.clear(&charges);
        fmm.evaluate();

        let fmm = Box::new(fmm);
        test_single_node_fmm_vector_helper(fmm, EvalType::Value, &sources, &charges, threshold_pot);
    }

    #[test]
    fn test_laplace_fmm_vector() {
        // Setup random sources and targets
        let nsources = 9000;
        let ntargets = 10000;

        let min = None;
        let max = None;
        let sources = points_fixture::<f64>(nsources, min, max, Some(0));
        let targets = points_fixture::<f64>(ntargets, min, max, Some(1));

        // FMM parameters
        let n_crit = Some(100);
        let expansion_order = 6;
        let sparse = true;
        let threshold_pot = 1e-5;
        let threshold_deriv = 1e-4;
        let threshold_deriv_blas = 1e-3;
        let singular_value_threshold = Some(1e-2);

        // Charge data
        let nvecs = 1;
        let mut rng = StdRng::seed_from_u64(0);
        let mut charges = rlst_dynamic_array2!(f64, [nsources, nvecs]);
        charges.data_mut().iter_mut().for_each(|c| *c = rng.gen());

        // fmm with fft based field translation
        {
            // Evaluate potentials
            let fmm_fft = SingleNodeBuilder::new()
                .tree(&sources, &targets, n_crit, sparse)
                .unwrap()
                .parameters(
                    &charges,
                    expansion_order,
                    Laplace3dKernel::new(),
                    EvalType::Value,
                    FftFieldTranslation::new(),
                )
                .unwrap()
                .build()
                .unwrap();
            fmm_fft.evaluate();
            let eval_type = fmm_fft.kernel_eval_type;
            let fmm_fft = Box::new(fmm_fft);
            test_single_node_fmm_vector_helper(
                fmm_fft,
                eval_type,
                &sources,
                &charges,
                threshold_pot,
            );

            // Evaluate potentials + derivatives
            let fmm_fft = SingleNodeBuilder::new()
                .tree(&sources, &targets, n_crit, sparse)
                .unwrap()
                .parameters(
                    &charges,
                    expansion_order,
                    Laplace3dKernel::new(),
                    EvalType::ValueDeriv,
                    FftFieldTranslation::new(),
                )
                .unwrap()
                .build()
                .unwrap();
            fmm_fft.evaluate();
            let eval_type = fmm_fft.kernel_eval_type;
            let fmm_fft = Box::new(fmm_fft);
            test_single_node_fmm_vector_helper(
                fmm_fft,
                eval_type,
                &sources,
                &charges,
                threshold_deriv,
            );
        }

        // fmm with BLAS based field translation
        {
            // Evaluate potentials
            let eval_type = EvalType::Value;
            let fmm_blas = SingleNodeBuilder::new()
                .tree(&sources, &targets, n_crit, sparse)
                .unwrap()
                .parameters(
                    &charges,
                    expansion_order,
                    Laplace3dKernel::new(),
                    eval_type,
                    BlasFieldTranslation::new(singular_value_threshold),
                )
                .unwrap()
                .build()
                .unwrap();
            fmm_blas.evaluate();
            let fmm_blas = Box::new(fmm_blas);
            test_single_node_fmm_vector_helper(
                fmm_blas,
                eval_type,
                &sources,
                &charges,
                threshold_pot,
            );

            // Evaluate potentials + derivatives
            let eval_type = EvalType::ValueDeriv;
            let fmm_blas = SingleNodeBuilder::new()
                .tree(&sources, &targets, n_crit, sparse)
                .unwrap()
                .parameters(
                    &charges,
                    expansion_order,
                    Laplace3dKernel::new(),
                    eval_type,
                    BlasFieldTranslation::new(singular_value_threshold),
                )
                .unwrap()
                .build()
                .unwrap();
            fmm_blas.evaluate();
            let fmm_blas = Box::new(fmm_blas);
            test_single_node_fmm_vector_helper(
                fmm_blas,
                eval_type,
                &sources,
                &charges,
                threshold_deriv_blas,
            );
        }
    }

    #[test]
    fn test_laplace_fmm_matrix() {
        // Setup random sources and targets
        let nsources = 9000;
        let ntargets = 10000;

        let min = None;
        let max = None;
        let sources = points_fixture::<f64>(nsources, min, max, Some(0));
        let targets = points_fixture::<f64>(ntargets, min, max, Some(1));
        // FMM parameters
        let n_crit = Some(10);
        let expansion_order = 6;
        let sparse = true;
        let threshold = 1e-5;
        let threshold_deriv = 1e-3;
        let singular_value_threshold = Some(1e-2);

        // Charge data
        let nvecs = 5;
        let mut charges = rlst_dynamic_array2!(f64, [nsources, nvecs]);
        let mut rng = StdRng::seed_from_u64(0);
        charges
            .data_mut()
            .chunks_exact_mut(nsources)
            .for_each(|chunk| chunk.iter_mut().for_each(|elem| *elem += rng.gen::<f64>()));

        // fmm with blas based field translation
        {
            // Evaluate potentials
            let eval_type = EvalType::Value;
            let fmm_blas = SingleNodeBuilder::new()
                .tree(&sources, &targets, n_crit, sparse)
                .unwrap()
                .parameters(
                    &charges,
                    expansion_order,
                    Laplace3dKernel::new(),
                    eval_type,
                    BlasFieldTranslation::new(singular_value_threshold),
                )
                .unwrap()
                .build()
                .unwrap();
            fmm_blas.evaluate();

            let fmm_blas = Box::new(fmm_blas);
            test_single_node_fmm_matrix_helper(fmm_blas, eval_type, &sources, &charges, threshold);

            // Evaluate potentials + derivatives
            let eval_type = EvalType::ValueDeriv;
            let fmm_blas = SingleNodeBuilder::new()
                .tree(&sources, &targets, n_crit, sparse)
                .unwrap()
                .parameters(
                    &charges,
                    expansion_order,
                    Laplace3dKernel::new(),
                    eval_type,
                    BlasFieldTranslation::new(singular_value_threshold),
                )
                .unwrap()
                .build()
                .unwrap();
            fmm_blas.evaluate();
            let fmm_blas = Box::new(fmm_blas);
            test_single_node_fmm_matrix_helper(
                fmm_blas,
                eval_type,
                &sources,
                &charges,
                threshold_deriv,
            );
        }
    }

    fn test_root_multipole_laplace_single_node<T: RlstScalar<Real = T> + Float + Default>(
        fmm: Box<
            dyn Fmm<
                Scalar = T,
                Node = MortonKey,
                Kernel = Laplace3dKernel<T>,
                Tree = SingleNodeFmmTree<T>,
            >,
        >,
        sources: &Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>,
        charges: &Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>,
        threshold: T,
    ) {
        let multipole = fmm.multipole(&ROOT).unwrap();
        let upward_equivalent_surface = ROOT.compute_kifmm_surface(
            fmm.tree().domain(),
            fmm.expansion_order(),
            T::from(ALPHA_INNER).unwrap(),
        );

        let test_point = vec![T::from(100000.).unwrap(), T::zero(), T::zero()];
        let mut expected = vec![T::zero()];
        let mut found = vec![T::zero()];

        fmm.kernel().evaluate_st(
            EvalType::Value,
            sources.data(),
            &test_point,
            charges.data(),
            &mut expected,
        );

        fmm.kernel().evaluate_st(
            EvalType::Value,
            &upward_equivalent_surface,
            &test_point,
            multipole,
            &mut found,
        );

        let abs_error = num::Float::abs(expected[0] - found[0]);
        let rel_error = abs_error / expected[0];

        assert!(rel_error <= threshold);
    }

    #[test]
    fn test_upward_pass_vector() {
        // Setup random sources and targets
        let nsources = 10000;
        let ntargets = 10000;
        let sources = points_fixture::<f64>(nsources, None, None, Some(1));
        let targets = points_fixture::<f64>(ntargets, None, None, Some(1));

        // FMM parameters
        let n_crit = Some(100);
        let expansion_order = 6;
        let sparse = true;

        // Charge data
        let nvecs = 1;
        let mut rng = StdRng::seed_from_u64(0);
        let mut charges = rlst_dynamic_array2!(f64, [nsources, nvecs]);
        charges.data_mut().iter_mut().for_each(|c| *c = rng.gen());

        let fmm_fft = SingleNodeBuilder::new()
            .tree(&sources, &targets, n_crit, sparse)
            .unwrap()
            .parameters(
                &charges,
                expansion_order,
                Laplace3dKernel::new(),
                EvalType::Value,
                FftFieldTranslation::new(),
            )
            .unwrap()
            .build()
            .unwrap();
        fmm_fft.evaluate();

        let svd_threshold = Some(1e-5);
        let fmm_svd = SingleNodeBuilder::new()
            .tree(&sources, &targets, n_crit, sparse)
            .unwrap()
            .parameters(
                &charges,
                expansion_order,
                Laplace3dKernel::new(),
                EvalType::Value,
                BlasFieldTranslation::new(svd_threshold),
            )
            .unwrap()
            .build()
            .unwrap();
        fmm_svd.evaluate();

        let fmm_fft = Box::new(fmm_fft);
        let fmm_svd = Box::new(fmm_svd);
        test_root_multipole_laplace_single_node(fmm_fft, &sources, &charges, 1e-5);
        test_root_multipole_laplace_single_node(fmm_svd, &sources, &charges, 1e-5);
    }
}
