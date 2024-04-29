//! Single Node FMM
use green_kernels::traits::Kernel as KernelTrait;

use rlst::{RawAccess, RlstScalar, Shape};

use crate::{
    fmm::types::{FmmEvalType, KiFmm},
    traits::{
        field::SourceToTargetData as SourceToTargetDataTrait,
        fmm::{
            FmmOperatorData, HomogenousKernel, SourceToTargetTranslation, SourceTranslation,
            TargetTranslation,
        },
        tree::{FmmTree, Tree},
        types::FmmError,
    },
    Fmm, SingleNodeFmmTree,
};

use super::{
    helpers::{leaf_expansion_pointers, level_expansion_pointers, map_charges, potential_pointers},
    types::Charges,
};

impl<Scalar, Kernel, SourceToTargetData> Fmm for KiFmm<Scalar, Kernel, SourceToTargetData>
where
    Scalar: RlstScalar + Default,
    Kernel: KernelTrait<T = Scalar> + HomogenousKernel + Default + Send + Sync,
    SourceToTargetData: SourceToTargetDataTrait + Send + Sync,
    <Scalar as RlstScalar>::Real: Default,
    Self: SourceToTargetTranslation + FmmOperatorData,
{
    type Scalar = Scalar;
    type Kernel = Kernel;
    type Tree = SingleNodeFmmTree<Scalar::Real>;

    fn dim(&self) -> usize {
        self.dim
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

    fn multipole(
        &self,
        key: &<<Self::Tree as crate::traits::tree::FmmTree>::Tree as crate::traits::tree::Tree>::Node,
    ) -> Option<&[Self::Scalar]> {
        if let Some(index) = self.tree().source_tree().index(key) {
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

    fn local(
        &self,
        key: &<<Self::Tree as FmmTree>::Tree as Tree>::Node,
    ) -> Option<&[Self::Scalar]> {
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

    fn potential(
        &self,
        leaf: &<<Self::Tree as FmmTree>::Tree as Tree>::Node,
    ) -> Option<Vec<&[Self::Scalar]>> {
        if let Some(&leaf_idx) = self.tree.target_tree().leaf_index(leaf) {
            let (l, r) = self.charge_index_pointer_targets[leaf_idx];
            let ntargets = r - l;

            match self.fmm_eval_type {
                FmmEvalType::Vector => Some(vec![
                    &self.potentials[l * self.kernel_eval_size..r * self.kernel_eval_size],
                ]),
                FmmEvalType::Matrix(nmatvecs) => {
                    let n_leaves = self.tree.target_tree().n_leaves().unwrap();
                    let mut slices = Vec::new();
                    for eval_idx in 0..nmatvecs {
                        let potentials_pointer =
                            self.potentials_send_pointers[eval_idx * n_leaves + leaf_idx].raw;
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

    fn evaluate(&self) -> Result<(), FmmError> {
        // Upward pass
        {
            self.p2m()?;
            for level in (1..=self.tree().source_tree().depth()).rev() {
                self.m2m(level)?;
            }
        }

        // Downward pass
        {
            for level in 2..=self.tree().target_tree().depth() {
                if level > 2 {
                    self.l2l(level)?;
                }
                self.m2l(level)?;
            }

            // Leaf level computation
            self.p2p()?;
            self.l2p()?;
        }

        Ok(())
    }

    fn clear(&mut self, charges: &Charges<Self::Scalar>) {
        let [_ncharges, nmatvecs] = charges.shape();
        let ntarget_points = self.tree().target_tree().n_coordinates_tot().unwrap();
        let nsource_leaves = self.tree().source_tree().n_leaves().unwrap();
        let ntarget_leaves = self.tree().target_tree().n_leaves().unwrap();

        // Clear buffers and set new buffers
        self.multipoles = vec![Scalar::default(); self.multipoles.len()];
        self.locals = vec![Scalar::default(); self.locals.len()];
        self.potentials = vec![Scalar::default(); self.potentials.len()];
        self.charges = vec![Scalar::default(); self.charges.len()];

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

#[allow(clippy::type_complexity)]
#[cfg(test)]
mod test {
    use green_kernels::{
        helmholtz_3d::Helmholtz3dKernel, laplace_3d::Laplace3dKernel, traits::Kernel,
        types::EvalType,
    };
    use num::{Float, One, Zero};
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use rlst::{
        c64, rlst_array_from_slice2, rlst_dynamic_array2, Array, BaseArray, RawAccess,
        RawAccessMut, RlstScalar, Shape, VectorContainer,
    };

    use crate::{
        fmm::types::BlasFieldTranslationIa,
        traits::tree::{FmmTree, FmmTreeNode, Tree},
        tree::{constants::ALPHA_INNER, helpers::points_fixture, types::MortonKey},
        BlasFieldTranslationSaRcmp, FftFieldTranslation, Fmm, SingleNodeBuilder, SingleNodeFmmTree,
    };

    fn test_single_node_laplace_fmm_matrix_helper<T: RlstScalar<Real = T> + Float + Default>(
        fmm: Box<
            dyn Fmm<
                Scalar = T::Real,
                Kernel = Laplace3dKernel<T::Real>,
                Tree = SingleNodeFmmTree<T::Real>,
            >,
        >,
        eval_type: EvalType,
        sources: &Array<T::Real, BaseArray<T::Real, VectorContainer<T::Real>, 2>, 2>,
        charges: &Array<T::Real, BaseArray<T::Real, VectorContainer<T::Real>, 2>, 2>,
        threshold: T::Real,
    ) where
        T::Real: Default,
    {
        let eval_size = match eval_type {
            EvalType::Value => 1,
            EvalType::ValueDeriv => 4,
        };

        let leaf_idx = 0;
        let leaf = fmm.tree().target_tree().all_leaves().unwrap()[leaf_idx];

        let leaf_targets = fmm.tree().target_tree().coordinates(&leaf).unwrap();

        let ntargets = leaf_targets.len() / fmm.dim();

        let leaf_coordinates_row_major =
            rlst_array_from_slice2!(leaf_targets, [ntargets, fmm.dim()], [fmm.dim(), 1]);

        let mut leaf_coordinates_col_major = rlst_dynamic_array2!(T::Real, [ntargets, fmm.dim()]);
        leaf_coordinates_col_major.fill_from(leaf_coordinates_row_major.view());

        let [nsources, nmatvecs] = charges.shape();

        for i in 0..nmatvecs {
            let potential_i = fmm.potential(&leaf).unwrap()[i];
            let charges_i = &charges.data()[nsources * i..nsources * (i + 1)];
            let mut direct_i = vec![T::Real::zero(); ntargets * eval_size];
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
                let abs_error = RlstScalar::abs(d - p);
                let rel_error = abs_error / p;
                assert!(rel_error <= threshold)
            })
        }
    }

    fn test_single_node_helmholtz_fmm_matrix_helper<T: RlstScalar<Complex = T> + Default>(
        fmm: Box<
            dyn Fmm<Scalar = T, Kernel = Helmholtz3dKernel<T>, Tree = SingleNodeFmmTree<T::Real>>,
        >,
        eval_type: EvalType,
        sources: &Array<T::Real, BaseArray<T::Real, VectorContainer<T::Real>, 2>, 2>,
        charges: &Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>,
        threshold: T::Real,
    ) where
        T::Real: Default,
    {
        let eval_size = match eval_type {
            EvalType::Value => 1,
            EvalType::ValueDeriv => 4,
        };

        let leaf_idx = 0;
        let leaf = fmm.tree().target_tree().all_leaves().unwrap()[leaf_idx];

        let leaf_targets = fmm.tree().target_tree().coordinates(&leaf).unwrap();

        let ntargets = leaf_targets.len() / fmm.dim();

        let leaf_coordinates_row_major =
            rlst_array_from_slice2!(leaf_targets, [ntargets, fmm.dim()], [fmm.dim(), 1]);

        let mut leaf_coordinates_col_major = rlst_dynamic_array2!(T::Real, [ntargets, fmm.dim()]);
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
                let abs_error = (d - p).abs();
                let rel_error = abs_error / p.abs();
                assert!(rel_error <= threshold)
            })
        }
    }

    fn test_single_node_helmholtz_fmm_vector_helper<T: RlstScalar<Complex = T> + Default>(
        fmm: Box<
            dyn Fmm<Scalar = T, Kernel = Helmholtz3dKernel<T>, Tree = SingleNodeFmmTree<T::Real>>,
        >,
        eval_type: EvalType,
        sources: &Array<T::Real, BaseArray<T::Real, VectorContainer<T::Real>, 2>, 2>,
        charges: &Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>,
        threshold: T::Real,
    ) where
        T::Real: Default,
    {
        let eval_size = match eval_type {
            EvalType::Value => 1,
            EvalType::ValueDeriv => 4,
        };

        let leaf_idx = 0;
        let leaf = fmm.tree().target_tree().all_leaves().unwrap()[leaf_idx];
        let potential = fmm.potential(&leaf).unwrap()[0];

        let leaf_targets = fmm.tree().target_tree().coordinates(&leaf).unwrap();

        let ntargets = leaf_targets.len() / fmm.dim();
        let mut direct = vec![T::zero(); ntargets * eval_size];

        let leaf_coordinates_row_major =
            rlst_array_from_slice2!(leaf_targets, [ntargets, fmm.dim()], [fmm.dim(), 1]);
        let mut leaf_coordinates_col_major = rlst_dynamic_array2!(T::Real, [ntargets, fmm.dim()]);
        leaf_coordinates_col_major.fill_from(leaf_coordinates_row_major.view());

        fmm.kernel().evaluate_st(
            eval_type,
            sources.data(),
            leaf_coordinates_col_major.data(),
            charges.data(),
            &mut direct,
        );

        direct.iter().zip(potential).for_each(|(&d, &p)| {
            let abs_error = (d - p).abs();
            let rel_error = abs_error / p.abs();

            println!("err {:?} \nd {:?} \np {:?}", rel_error, direct, potential);
            assert!(rel_error <= threshold)
        });
    }

    fn test_single_node_laplace_fmm_vector_helper<T: RlstScalar + Float + Default>(
        fmm: Box<
            dyn Fmm<
                Scalar = T::Real,
                Kernel = Laplace3dKernel<T::Real>,
                Tree = SingleNodeFmmTree<T::Real>,
            >,
        >,
        eval_type: EvalType,
        sources: &Array<T::Real, BaseArray<T::Real, VectorContainer<T::Real>, 2>, 2>,
        charges: &Array<T::Real, BaseArray<T::Real, VectorContainer<T::Real>, 2>, 2>,
        threshold: T::Real,
    ) where
        T::Real: Default,
    {
        let eval_size = match eval_type {
            EvalType::Value => 1,
            EvalType::ValueDeriv => 4,
        };

        let leaf_idx = 0;
        let leaf = fmm.tree().target_tree().all_leaves().unwrap()[leaf_idx];
        let potential = fmm.potential(&leaf).unwrap()[0];

        let leaf_targets = fmm.tree().target_tree().coordinates(&leaf).unwrap();

        let ntargets = leaf_targets.len() / fmm.dim();
        let mut direct = vec![T::Real::zero(); ntargets * eval_size];

        let leaf_coordinates_row_major =
            rlst_array_from_slice2!(leaf_targets, [ntargets, fmm.dim()], [fmm.dim(), 1]);
        let mut leaf_coordinates_col_major = rlst_dynamic_array2!(T::Real, [ntargets, fmm.dim()]);
        leaf_coordinates_col_major.fill_from(leaf_coordinates_row_major.view());

        fmm.kernel().evaluate_st(
            eval_type,
            sources.data(),
            leaf_coordinates_col_major.data(),
            charges.data(),
            &mut direct,
        );

        direct.iter().zip(potential).for_each(|(&d, &p)| {
            let abs_error = RlstScalar::abs(d - p);
            let rel_error = abs_error / p;
            println!("err {:?} \nd {:?} \np {:?}", rel_error, direct, potential);
            assert!(rel_error <= threshold)
        });
    }

    fn test_root_multipole_laplace_single_node<T: RlstScalar + Float + Default>(
        fmm: Box<
            dyn Fmm<
                Scalar = T::Real,
                Kernel = Laplace3dKernel<T::Real>,
                Tree = SingleNodeFmmTree<T::Real>,
            >,
        >,
        sources: &Array<T::Real, BaseArray<T::Real, VectorContainer<T::Real>, 2>, 2>,
        charges: &Array<T::Real, BaseArray<T::Real, VectorContainer<T::Real>, 2>, 2>,
        threshold: T::Real,
    ) where
        T::Real: Default,
    {
        let root = MortonKey::root();

        let multipole = fmm.multipole(&root).unwrap();
        let upward_equivalent_surface = root.surface_grid(
            fmm.expansion_order(),
            fmm.tree().domain(),
            T::from(ALPHA_INNER).unwrap().re(),
        );

        let test_point = vec![T::real(100000.), T::Real::zero(), T::Real::zero()];
        let mut expected = vec![T::Real::zero()];
        let mut found = vec![T::Real::zero()];

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

        let abs_error = RlstScalar::abs(expected[0] - found[0]);
        let rel_error = abs_error / expected[0];

        assert!(rel_error <= threshold);
    }

    fn test_root_multipole_helmholtz_single_node<T: RlstScalar<Complex = T> + Default>(
        fmm: Box<
            dyn Fmm<Scalar = T, Kernel = Helmholtz3dKernel<T>, Tree = SingleNodeFmmTree<T::Real>>,
        >,
        sources: &Array<T::Real, BaseArray<T::Real, VectorContainer<T::Real>, 2>, 2>,
        charges: &Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>,
        threshold: T::Real,
    ) where
        T::Real: Default,
    {
        let root = MortonKey::<T::Real>::root();
        let multipole = fmm.multipole(&root).unwrap();

        let upward_equivalent_surface = root.surface_grid(
            fmm.expansion_order(),
            fmm.tree().domain(),
            T::from(ALPHA_INNER).unwrap().re(),
        );

        let test_point = vec![T::real(1000.), T::Real::zero(), T::Real::zero()];
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

        let abs_error = (expected[0] - found[0]).abs();
        let rel_error = abs_error / expected[0].abs();
        println!(
            "abs {:?} rel {:?} \n expected {:?} found {:?}",
            abs_error, rel_error, expected, found
        );
        assert!(rel_error <= threshold);
    }

    #[test]
    fn test_upward_pass_vector_laplace() {
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
        fmm_fft.evaluate().unwrap();

        let svd_threshold = Some(1e-5);
        let fmm_svd = SingleNodeBuilder::new()
            .tree(&sources, &targets, n_crit, sparse)
            .unwrap()
            .parameters(
                &charges,
                expansion_order,
                Laplace3dKernel::new(),
                EvalType::Value,
                BlasFieldTranslationSaRcmp::new(svd_threshold),
            )
            .unwrap()
            .build()
            .unwrap();
        fmm_svd.evaluate().unwrap();

        let fmm_fft = Box::new(fmm_fft);
        let fmm_svd = Box::new(fmm_svd);
        test_root_multipole_laplace_single_node::<f64>(fmm_fft, &sources, &charges, 1e-5);
        test_root_multipole_laplace_single_node::<f64>(fmm_svd, &sources, &charges, 1e-5);
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
        fmm.evaluate().unwrap();

        // Reset Charge data and re-evaluate potential
        let mut rng = StdRng::seed_from_u64(1);
        charges.data_mut().iter_mut().for_each(|c| *c = rng.gen());

        fmm.clear(&charges);
        fmm.evaluate().unwrap();

        let fmm = Box::new(fmm);
        test_single_node_laplace_fmm_vector_helper::<f64>(
            fmm,
            EvalType::Value,
            &sources,
            &charges,
            threshold_pot,
        );
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

        // FFT based field translation
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
            fmm_fft.evaluate().unwrap();
            let eval_type = fmm_fft.kernel_eval_type;
            let fmm_fft = Box::new(fmm_fft);
            test_single_node_laplace_fmm_vector_helper::<f64>(
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
            fmm_fft.evaluate().unwrap();
            let eval_type = fmm_fft.kernel_eval_type;
            let fmm_fft = Box::new(fmm_fft);
            test_single_node_laplace_fmm_vector_helper::<f64>(
                fmm_fft,
                eval_type,
                &sources,
                &charges,
                threshold_deriv,
            );
        }

        // BLAS based field translation
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
                    BlasFieldTranslationSaRcmp::new(singular_value_threshold),
                )
                .unwrap()
                .build()
                .unwrap();
            fmm_blas.evaluate().unwrap();
            let fmm_blas = Box::new(fmm_blas);
            test_single_node_laplace_fmm_vector_helper::<f64>(
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
                    BlasFieldTranslationSaRcmp::new(singular_value_threshold),
                )
                .unwrap()
                .build()
                .unwrap();
            fmm_blas.evaluate().unwrap();
            let fmm_blas = Box::new(fmm_blas);
            test_single_node_laplace_fmm_vector_helper::<f64>(
                fmm_blas,
                eval_type,
                &sources,
                &charges,
                threshold_deriv_blas,
            );
        }
    }

    #[test]
    fn test_upward_pass_vector_helmholtz() {
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
        let mut charges = rlst_dynamic_array2!(c64, [nsources, nvecs]);
        charges.data_mut().iter_mut().for_each(|c| *c = rng.gen());

        let wavenumber = 2.5;

        let fmm_fft = SingleNodeBuilder::new()
            .tree(&sources, &targets, n_crit, sparse)
            .unwrap()
            .parameters(
                &charges,
                expansion_order,
                Helmholtz3dKernel::new(wavenumber),
                EvalType::Value,
                FftFieldTranslation::new(),
            )
            .unwrap()
            .build()
            .unwrap();

        fmm_fft.evaluate().unwrap();
        let fmm_fft = Box::new(fmm_fft);
        test_root_multipole_helmholtz_single_node(fmm_fft, &sources, &charges, 1e-5);
    }

    #[test]
    fn test_helmholtz_fmm_vector() {
        // Setup random sources and targets
        let nsources = 9000;
        let ntargets = 10000;
        let sources = points_fixture::<f64>(nsources, None, None, Some(1));
        let targets = points_fixture::<f64>(ntargets, None, None, Some(1));
        let threshold = 1e-5;
        let threshold_deriv = 1e-3;

        // FMM parameters
        let n_crit = Some(100);
        let expansion_order = 6;
        let sparse = true;
        let wavenumber = 2.5;

        // Charge data
        let nvecs = 1;
        let mut rng = StdRng::seed_from_u64(0);
        let mut charges = rlst_dynamic_array2!(c64, [nsources, nvecs]);
        charges.data_mut().iter_mut().for_each(|c| *c = rng.gen());

        // BLAS based field translation
        {
            // Evaluate potentials
            let fmm = SingleNodeBuilder::new()
                .tree(&sources, &targets, n_crit, sparse)
                .unwrap()
                .parameters(
                    &charges,
                    expansion_order,
                    Helmholtz3dKernel::new(wavenumber),
                    EvalType::Value,
                    BlasFieldTranslationIa::new(None),
                )
                .unwrap()
                .build()
                .unwrap();
            fmm.evaluate().unwrap();

            let fmm: Box<_> = Box::new(fmm);
            let eval_type = fmm.kernel_eval_type;
            test_single_node_helmholtz_fmm_vector_helper::<c64>(
                fmm, eval_type, &sources, &charges, threshold,
            );

            // Evaluate potentials + derivatives
            let fmm = SingleNodeBuilder::new()
                .tree(&sources, &targets, n_crit, sparse)
                .unwrap()
                .parameters(
                    &charges,
                    expansion_order,
                    Helmholtz3dKernel::new(wavenumber),
                    EvalType::ValueDeriv,
                    BlasFieldTranslationIa::new(None),
                )
                .unwrap()
                .build()
                .unwrap();
            fmm.evaluate().unwrap();
            let eval_type = fmm.kernel_eval_type;
            let fmm = Box::new(fmm);
            test_single_node_helmholtz_fmm_vector_helper::<c64>(
                fmm,
                eval_type,
                &sources,
                &charges,
                threshold_deriv,
            );
        }

        // FFT based field translation
        {
            // Evaluate potentials
            let fmm = SingleNodeBuilder::new()
                .tree(&sources, &targets, n_crit, sparse)
                .unwrap()
                .parameters(
                    &charges,
                    expansion_order,
                    Helmholtz3dKernel::new(wavenumber),
                    EvalType::Value,
                    FftFieldTranslation::new(),
                )
                .unwrap()
                .build()
                .unwrap();
            fmm.evaluate().unwrap();

            let fmm: Box<_> = Box::new(fmm);
            let eval_type = fmm.kernel_eval_type;
            test_single_node_helmholtz_fmm_vector_helper::<c64>(
                fmm, eval_type, &sources, &charges, 1e-5,
            );

            // Evaluate potentials + derivatives
            let fmm = SingleNodeBuilder::new()
                .tree(&sources, &targets, n_crit, sparse)
                .unwrap()
                .parameters(
                    &charges,
                    expansion_order,
                    Helmholtz3dKernel::new(wavenumber),
                    EvalType::ValueDeriv,
                    FftFieldTranslation::new(),
                )
                .unwrap()
                .build()
                .unwrap();
            fmm.evaluate().unwrap();
            let eval_type = fmm.kernel_eval_type;
            let fmm = Box::new(fmm);
            test_single_node_helmholtz_fmm_vector_helper::<c64>(
                fmm,
                eval_type,
                &sources,
                &charges,
                threshold_deriv,
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
        let mut rng = StdRng::seed_from_u64(0);
        let mut charges = rlst_dynamic_array2!(f64, [nsources, nvecs]);
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
                    BlasFieldTranslationSaRcmp::new(singular_value_threshold),
                )
                .unwrap()
                .build()
                .unwrap();
            fmm_blas.evaluate().unwrap();

            let fmm_blas = Box::new(fmm_blas);
            test_single_node_laplace_fmm_matrix_helper::<f64>(
                fmm_blas, eval_type, &sources, &charges, threshold,
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
                    BlasFieldTranslationSaRcmp::new(singular_value_threshold),
                )
                .unwrap()
                .build()
                .unwrap();
            fmm_blas.evaluate().unwrap();
            let fmm_blas = Box::new(fmm_blas);
            test_single_node_laplace_fmm_matrix_helper::<f64>(
                fmm_blas,
                eval_type,
                &sources,
                &charges,
                threshold_deriv,
            );
        }
    }

    #[test]
    fn test_helmholtz_fmm_matrix() {
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
        let wavenumber = 1.0;

        // Charge data
        let nvecs = 2;
        let mut rng = StdRng::seed_from_u64(0);
        let mut charges = rlst_dynamic_array2!(c64, [nsources, nvecs]);
        charges
            .data_mut()
            .chunks_exact_mut(nsources)
            .for_each(|chunk| chunk.iter_mut().for_each(|elem| *elem += rng.gen::<c64>()));

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
                    Helmholtz3dKernel::new(wavenumber),
                    eval_type,
                    BlasFieldTranslationIa::new(singular_value_threshold),
                )
                .unwrap()
                .build()
                .unwrap();
            fmm_blas.evaluate().unwrap();

            let fmm_blas = Box::new(fmm_blas);
            test_single_node_helmholtz_fmm_matrix_helper::<c64>(
                fmm_blas, eval_type, &sources, &charges, threshold,
            );

            // Evaluate potentials + derivatives
            let eval_type = EvalType::ValueDeriv;
            let fmm_blas = SingleNodeBuilder::new()
                .tree(&sources, &targets, n_crit, sparse)
                .unwrap()
                .parameters(
                    &charges,
                    expansion_order,
                    Helmholtz3dKernel::new(wavenumber),
                    eval_type,
                    BlasFieldTranslationIa::new(singular_value_threshold),
                )
                .unwrap()
                .build()
                .unwrap();
            fmm_blas.evaluate().unwrap();
            let fmm_blas = Box::new(fmm_blas);
            test_single_node_helmholtz_fmm_matrix_helper::<c64>(
                fmm_blas,
                eval_type,
                &sources,
                &charges,
                threshold_deriv,
            );
        }
    }
}
