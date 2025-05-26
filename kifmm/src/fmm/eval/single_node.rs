//! Single Node FMM
use green_kernels::traits::Kernel as KernelTrait;
use rlst::RlstScalar;

use crate::{
    fmm::helpers::single_node::optionally_time,
    traits::{
        field::{
            FieldTranslation as FieldTranslationTrait, SourceToTargetTranslation,
            SourceTranslation, TargetTranslation,
        },
        fmm::{DataAccess, HomogenousKernel},
        tree::{SingleFmmTree, SingleTree},
        types::{FmmError, FmmOperatorTime, FmmOperatorType},
    },
    Evaluate, KiFmm,
};

impl<Scalar, Kernel, FieldTranslation> Evaluate for KiFmm<Scalar, Kernel, FieldTranslation>
where
    Scalar: RlstScalar + Default,
    Kernel: KernelTrait<T = Scalar> + HomogenousKernel + Default + Send + Sync,
    FieldTranslation: FieldTranslationTrait + Send + Sync,
    <Scalar as RlstScalar>::Real: Default,
    Self: DataAccess + SourceTranslation + TargetTranslation + SourceToTargetTranslation,
{
    #[inline(always)]
    fn evaluate_leaf_sources(&mut self) -> Result<(), FmmError> {
        let (result, duration) = optionally_time(self.timed, || self.p2m());

        result?;

        if let Some(d) = duration {
            self.operator_times
                .push(FmmOperatorTime::from_duration(FmmOperatorType::P2M, d));
        }

        Ok(())
    }

    #[inline(always)]
    fn evaluate_upward_pass(&mut self) -> Result<(), FmmError> {
        for level in (1..=self.tree().source_tree().depth()).rev() {
            let (result, duration) = optionally_time(self.timed, || self.m2m(level));

            result?;

            if let Some(d) = duration {
                self.operator_times.push(FmmOperatorTime::from_duration(
                    FmmOperatorType::M2M(level),
                    d,
                ));
            }
        }

        Ok(())
    }

    #[inline(always)]
    fn evaluate_downward_pass(&mut self) -> Result<(), FmmError> {
        for level in 2..=self.tree().target_tree().depth() {
            if level > 2 {
                let (result, duration) = optionally_time(self.timed, || self.l2l(level));

                result?;

                if let Some(d) = duration {
                    self.operator_times.push(FmmOperatorTime::from_duration(
                        FmmOperatorType::L2L(level),
                        d,
                    ));
                }
            }

            let (result, duration) = optionally_time(self.timed, || self.m2l(level));

            result?;

            if let Some(d) = duration {
                self.operator_times.push(FmmOperatorTime::from_duration(
                    FmmOperatorType::M2L(level),
                    d,
                ));
            }
        }
        Ok(())
    }

    #[inline(always)]
    fn evaluate_leaf_targets(&mut self) -> Result<(), FmmError> {
        let (result, duration) = optionally_time(self.timed, || self.p2p());

        result?;

        if let Some(d) = duration {
            self.operator_times
                .push(FmmOperatorTime::from_duration(FmmOperatorType::P2P, d));
        }

        let (result, duration) = optionally_time(self.timed, || self.l2p());

        result?;

        if let Some(d) = duration {
            self.operator_times
                .push(FmmOperatorTime::from_duration(FmmOperatorType::L2P, d));
        }

        Ok(())
    }

    fn evaluate(&mut self) -> Result<(), FmmError> {
        self.evaluate_leaf_sources()?;
        self.evaluate_upward_pass()?;
        self.evaluate_downward_pass()?;
        self.evaluate_leaf_targets()?;
        Ok(())
    }
}

#[allow(clippy::type_complexity)]
#[cfg(test)]
mod test {
    use green_kernels::{
        helmholtz_3d::Helmholtz3dKernel, laplace_3d::Laplace3dKernel, traits::Kernel,
        types::GreenKernelEvalType,
    };
    use num::{Float, Zero};
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use rlst::{
        c64, rlst_dynamic_array2, Array, BaseArray, RawAccess, RawAccessMut, RlstScalar, Shape,
        VectorContainer,
    };

    use crate::{
        fmm::{helpers::single_node::l2_error, types::BlasFieldTranslationIa},
        traits::{
            fmm::ChargeHandler,
            tree::{FmmTreeNode, SingleFmmTree, SingleTree},
        },
        tree::{constants::ALPHA_INNER, helpers::points_fixture, types::MortonKey},
        BlasFieldTranslationSaRcmp, Evaluate, FftFieldTranslation, FmmSvdMode, SingleNodeBuilder,
        SingleNodeFmmTree,
    };

    fn test_single_node_laplace_fmm_matrix_helper<T: RlstScalar<Real = T> + Float + Default>(
        fmm: Box<
            dyn Evaluate<
                Scalar = T::Real,
                Kernel = Laplace3dKernel<T::Real>,
                Tree = SingleNodeFmmTree<T::Real>,
            >,
        >,
        eval_type: GreenKernelEvalType,
        sources: &Array<T::Real, BaseArray<T::Real, VectorContainer<T::Real>, 2>, 2>,
        charges: &Array<T::Real, BaseArray<T::Real, VectorContainer<T::Real>, 2>, 2>,
        threshold: T::Real,
    ) where
        T::Real: Default,
    {
        let eval_size = match eval_type {
            GreenKernelEvalType::Value => 1,
            GreenKernelEvalType::ValueDeriv => 4,
        };

        let leaf_idx = 0;
        let leaf = fmm.tree().target_tree().all_leaves().unwrap()[leaf_idx];

        let leaf_targets = fmm.tree().target_tree().coordinates(&leaf).unwrap();

        let n_targets = leaf_targets.len() / fmm.dim();

        let [n_sources, n_matvecs] = charges.shape();

        for i in 0..n_matvecs {
            let potential_i = fmm.potential(&leaf).unwrap()[i];
            let charges_i = &charges.data()[n_sources * i..n_sources * (i + 1)];
            let mut direct_i = vec![T::Real::zero(); n_targets * eval_size];
            fmm.kernel().evaluate_st(
                eval_type,
                sources.data(),
                leaf_targets,
                charges_i,
                &mut direct_i,
            );

            println!(
                "i {:?} \n direct_i {:?}\n potential_i {:?}",
                i, direct_i, potential_i
            );

            let l2_error = l2_error(&direct_i, &potential_i);
            assert!(l2_error <= threshold);
        }
    }

    fn test_single_node_helmholtz_fmm_matrix_helper<T: RlstScalar<Complex = T> + Default>(
        fmm: Box<
            dyn Evaluate<
                Scalar = T,
                Kernel = Helmholtz3dKernel<T>,
                Tree = SingleNodeFmmTree<T::Real>,
            >,
        >,
        eval_type: GreenKernelEvalType,
        sources: &Array<T::Real, BaseArray<T::Real, VectorContainer<T::Real>, 2>, 2>,
        charges: &Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>,
        threshold: T::Real,
    ) where
        T::Real: Default,
    {
        let eval_size = match eval_type {
            GreenKernelEvalType::Value => 1,
            GreenKernelEvalType::ValueDeriv => 4,
        };

        let leaf_idx = 0;
        let leaf = fmm.tree().target_tree().all_leaves().unwrap()[leaf_idx];

        let leaf_targets = fmm.tree().target_tree().coordinates(&leaf).unwrap();

        let n_targets = leaf_targets.len() / fmm.dim();

        let [n_sources, n_matvecs] = charges.shape();

        for i in 0..n_matvecs {
            let potential_i = fmm.potential(&leaf).unwrap()[i];
            let charges_i = &charges.data()[n_sources * i..n_sources * (i + 1)];
            let mut direct_i = vec![T::zero(); n_targets * eval_size];
            fmm.kernel().evaluate_st(
                eval_type,
                sources.data(),
                leaf_targets,
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
            dyn Evaluate<
                Scalar = T,
                Kernel = Helmholtz3dKernel<T>,
                Tree = SingleNodeFmmTree<T::Real>,
            >,
        >,
        eval_type: GreenKernelEvalType,
        sources: &Array<T::Real, BaseArray<T::Real, VectorContainer<T::Real>, 2>, 2>,
        charges: &Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>,
        threshold: T::Real,
    ) where
        T::Real: Default,
    {
        let eval_size = match eval_type {
            GreenKernelEvalType::Value => 1,
            GreenKernelEvalType::ValueDeriv => 4,
        };

        let leaf_idx = 0;
        let leaf = fmm.tree().target_tree().all_leaves().unwrap()[leaf_idx];
        let potential = fmm.potential(&leaf).unwrap()[0];

        let leaf_targets = fmm.tree().target_tree().coordinates(&leaf).unwrap();

        let n_targets = leaf_targets.len() / fmm.dim();
        let mut direct = vec![T::zero(); n_targets * eval_size];

        fmm.kernel().evaluate_st(
            eval_type,
            sources.data(),
            leaf_targets,
            charges.data(),
            &mut direct,
        );

        direct.iter().zip(potential).for_each(|(&d, &p)| {
            let abs_error = (d - p).abs();
            let rel_error = abs_error / p.abs();

            println!(
                "err {:?} \nd {:?} \np {:?}",
                rel_error,
                &direct[0..5],
                &potential[0..5]
            );
            assert!(rel_error <= threshold)
        });
    }

    fn test_single_node_laplace_fmm_vector_helper<T: RlstScalar + Float + Default>(
        fmm: Box<
            dyn Evaluate<
                Scalar = T::Real,
                Kernel = Laplace3dKernel<T::Real>,
                Tree = SingleNodeFmmTree<T::Real>,
            >,
        >,
        eval_type: GreenKernelEvalType,
        sources: &Array<T::Real, BaseArray<T::Real, VectorContainer<T::Real>, 2>, 2>,
        charges: &Array<T::Real, BaseArray<T::Real, VectorContainer<T::Real>, 2>, 2>,
        threshold: T::Real,
    ) where
        T::Real: Default,
    {
        let eval_size = match eval_type {
            GreenKernelEvalType::Value => 1,
            GreenKernelEvalType::ValueDeriv => 4,
        };

        let leaf_idx = 0;
        let leaf = fmm.tree().target_tree().all_leaves().unwrap()[leaf_idx];
        let potential = fmm.potential(&leaf).unwrap()[0];

        let leaf_targets = fmm.tree().target_tree().coordinates(&leaf).unwrap();

        let n_targets = leaf_targets.len() / fmm.dim();
        let mut direct = vec![T::Real::zero(); n_targets * eval_size];

        fmm.kernel().evaluate_st(
            eval_type,
            sources.data(),
            leaf_targets,
            charges.data(),
            &mut direct,
        );

        let l2_error = l2_error(&direct, &potential);
        assert!(l2_error <= threshold);
    }

    fn test_root_multipole_laplace_single_node<T: RlstScalar + Float + Default>(
        fmm: Box<
            dyn Evaluate<
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

        let multipoles = fmm.multipole(&root).unwrap();
        let upward_equivalent_surface = root.surface_grid(
            fmm.equivalent_surface_order(0),
            fmm.tree().domain(),
            T::from(ALPHA_INNER).unwrap().re(),
        );

        let ncoeffs = fmm.n_coeffs_equivalent_surface(0);

        let test_point = vec![T::real(100000.), T::Real::zero(), T::Real::zero()];
        let mut expected = vec![T::Real::zero()];
        let mut found = vec![T::Real::zero()];

        let [n_sources, nvecs] = charges.shape();

        for i in 0..nvecs {
            let charges_i = &charges.data()[n_sources * i..n_sources * (i + 1)];
            let multipole_i = &multipoles[ncoeffs * i..(i + 1) * ncoeffs];

            println!(
                "root multipole {:?}, {:?} {:?}",
                &multipole_i[0..5],
                multipoles.len(),
                multipole_i.len()
            );

            fmm.kernel().evaluate_st(
                GreenKernelEvalType::Value,
                sources.data(),
                &test_point,
                charges_i,
                &mut expected,
            );

            fmm.kernel().evaluate_st(
                GreenKernelEvalType::Value,
                &upward_equivalent_surface,
                &test_point,
                multipole_i,
                &mut found,
            );

            let l2_error = l2_error(&found, &expected);
            assert!(l2_error <= threshold);
        }
    }

    fn test_root_multipole_helmholtz_single_node<T: RlstScalar<Complex = T> + Default>(
        fmm: Box<
            dyn Evaluate<
                Scalar = T,
                Kernel = Helmholtz3dKernel<T>,
                Tree = SingleNodeFmmTree<T::Real>,
            >,
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
            fmm.equivalent_surface_order(0),
            fmm.tree().domain(),
            T::from(ALPHA_INNER).unwrap().re(),
        );

        let test_point = vec![T::real(1000.), T::Real::zero(), T::Real::zero()];
        let mut expected = vec![T::zero()];
        let mut found = vec![T::zero()];

        fmm.kernel().evaluate_st(
            GreenKernelEvalType::Value,
            sources.data(),
            &test_point,
            charges.data(),
            &mut expected,
        );

        fmm.kernel().evaluate_st(
            GreenKernelEvalType::Value,
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

        let mut fmm_fft = SingleNodeBuilder::new(false)
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
        fmm_fft.evaluate().unwrap();

        let svd_threshold = Some(1e-5);
        let mut fmm_svd = SingleNodeBuilder::new(false)
            .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
            .unwrap()
            .parameters(
                charges.data(),
                &expansion_order,
                Laplace3dKernel::new(),
                GreenKernelEvalType::Value,
                BlasFieldTranslationSaRcmp::new(
                    svd_threshold,
                    None,
                    crate::fmm::types::FmmSvdMode::Deterministic,
                ),
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
    fn test_upward_pass_matrix_laplace() {
        // Setup random sources and targets
        let n_sources = 10000;
        let n_targets = 10000;
        let sources = points_fixture::<f64>(n_sources, None, None, Some(1));
        let targets = points_fixture::<f64>(n_targets, None, None, Some(1));

        // FMM parameters

        let n_crit = None;
        let depth = Some(3);
        let expansion_order = [6, 6, 6, 6];

        let prune_empty = true;

        // Charge data
        let nvecs = 2;
        let mut rng = StdRng::seed_from_u64(0);
        let mut charges = rlst_dynamic_array2!(f64, [n_sources, nvecs]);
        charges.data_mut().iter_mut().for_each(|c| *c = rng.gen());

        let svd_threshold = Some(1e-5);
        let mut fmm_svd = SingleNodeBuilder::new(false)
            .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
            .unwrap()
            .parameters(
                charges.data(),
                &expansion_order,
                Laplace3dKernel::new(),
                GreenKernelEvalType::Value,
                BlasFieldTranslationSaRcmp::new(
                    svd_threshold,
                    None,
                    crate::fmm::types::FmmSvdMode::Deterministic,
                ),
            )
            .unwrap()
            .build()
            .unwrap();
        fmm_svd.evaluate().unwrap();

        let fmm_svd = Box::new(fmm_svd);
        test_root_multipole_laplace_single_node::<f64>(fmm_svd, &sources, &charges, 1e-5);
    }

    #[test]
    fn test_fmm_api() {
        // Setup random sources and targets
        let n_sources = 9000;
        let n_targets = 10000;

        let min = None;
        let max = None;
        let sources = points_fixture::<f64>(n_sources, min, max, Some(0));
        let targets = points_fixture::<f64>(n_targets, min, max, Some(1));

        // FMM parameters
        let n_crit = Some(100);
        let depth = None;
        let expansion_order = [6];
        let prune_empty = true;
        let threshold_pot = 1e-5;

        // Set charge data and evaluate an FMM
        let nvecs = 1;
        let mut rng = StdRng::seed_from_u64(0);
        let mut charges = rlst_dynamic_array2!(f64, [n_sources, nvecs]);
        charges.data_mut().iter_mut().for_each(|c| *c = rng.gen());

        let svd_mode = crate::fmm::types::FmmSvdMode::new(false, None, None, None, None);
        let mut fmm = SingleNodeBuilder::new(false)
            .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
            .unwrap()
            .parameters(
                charges.data(),
                &expansion_order,
                Laplace3dKernel::new(),
                GreenKernelEvalType::Value,
                BlasFieldTranslationSaRcmp::new(None, None, svd_mode),
            )
            .unwrap()
            .build()
            .unwrap();

        fmm.evaluate().unwrap();
        // Reset Charge data and re-evaluate potential
        let mut rng = StdRng::seed_from_u64(1);
        charges.data_mut().iter_mut().for_each(|c| *c = rng.gen());

        let _ = fmm.attach_charges_unordered(charges.data());
        fmm.evaluate().unwrap();

        let fmm = Box::new(fmm);
        test_single_node_laplace_fmm_vector_helper::<f64>(
            fmm,
            GreenKernelEvalType::Value,
            &sources,
            &charges,
            threshold_pot,
        );
    }

    #[test]
    fn test_laplace_fmm_vector_variable_expansion_order() {
        // Setup random sources and targets
        let n_sources = 9000;
        let n_targets = 10000;

        let min = None;
        let max = None;
        let sources = points_fixture::<f64>(n_sources, min, max, Some(0));
        let targets = points_fixture::<f64>(n_targets, min, max, Some(1));

        // FMM parameters
        let n_crit = None;
        let depth = Some(3);
        let expansion_order = [5, 6, 5, 6];

        let prune_empty = true;
        let threshold_pot = 1e-3;
        let threshold_deriv = 1e-4;
        let threshold_deriv_blas = 1e-3;
        let singular_value_threshold = Some(1e-2);

        // Charge data
        let nvecs = 1;
        let mut rng = StdRng::seed_from_u64(0);
        let mut charges = rlst_dynamic_array2!(f64, [n_sources, nvecs]);
        charges.data_mut().iter_mut().for_each(|c| *c = rng.gen());

        // FFT based field translation
        {
            // Evaluate potentials
            let mut fmm_fft = SingleNodeBuilder::new(false)
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
            let mut fmm_fft = SingleNodeBuilder::new(false)
                .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
                .unwrap()
                .parameters(
                    charges.data(),
                    &expansion_order,
                    Laplace3dKernel::new(),
                    GreenKernelEvalType::ValueDeriv,
                    FftFieldTranslation::new(None),
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
            let eval_type = GreenKernelEvalType::Value;
            let mut fmm_blas = SingleNodeBuilder::new(false)
                .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
                .unwrap()
                .parameters(
                    charges.data(),
                    &expansion_order,
                    Laplace3dKernel::new(),
                    eval_type,
                    BlasFieldTranslationSaRcmp::new(
                        singular_value_threshold,
                        None,
                        crate::fmm::types::FmmSvdMode::Deterministic,
                    ),
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
            let eval_type = GreenKernelEvalType::ValueDeriv;
            let mut fmm_blas = SingleNodeBuilder::new(false)
                .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
                .unwrap()
                .parameters(
                    charges.data(),
                    &expansion_order,
                    Laplace3dKernel::new(),
                    eval_type,
                    BlasFieldTranslationSaRcmp::new(
                        singular_value_threshold,
                        None,
                        crate::fmm::types::FmmSvdMode::Deterministic,
                    ),
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
    fn test_laplace_fmm_vector_variable_surfaces() {
        // Setup random sources and targets
        let n_sources = 9000;
        let n_targets = 10000;

        let min = None;
        let max = None;
        let sources = points_fixture::<f64>(n_sources, min, max, Some(0));
        let targets = points_fixture::<f64>(n_targets, min, max, Some(1));

        // FMM parameters
        let n_crit = Some(150);
        let depth = None;
        let expansion_order = [6];
        let surface_diff = Some(1);
        let prune_empty = true;
        let threshold_pot = 1e-6;
        let threshold_deriv_blas = 1e-4;
        let singular_value_threshold = None;

        // Charge data
        let nvecs = 1;
        let mut rng = StdRng::seed_from_u64(0);
        let mut charges = rlst_dynamic_array2!(f64, [n_sources, nvecs]);
        charges.data_mut().iter_mut().for_each(|c| *c = rng.gen());

        // BLAS based field translations allow variable check/equiv surfaces
        {
            // Evaluate potentials
            let eval_type = GreenKernelEvalType::Value;
            let mut fmm_blas = SingleNodeBuilder::new(false)
                .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
                .unwrap()
                .parameters(
                    charges.data(),
                    &expansion_order,
                    Laplace3dKernel::new(),
                    eval_type,
                    BlasFieldTranslationSaRcmp::new(
                        singular_value_threshold,
                        surface_diff,
                        crate::fmm::types::FmmSvdMode::Deterministic,
                    ),
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
            let eval_type = GreenKernelEvalType::ValueDeriv;
            let mut fmm_blas = SingleNodeBuilder::new(false)
                .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
                .unwrap()
                .parameters(
                    charges.data(),
                    &expansion_order,
                    Laplace3dKernel::new(),
                    eval_type,
                    BlasFieldTranslationSaRcmp::new(
                        singular_value_threshold,
                        None,
                        crate::fmm::types::FmmSvdMode::Deterministic,
                    ),
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
    fn test_laplace_fmm_matrix_variable_surfaces() {
        // Setup random sources and targets
        let n_sources = 9000;
        let n_targets = 10000;

        let min = None;
        let max = None;
        let sources = points_fixture::<f64>(n_sources, min, max, Some(0));
        let targets = points_fixture::<f64>(n_targets, min, max, Some(1));

        // FMM parameters
        let n_crit = Some(150);
        let depth = None;
        let expansion_order = [6];
        let surface_diff = Some(1);
        let prune_empty = true;
        let threshold_pot = 1e-6;
        let singular_value_threshold = None;

        // Charge data
        let nvecs = 2;
        let mut rng = StdRng::seed_from_u64(0);
        let mut charges = rlst_dynamic_array2!(f64, [n_sources, nvecs]);
        charges.data_mut().iter_mut().for_each(|c| *c = rng.gen());

        // BLAS based field translations allow variable check/equiv surfaces
        {
            // Evaluate potentials
            let eval_type = GreenKernelEvalType::Value;
            let mut fmm_blas = SingleNodeBuilder::new(false)
                .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
                .unwrap()
                .parameters(
                    charges.data(),
                    &expansion_order,
                    Laplace3dKernel::new(),
                    eval_type,
                    BlasFieldTranslationSaRcmp::new(
                        singular_value_threshold,
                        surface_diff,
                        crate::fmm::types::FmmSvdMode::Deterministic,
                    ),
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
                threshold_pot,
            );
        }
    }

    #[test]
    fn test_laplace_fmm_vector_variable_surfaces_variable_expansion_order() {
        // Setup random sources and targets
        let n_sources = 9000;
        let n_targets = 10000;

        let min = None;
        let max = None;
        let sources = points_fixture::<f64>(n_sources, min, max, Some(0));
        let targets = points_fixture::<f64>(n_targets, min, max, Some(1));

        // FMM parameters
        let n_crit = None;
        let depth = Some(3);
        let expansion_order = [6, 5, 6, 5];
        let surface_diff = Some(1);
        let prune_empty = true;
        let threshold_pot = 1e-5;
        let threshold_deriv_blas = 1e-3;
        let singular_value_threshold = None;

        // Charge data
        let nvecs = 1;
        let mut rng = StdRng::seed_from_u64(0);
        let mut charges = rlst_dynamic_array2!(f64, [n_sources, nvecs]);
        charges.data_mut().iter_mut().for_each(|c| *c = rng.gen());

        // BLAS based field translations allow variable check/equiv surfaces
        {
            // Evaluate potentials
            let eval_type = GreenKernelEvalType::Value;
            let mut fmm_blas = SingleNodeBuilder::new(false)
                .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
                .unwrap()
                .parameters(
                    charges.data(),
                    &expansion_order,
                    Laplace3dKernel::new(),
                    eval_type,
                    BlasFieldTranslationSaRcmp::new(
                        singular_value_threshold,
                        surface_diff,
                        crate::fmm::types::FmmSvdMode::Deterministic,
                    ),
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
            let eval_type = GreenKernelEvalType::ValueDeriv;
            let mut fmm_blas = SingleNodeBuilder::new(false)
                .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
                .unwrap()
                .parameters(
                    charges.data(),
                    &expansion_order,
                    Laplace3dKernel::new(),
                    eval_type,
                    BlasFieldTranslationSaRcmp::new(
                        singular_value_threshold,
                        None,
                        crate::fmm::types::FmmSvdMode::Deterministic,
                    ),
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
        let n_sources = 10000;
        let n_targets = 10000;
        let sources = points_fixture::<f64>(n_sources, None, None, Some(1));
        let targets = points_fixture::<f64>(n_targets, None, None, Some(1));

        // FMM parameters
        // let n_crit = Some(100);
        // let depth = None;
        // let expansion_order = [6];

        let n_crit = None;
        let depth = Some(3);
        let expansion_order = [6, 6, 5, 6];

        let prune_empty = true;
        // Charge data
        let nvecs = 1;
        let mut rng = StdRng::seed_from_u64(0);
        let mut charges = rlst_dynamic_array2!(c64, [n_sources, nvecs]);
        charges.data_mut().iter_mut().for_each(|c| *c = rng.gen());

        let wavenumber = 2.5;

        let mut fmm_fft = SingleNodeBuilder::new(false)
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

        fmm_fft.evaluate().unwrap();
        let fmm_fft = Box::new(fmm_fft);
        test_root_multipole_helmholtz_single_node(fmm_fft, &sources, &charges, 1e-5);
    }

    #[test]
    fn test_helmholtz_fmm_vector() {
        // Setup random sources and targets
        let n_sources = 9000;
        let n_targets = 10000;
        let sources = points_fixture::<f64>(n_sources, None, None, Some(1));
        let targets = points_fixture::<f64>(n_targets, None, None, Some(1));
        let threshold = 1e-5;
        let threshold_deriv = 1e-3;

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

        // BLAS based field translation
        {
            // Evaluate potentials
            let mut fmm = SingleNodeBuilder::new(false)
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
            fmm.evaluate().unwrap();

            let fmm: Box<_> = Box::new(fmm);
            let eval_type = fmm.kernel_eval_type;
            test_single_node_helmholtz_fmm_vector_helper::<c64>(
                fmm, eval_type, &sources, &charges, threshold,
            );

            // Evaluate potentials + derivatives
            let mut fmm = SingleNodeBuilder::new(false)
                .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
                .unwrap()
                .parameters(
                    charges.data(),
                    &expansion_order,
                    Helmholtz3dKernel::new(wavenumber),
                    GreenKernelEvalType::ValueDeriv,
                    BlasFieldTranslationIa::new(None, None, FmmSvdMode::Deterministic),
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
            let mut fmm = SingleNodeBuilder::new(false)
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
            fmm.evaluate().unwrap();

            let fmm: Box<_> = Box::new(fmm);
            let eval_type = fmm.kernel_eval_type;
            test_single_node_helmholtz_fmm_vector_helper::<c64>(
                fmm, eval_type, &sources, &charges, 1e-5,
            );

            // Evaluate potentials + derivatives
            let mut fmm = SingleNodeBuilder::new(false)
                .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
                .unwrap()
                .parameters(
                    charges.data(),
                    &expansion_order,
                    Helmholtz3dKernel::new(wavenumber),
                    GreenKernelEvalType::ValueDeriv,
                    FftFieldTranslation::new(None),
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
    fn test_helmholtz_fmm_vector_variable_expansion_order() {
        // Setup random sources and targets
        let n_sources = 9000;
        let n_targets = 10000;
        let sources = points_fixture::<f64>(n_sources, None, None, Some(1));
        let targets = points_fixture::<f64>(n_targets, None, None, Some(1));
        let threshold = 1e-5;
        let threshold_deriv = 1e-3;

        // FMM parameters
        let n_crit = None;
        let depth = Some(3);
        let expansion_order = [5, 6, 5, 6];

        let prune_empty = true;
        let wavenumber = 2.5;

        // Charge data
        let nvecs = 1;
        let mut rng = StdRng::seed_from_u64(0);
        let mut charges = rlst_dynamic_array2!(c64, [n_sources, nvecs]);
        charges.data_mut().iter_mut().for_each(|c| *c = rng.gen());

        // BLAS based field translation
        {
            // Evaluate potentials
            let mut fmm = SingleNodeBuilder::new(false)
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
            fmm.evaluate().unwrap();

            let fmm: Box<_> = Box::new(fmm);
            let eval_type = fmm.kernel_eval_type;
            test_single_node_helmholtz_fmm_vector_helper::<c64>(
                fmm, eval_type, &sources, &charges, threshold,
            );

            // Evaluate potentials + derivatives
            let mut fmm = SingleNodeBuilder::new(false)
                .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
                .unwrap()
                .parameters(
                    charges.data(),
                    &expansion_order,
                    Helmholtz3dKernel::new(wavenumber),
                    GreenKernelEvalType::ValueDeriv,
                    BlasFieldTranslationIa::new(None, None, FmmSvdMode::Deterministic),
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
            let mut fmm = SingleNodeBuilder::new(false)
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
            fmm.evaluate().unwrap();

            let fmm: Box<_> = Box::new(fmm);
            let eval_type = fmm.kernel_eval_type;
            test_single_node_helmholtz_fmm_vector_helper::<c64>(
                fmm, eval_type, &sources, &charges, 1e-5,
            );

            // Evaluate potentials + derivatives
            let mut fmm = SingleNodeBuilder::new(false)
                .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
                .unwrap()
                .parameters(
                    charges.data(),
                    &expansion_order,
                    Helmholtz3dKernel::new(wavenumber),
                    GreenKernelEvalType::ValueDeriv,
                    FftFieldTranslation::new(None),
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
    fn test_helmholtz_fmm_vector_variable_surfaces() {
        // Setup random sources and targets
        let n_sources = 9000;
        let n_targets = 10000;
        let sources = points_fixture::<f64>(n_sources, None, None, Some(1));
        let targets = points_fixture::<f64>(n_targets, None, None, Some(1));
        let threshold = 1e-5;
        let threshold_deriv = 1e-3;

        // FMM parameters
        let n_crit = Some(100);
        let depth = None;
        let surface_diff = Some(1);
        let expansion_order = [6];

        let prune_empty = true;
        let wavenumber = 2.5;

        // Charge data
        let nvecs = 1;
        let mut rng = StdRng::seed_from_u64(0);
        let mut charges = rlst_dynamic_array2!(c64, [n_sources, nvecs]);
        charges.data_mut().iter_mut().for_each(|c| *c = rng.gen());

        // BLAS based field translation
        {
            // Evaluate potentials
            let mut fmm = SingleNodeBuilder::new(false)
                .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
                .unwrap()
                .parameters(
                    charges.data(),
                    &expansion_order,
                    Helmholtz3dKernel::new(wavenumber),
                    GreenKernelEvalType::Value,
                    BlasFieldTranslationIa::new(None, surface_diff, FmmSvdMode::Deterministic),
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
            let mut fmm = SingleNodeBuilder::new(false)
                .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
                .unwrap()
                .parameters(
                    charges.data(),
                    &expansion_order,
                    Helmholtz3dKernel::new(wavenumber),
                    GreenKernelEvalType::ValueDeriv,
                    BlasFieldTranslationIa::new(None, None, FmmSvdMode::Deterministic),
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
    fn test_helmholtz_fmm_matrix_variable_surfaces() {
        // Setup random sources and targets
        let n_sources = 9000;
        let n_targets = 10000;
        let sources = points_fixture::<f64>(n_sources, None, None, Some(1));
        let targets = points_fixture::<f64>(n_targets, None, None, Some(1));
        let threshold = 1e-5;

        // FMM parameters
        let n_crit = Some(100);
        let depth = None;
        let surface_diff = Some(1);
        let expansion_order = [6];

        let prune_empty = true;
        let wavenumber = 2.5;

        // Charge data
        let nvecs = 2;
        let mut rng = StdRng::seed_from_u64(0);
        let mut charges = rlst_dynamic_array2!(c64, [n_sources, nvecs]);
        charges.data_mut().iter_mut().for_each(|c| *c = rng.gen());

        // BLAS based field translation
        {
            // Evaluate potentials
            let mut fmm = SingleNodeBuilder::new(false)
                .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
                .unwrap()
                .parameters(
                    charges.data(),
                    &expansion_order,
                    Helmholtz3dKernel::new(wavenumber),
                    GreenKernelEvalType::Value,
                    BlasFieldTranslationIa::new(None, surface_diff, FmmSvdMode::Deterministic),
                )
                .unwrap()
                .build()
                .unwrap();
            fmm.evaluate().unwrap();

            let fmm: Box<_> = Box::new(fmm);
            let eval_type = fmm.kernel_eval_type;

            test_single_node_helmholtz_fmm_matrix_helper::<c64>(
                fmm, eval_type, &sources, &charges, threshold,
            );
        }
    }

    #[test]
    fn test_laplace_fmm_matrix() {
        // Setup random sources and targets
        let n_sources = 9000;
        let n_targets = 10000;

        let min = None;
        let max = None;
        let sources = points_fixture::<f64>(n_sources, min, max, Some(0));
        let targets = points_fixture::<f64>(n_targets, min, max, Some(1));

        // FMM parameters
        let n_crit = None;
        let depth = Some(3);
        let expansion_order = [6, 5, 6, 5];

        let prune_empty = true;
        let threshold = 1e-5;
        let threshold_deriv = 1e-3;
        let singular_value_threshold = Some(1e-2);

        // Charge data
        let nvecs = 5;
        let mut rng = StdRng::seed_from_u64(0);
        let mut charges = rlst_dynamic_array2!(f64, [n_sources, nvecs]);
        charges
            .data_mut()
            .chunks_exact_mut(n_sources)
            .for_each(|chunk| chunk.iter_mut().for_each(|elem| *elem += rng.gen::<f64>()));

        // fmm with blas based field translation
        {
            // Evaluate potentials
            let eval_type = GreenKernelEvalType::Value;
            let mut fmm_blas = SingleNodeBuilder::new(false)
                .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
                .unwrap()
                .parameters(
                    charges.data(),
                    &expansion_order,
                    Laplace3dKernel::new(),
                    eval_type,
                    BlasFieldTranslationSaRcmp::new(
                        singular_value_threshold,
                        None,
                        crate::fmm::types::FmmSvdMode::Deterministic,
                    ),
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
            let eval_type = GreenKernelEvalType::ValueDeriv;
            let mut fmm_blas = SingleNodeBuilder::new(false)
                .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
                .unwrap()
                .parameters(
                    charges.data(),
                    &expansion_order,
                    Laplace3dKernel::new(),
                    eval_type,
                    BlasFieldTranslationSaRcmp::new(
                        singular_value_threshold,
                        None,
                        crate::fmm::types::FmmSvdMode::Deterministic,
                    ),
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
        let n_sources = 9000;
        let n_targets = 10000;

        let min = None;
        let max = None;
        let sources = points_fixture::<f64>(n_sources, min, max, Some(0));
        let targets = points_fixture::<f64>(n_targets, min, max, Some(1));

        // FMM parameters
        let n_crit = None;
        let depth = Some(3);
        let expansion_order = [6, 6, 6, 6];

        let prune_empty = true;
        let threshold = 1e-5;
        let threshold_deriv = 1e-3;
        let singular_value_threshold = Some(1e-2);
        let wavenumber = 1.0;

        // Charge data
        let nvecs = 2;
        let mut rng = StdRng::seed_from_u64(0);
        let mut charges = rlst_dynamic_array2!(c64, [n_sources, nvecs]);
        charges
            .data_mut()
            .chunks_exact_mut(n_sources)
            .for_each(|chunk| chunk.iter_mut().for_each(|elem| *elem += rng.gen::<c64>()));

        // fmm with blas based field translation
        {
            // Evaluate potentials
            let eval_type = GreenKernelEvalType::Value;
            let mut fmm_blas = SingleNodeBuilder::new(false)
                .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
                .unwrap()
                .parameters(
                    charges.data(),
                    &expansion_order,
                    Helmholtz3dKernel::new(wavenumber),
                    eval_type,
                    BlasFieldTranslationIa::new(
                        singular_value_threshold,
                        None,
                        FmmSvdMode::Deterministic,
                    ),
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
            let eval_type = GreenKernelEvalType::ValueDeriv;
            let mut fmm_blas = SingleNodeBuilder::new(false)
                .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
                .unwrap()
                .parameters(
                    charges.data(),
                    &expansion_order,
                    Helmholtz3dKernel::new(wavenumber),
                    eval_type,
                    BlasFieldTranslationIa::new(
                        singular_value_threshold,
                        None,
                        FmmSvdMode::Deterministic,
                    ),
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
