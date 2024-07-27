//! Single Node FMM
use std::time::Instant;

use green_kernels::traits::Kernel as KernelTrait;

use rlst::RlstScalar;

use crate::{
    fmm::types::{FmmEvalType, KiFmm},
    traits::{
        field::SourceToTargetData as SourceToTargetDataTrait,
        fmm::{
            FmmOperatorData, HomogenousKernel, SourceToTargetTranslation, SourceTranslation,
            TargetTranslation,
        },
        tree::{FmmTree, Tree},
        types::{FmmError, FmmOperatorTime, FmmOperatorType},
    },
    Fmm, SingleNodeFmmTree,
};

use super::helpers::{
    leaf_expansion_pointers, level_expansion_pointers, map_charges, potential_pointers,
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

    fn variable_expansion_order(&self) -> bool {
        self.variable_expansion_order
    }

    fn equivalent_surface_order(&self, level: u64) -> usize {
        self.equivalent_surface_order[self.expansion_index(level)]
    }

    fn check_surface_order(&self, level: u64) -> usize {
        self.check_surface_order[self.expansion_index(level)]
    }

    fn ncoeffs_equivalent_surface(&self, level: u64) -> usize {
        self.ncoeffs_equivalent_surface[self.expansion_index(level)]
    }

    fn ncoeffs_check_surface(&self, level: u64) -> usize {
        self.ncoeffs_check_surface[self.expansion_index(level)]
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
        if let Some(&key_idx) = self.tree().source_tree().level_index(key) {
            let multipole_ptr = &self.level_multipoles[key.level() as usize][key_idx][0];

            unsafe {
                match self.fmm_eval_type {
                    FmmEvalType::Vector => Some(std::slice::from_raw_parts(
                        multipole_ptr.raw,
                        self.ncoeffs_equivalent_surface(key.level()),
                    )),
                    FmmEvalType::Matrix(nmatvecs) => Some(std::slice::from_raw_parts(
                        multipole_ptr.raw,
                        self.ncoeffs_equivalent_surface(key.level()) * nmatvecs,
                    )),
                }
            }
        } else {
            None
        }
    }

    fn multipoles(&self, level: u64) -> Option<&[Self::Scalar]> {
        let multipole_ptr = &self.level_multipoles[level as usize][0][0];
        let nsources = self.tree.source_tree.n_keys(level).unwrap();
        unsafe {
            match self.fmm_eval_type {
                FmmEvalType::Vector => Some(std::slice::from_raw_parts(
                    multipole_ptr.raw,
                    self.ncoeffs_equivalent_surface(level) * nsources,
                )),
                FmmEvalType::Matrix(nmatvecs) => Some(std::slice::from_raw_parts(
                    multipole_ptr.raw,
                    self.ncoeffs_equivalent_surface(level) * nsources * nmatvecs,
                )),
            }
        }
    }

    fn local(
        &self,
        key: &<<Self::Tree as FmmTree>::Tree as Tree>::Node,
    ) -> Option<&[Self::Scalar]> {
        if let Some(&key_idx) = self.tree().target_tree().level_index(key) {
            let local_ptr = &self.level_locals[key.level() as usize][key_idx][0];

            unsafe {
                match self.fmm_eval_type {
                    FmmEvalType::Vector => Some(std::slice::from_raw_parts(
                        local_ptr.raw,
                        self.ncoeffs_equivalent_surface(key.level()),
                    )),
                    FmmEvalType::Matrix(nmatvecs) => Some(std::slice::from_raw_parts(
                        local_ptr.raw,
                        self.ncoeffs_equivalent_surface(key.level()) * nmatvecs,
                    )),
                }
            }
        } else {
            None
        }
    }

    fn locals(&self, level: u64) -> Option<&[Self::Scalar]> {
        let local_ptr = &self.level_locals[level as usize][0][0];
        let ntargets = self.tree.target_tree.n_keys(level).unwrap();
        unsafe {
            match self.fmm_eval_type {
                FmmEvalType::Vector => Some(std::slice::from_raw_parts(
                    local_ptr.raw,
                    self.ncoeffs_equivalent_surface(level) * ntargets,
                )),
                FmmEvalType::Matrix(nmatvecs) => Some(std::slice::from_raw_parts(
                    local_ptr.raw,
                    self.ncoeffs_equivalent_surface(level) * ntargets * nmatvecs,
                )),
            }
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

    fn potentials(&self) -> Option<&Vec<Self::Scalar>> {
        Some(&self.potentials)
    }

    fn evaluate(&mut self, timed: bool) -> Result<(), FmmError> {
        // Upward pass
        if timed {
            {
                let s = Instant::now();
                self.p2m()?;
                self.times.push(FmmOperatorTime::from_instant(FmmOperatorType::P2M, s));

                for level in (1..=self.tree().source_tree().depth()).rev() {
                    let s = Instant::now();
                    self.m2m(level)?;
                    self.times.push(FmmOperatorTime::from_instant(
                        FmmOperatorType::M2M(level),
                        s,
                    ));
                }
            }

            // Downward pass
            {
                for level in 2..=self.tree().target_tree().depth() {
                    if level > 2 {
                        let s = Instant::now();
                        self.l2l(level)?;
                        self.times.push(FmmOperatorTime::from_instant(
                            FmmOperatorType::L2L(level),
                            s,
                        ));
                    }
                    let s = Instant::now();
                    self.m2l(level)?;
                    self.times.push(FmmOperatorTime::from_instant(
                        FmmOperatorType::M2L(level),
                        s,
                    ));
                }

                // Leaf level computation
                let s = Instant::now();
                self.p2p()?;
                self.times.push(FmmOperatorTime::from_instant(FmmOperatorType::P2P, s));
                let s = Instant::now();
                self.l2p()?;
                self.times.push(FmmOperatorTime::from_instant(FmmOperatorType::L2P, s));
            }
            Ok(())
        } else {
            // Upward pass
            {
                self.p2m()?;

                for level in (1..=self.tree().source_tree().depth()).rev() {
                    self.m2m(level)?;
                }
            }

            // // Downward pass
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

    }

    fn clear(&mut self, charges: &[Self::Scalar]) {
        let ntarget_points = self.tree().target_tree().n_coordinates_tot().unwrap();
        let nsource_points = self.tree().source_tree().n_coordinates_tot().unwrap();
        let nmatvecs = charges.len() / nsource_points;
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
            &self.ncoeffs_equivalent_surface,
            nmatvecs,
            nsource_leaves,
            &self.multipoles,
        );

        let level_multipoles = level_expansion_pointers(
            self.tree().source_tree(),
            &self.ncoeffs_equivalent_surface,
            nmatvecs,
            &self.multipoles,
        );

        let level_locals = level_expansion_pointers(
            self.tree().target_tree(),
            &self.ncoeffs_equivalent_surface,
            nmatvecs,
            &self.locals,
        );

        let leaf_locals = leaf_expansion_pointers(
            self.tree().target_tree(),
            &self.ncoeffs_equivalent_surface,
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
            nmatvecs,
        )
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
    use num::{Float, Zero};
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use rlst::{
        c64, rlst_dynamic_array2, Array, BaseArray, RawAccess, RawAccessMut, RlstScalar, Shape,
        VectorContainer,
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

        let [nsources, nmatvecs] = charges.shape();

        for i in 0..nmatvecs {
            let potential_i = fmm.potential(&leaf).unwrap()[i];
            let charges_i = &charges.data()[nsources * i..nsources * (i + 1)];
            let mut direct_i = vec![T::Real::zero(); ntargets * eval_size];
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

        let [nsources, nmatvecs] = charges.shape();

        for i in 0..nmatvecs {
            let potential_i = fmm.potential(&leaf).unwrap()[i];
            let charges_i = &charges.data()[nsources * i..nsources * (i + 1)];
            let mut direct_i = vec![T::zero(); ntargets * eval_size];
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

        fmm.kernel().evaluate_st(
            eval_type,
            sources.data(),
            leaf_targets,
            charges.data(),
            &mut direct,
        );

        direct.iter().zip(potential).for_each(|(&d, &p)| {
            let abs_error = RlstScalar::abs(d - p);
            let rel_error = abs_error / p;
            // println!("err {:?} \nd {:?} \np {:?}", rel_error, d, &p);
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

        let multipoles = fmm.multipole(&root).unwrap();
        let upward_equivalent_surface = root.surface_grid(
            fmm.equivalent_surface_order(0),
            fmm.tree().domain(),
            T::from(ALPHA_INNER).unwrap().re(),
        );

        let ncoeffs = fmm.ncoeffs_equivalent_surface(0);

        let test_point = vec![T::real(100000.), T::Real::zero(), T::Real::zero()];
        let mut expected = vec![T::Real::zero()];
        let mut found = vec![T::Real::zero()];

        let [nsources, nvecs] = charges.shape();

        for i in 0..nvecs {
            let charges_i = &charges.data()[nsources * i..nsources * (i + 1)];
            let multipole_i = &multipoles[ncoeffs * i..(i + 1) * ncoeffs];

            println!(
                "root multipole {:?}, {:?} {:?}",
                &multipole_i[0..5],
                multipoles.len(),
                multipole_i.len()
            );

            fmm.kernel().evaluate_st(
                EvalType::Value,
                sources.data(),
                &test_point,
                charges_i,
                &mut expected,
            );

            fmm.kernel().evaluate_st(
                EvalType::Value,
                &upward_equivalent_surface,
                &test_point,
                multipole_i,
                &mut found,
            );

            let abs_error = RlstScalar::abs(expected[0] - found[0]);
            let rel_error = abs_error / expected[0];

            println!(
                "i {:?} abs {:?} rel {:?} \n expected {:?} found {:?}",
                i, abs_error, rel_error, expected, found
            );
            assert!(rel_error <= threshold);
        }
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
            fmm.equivalent_surface_order(0),
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
        let depth = None;
        let expansion_order = [6];

        // let n_crit = None;
        // let depth = Some(3);
        // let expansion_order = [5, 6, 5, 6];

        let prune_empty = true;

        // Charge data
        let nvecs = 1;
        let mut rng = StdRng::seed_from_u64(0);
        let mut charges = rlst_dynamic_array2!(f64, [nsources, nvecs]);
        charges.data_mut().iter_mut().for_each(|c| *c = rng.gen());

        let mut fmm_fft = SingleNodeBuilder::new()
            .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
            .unwrap()
            .parameters(
                charges.data(),
                &expansion_order,
                Laplace3dKernel::new(),
                EvalType::Value,
                FftFieldTranslation::new(None),
            )
            .unwrap()
            .build()
            .unwrap();
        fmm_fft.evaluate(false).unwrap();

        let svd_threshold = Some(1e-5);
        let mut fmm_svd = SingleNodeBuilder::new()
            .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
            .unwrap()
            .parameters(
                charges.data(),
                &expansion_order,
                Laplace3dKernel::new(),
                EvalType::Value,
                BlasFieldTranslationSaRcmp::new(
                    svd_threshold,
                    None,
                    crate::fmm::types::FmmSvdMode::Deterministic,
                ),
            )
            .unwrap()
            .build()
            .unwrap();
        fmm_svd.evaluate(false).unwrap();

        let mut fmm_fft = Box::new(fmm_fft);
        let mut fmm_svd = Box::new(fmm_svd);
        test_root_multipole_laplace_single_node::<f64>(fmm_fft, &sources, &charges, 1e-5);
        test_root_multipole_laplace_single_node::<f64>(fmm_svd, &sources, &charges, 1e-5);
    }

    #[test]
    fn test_upward_pass_matrix_laplace() {
        // Setup random sources and targets
        let nsources = 10000;
        let ntargets = 10000;
        let sources = points_fixture::<f64>(nsources, None, None, Some(1));
        let targets = points_fixture::<f64>(ntargets, None, None, Some(1));

        // FMM parameters
        // let n_crit = Some(100);
        // let depth = None;
        // let expansion_order = [6];

        let n_crit = None;
        let depth = Some(3);
        let expansion_order = [6, 6, 6, 6];

        let prune_empty = true;

        // Charge data
        let nvecs = 2;
        let mut rng = StdRng::seed_from_u64(0);
        let mut charges = rlst_dynamic_array2!(f64, [nsources, nvecs]);
        charges.data_mut().iter_mut().for_each(|c| *c = rng.gen());

        let svd_threshold = Some(1e-5);
        let mut fmm_svd = SingleNodeBuilder::new()
            .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
            .unwrap()
            .parameters(
                charges.data(),
                &expansion_order,
                Laplace3dKernel::new(),
                EvalType::Value,
                BlasFieldTranslationSaRcmp::new(
                    svd_threshold,
                    None,
                    crate::fmm::types::FmmSvdMode::Deterministic,
                ),
            )
            .unwrap()
            .build()
            .unwrap();
        fmm_svd.evaluate(false).unwrap();

        let mut fmm_svd = Box::new(fmm_svd);
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
        let depth = None;
        let expansion_order = [6];
        let prune_empty = true;
        let threshold_pot = 1e-5;

        // Set charge data and evaluate an FMM
        let nvecs = 1;
        let mut rng = StdRng::seed_from_u64(0);
        let mut charges = rlst_dynamic_array2!(f64, [nsources, nvecs]);
        charges.data_mut().iter_mut().for_each(|c| *c = rng.gen());

        let svd_mode = crate::fmm::types::FmmSvdMode::new(false, None, None, None, None);
        let mut fmm = SingleNodeBuilder::new()
            .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
            .unwrap()
            .parameters(
                charges.data(),
                &expansion_order,
                Laplace3dKernel::new(),
                EvalType::Value,
                BlasFieldTranslationSaRcmp::new(None, None, svd_mode),
            )
            .unwrap()
            .build()
            .unwrap();

        fmm.evaluate(false).unwrap();
        // Reset Charge data and re-evaluate potential
        let mut rng = StdRng::seed_from_u64(1);
        charges.data_mut().iter_mut().for_each(|c| *c = rng.gen());

        fmm.clear(charges.data());
        fmm.evaluate(false).unwrap();

        let mut fmm = Box::new(fmm);
        test_single_node_laplace_fmm_vector_helper::<f64>(
            fmm,
            EvalType::Value,
            &sources,
            &charges,
            threshold_pot,
        );
    }

    #[test]
    fn test_laplace_fmm_vector_variable_expansion_order() {
        // Setup random sources and targets
        let nsources = 9000;
        let ntargets = 10000;

        let min = None;
        let max = None;
        let sources = points_fixture::<f64>(nsources, min, max, Some(0));
        let targets = points_fixture::<f64>(ntargets, min, max, Some(1));

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
        let mut charges = rlst_dynamic_array2!(f64, [nsources, nvecs]);
        charges.data_mut().iter_mut().for_each(|c| *c = rng.gen());

        // FFT based field translation
        {
            // Evaluate potentials
            let mut fmm_fft = SingleNodeBuilder::new()
                .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
                .unwrap()
                .parameters(
                    charges.data(),
                    &expansion_order,
                    Laplace3dKernel::new(),
                    EvalType::Value,
                    FftFieldTranslation::new(None),
                )
                .unwrap()
                .build()
                .unwrap();
            fmm_fft.evaluate(false).unwrap();
            let eval_type = fmm_fft.kernel_eval_type;
            let mut fmm_fft = Box::new(fmm_fft);
            test_single_node_laplace_fmm_vector_helper::<f64>(
                fmm_fft,
                eval_type,
                &sources,
                &charges,
                threshold_pot,
            );

            // Evaluate potentials + derivatives
            let mut fmm_fft = SingleNodeBuilder::new()
                .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
                .unwrap()
                .parameters(
                    charges.data(),
                    &expansion_order,
                    Laplace3dKernel::new(),
                    EvalType::ValueDeriv,
                    FftFieldTranslation::new(None),
                )
                .unwrap()
                .build()
                .unwrap();
            fmm_fft.evaluate(false).unwrap();

            let eval_type = fmm_fft.kernel_eval_type;
            let mut fmm_fft = Box::new(fmm_fft);
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
            let mut fmm_blas = SingleNodeBuilder::new()
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
            fmm_blas.evaluate(false).unwrap();
            let mut fmm_blas = Box::new(fmm_blas);
            test_single_node_laplace_fmm_vector_helper::<f64>(
                fmm_blas,
                eval_type,
                &sources,
                &charges,
                threshold_pot,
            );

            // Evaluate potentials + derivatives
            let eval_type = EvalType::ValueDeriv;
            let mut fmm_blas = SingleNodeBuilder::new()
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
            fmm_blas.evaluate(false).unwrap();
            let mut fmm_blas = Box::new(fmm_blas);
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
        let nsources = 9000;
        let ntargets = 10000;

        let min = None;
        let max = None;
        let sources = points_fixture::<f64>(nsources, min, max, Some(0));
        let targets = points_fixture::<f64>(ntargets, min, max, Some(1));

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
        let mut charges = rlst_dynamic_array2!(f64, [nsources, nvecs]);
        charges.data_mut().iter_mut().for_each(|c| *c = rng.gen());

        // BLAS based field translations allow variable check/equiv surfaces
        {
            // Evaluate potentials
            let eval_type = EvalType::Value;
            let mut fmm_blas = SingleNodeBuilder::new()
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
            fmm_blas.evaluate(false).unwrap();
            let mut fmm_blas = Box::new(fmm_blas);

            test_single_node_laplace_fmm_vector_helper::<f64>(
                fmm_blas,
                eval_type,
                &sources,
                &charges,
                threshold_pot,
            );

            // Evaluate potentials + derivatives
            let eval_type = EvalType::ValueDeriv;
            let mut fmm_blas = SingleNodeBuilder::new()
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
            fmm_blas.evaluate(false).unwrap();
            let mut fmm_blas = Box::new(fmm_blas);
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
        let nsources = 9000;
        let ntargets = 10000;

        let min = None;
        let max = None;
        let sources = points_fixture::<f64>(nsources, min, max, Some(0));
        let targets = points_fixture::<f64>(ntargets, min, max, Some(1));

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
        let mut charges = rlst_dynamic_array2!(f64, [nsources, nvecs]);
        charges.data_mut().iter_mut().for_each(|c| *c = rng.gen());

        // BLAS based field translations allow variable check/equiv surfaces
        {
            // Evaluate potentials
            let eval_type = EvalType::Value;
            let mut fmm_blas = SingleNodeBuilder::new()
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
            fmm_blas.evaluate(false).unwrap();
            let mut fmm_blas = Box::new(fmm_blas);

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
        let nsources = 9000;
        let ntargets = 10000;

        let min = None;
        let max = None;
        let sources = points_fixture::<f64>(nsources, min, max, Some(0));
        let targets = points_fixture::<f64>(ntargets, min, max, Some(1));

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
        let mut charges = rlst_dynamic_array2!(f64, [nsources, nvecs]);
        charges.data_mut().iter_mut().for_each(|c| *c = rng.gen());

        // BLAS based field translations allow variable check/equiv surfaces
        {
            // Evaluate potentials
            let eval_type = EvalType::Value;
            let mut fmm_blas = SingleNodeBuilder::new()
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
            fmm_blas.evaluate(false).unwrap();
            let mut fmm_blas = Box::new(fmm_blas);

            test_single_node_laplace_fmm_vector_helper::<f64>(
                fmm_blas,
                eval_type,
                &sources,
                &charges,
                threshold_pot,
            );

            // Evaluate potentials + derivatives
            let eval_type = EvalType::ValueDeriv;
            let mut fmm_blas = SingleNodeBuilder::new()
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
            fmm_blas.evaluate(false).unwrap();
            let mut fmm_blas = Box::new(fmm_blas);
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
        let mut charges = rlst_dynamic_array2!(c64, [nsources, nvecs]);
        charges.data_mut().iter_mut().for_each(|c| *c = rng.gen());

        let wavenumber = 2.5;

        let mut fmm_fft = SingleNodeBuilder::new()
            .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
            .unwrap()
            .parameters(
                charges.data(),
                &expansion_order,
                Helmholtz3dKernel::new(wavenumber),
                EvalType::Value,
                FftFieldTranslation::new(None),
            )
            .unwrap()
            .build()
            .unwrap();

        fmm_fft.evaluate(false).unwrap();
        let mut fmm_fft = Box::new(fmm_fft);
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
        let depth = None;
        let expansion_order = [6];

        let prune_empty = true;
        let wavenumber = 2.5;

        // Charge data
        let nvecs = 1;
        let mut rng = StdRng::seed_from_u64(0);
        let mut charges = rlst_dynamic_array2!(c64, [nsources, nvecs]);
        charges.data_mut().iter_mut().for_each(|c| *c = rng.gen());

        // BLAS based field translation
        {
            // Evaluate potentials
            let mut fmm = SingleNodeBuilder::new()
                .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
                .unwrap()
                .parameters(
                    charges.data(),
                    &expansion_order,
                    Helmholtz3dKernel::new(wavenumber),
                    EvalType::Value,
                    BlasFieldTranslationIa::new(None, None),
                )
                .unwrap()
                .build()
                .unwrap();
            fmm.evaluate(false).unwrap();

            let mut fmm: Box<_> = Box::new(fmm);
            let eval_type = fmm.kernel_eval_type;
            test_single_node_helmholtz_fmm_vector_helper::<c64>(
                fmm, eval_type, &sources, &charges, threshold,
            );

            // Evaluate potentials + derivatives
            let mut fmm = SingleNodeBuilder::new()
                .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
                .unwrap()
                .parameters(
                    charges.data(),
                    &expansion_order,
                    Helmholtz3dKernel::new(wavenumber),
                    EvalType::ValueDeriv,
                    BlasFieldTranslationIa::new(None, None),
                )
                .unwrap()
                .build()
                .unwrap();
            fmm.evaluate(false).unwrap();
            let eval_type = fmm.kernel_eval_type;
            let mut fmm = Box::new(fmm);
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
            let mut fmm = SingleNodeBuilder::new()
                .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
                .unwrap()
                .parameters(
                    charges.data(),
                    &expansion_order,
                    Helmholtz3dKernel::new(wavenumber),
                    EvalType::Value,
                    FftFieldTranslation::new(None),
                )
                .unwrap()
                .build()
                .unwrap();
            fmm.evaluate(false).unwrap();

            let mut fmm: Box<_> = Box::new(fmm);
            let eval_type = fmm.kernel_eval_type;
            test_single_node_helmholtz_fmm_vector_helper::<c64>(
                fmm, eval_type, &sources, &charges, 1e-5,
            );

            // Evaluate potentials + derivatives
            let mut fmm = SingleNodeBuilder::new()
                .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
                .unwrap()
                .parameters(
                    charges.data(),
                    &expansion_order,
                    Helmholtz3dKernel::new(wavenumber),
                    EvalType::ValueDeriv,
                    FftFieldTranslation::new(None),
                )
                .unwrap()
                .build()
                .unwrap();
            fmm.evaluate(false).unwrap();
            let eval_type = fmm.kernel_eval_type;
            let mut fmm = Box::new(fmm);
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
        let nsources = 9000;
        let ntargets = 10000;
        let sources = points_fixture::<f64>(nsources, None, None, Some(1));
        let targets = points_fixture::<f64>(ntargets, None, None, Some(1));
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
        let mut charges = rlst_dynamic_array2!(c64, [nsources, nvecs]);
        charges.data_mut().iter_mut().for_each(|c| *c = rng.gen());

        // BLAS based field translation
        {
            // Evaluate potentials
            let mut fmm = SingleNodeBuilder::new()
                .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
                .unwrap()
                .parameters(
                    charges.data(),
                    &expansion_order,
                    Helmholtz3dKernel::new(wavenumber),
                    EvalType::Value,
                    BlasFieldTranslationIa::new(None, None),
                )
                .unwrap()
                .build()
                .unwrap();
            fmm.evaluate(false).unwrap();

            let mut fmm: Box<_> = Box::new(fmm);
            let eval_type = fmm.kernel_eval_type;
            test_single_node_helmholtz_fmm_vector_helper::<c64>(
                fmm, eval_type, &sources, &charges, threshold,
            );

            // Evaluate potentials + derivatives
            let mut fmm = SingleNodeBuilder::new()
                .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
                .unwrap()
                .parameters(
                    charges.data(),
                    &expansion_order,
                    Helmholtz3dKernel::new(wavenumber),
                    EvalType::ValueDeriv,
                    BlasFieldTranslationIa::new(None, None),
                )
                .unwrap()
                .build()
                .unwrap();
            fmm.evaluate(false).unwrap();
            let eval_type = fmm.kernel_eval_type;
            let mut fmm = Box::new(fmm);
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
            let mut fmm = SingleNodeBuilder::new()
                .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
                .unwrap()
                .parameters(
                    charges.data(),
                    &expansion_order,
                    Helmholtz3dKernel::new(wavenumber),
                    EvalType::Value,
                    FftFieldTranslation::new(None),
                )
                .unwrap()
                .build()
                .unwrap();
            fmm.evaluate(false).unwrap();

            let mut fmm: Box<_> = Box::new(fmm);
            let eval_type = fmm.kernel_eval_type;
            test_single_node_helmholtz_fmm_vector_helper::<c64>(
                fmm, eval_type, &sources, &charges, 1e-5,
            );

            // Evaluate potentials + derivatives
            let mut fmm = SingleNodeBuilder::new()
                .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
                .unwrap()
                .parameters(
                    charges.data(),
                    &expansion_order,
                    Helmholtz3dKernel::new(wavenumber),
                    EvalType::ValueDeriv,
                    FftFieldTranslation::new(None),
                )
                .unwrap()
                .build()
                .unwrap();
            fmm.evaluate(false).unwrap();
            let eval_type = fmm.kernel_eval_type;
            let mut fmm = Box::new(fmm);
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
        let nsources = 9000;
        let ntargets = 10000;
        let sources = points_fixture::<f64>(nsources, None, None, Some(1));
        let targets = points_fixture::<f64>(ntargets, None, None, Some(1));
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
        let mut charges = rlst_dynamic_array2!(c64, [nsources, nvecs]);
        charges.data_mut().iter_mut().for_each(|c| *c = rng.gen());

        // BLAS based field translation
        {
            // Evaluate potentials
            let mut fmm = SingleNodeBuilder::new()
                .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
                .unwrap()
                .parameters(
                    charges.data(),
                    &expansion_order,
                    Helmholtz3dKernel::new(wavenumber),
                    EvalType::Value,
                    BlasFieldTranslationIa::new(None, surface_diff),
                )
                .unwrap()
                .build()
                .unwrap();
            fmm.evaluate(false).unwrap();

            let mut fmm: Box<_> = Box::new(fmm);
            let eval_type = fmm.kernel_eval_type;

            test_single_node_helmholtz_fmm_vector_helper::<c64>(
                fmm, eval_type, &sources, &charges, threshold,
            );

            // Evaluate potentials + derivatives
            let mut fmm = SingleNodeBuilder::new()
                .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
                .unwrap()
                .parameters(
                    charges.data(),
                    &expansion_order,
                    Helmholtz3dKernel::new(wavenumber),
                    EvalType::ValueDeriv,
                    BlasFieldTranslationIa::new(None, None),
                )
                .unwrap()
                .build()
                .unwrap();
            fmm.evaluate(false).unwrap();
            let eval_type = fmm.kernel_eval_type;
            let mut fmm = Box::new(fmm);
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
        let nsources = 9000;
        let ntargets = 10000;
        let sources = points_fixture::<f64>(nsources, None, None, Some(1));
        let targets = points_fixture::<f64>(ntargets, None, None, Some(1));
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
        let mut charges = rlst_dynamic_array2!(c64, [nsources, nvecs]);
        charges.data_mut().iter_mut().for_each(|c| *c = rng.gen());

        // BLAS based field translation
        {
            // Evaluate potentials
            let mut fmm = SingleNodeBuilder::new()
                .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
                .unwrap()
                .parameters(
                    charges.data(),
                    &expansion_order,
                    Helmholtz3dKernel::new(wavenumber),
                    EvalType::Value,
                    BlasFieldTranslationIa::new(None, surface_diff),
                )
                .unwrap()
                .build()
                .unwrap();
            fmm.evaluate(false).unwrap();

            let mut fmm: Box<_> = Box::new(fmm);
            let eval_type = fmm.kernel_eval_type;

            test_single_node_helmholtz_fmm_matrix_helper::<c64>(
                fmm, eval_type, &sources, &charges, threshold,
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
        let mut charges = rlst_dynamic_array2!(f64, [nsources, nvecs]);
        charges
            .data_mut()
            .chunks_exact_mut(nsources)
            .for_each(|chunk| chunk.iter_mut().for_each(|elem| *elem += rng.gen::<f64>()));

        // fmm with blas based field translation
        {
            // Evaluate potentials
            let eval_type = EvalType::Value;
            let mut fmm_blas = SingleNodeBuilder::new()
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
            fmm_blas.evaluate(false).unwrap();

            let mut fmm_blas = Box::new(fmm_blas);
            test_single_node_laplace_fmm_matrix_helper::<f64>(
                fmm_blas, eval_type, &sources, &charges, threshold,
            );

            // Evaluate potentials + derivatives
            let eval_type = EvalType::ValueDeriv;
            let mut fmm_blas = SingleNodeBuilder::new()
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
            fmm_blas.evaluate(false).unwrap();
            let mut fmm_blas = Box::new(fmm_blas);
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
        let mut charges = rlst_dynamic_array2!(c64, [nsources, nvecs]);
        charges
            .data_mut()
            .chunks_exact_mut(nsources)
            .for_each(|chunk| chunk.iter_mut().for_each(|elem| *elem += rng.gen::<c64>()));

        // fmm with blas based field translation
        {
            // Evaluate potentials
            let eval_type = EvalType::Value;
            let mut fmm_blas = SingleNodeBuilder::new()
                .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
                .unwrap()
                .parameters(
                    charges.data(),
                    &expansion_order,
                    Helmholtz3dKernel::new(wavenumber),
                    eval_type,
                    BlasFieldTranslationIa::new(singular_value_threshold, None),
                )
                .unwrap()
                .build()
                .unwrap();
            fmm_blas.evaluate(false).unwrap();

            let mut fmm_blas = Box::new(fmm_blas);
            test_single_node_helmholtz_fmm_matrix_helper::<c64>(
                fmm_blas, eval_type, &sources, &charges, threshold,
            );

            // Evaluate potentials + derivatives
            let eval_type = EvalType::ValueDeriv;
            let mut fmm_blas = SingleNodeBuilder::new()
                .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
                .unwrap()
                .parameters(
                    charges.data(),
                    &expansion_order,
                    Helmholtz3dKernel::new(wavenumber),
                    eval_type,
                    BlasFieldTranslationIa::new(singular_value_threshold, None),
                )
                .unwrap()
                .build()
                .unwrap();
            fmm_blas.evaluate(false).unwrap();
            let mut fmm_blas = Box::new(fmm_blas);
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
