use green_kernels::{helmholtz_3d::Helmholtz3dKernel, types::EvalType};
use kifmm::fmm::types::BlasFieldTranslationIa;
use kifmm::{FftFieldTranslation, SingleFmm, SingleNodeBuilder};

use kifmm::tree::helpers::points_fixture;
use num::{FromPrimitive, One};
use rlst::{c32, rlst_dynamic_array2, RawAccess, RawAccessMut};

fn main() {
    // Setup random sources and targets
    let n_sources = 1000;
    let n_targets = 2000;
    let sources = points_fixture::<f32>(n_sources, None, None, Some(0));
    let targets = points_fixture::<f32>(n_targets, None, None, Some(1));

    // FMM parameters
    let n_crit = Some(150);
    let depth = None;
    let expansion_order = [5];
    let prune_empty = true;

    // Kernel parameter
    let wavenumber = 2.5;

    // FFT based M2L for a vector of charges
    {
        let nvecs = 1;
        let tmp = vec![c32::one(); n_sources * nvecs];
        let mut charges = rlst_dynamic_array2!(c32, [n_sources, nvecs]);
        charges.data_mut().copy_from_slice(&tmp);

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
    }

    // BLAS based M2L
    {
        // Vector of charges
        let nvecs = 1;
        let mut charges = rlst_dynamic_array2!(c32, [n_sources, nvecs]);
        charges
            .data_mut()
            .chunks_exact_mut(n_sources)
            .enumerate()
            .for_each(|(i, chunk)| {
                chunk
                    .iter_mut()
                    .for_each(|elem| *elem += c32::from_f32(1. + i as f32).unwrap())
            });

        let singular_value_threshold = Some(1e-5);

        let mut fmm_vec = SingleNodeBuilder::new()
            .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
            .unwrap()
            .parameters(
                charges.data(),
                &expansion_order,
                Helmholtz3dKernel::new(wavenumber),
                EvalType::Value,
                BlasFieldTranslationIa::new(singular_value_threshold, None),
            )
            .unwrap()
            .build()
            .unwrap();

        fmm_vec.evaluate(false).unwrap();

        // Matrix of charges
        let nvecs = 5;
        let mut charges = rlst_dynamic_array2!(c32, [n_sources, nvecs]);
        charges
            .data_mut()
            .chunks_exact_mut(n_sources)
            .enumerate()
            .for_each(|(i, chunk)| {
                chunk
                    .iter_mut()
                    .for_each(|elem| *elem += c32::from_f32(1. + i as f32).unwrap())
            });

        let mut fmm_mat = SingleNodeBuilder::new()
            .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
            .unwrap()
            .parameters(
                charges.data(),
                &expansion_order,
                Helmholtz3dKernel::new(wavenumber),
                EvalType::Value,
                BlasFieldTranslationIa::new(singular_value_threshold, None),
            )
            .unwrap()
            .build()
            .unwrap();
        fmm_mat.evaluate(false).unwrap();
    }
}
