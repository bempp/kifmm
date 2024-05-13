use green_kernels::{helmholtz_3d::Helmholtz3dKernel, types::EvalType};
use kifmm::fmm::types::BlasFieldTranslationIa;
use kifmm::{FftFieldTranslation, Fmm, SingleNodeBuilder};

use kifmm::tree::helpers::points_fixture;
use num::{FromPrimitive, One};
use rlst::{c32, rlst_dynamic_array2, RawAccessMut};

extern crate blas_src;
extern crate lapack_src;

fn main() {
    // Setup random sources and targets
    let nsources = 1000;
    let ntargets = 2000;
    let sources = points_fixture::<f32>(nsources, None, None, Some(0));
    let targets = points_fixture::<f32>(ntargets, None, None, Some(1));

    // FMM parameters
    let n_crit = Some(150);
    let expansion_order = 5;
    let sparse = true;

    // Kernel parameter
    let wavenumber = 2.5;

    // FFT based M2L for a vector of charges
    {
        let nvecs = 1;
        let tmp = vec![c32::one(); nsources * nvecs];
        let mut charges = rlst_dynamic_array2!(c32, [nsources, nvecs]);
        charges.data_mut().copy_from_slice(&tmp);

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
        fmm_fft.evaluate(false).unwrap();
    }

    // BLAS based M2L
    {
        // Vector of charges
        let nvecs = 1;
        let mut charges = rlst_dynamic_array2!(c32, [nsources, nvecs]);
        charges
            .data_mut()
            .chunks_exact_mut(nsources)
            .enumerate()
            .for_each(|(i, chunk)| {
                chunk
                    .iter_mut()
                    .for_each(|elem| *elem += c32::from_f32(1. + i as f32).unwrap())
            });

        let singular_value_threshold = Some(1e-5);

        let fmm_vec = SingleNodeBuilder::new()
            .tree(&sources, &targets, n_crit, sparse)
            .unwrap()
            .parameters(
                &charges,
                expansion_order,
                Helmholtz3dKernel::new(wavenumber),
                EvalType::Value,
                BlasFieldTranslationIa::new(singular_value_threshold),
            )
            .unwrap()
            .build()
            .unwrap();

        fmm_vec.evaluate(false).unwrap();

        // Matrix of charges
        let nvecs = 5;
        let mut charges = rlst_dynamic_array2!(c32, [nsources, nvecs]);
        charges
            .data_mut()
            .chunks_exact_mut(nsources)
            .enumerate()
            .for_each(|(i, chunk)| {
                chunk
                    .iter_mut()
                    .for_each(|elem| *elem += c32::from_f32(1. + i as f32).unwrap())
            });

        let fmm_mat = SingleNodeBuilder::new()
            .tree(&sources, &targets, n_crit, sparse)
            .unwrap()
            .parameters(
                &charges,
                expansion_order,
                Helmholtz3dKernel::new(wavenumber),
                EvalType::Value,
                BlasFieldTranslationIa::new(singular_value_threshold),
            )
            .unwrap()
            .build()
            .unwrap();
        fmm_mat.evaluate(false).unwrap();
    }
}
