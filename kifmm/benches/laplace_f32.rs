use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion};
use green_kernels::{laplace_3d::Laplace3dKernel, types::EvalType};
use kifmm::fmm::types::{BlasFieldTranslationSaRcmp, FftFieldTranslation, SingleNodeBuilder};
use kifmm::traits::fmm::Fmm;
use kifmm::tree::helpers::points_fixture;
use rlst::{rlst_dynamic_array2, RawAccess, RawAccessMut};

extern crate blas_src;
extern crate lapack_src;

fn laplace_potentials_f32(c: &mut Criterion) {
    // Setup random sources and targets
    let nsources = 1000000;
    let ntargets = 1000000;
    let sources = points_fixture::<f32>(nsources, None, None, Some(0));
    let targets = points_fixture::<f32>(ntargets, None, None, Some(1));

    // FMM parameters
    let n_crit = Some(150);
    let depth = None;
    let expansion_order = [5];
    let prune_empty = true;
    let svd_threshold = Some(2e-1);

    // FFT based M2L for a vector of charges
    let nvecs = 1;
    let tmp = vec![1.0; nsources * nvecs];
    let mut charges = rlst_dynamic_array2!(f32, [nsources, nvecs]);
    charges.data_mut().copy_from_slice(&tmp);

    let fmm_fft = SingleNodeBuilder::new()
        .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
        .unwrap()
        .parameters(
            charges.data(),
            &expansion_order,
            Laplace3dKernel::new(),
            EvalType::Value,
            FftFieldTranslation::new(),
        )
        .unwrap()
        .build()
        .unwrap();

    let fmm_blas = SingleNodeBuilder::new()
        .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
        .unwrap()
        .parameters(
            charges.data(),
            &expansion_order,
            Laplace3dKernel::new(),
            EvalType::Value,
            BlasFieldTranslationSaRcmp::new(svd_threshold, None),
        )
        .unwrap()
        .build()
        .unwrap();

    let mut group = c.benchmark_group("Laplace Potentials f32");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(15));

    group.bench_function(format!("M2L=FFT, N={nsources}"), |b| {
        b.iter(|| fmm_fft.evaluate(false).unwrap())
    });

    group.bench_function(format!("M2L=BLAS, N={nsources}"), |b| {
        b.iter(|| fmm_blas.evaluate(false).unwrap())
    });
}

fn laplace_potentials_gradients_f32(c: &mut Criterion) {
    // Setup random sources and targets
    let nsources = 1000000;
    let ntargets = 1000000;
    let sources = points_fixture::<f32>(nsources, None, None, Some(0));
    let targets = points_fixture::<f32>(ntargets, None, None, Some(1));

    // FMM parameters
    let n_crit = Some(150);
    let depth = None;
    let expansion_order = [5];
    let prune_empty = true;
    let svd_threshold = Some(2e-1);

    // FFT based M2L for a vector of charges
    let nvecs = 1;
    let tmp = vec![1.0; nsources * nvecs];
    let mut charges = rlst_dynamic_array2!(f32, [nsources, nvecs]);
    charges.data_mut().copy_from_slice(&tmp);

    let fmm_fft = SingleNodeBuilder::new()
        .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
        .unwrap()
        .parameters(
            charges.data(),
            &expansion_order,
            Laplace3dKernel::new(),
            EvalType::ValueDeriv,
            FftFieldTranslation::new(),
        )
        .unwrap()
        .build()
        .unwrap();

    let fmm_blas = SingleNodeBuilder::new()
        .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
        .unwrap()
        .parameters(
            charges.data(),
            &expansion_order,
            Laplace3dKernel::new(),
            EvalType::ValueDeriv,
            BlasFieldTranslationSaRcmp::new(svd_threshold, None),
        )
        .unwrap()
        .build()
        .unwrap();

    let mut group = c.benchmark_group("Laplace Gradients f32");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(20));

    group.bench_function(format!("M2L=FFT, N={nsources}"), |b| {
        b.iter(|| fmm_fft.evaluate(false).unwrap())
    });

    group.bench_function(format!("M2L=BLAS, N={nsources}"), |b| {
        b.iter(|| fmm_blas.evaluate(false).unwrap())
    });
}

criterion_group!(laplace_p_f32, laplace_potentials_f32);
criterion_group!(laplace_g_f32, laplace_potentials_gradients_f32);

criterion_main!(laplace_p_f32, laplace_g_f32,);
