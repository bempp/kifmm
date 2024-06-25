use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion};
use green_kernels::{laplace_3d::Laplace3dKernel, types::EvalType};
use kifmm::fmm::types::{BlasFieldTranslationSaRcmp, SingleNodeBuilder};
use kifmm::traits::fmm::Fmm;
use kifmm::fmm::types::FmmSvdMode;
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
    let n_crit = Some(400);
    let depth = None;
    let expansion_order = [5];
    let prune_empty = true;
    let svd_threshold = Some(2e-1);

    // FFT based M2L for a vector of charges
    let nvecs = 5;
    let tmp = vec![1.0; nsources * nvecs];
    let mut charges = rlst_dynamic_array2!(f32, [nsources, nvecs]);
    charges.data_mut().copy_from_slice(&tmp);

    let fmm_blas_5 = SingleNodeBuilder::new()
        .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
        .unwrap()
        .parameters(
            charges.data(),
            &expansion_order,
            Laplace3dKernel::new(),
            EvalType::Value,
            BlasFieldTranslationSaRcmp::new(svd_threshold, None, FmmSvdMode::new(false, None, None, None, None)),
        )
        .unwrap()
        .build()
        .unwrap();

    let nvecs = 10;
    let tmp = vec![1.0; nsources * nvecs];
    let mut charges = rlst_dynamic_array2!(f32, [nsources, nvecs]);
    charges.data_mut().copy_from_slice(&tmp);

    let fmm_blas_10 = SingleNodeBuilder::new()
        .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
        .unwrap()
        .parameters(
            charges.data(),
            &expansion_order,
            Laplace3dKernel::new(),
            EvalType::Value,
            BlasFieldTranslationSaRcmp::new(svd_threshold, None, FmmSvdMode::new(false, None, None, None, None)),
        )
        .unwrap()
        .build()
        .unwrap();

    let mut group = c.benchmark_group("Laplace Potentials mat f32");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(100));

    group.bench_function(format!("M2L=BLAS, N={nsources} NVecs=5"), |b| {
        b.iter(|| fmm_blas_5.evaluate(false).unwrap())
    });

    group.bench_function(format!("M2L=BLAS, N={nsources} NVecs=10"), |b| {
        b.iter(|| fmm_blas_10.evaluate(false).unwrap())
    });
}

fn laplace_potentials_gradients_f32(c: &mut Criterion) {
    // Setup random sources and targets
    let nsources = 1000000;
    let ntargets = 1000000;
    let sources = points_fixture::<f32>(nsources, None, None, Some(0));
    let targets = points_fixture::<f32>(ntargets, None, None, Some(1));

    // FMM parameters
    let n_crit = Some(400);
    let depth = None;
    let expansion_order = [5];
    let prune_empty = true;
    let svd_threshold = Some(2e-1);

    // FFT based M2L for a vector of charges
    let nvecs = 5;
    let tmp = vec![1.0; nsources * nvecs];
    let mut charges = rlst_dynamic_array2!(f32, [nsources, nvecs]);
    charges.data_mut().copy_from_slice(&tmp);

    let fmm_blas_5 = SingleNodeBuilder::new()
        .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
        .unwrap()
        .parameters(
            charges.data(),
            &expansion_order,
            Laplace3dKernel::new(),
            EvalType::ValueDeriv,
            BlasFieldTranslationSaRcmp::new(svd_threshold, None, FmmSvdMode::new(false, None, None, None, None)),
        )
        .unwrap()
        .build()
        .unwrap();

    let nvecs = 10;
    let tmp = vec![1.0; nsources * nvecs];
    let mut charges = rlst_dynamic_array2!(f32, [nsources, nvecs]);
    charges.data_mut().copy_from_slice(&tmp);

    let fmm_blas_10 = SingleNodeBuilder::new()
        .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
        .unwrap()
        .parameters(
            charges.data(),
            &expansion_order,
            Laplace3dKernel::new(),
            EvalType::ValueDeriv,
            BlasFieldTranslationSaRcmp::new(svd_threshold, None, FmmSvdMode::new(false, None, None, None, None)),
        )
        .unwrap()
        .build()
        .unwrap();

    let mut group = c.benchmark_group("Laplace Gradients mat f32");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(150));

    group.bench_function(format!("M2L=BLAS, N={nsources}, NVecs=5"), |b| {
        b.iter(|| fmm_blas_5.evaluate(false).unwrap())
    });

    group.bench_function(format!("M2L=BLAS, N={nsources}, NVecs=10"), |b| {
        b.iter(|| fmm_blas_10.evaluate(false).unwrap())
    });
}

criterion_group!(laplace_p_f32, laplace_potentials_f32);
criterion_group!(laplace_g_f32, laplace_potentials_gradients_f32);

criterion_main!(laplace_p_f32, laplace_g_f32,);
