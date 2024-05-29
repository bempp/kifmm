use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion};
use green_kernels::{helmholtz_3d::Helmholtz3dKernel, types::EvalType};
use kifmm::fmm::types::SingleNodeBuilder;
use kifmm::traits::fmm::Fmm;
use kifmm::tree::helpers::points_fixture;
use kifmm::BlasFieldTranslationIa;
use num::{Complex, One};
use rlst::{c32, rlst_dynamic_array2, RawAccess, RawAccessMut};

extern crate blas_src;
extern crate lapack_src;

fn helmholtz_potentials_f32(c: &mut Criterion) {
    // Setup random sources and targets
    let nsources = 1000000;
    let ntargets = 1000000;
    let sources = points_fixture::<f32>(nsources, None, None, Some(0));
    let targets = points_fixture::<f32>(ntargets, None, None, Some(1));

    // FMM parameters
    let n_crit = Some(400);
    let expansion_order = 5;
    let prune_empty = true;
    let svd_threshold = Some(1e-2);
    let wavenumber = 1.0;

    // FFT based M2L for a vector of charges
    let nvecs = 5;
    let tmp = vec![c32::one(); nsources * nvecs];
    let mut charges = rlst_dynamic_array2!(c32, [nsources, nvecs]);
    charges.data_mut().copy_from_slice(&tmp);

    let fmm_blas_5 = SingleNodeBuilder::new()
        .tree(sources.data(), targets.data(), n_crit, prune_empty)
        .unwrap()
        .parameters(
            charges.data(),
            expansion_order,
            Helmholtz3dKernel::new(wavenumber),
            EvalType::Value,
            BlasFieldTranslationIa::new(svd_threshold),
        )
        .unwrap()
        .build()
        .unwrap();

    let nvecs = 10;
    let tmp = vec![c32::one(); nsources * nvecs];
    let mut charges = rlst_dynamic_array2!(c32, [nsources, nvecs]);
    charges.data_mut().copy_from_slice(&tmp);

    let fmm_blas_10 = SingleNodeBuilder::new()
        .tree(sources.data(), targets.data(), n_crit, prune_empty)
        .unwrap()
        .parameters(
            charges.data(),
            expansion_order,
            Helmholtz3dKernel::new(wavenumber),
            EvalType::Value,
            BlasFieldTranslationIa::new(svd_threshold),
        )
        .unwrap()
        .build()
        .unwrap();

    let mut group = c.benchmark_group("Helmholtz Potentials mat f32");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(200));

    group.bench_function(
        format!("M2L=BLAS, N={nsources}, NVecs=5, wavenumber={wavenumber}"),
        |b| b.iter(|| fmm_blas_5.evaluate(false).unwrap()),
    );

    group.bench_function(
        format!("M2L=BLAS, N={nsources}, NVecs=10, wavenumber={wavenumber}"),
        |b| b.iter(|| fmm_blas_10.evaluate(false).unwrap()),
    );
}

fn helmholtz_potentials_gradients_f32(c: &mut Criterion) {
    // Setup random sources and targets
    let nsources = 1000000;
    let ntargets = 1000000;
    let sources = points_fixture::<f32>(nsources, None, None, Some(0));
    let targets = points_fixture::<f32>(ntargets, None, None, Some(1));

    // FMM parameters
    let n_crit = Some(400);
    let expansion_order = 5;
    let prune_empty = true;
    let svd_threshold = Some(1e-2);
    let wavenumber = 1.0;

    // FFT based M2L for a vector of charges
    let nvecs = 5;
    let tmp = vec![Complex::one(); nsources * nvecs];
    let mut charges = rlst_dynamic_array2!(c32, [nsources, nvecs]);
    charges.data_mut().copy_from_slice(&tmp);

    let fmm_blas_5 = SingleNodeBuilder::new()
        .tree(sources.data(), targets.data(), n_crit, prune_empty)
        .unwrap()
        .parameters(
            charges.data(),
            expansion_order,
            Helmholtz3dKernel::new(wavenumber),
            EvalType::ValueDeriv,
            BlasFieldTranslationIa::new(svd_threshold),
        )
        .unwrap()
        .build()
        .unwrap();

    let nvecs = 10;
    let tmp = vec![Complex::one(); nsources * nvecs];
    let mut charges = rlst_dynamic_array2!(c32, [nsources, nvecs]);
    charges.data_mut().copy_from_slice(&tmp);

    let fmm_blas_10 = SingleNodeBuilder::new()
        .tree(sources.data(), targets.data(), n_crit, prune_empty)
        .unwrap()
        .parameters(
            charges.data(),
            expansion_order,
            Helmholtz3dKernel::new(wavenumber),
            EvalType::ValueDeriv,
            BlasFieldTranslationIa::new(svd_threshold),
        )
        .unwrap()
        .build()
        .unwrap();

    let mut group = c.benchmark_group("Helmholtz Gradients mat f32");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(300));

    group.bench_function(
        format!("M2L=BLAS, N={nsources}, NVecs=5, wavenumber={wavenumber}"),
        |b| b.iter(|| fmm_blas_5.evaluate(false).unwrap()),
    );

    group.bench_function(
        format!("M2L=BLAS, N={nsources}, NVecs=10, wavenumber={wavenumber}"),
        |b| b.iter(|| fmm_blas_10.evaluate(false).unwrap()),
    );
}

criterion_group!(helmholtz_p_f32, helmholtz_potentials_f32);
criterion_group!(helmholtz_g_f32, helmholtz_potentials_gradients_f32);

criterion_main!(helmholtz_p_f32, helmholtz_g_f32);
