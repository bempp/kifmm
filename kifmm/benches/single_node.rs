use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion};
use green_kernels::helmholtz_3d::Helmholtz3dKernel;
use green_kernels::{laplace_3d::Laplace3dKernel, types::EvalType};
use kifmm::fmm::types::{BlasFieldTranslationSaRcmp, FftFieldTranslation, SingleNodeBuilder};
use kifmm::traits::fmm::Fmm;
use kifmm::tree::helpers::points_fixture;
use kifmm::BlasFieldTranslationIa;
use num::{Complex, One};
use rlst::{rlst_dynamic_array2, RawAccessMut, c32};

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
    let expansion_order = 5;
    let sparse = true;
    let svd_threshold = Some(1e-2);

    // FFT based M2L for a vector of charges
    let nvecs = 1;
    let tmp = vec![1.0; nsources * nvecs];
    let mut charges = rlst_dynamic_array2!(f32, [nsources, nvecs]);
    charges.data_mut().copy_from_slice(&tmp);

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

    let fmm_blas = SingleNodeBuilder::new()
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

    let mut group = c.benchmark_group("Laplace Potentials f32");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(15));

    group.bench_function(format!("M2L=FFT, N={nsources}"), |b| {
        b.iter(|| fmm_fft.evaluate())
    });

    group.bench_function(format!("M2L=BLAS, N={nsources}"), |b| {
        b.iter(|| fmm_blas.evaluate())
    });
}

fn helmholtz_potentials_f32(c: &mut Criterion) {
    // Setup random sources and targets
    let nsources = 1000000;
    let ntargets = 1000000;
    let sources = points_fixture::<f32>(nsources, None, None, Some(0));
    let targets = points_fixture::<f32>(ntargets, None, None, Some(1));

    // FMM parameters
    let n_crit = Some(400);
    let expansion_order = 5;
    let sparse = true;
    let svd_threshold = Some(1e-2);

    // FFT based M2L for a vector of charges
    let nvecs = 1;
    let tmp = vec![c32::one(); nsources * nvecs];
    let mut charges = rlst_dynamic_array2!(c32, [nsources, nvecs]);
    charges.data_mut().copy_from_slice(&tmp);

    let wavenumber = 1.0;

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

    let fmm_blas = SingleNodeBuilder::new()
        .tree(&sources, &targets, n_crit, sparse)
        .unwrap()
        .parameters(
            &charges,
            expansion_order,
            Helmholtz3dKernel::new(wavenumber),
            EvalType::Value,
            BlasFieldTranslationIa::new(svd_threshold),
        )
        .unwrap()
        .build()
        .unwrap();

    let mut group = c.benchmark_group("Helmholtz Potentials f32");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(40));

    group.bench_function(format!("M2L=FFT, N={nsources}, wavenumber={wavenumber}"), |b| {
        b.iter(|| fmm_fft.evaluate())
    });

    group.bench_function(
        format!("M2L=BLAS, N={nsources}, wavenumber={wavenumber}"),
        |b| b.iter(|| fmm_blas.evaluate()),
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
    let sparse = true;
    let svd_threshold = Some(1e-2);

    // FFT based M2L for a vector of charges
    let nvecs = 1;
    let tmp = vec![Complex::one(); nsources * nvecs];
    let mut charges = rlst_dynamic_array2!(c32, [nsources, nvecs]);
    charges.data_mut().copy_from_slice(&tmp);

    let wavenumber = 1.0;

    let fmm_fft = SingleNodeBuilder::new()
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

    let fmm_blas = SingleNodeBuilder::new()
        .tree(&sources, &targets, n_crit, sparse)
        .unwrap()
        .parameters(
            &charges,
            expansion_order,
            Helmholtz3dKernel::new(wavenumber),
            EvalType::ValueDeriv,
            BlasFieldTranslationIa::new(svd_threshold),
        )
        .unwrap()
        .build()
        .unwrap();

    let mut group = c.benchmark_group("Helmholtz Gradients f32");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(75));

    group.bench_function(
        format!("M2L=FFT, N={nsources}, wavenumber={wavenumber}"),
        |b| b.iter(|| fmm_fft.evaluate()),
    );

    group.bench_function(
        format!("M2L=BLAS, N={nsources}, wavenumber={wavenumber}"),
        |b| b.iter(|| fmm_blas.evaluate()),
    );
}

fn laplace_potentials_gradients_f32(c: &mut Criterion) {
    // Setup random sources and targets
    let nsources = 1000000;
    let ntargets = 1000000;
    let sources = points_fixture::<f32>(nsources, None, None, Some(0));
    let targets = points_fixture::<f32>(ntargets, None, None, Some(1));

    // FMM parameters
    let n_crit = Some(400);
    let expansion_order = 5;
    let sparse = true;
    let svd_threshold = Some(1e-2);

    // FFT based M2L for a vector of charges
    let nvecs = 1;
    let tmp = vec![1.0; nsources * nvecs];
    let mut charges = rlst_dynamic_array2!(f32, [nsources, nvecs]);
    charges.data_mut().copy_from_slice(&tmp);

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

    let fmm_blas = SingleNodeBuilder::new()
        .tree(&sources, &targets, n_crit, sparse)
        .unwrap()
        .parameters(
            &charges,
            expansion_order,
            Laplace3dKernel::new(),
            EvalType::ValueDeriv,
            BlasFieldTranslationSaRcmp::new(svd_threshold),
        )
        .unwrap()
        .build()
        .unwrap();

    let mut group = c.benchmark_group("Laplace Gradients f32");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(20));

    group.bench_function(format!("M2L=FFT, N={nsources}"), |b| {
        b.iter(|| fmm_fft.evaluate())
    });

    group.bench_function(format!("M2L=BLAS, N={nsources}"), |b| {
        b.iter(|| fmm_blas.evaluate())
    });
}

criterion_group!(laplace_p_f32, laplace_potentials_f32);
criterion_group!(laplace_g_f32, laplace_potentials_gradients_f32);
criterion_group!(helmholtz_p_f32, helmholtz_potentials_f32);
criterion_group!(helmholtz_g_f32, helmholtz_potentials_gradients_f32);

criterion_main!(
    laplace_p_f32,
    laplace_g_f32,
    helmholtz_p_f32,
    helmholtz_g_f32
);
