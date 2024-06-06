use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion};
use green_kernels::{laplace_3d::Laplace3dKernel, types::EvalType};
use kifmm::fmm::types::{BlasFieldTranslationSaRcmp, FftFieldTranslation, SingleNodeBuilder};
use kifmm::traits::fmm::Fmm;
use kifmm::tree::helpers::points_fixture;
use rlst::{rlst_dynamic_array2, RawAccess, RawAccessMut};

extern crate blas_src;
extern crate lapack_src;

fn laplace_6_digits(c: &mut Criterion) {
    // Setup random sources and targets
    let nsources = 1000000;
    let ntargets = 1000000;
    let sources = points_fixture::<f64>(nsources, None, None, Some(0));
    let targets = points_fixture::<f64>(ntargets, None, None, Some(1));

    // FFT based M2L for a vector of charges
    let nvecs = 1;
    let tmp = vec![1.0; nsources * nvecs];
    let mut charges = rlst_dynamic_array2!(f64, [nsources, nvecs]);
    charges.data_mut().copy_from_slice(&tmp);

    // FMM parameters
    let n_crit = None;
    let depth = Some(5);
    let expansion_order = [6, 6, 6, 6, 6, 6];
    let prune_empty = true;
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


    // FMM parameters
    let surface_diff = Some(1);
    let depth = Some(5);
    let expansion_order = [5, 5, 5, 5, 5, 5];
    let prune_empty = true;
    let svd_threshold = Some(1e-6);

    let fmm_blas = SingleNodeBuilder::new()
        .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
        .unwrap()
        .parameters(
            charges.data(),
            &expansion_order,
            Laplace3dKernel::new(),
            EvalType::Value,
            BlasFieldTranslationSaRcmp::new(svd_threshold, surface_diff),
        )
        .unwrap()
        .build()
        .unwrap();

    let mut group = c.benchmark_group("Laplace Potentials 6 Digits");
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


fn laplace_7_digits(c: &mut Criterion) {
    // Setup random sources and targets
    let nsources = 1000000;
    let ntargets = 1000000;
    let sources = points_fixture::<f64>(nsources, None, None, Some(0));
    let targets = points_fixture::<f64>(ntargets, None, None, Some(1));


    // FFT based M2L for a vector of charges
    let nvecs = 1;
    let tmp = vec![1.0; nsources * nvecs];
    let mut charges = rlst_dynamic_array2!(f64, [nsources, nvecs]);
    charges.data_mut().copy_from_slice(&tmp);

    // FMM parameters
    let n_crit = None;
    let depth = Some(4);
    let expansion_order = [7, 7, 7, 7, 7];
    let prune_empty = true;

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


    // FMM parameters
    let surface_diff = None;
    let depth = Some(4);
    let expansion_order = [7, 7, 7, 7, 7];
    let prune_empty = true;
    let svd_threshold = Some(1e-6);

    let fmm_blas = SingleNodeBuilder::new()
        .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
        .unwrap()
        .parameters(
            charges.data(),
            &expansion_order,
            Laplace3dKernel::new(),
            EvalType::Value,
            BlasFieldTranslationSaRcmp::new(svd_threshold, surface_diff),
        )
        .unwrap()
        .build()
        .unwrap();

    let mut group = c.benchmark_group("Laplace Potentials 7 Digits");
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

fn laplace_8_digits(c: &mut Criterion) {
    // Setup random sources and targets
    let nsources = 1000000;
    let ntargets = 1000000;
    let sources = points_fixture::<f64>(nsources, None, None, Some(0));
    let targets = points_fixture::<f64>(ntargets, None, None, Some(1));

    // FFT based M2L for a vector of charges
    let nvecs = 1;
    let tmp = vec![1.0; nsources * nvecs];
    let mut charges = rlst_dynamic_array2!(f64, [nsources, nvecs]);
    charges.data_mut().copy_from_slice(&tmp);

    // FMM parameters
    let n_crit = None;
    let depth = Some(4);
    let expansion_order = [8, 8, 8, 8, 8];
    let prune_empty = true;

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

    // FMM parameters
    let n_crit = None;
    let depth = Some(4);
    let expansion_order = [7, 7, 7, 7, 7];
    let prune_empty = true;
    let svd_threshold = Some(1e-6);
    let surface_diff = Some(2);

    let fmm_blas = SingleNodeBuilder::new()
        .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
        .unwrap()
        .parameters(
            charges.data(),
            &expansion_order,
            Laplace3dKernel::new(),
            EvalType::Value,
            BlasFieldTranslationSaRcmp::new(svd_threshold, surface_diff),
        )
        .unwrap()
        .build()
        .unwrap();

    let mut group = c.benchmark_group("Laplace Potentials 8 Digits");
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

fn laplace_9_digits(c: &mut Criterion) {
    // Setup random sources and targets
    let nsources = 1000000;
    let ntargets = 1000000;
    let sources = points_fixture::<f64>(nsources, None, None, Some(0));
    let targets = points_fixture::<f64>(ntargets, None, None, Some(1));

    // FFT based M2L for a vector of charges
    let nvecs = 1;
    let tmp = vec![1.0; nsources * nvecs];
    let mut charges = rlst_dynamic_array2!(f64, [nsources, nvecs]);
    charges.data_mut().copy_from_slice(&tmp);

    // FMM parameters
    let n_crit = None;
    let depth = Some(4);
    let expansion_order = [9, 9, 9, 9, 9];
    let prune_empty = true;

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

    // FMM parameters
    let n_crit = None;
    let depth = Some(4);
    let expansion_order = [9, 9, 9, 9, 9];
    let prune_empty = true;
    let svd_threshold = Some(1e-6);
    let surface_diff = Some(0);

    let fmm_blas = SingleNodeBuilder::new()
        .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
        .unwrap()
        .parameters(
            charges.data(),
            &expansion_order,
            Laplace3dKernel::new(),
            EvalType::Value,
            BlasFieldTranslationSaRcmp::new(svd_threshold, surface_diff),
        )
        .unwrap()
        .build()
        .unwrap();

    let mut group = c.benchmark_group("Laplace Potentials 9 Digits");
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


fn laplace_10_digits(c: &mut Criterion) {
    // Setup random sources and targets
    let nsources = 1000000;
    let ntargets = 1000000;
    let sources = points_fixture::<f64>(nsources, None, None, Some(0));
    let targets = points_fixture::<f64>(ntargets, None, None, Some(1));

    // FFT based M2L for a vector of charges
    let nvecs = 1;
    let tmp = vec![1.0; nsources * nvecs];
    let mut charges = rlst_dynamic_array2!(f64, [nsources, nvecs]);
    charges.data_mut().copy_from_slice(&tmp);

    // FMM parameters
    let n_crit = None;
    let depth = Some(5);
    let expansion_order = [10, 10, 10, 10, 10, 10];
    let prune_empty = true;

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

    // FMM parameters
    let n_crit = None;
    let depth = Some(4);
    let expansion_order = [9, 9, 9, 9, 9];
    let prune_empty = true;
    let svd_threshold = Some(1e-9);
    let surface_diff = Some(2);

    let fmm_blas = SingleNodeBuilder::new()
        .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
        .unwrap()
        .parameters(
            charges.data(),
            &expansion_order,
            Laplace3dKernel::new(),
            EvalType::Value,
            BlasFieldTranslationSaRcmp::new(svd_threshold, surface_diff),
        )
        .unwrap()
        .build()
        .unwrap();

    let mut group = c.benchmark_group("Laplace Potentials 10 Digits");
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

criterion_group!(laplace_p_f64, laplace_6_digits, laplace_7_digits, laplace_8_digits, laplace_9_digits, laplace_10_digits);

criterion_main!(laplace_p_f64);
