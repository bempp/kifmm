use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion};
use green_kernels::{laplace_3d::Laplace3dKernel, types::EvalType};
use kifmm::fmm::types::FmmSvdMode;
use kifmm::fmm::types::{BlasFieldTranslationSaRcmp, FftFieldTranslation, SingleNodeBuilder};
use kifmm::traits::fmm::Fmm;
use kifmm::tree::helpers::points_fixture;
use rlst::{rlst_dynamic_array2, RawAccess, RawAccessMut};

extern crate blas_src;
extern crate lapack_src;

fn laplace_potentials_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("F64 Potentials");

    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(15));

    let nsources = 1000000;
    let ntargets = 1000000;
    let sources = points_fixture::<f64>(nsources, None, None, Some(0));
    let targets = points_fixture::<f64>(ntargets, None, None, Some(1));

    let nvecs = 1;
    let tmp = vec![1.0; nsources * nvecs];
    let mut charges = rlst_dynamic_array2!(f64, [nsources, nvecs]);
    charges.data_mut().copy_from_slice(&tmp);

    // 6 Digits
    {
        // FFT based M2L for a vector of charges
        // FMM parameters
        let n_crit = None;
        let depth = Some(5);
        let e = 6;
        let expansion_order = vec![e; depth.unwrap() as usize + 1];
        let prune_empty = true;
        let block_size = Some(128);

        let fmm_fft = SingleNodeBuilder::new()
            .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
            .unwrap()
            .parameters(
                charges.data(),
                &expansion_order,
                Laplace3dKernel::new(),
                EvalType::Value,
                FftFieldTranslation::new(block_size),
            )
            .unwrap()
            .build()
            .unwrap();

        group.bench_function(format!("M2L=FFT digits=6"), |b| {
            b.iter(|| fmm_fft.evaluate(false))
        });

        // BLAS based M2L for a vector of charges
        // FMM parameters
        let n_crit = None;
        let depth = Some(5);
        let e = 6;
        let surface_diff = None;
        let svd_threshold = Some(1e-6);
        let svd_mode = crate::FmmSvdMode::new(true, None, None, Some(10), None);
        let expansion_order = vec![e; depth.unwrap() as usize + 1];
        let prune_empty = true;

        let fmm_blas = SingleNodeBuilder::new()
            .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
            .unwrap()
            .parameters(
                charges.data(),
                &expansion_order,
                Laplace3dKernel::new(),
                EvalType::Value,
                BlasFieldTranslationSaRcmp::new(svd_threshold, surface_diff, svd_mode),
            )
            .unwrap()
            .build()
            .unwrap();

        group.bench_function(format!("M2L=BLAS digits=6"), |b| {
            b.iter(|| fmm_blas.evaluate(false))
        });
    }

    // 7 Digits
    {
        // FFT based M2L for a vector of charges
        // FMM parameters
        let n_crit = None;
        let depth = Some(4);
        let e = 7;
        let block_size = Some(128);
        let expansion_order = vec![e; depth.unwrap() as usize + 1];
        let prune_empty = true;

        let fmm_fft = SingleNodeBuilder::new()
            .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
            .unwrap()
            .parameters(
                charges.data(),
                &expansion_order,
                Laplace3dKernel::new(),
                EvalType::Value,
                FftFieldTranslation::new(block_size),
            )
            .unwrap()
            .build()
            .unwrap();

        group.bench_function(format!("M2L=FFT digits=7"), |b| {
            b.iter(|| fmm_fft.evaluate(false))
        });

        // BLAS based M2L for a vector of charges
        // FMM parameters
        let n_crit = None;
        let depth = Some(5);
        let e = 6;
        let surface_diff = Some(2);
        let svd_threshold = Some(1e-6);
        let svd_mode = crate::FmmSvdMode::new(true, None, None, Some(5), None);
        let expansion_order = vec![e; depth.unwrap() as usize + 1];
        let prune_empty = true;

        let fmm_blas = SingleNodeBuilder::new()
            .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
            .unwrap()
            .parameters(
                charges.data(),
                &expansion_order,
                Laplace3dKernel::new(),
                EvalType::Value,
                BlasFieldTranslationSaRcmp::new(svd_threshold, surface_diff, svd_mode),
            )
            .unwrap()
            .build()
            .unwrap();

        group.bench_function(format!("M2L=BLAS digits=7"), |b| {
            b.iter(|| fmm_blas.evaluate(false))
        });
    }

    // 8 Digits
    {
        // FFT based M2L for a vector of charges
        // FMM parameters
        let n_crit = None;
        let depth = Some(4);
        let e = 8;
        let block_size = Some(128);
        let expansion_order = vec![e; depth.unwrap() as usize + 1];
        let prune_empty = true;

        let fmm_fft = SingleNodeBuilder::new()
            .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
            .unwrap()
            .parameters(
                charges.data(),
                &expansion_order,
                Laplace3dKernel::new(),
                EvalType::Value,
                FftFieldTranslation::new(block_size),
            )
            .unwrap()
            .build()
            .unwrap();

        group.bench_function(format!("M2L=FFT digits=8"), |b| {
            b.iter(|| fmm_fft.evaluate(false))
        });

        // BLAS based M2L for a vector of charges
        // FMM parameters
        let n_crit = None;
        let depth = Some(4);
        let e = 7;
        let surface_diff = Some(1);
        let svd_threshold = Some(1e-6);
        let svd_mode = crate::FmmSvdMode::new(true, None, None, Some(20), None);
        let expansion_order = vec![e; depth.unwrap() as usize + 1];
        let prune_empty = true;

        let fmm_blas = SingleNodeBuilder::new()
            .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
            .unwrap()
            .parameters(
                charges.data(),
                &expansion_order,
                Laplace3dKernel::new(),
                EvalType::Value,
                BlasFieldTranslationSaRcmp::new(svd_threshold, surface_diff, svd_mode),
            )
            .unwrap()
            .build()
            .unwrap();

        group.bench_function(format!("M2L=BLAS digits=8"), |b| {
            b.iter(|| fmm_blas.evaluate(false))
        });
    }

    // 9 Digits
    {
        // FFT based M2L for a vector of charges
        // FMM parameters
        let n_crit = None;
        let depth = Some(4);
        let e = 9;
        let block_size = Some(32);
        let expansion_order = vec![e; depth.unwrap() as usize + 1];
        let prune_empty = true;

        let fmm_fft = SingleNodeBuilder::new()
            .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
            .unwrap()
            .parameters(
                charges.data(),
                &expansion_order,
                Laplace3dKernel::new(),
                EvalType::Value,
                FftFieldTranslation::new(block_size),
            )
            .unwrap()
            .build()
            .unwrap();

        group.bench_function(format!("M2L=FFT digits=9"), |b| {
            b.iter(|| fmm_fft.evaluate(false))
        });

        // BLAS based M2L for a vector of charges
        // FMM parameters
        let n_crit = None;
        let depth = Some(4);
        let e = 8;
        let surface_diff = Some(2);
        let svd_threshold = Some(1e-6);
        let svd_mode = crate::FmmSvdMode::new(true, None, None, Some(5), None);
        let expansion_order = vec![e; depth.unwrap() as usize + 1];
        let prune_empty = true;

        let fmm_blas = SingleNodeBuilder::new()
            .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
            .unwrap()
            .parameters(
                charges.data(),
                &expansion_order,
                Laplace3dKernel::new(),
                EvalType::Value,
                BlasFieldTranslationSaRcmp::new(svd_threshold, surface_diff, svd_mode),
            )
            .unwrap()
            .build()
            .unwrap();

        group.bench_function(format!("M2L=BLAS digits=9"), |b| {
            b.iter(|| fmm_blas.evaluate(false))
        });
    }

    // 10 Digits
    {
        // FFT based M2L for a vector of charges
        // FMM parameters
        let n_crit = None;
        let depth = Some(4);
        let e = 10;
        let block_size = Some(64);
        let expansion_order = vec![e; depth.unwrap() as usize + 1];
        let prune_empty = true;

        let fmm_fft = SingleNodeBuilder::new()
            .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
            .unwrap()
            .parameters(
                charges.data(),
                &expansion_order,
                Laplace3dKernel::new(),
                EvalType::Value,
                FftFieldTranslation::new(block_size),
            )
            .unwrap()
            .build()
            .unwrap();

        group.bench_function(format!("M2L=FFT digits=10"), |b| {
            b.iter(|| fmm_fft.evaluate(false))
        });

        // BLAS based M2L for a vector of charges
        // FMM parameters
        let n_crit = None;
        let depth = Some(4);
        let e = 9;
        let surface_diff = Some(2);
        let svd_threshold = Some(1e-6);
        let svd_mode = crate::FmmSvdMode::new(true, None, None, Some(10), None);
        let expansion_order = vec![e; depth.unwrap() as usize + 1];
        let prune_empty = true;

        let fmm_blas = SingleNodeBuilder::new()
            .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
            .unwrap()
            .parameters(
                charges.data(),
                &expansion_order,
                Laplace3dKernel::new(),
                EvalType::Value,
                BlasFieldTranslationSaRcmp::new(svd_threshold, surface_diff, svd_mode),
            )
            .unwrap()
            .build()
            .unwrap();

        group.bench_function(format!("M2L=BLAS digits=10"), |b| {
            b.iter(|| fmm_blas.evaluate(false))
        });
    }
}

criterion_group!(laplace_p_f64, laplace_potentials_f64);
criterion_main!(laplace_p_f64);