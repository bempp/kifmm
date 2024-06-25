use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion};
use green_kernels::{laplace_3d::Laplace3dKernel, types::EvalType};
use kifmm::fmm::types::{BlasFieldTranslationSaRcmp, FftFieldTranslation, SingleNodeBuilder, FmmSvdMode};
use kifmm::tree::helpers::points_fixture;
use rlst::{rlst_dynamic_array2, RawAccess, RawAccessMut};

extern crate blas_src;
extern crate lapack_src;

fn precomputation_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("F64 Setup");

    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(15));

    {
        let nsources = 5000;
        let ntargets = 5000;
        let sources = points_fixture::<f64>(nsources, None, None, Some(0));
        let targets = points_fixture::<f64>(ntargets, None, None, Some(1));

        // FMM parameters
        let n_crit = Some(150);
        let depth = None;
        let e = 6;
        let expansion_order = [e];
        let prune_empty = true;

        // FFT based M2L for a vector of charges
        let nvecs = 1;
        let tmp = vec![1.0; nsources * nvecs];
        let mut charges = rlst_dynamic_array2!(f64, [nsources, nvecs]);
        charges.data_mut().copy_from_slice(&tmp);

        group.bench_function(format!("M2L=FFT expansion_order={e}"), |b| {
            b.iter(|| {
                let fmm_fft = SingleNodeBuilder::new()
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
            })
        });

        group.bench_function(format!("M2L=BLAS expansion_order={e}"), |b| {
            b.iter(|| {
                let fmm_blas = SingleNodeBuilder::new()
                    .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
                    .unwrap()
                    .parameters(
                        charges.data(),
                        &expansion_order,
                        Laplace3dKernel::new(),
                        EvalType::Value,
                        BlasFieldTranslationSaRcmp::new(None, None, FmmSvdMode::new(false, None, None, None, None))
                    )
                    .unwrap()
                    .build()
                    .unwrap();
            })
        });
    }

    {
        let nsources = 5000;
        let ntargets = 5000;
        let sources = points_fixture::<f64>(nsources, None, None, Some(0));
        let targets = points_fixture::<f64>(ntargets, None, None, Some(1));

        // FMM parameters
        let n_crit = Some(150);
        let depth = None;
        let e = 8;
        let expansion_order = [e];
        let prune_empty = true;

        // FFT based M2L for a vector of charges
        let nvecs = 1;
        let tmp = vec![1.0; nsources * nvecs];
        let mut charges = rlst_dynamic_array2!(f64, [nsources, nvecs]);
        charges.data_mut().copy_from_slice(&tmp);

        group.bench_function(format!("M2L=FFT expansion_order={e}"), |b| {
            b.iter(|| {
                let fmm_fft = SingleNodeBuilder::new()
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
            })
        });

        group.bench_function(format!("M2L=BLAS expansion_order={e}"), |b| {
            b.iter(|| {
                let fmm_blas = SingleNodeBuilder::new()
                    .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
                    .unwrap()
                    .parameters(
                        charges.data(),
                        &expansion_order,
                        Laplace3dKernel::new(),
                        EvalType::Value,
                        BlasFieldTranslationSaRcmp::new(None, None, FmmSvdMode::new(false, None, None, None, None))
                    )
                    .unwrap()
                    .build()
                    .unwrap();
            })
        });
    }

    {
        let nsources = 5000;
        let ntargets = 5000;
        let sources = points_fixture::<f64>(nsources, None, None, Some(0));
        let targets = points_fixture::<f64>(ntargets, None, None, Some(1));

        // FMM parameters
        let n_crit = Some(150);
        let depth = None;
        let e = 10;
        let expansion_order = [e];
        let prune_empty = true;

        // FFT based M2L for a vector of charges
        let nvecs = 1;
        let tmp = vec![1.0; nsources * nvecs];
        let mut charges = rlst_dynamic_array2!(f64, [nsources, nvecs]);
        charges.data_mut().copy_from_slice(&tmp);

        group.bench_function(format!("M2L=FFT expansion_order={e}"), |b| {
            b.iter(|| {
                let fmm_fft = SingleNodeBuilder::new()
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
            })
        });

        group.bench_function(format!("M2L=BLAS expansion_order={e}"), |b| {
            b.iter(|| {
                let fmm_blas = SingleNodeBuilder::new()
                    .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
                    .unwrap()
                    .parameters(
                        charges.data(),
                        &expansion_order,
                        Laplace3dKernel::new(),
                        EvalType::Value,
                        BlasFieldTranslationSaRcmp::new(None, None, FmmSvdMode::new(false, None, None, None, None))
                    )
                    .unwrap()
                    .build()
                    .unwrap();
            })
        });
    }
}

criterion_group!(p_f64, precomputation_f64);

criterion_main!(p_f64);
