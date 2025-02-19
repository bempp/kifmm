use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion};
use green_kernels::{laplace_3d::Laplace3dKernel, types::GreenKernelEvalType};
use kifmm::fmm::types::FmmSvdMode;
use kifmm::fmm::types::{BlasFieldTranslationSaRcmp, FftFieldTranslation, SingleNodeBuilder};
use kifmm::traits::field::{SourceToTargetTranslation, TargetTranslation};
use kifmm::traits::fmm::{DataAccess, Evaluate};
use kifmm::traits::tree::{SingleFmmTree, SingleTree};
use kifmm::tree::helpers::points_fixture;
use rlst::{rlst_dynamic_array2, RawAccess, RawAccessMut};

fn laplace_potentials_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("F32 Potentials");

    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(15));

    let n_sources = 1000000;
    let n_targets = 1000000;
    let sources = points_fixture::<f32>(n_sources, None, None, Some(0));
    let targets = points_fixture::<f32>(n_targets, None, None, Some(1));

    let nvecs = 1;
    let tmp = vec![1.0; n_sources * nvecs];
    let mut charges = rlst_dynamic_array2!(f32, [n_sources, nvecs]);
    charges.data_mut().copy_from_slice(&tmp);

    // 3 Digits
    {
        // FFT based M2L for a vector of charges
        // FMM parameters
        let n_crit = None;
        let depth = Some(5);
        let e = 3;
        let expansion_order = vec![e; depth.unwrap() as usize + 1];
        let prune_empty = true;
        let block_size = Some(256);

        let mut fmm_fft = SingleNodeBuilder::new(true)
            .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
            .unwrap()
            .parameters(
                charges.data(),
                &expansion_order,
                Laplace3dKernel::new(),
                GreenKernelEvalType::Value,
                FftFieldTranslation::new(block_size),
            )
            .unwrap()
            .build()
            .unwrap();

        group.bench_function(format!("M2L=FFT digits=3"), |b| {
            b.iter(|| fmm_fft.evaluate())
        });

        group.bench_function(format!("M2L=FFT digits=3, M2L "), |b| {
            b.iter(|| {
                for level in 2..=fmm_fft.tree().target_tree().depth() {
                    fmm_fft.m2l(level).unwrap();
                }
            })
        });

        group.bench_function(format!("M2L=FFT digits=3, P2P "), |b| {
            b.iter(|| fmm_fft.p2p().unwrap())
        });

        // BLAS based M2L for a vector of charges
        // FMM parameters
        let n_crit = None;
        let depth = Some(5);
        let e = 3;
        let expansion_order = vec![e; depth.unwrap() as usize + 1];
        let prune_empty = true;
        let surface_diff = None;
        let svd_mode = crate::FmmSvdMode::new(true, None, None, Some(5), None);
        let svd_threshold = Some(1e-7);

        let mut fmm_blas = SingleNodeBuilder::new(false)
            .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
            .unwrap()
            .parameters(
                charges.data(),
                &expansion_order,
                Laplace3dKernel::new(),
                GreenKernelEvalType::Value,
                BlasFieldTranslationSaRcmp::new(svd_threshold, surface_diff, svd_mode),
            )
            .unwrap()
            .build()
            .unwrap();

        group.bench_function(format!("M2L=BLAS digits=3"), |b| {
            b.iter(|| fmm_blas.evaluate())
        });

        group.bench_function(format!("M2L=BLAS digits=3, M2L "), |b| {
            b.iter(|| {
                for level in 2..=fmm_blas.tree().target_tree().depth() {
                    fmm_blas.m2l(level).unwrap();
                }
            })
        });

        group.bench_function(format!("M2L=BLAS digits=3, P2P "), |b| {
            b.iter(|| fmm_blas.p2p().unwrap())
        });
    }

    // 4 Digits
    {
        // FFT based M2L for a vector of charges
        // FMM parameters
        let n_crit = None;
        let depth = Some(5);
        let e = 4;
        let expansion_order = vec![e; depth.unwrap() as usize + 1];
        let prune_empty = true;
        let block_size = Some(128);

        let mut fmm_fft = SingleNodeBuilder::new(false)
            .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
            .unwrap()
            .parameters(
                charges.data(),
                &expansion_order,
                Laplace3dKernel::new(),
                GreenKernelEvalType::Value,
                FftFieldTranslation::new(block_size),
            )
            .unwrap()
            .build()
            .unwrap();

        group.bench_function(format!("M2L=FFT digits=4"), |b| {
            b.iter(|| fmm_fft.evaluate())
        });

        group.bench_function(format!("M2L=FFT digits=4, M2L "), |b| {
            b.iter(|| {
                for level in 2..=fmm_fft.tree().target_tree().depth() {
                    fmm_fft.m2l(level).unwrap();
                }
            })
        });

        group.bench_function(format!("M2L=FFT digits=4, P2P "), |b| {
            b.iter(|| fmm_fft.p2p().unwrap())
        });

        // BLAS based M2L for a vector of charges
        // FMM parameters
        let n_crit = None;
        let depth = Some(5);
        let e = 3;
        let expansion_order = vec![e; depth.unwrap() as usize + 1];
        let prune_empty = true;
        let surface_diff = Some(1);
        let svd_mode = crate::FmmSvdMode::new(true, None, None, Some(5), None);
        let svd_threshold = Some(1e-4);

        let mut fmm_blas = SingleNodeBuilder::new(false)
            .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
            .unwrap()
            .parameters(
                charges.data(),
                &expansion_order,
                Laplace3dKernel::new(),
                GreenKernelEvalType::Value,
                BlasFieldTranslationSaRcmp::new(svd_threshold, surface_diff, svd_mode),
            )
            .unwrap()
            .build()
            .unwrap();

        group.bench_function(format!("M2L=BLAS digits=4"), |b| {
            b.iter(|| fmm_blas.evaluate())
        });

        group.bench_function(format!("M2L=BLAS digits=4, M2L "), |b| {
            b.iter(|| {
                for level in 2..=fmm_blas.tree().target_tree().depth() {
                    fmm_blas.m2l(level).unwrap();
                }
            })
        });

        group.bench_function(format!("M2L=BLAS digits=4, P2P "), |b| {
            b.iter(|| fmm_blas.p2p().unwrap())
        });
    }
}

criterion_group!(laplace_p_f32, laplace_potentials_f32);
criterion_main!(laplace_p_f32);
