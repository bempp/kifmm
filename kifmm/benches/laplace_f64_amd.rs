use core::panic;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion};
use green_kernels::{laplace_3d::Laplace3dKernel, types::GreenKernelEvalType};
use kifmm::fmm::types::FmmSvdMode;
use kifmm::fmm::types::{BlasFieldTranslationSaRcmp, FftFieldTranslation, SingleNodeBuilder};
use kifmm::traits::field::{SourceToTargetTranslation, TargetTranslation};
use kifmm::traits::fmm::{DataAccess, Evaluate};
use kifmm::traits::tree::{SingleFmmTree, SingleTree};
use kifmm::tree::helpers::{points_fixture, points_fixture_oblate_spheroid, points_fixture_sphere};
use rlst::{rlst_dynamic_array2, RawAccess, RawAccessMut};

fn fft_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("F64 Potentials FFT-M2L 3 Digits");

    let n_points = 1000000;
    let geometries = ["uniform", "sphere", "spheroid"];

    // Tree depth
    let depth_vec = vec![
        [4, 4, 5], // 6 digits, for each geometry
        [4, 4, 5], // 8 digits for each geometry
        [4, 4, 5], // 10 digits for each geometry
    ];

    // Expansion order
    let e_vec = vec![
        [7, 7, 7], // 6 digits, for each geometry
        [9, 9, 9], // 8 digits for each geometry
        [11, 11, 11], // 10 digits for each geometry
    ];

    // Block size
    let b_vec = vec![
        [64, 32, 128], // 6 digits, for each geometry
        [32, 16, 64], // 8 digits for each geometry
        [16, 64, 64], // 10 digits for each geometry
    ];

    let experiments = [6, 8, 10]; // number of digits sought

    let prune_empty = true;

    for (i, digits) in experiments.iter().enumerate() {
        for (j, &geometry) in geometries.iter().enumerate() {
            let sources;
            let targets;
            if geometry == "uniform" {
                sources = points_fixture::<f64>(n_points, Some(0.), Some(1.), None);
                targets = points_fixture::<f64>(n_points, Some(0.), Some(1.), None);
            } else if geometry == "sphere" {
                sources = points_fixture_sphere(n_points);
                targets = points_fixture_sphere(n_points);
            } else if geometry == "spheroid" {
                sources = points_fixture_oblate_spheroid(n_points, 1.0, 0.5);
                targets = points_fixture_oblate_spheroid(n_points, 1.0, 0.5);
            } else {
                panic!("invalid geometry");
            }

            let charges = vec![1f64; n_points];

            let depth = Some(depth_vec[i][j]);
            let e = e_vec[i][j];
            let b = b_vec[i][j];
            let expansion_order = vec![e; depth.unwrap() as usize + 1];
            let block_size = Some(b);

            let mut fmm_fft = SingleNodeBuilder::new(false)
                .tree(sources.data(), targets.data(), None, depth, prune_empty)
                .unwrap()
                .parameters(
                    &charges,
                    &expansion_order,
                    Laplace3dKernel::new(),
                    GreenKernelEvalType::Value,
                    FftFieldTranslation::new(block_size),
                )
                .unwrap()
                .build()
                .unwrap();

            group.bench_function(
                format!("M2L=FFT digits={} geometry={}", digits, geometry),
                |b| b.iter(|| fmm_fft.evaluate()),
            );

            group.bench_function(
                format!("M2L=FFT digits={}, geometry={} M2L ", digits, geometry),
                |b| {
                    b.iter(|| {
                        for level in 2..=fmm_fft.tree().target_tree().depth() {
                            fmm_fft.m2l(level).unwrap();
                        }
                    })
                },
            );

            group.bench_function(
                format!("M2L=FFT digits={} geometry={}, P2P ", digits, geometry),
                |b| b.iter(|| fmm_fft.p2p().unwrap()),
            );
        }
    }
}

fn blas_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("F64 Potentials BLAS-M2L");

    let n_points = 1000000;
    let geometries = ["uniform", "sphere", "spheroid"];

    // Tree depth
    let depth_vec = vec![
        [4, 4, 4], // 6 digits, for each geometry
        [4, 4, 4], // 8 digits for each geometry
        [4, 4, 4], // 10 digits for each geometry
    ];

    // Expansion order
    let e_vec = vec![
        [7, 7, 7], // 6 digits, for each geometry
        [8, 8, 9], // 8 digits for each geometry
        [10, 10, 11], // 10 digits for each geometry
    ];

    // SVD threshold
    let svd_threshold_vec = vec![
        [Some(1e-5), Some(1e-5), Some(1e-9)],              // 6 digits
        [Some(1e-5), Some(1e-7), Some(1e-7)], // 8 digits
        [Some(1e-9), Some(1e-7), Some(1e-11)], // 10 digits
    ];

    let svd_mode_vec = vec![
        [FmmSvdMode::new(true, None, None, Some(5), None), FmmSvdMode::new(true, None, None, Some(20), None), FmmSvdMode::new(true, None, None, Some(10), None)],
        [FmmSvdMode::new(false, None, None, None, None), FmmSvdMode::new(true, None, None, Some(20), None), FmmSvdMode::new(true, None, None, Some(10), None)],
        [FmmSvdMode::new(true, None, None, Some(20), None), FmmSvdMode::new(true, None, None, Some(20), None), FmmSvdMode::new(true, None, None, Some(20), None)],
    ];

    let surface_diff_vec = vec![
        [None, Some(1), None], // 6 digits
        [Some(1), Some(1), Some(1)], // 8 digits
        [Some(2), Some(2), Some(1)] // 10 digits
    ];

    let nvecs = vec![1, 5, 10];

    let experiments = [6, 8, 10]; // number of digits sought

    let prune_empty = true;

    for (i, digits) in experiments.iter().enumerate() {
        for (j, &geometry) in geometries.iter().enumerate() {
            for (k, nvec) in nvecs.iter().enumerate() {

                let sources;
                let targets;
                if geometry == "uniform" {
                    sources = points_fixture::<f64>(n_points, Some(0.), Some(1.), None);
                    targets = points_fixture::<f64>(n_points, Some(0.), Some(1.), None);
                } else if geometry == "sphere" {
                    sources = points_fixture_sphere(n_points);
                    targets = points_fixture_sphere(n_points);
                } else if geometry == "spheroid" {
                    sources = points_fixture_oblate_spheroid(n_points, 1.0, 0.5);
                    targets = points_fixture_oblate_spheroid(n_points, 1.0, 0.5);
                } else {
                    panic!("invalid geometry");
                }

                let charges = vec![1f64; n_points*nvec];

                let depth = Some(depth_vec[i][j]);
                let e = e_vec[i][j];
                let threshold = svd_threshold_vec[i][j];
                let surface_diff = surface_diff_vec[i][j];
                let svd_mode = svd_mode_vec[i][j];

                let expansion_order = vec![e; depth.unwrap() as usize + 1];

                let mut fmm = SingleNodeBuilder::new(false)
                    .tree(sources.data(), targets.data(), None, depth, prune_empty)
                    .unwrap()
                    .parameters(
                        &charges,
                        &expansion_order,
                        Laplace3dKernel::new(),
                        GreenKernelEvalType::Value,
                        BlasFieldTranslationSaRcmp::new(threshold, surface_diff, svd_mode)
                    )
                    .unwrap()
                    .build()
                    .unwrap();

                group.bench_function(
                    format!("M2L=BLAS digits={} geometry={} nvecs={}", digits, geometry, nvec),
                    |b| b.iter(|| fmm.evaluate()),
                );

                group.bench_function(
                    format!("M2L=BLAS digits={}, geometry={} nvecs={} M2L ", digits, geometry, nvec),
                    |b| {
                        b.iter(|| {
                            for level in 2..=fmm.tree().target_tree().depth() {
                                fmm.m2l(level).unwrap();
                            }
                        })
                    },
                );

                group.bench_function(
                    format!("M2L=BLAS digits={} geometry={} nvecs={} P2P ", digits, geometry, nvec),
                    |b| b.iter(|| fmm.p2p().unwrap()),
                );
            }
        }
    }
}


criterion_group!(laplace_p_f64, fft_f64, blas_f64);
criterion_main!(laplace_p_f64);
