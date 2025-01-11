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

fn fft_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("F32 Potentials FFT-M2L 3 Digits");

    let n_points = 1000000;
    let geometries = ["uniform", "sphere", "spheroid"];

    // TODO: 4 digits for sphere and spheroid are dummies for now

    // Tree depth
    let depth_vec = vec![
        [4, 5, 5], // 3 digits, for each geometry
        [4, 5, 5], // 4 digits for each geometry
    ];

    // Expansion order
    let e_vec = vec![
        [4, 5, 5], // 3 digits, for each geometry
        [5, 5, 5], // 4 digits for each geometry
    ];

    // Block size
    let b_vec = vec![
        [16, 64, 64], // 3 digits, for each geometry
        [16, 64, 64], // 4 digits for each geometry
    ];

    let experiments = [3, 4]; // number of digits sought

    let prune_empty = true;

    for (i, digits) in experiments.iter().enumerate() {
        for (j, &geometry) in geometries.iter().enumerate() {
            let sources;
            let targets;
            if geometry == "uniform" {
                sources = points_fixture::<f32>(n_points, Some(0.), Some(1.), None);
                targets = points_fixture::<f32>(n_points, Some(0.), Some(1.), None);
            } else if geometry == "sphere" {
                sources = points_fixture_sphere(n_points);
                targets = points_fixture_sphere(n_points);
            } else if geometry == "spheroid" {
                sources = points_fixture_oblate_spheroid(n_points, 1.0, 0.5);
                targets = points_fixture_oblate_spheroid(n_points, 1.0, 0.5);
            } else {
                panic!("invalid geometry");
            }

            let charges = vec![1f32; n_points];

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

fn blas_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("F32 Potentials BLAS-M2L");
}

fn blas_mat_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("F32 Potentials BLAS-M2L Matrix");
}

criterion_group!(laplace_p_f32, fft_f32);
criterion_main!(laplace_p_f32);
