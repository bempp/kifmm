use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion};
use green_kernels::{laplace_3d::Laplace3dKernel, types::EvalType};
use kifmm::fmm::types::FmmSvdMode;
use kifmm::fmm::types::{BlasFieldTranslationSaRcmp, FftFieldTranslation, SingleNodeBuilder};
use kifmm::traits::fmm::{Fmm, SourceToTargetTranslation, TargetTranslation};
use kifmm::traits::tree::{FmmTree, Tree};
use kifmm::tree::helpers::points_fixture;
use rlst::{rlst_dynamic_array2, RawAccess, RawAccessMut};

extern crate blas_src;
extern crate lapack_src;

fn laplace_potentials_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("F32 Potentials");

    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(15));

    let nsources = 1000000;
    let ntargets = 1000000;
    let sources = points_fixture::<f32>(nsources, None, None, Some(0));
    let targets = points_fixture::<f32>(ntargets, None, None, Some(1));


    // 3 Digits
    {

        let nvecs = 5;
        let tmp = vec![1.0; nsources * nvecs];
        let mut charges = rlst_dynamic_array2!(f32, [nsources, nvecs]);
        charges.data_mut().copy_from_slice(&tmp);

        // BLAS based M2L for a vector of charges
        // FMM parameters
        let n_crit = None;
        let depth = Some(4);
        let e = 3;
        let expansion_order = vec![e; depth.unwrap() as usize + 1];
        let prune_empty = true;
        let surface_diff = None;
        let svd_mode = crate::FmmSvdMode::new(true, None, None, Some(10), None);
        let svd_threshold = None;

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

        group.bench_function(format!("M2L=BLAS digits=3 vecs=5"), |b| {
            b.iter(|| fmm_blas.evaluate(false))
        });

        group.bench_function(format!("M2L=BLAS digits=3, M2L vecs=5"), |b| {
            b.iter(||
                for level in 2..= fmm_blas.tree().target_tree().depth() {
                    fmm_blas.m2l(level).unwrap();
                }
            )
        });

        group.bench_function(format!("M2L=BLAS digits=3, P2P vecs=5"), |b| {
            b.iter(||
                fmm_blas.p2p().unwrap()
            )
        });

        let nvecs = 10;
        let tmp = vec![1.0; nsources * nvecs];
        let mut charges = rlst_dynamic_array2!(f32, [nsources, nvecs]);
        charges.data_mut().copy_from_slice(&tmp);

        // BLAS based M2L for a vector of charges
        // FMM parameters
        let n_crit = None;
        let depth = Some(4);
        let e = 3;
        let expansion_order = vec![e; depth.unwrap() as usize + 1];
        let prune_empty = true;
        let surface_diff = None;
        let svd_mode = crate::FmmSvdMode::new(true, None, None, Some(10), None);
        let svd_threshold = None;

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

        group.bench_function(format!("M2L=BLAS digits=3 vecs=10"), |b| {
            b.iter(|| fmm_blas.evaluate(false))
        });

        group.bench_function(format!("M2L=BLAS digits=3, M2L vecs=10"), |b| {
            b.iter(||
                for level in 2..= fmm_blas.tree().target_tree().depth() {
                    fmm_blas.m2l(level).unwrap();
                }
            )
        });

        group.bench_function(format!("M2L=BLAS digits=3, P2P vecs=10"), |b| {
            b.iter(||
                fmm_blas.p2p().unwrap()
            )
        });
    }

    // 4 Digits
    {

        let nvecs = 5;
        let tmp = vec![1.0; nsources * nvecs];
        let mut charges = rlst_dynamic_array2!(f32, [nsources, nvecs]);
        charges.data_mut().copy_from_slice(&tmp);

        // BLAS based M2L for a vector of charges
        // FMM parameters
        let n_crit = None;
        let depth = Some(4);
        let e = 4;
        let expansion_order = vec![e; depth.unwrap() as usize + 1];
        let prune_empty = true;
        let surface_diff = Some(1);
        let svd_mode = crate::FmmSvdMode::new(true, None, None, Some(20), None);
        let svd_threshold = Some(0.001);

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

        group.bench_function(format!("M2L=BLAS digits=4 vecs=5"), |b| {
            b.iter(|| fmm_blas.evaluate(false))
        });

        group.bench_function(format!("M2L=BLAS digits=4, M2L vecs=5"), |b| {
            b.iter(||
                for level in 2..= fmm_blas.tree().target_tree().depth() {
                    fmm_blas.m2l(level).unwrap();
                }
            )
        });

        group.bench_function(format!("M2L=BLAS digits=4, P2P vecs=5"), |b| {
            b.iter(||
                fmm_blas.p2p().unwrap()
            )
        });

        let nvecs = 10;
        let tmp = vec![1.0; nsources * nvecs];
        let mut charges = rlst_dynamic_array2!(f32, [nsources, nvecs]);
        charges.data_mut().copy_from_slice(&tmp);

        // BLAS based M2L for a vector of charges
        // FMM parameters
        let n_crit = None;
        let depth = Some(4);
        let e = 4;
        let expansion_order = vec![e; depth.unwrap() as usize + 1];
        let prune_empty = true;
        let surface_diff = Some(1);
        let svd_mode = crate::FmmSvdMode::new(true, None, None, Some(20), None);
        let svd_threshold = Some(0.001);

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

        group.bench_function(format!("M2L=BLAS digits=4 vecs=10"), |b| {
            b.iter(|| fmm_blas.evaluate(false))
        });

        group.bench_function(format!("M2L=BLAS digits=4, M2L vecs=10"), |b| {
            b.iter(||
                for level in 2..= fmm_blas.tree().target_tree().depth() {
                    fmm_blas.m2l(level).unwrap();
                }
            )
        });

        group.bench_function(format!("M2L=BLAS digits=4, P2P vecs=10"), |b| {
            b.iter(||
                fmm_blas.p2p().unwrap()
            )
        });
    }

    // // 5 Digits
    // {
    //     let nvecs = 5;
    //     let tmp = vec![1.0; nsources * nvecs];
    //     let mut charges = rlst_dynamic_array2!(f32, [nsources, nvecs]);
    //     charges.data_mut().copy_from_slice(&tmp);

    //     // BLAS based M2L for a vector of charges
    //     // FMM parameters
    //     let n_crit = None;
    //     let depth = Some(4);
    //     let e = 5;
    //     let expansion_order = vec![e; depth.unwrap() as usize + 1];
    //     let prune_empty = true;
    //     let surface_diff = Some(2);
    //     let svd_mode = crate::FmmSvdMode::new(true, None, None, Some(5), None);
    //     let svd_threshold = Some(1e-2);

    //     let fmm_blas = SingleNodeBuilder::new()
    //         .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
    //         .unwrap()
    //         .parameters(
    //             charges.data(),
    //             &expansion_order,
    //             Laplace3dKernel::new(),
    //             EvalType::Value,
    //             BlasFieldTranslationSaRcmp::new(svd_threshold, surface_diff, svd_mode),
    //         )
    //         .unwrap()
    //         .build()
    //         .unwrap();
    //     group.bench_function(format!("M2L=BLAS digits=5 vecs=5"), |b| {
    //         b.iter(|| fmm_blas.evaluate(false))
    //     });

    //     let nvecs = 10;
    //     let tmp = vec![1.0; nsources * nvecs];
    //     let mut charges = rlst_dynamic_array2!(f32, [nsources, nvecs]);
    //     charges.data_mut().copy_from_slice(&tmp);

    //     // BLAS based M2L for a vector of charges
    //     // FMM parameters
    //     let n_crit = None;
    //     let depth = Some(4);
    //     let e = 5;
    //     let expansion_order = vec![e; depth.unwrap() as usize + 1];
    //     let prune_empty = true;
    //     let surface_diff = Some(2);
    //     let svd_mode = crate::FmmSvdMode::new(true, None, None, Some(5), None);
    //     let svd_threshold = Some(1e-2);

    //     let fmm_blas = SingleNodeBuilder::new()
    //         .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
    //         .unwrap()
    //         .parameters(
    //             charges.data(),
    //             &expansion_order,
    //             Laplace3dKernel::new(),
    //             EvalType::Value,
    //             BlasFieldTranslationSaRcmp::new(svd_threshold, surface_diff, svd_mode),
    //         )
    //         .unwrap()
    //         .build()
    //         .unwrap();
    //     group.bench_function(format!("M2L=BLAS digits=5 vecs=10"), |b| {
    //         b.iter(|| fmm_blas.evaluate(false))
    //     });
    // }
}

criterion_group!(laplace_p_f32, laplace_potentials_f32);
criterion_main!(laplace_p_f32);
