use std::time::Duration;

use criterion::measurement::{Measurement, WallTime};
use criterion::{criterion_group, criterion_main, Criterion};
use green_kernels::traits::Kernel;
use green_kernels::{laplace_3d::Laplace3dKernel, types::GreenKernelEvalType};
use kifmm::fftw::array::AlignedAllocable;
use kifmm::fmm::types::FmmSvdMode;
use kifmm::fmm::types::{BlasFieldTranslationSaRcmp, FftFieldTranslation, SingleNodeBuilder};
use kifmm::linalg::rsvd::MatrixRsvd;
use kifmm::traits::fftw::Dft;
use kifmm::traits::field::{SourceToTargetTranslation, TargetTranslation};
use kifmm::traits::fmm::{DataAccess, Evaluate};
use kifmm::traits::general::single_node::{AsComplex, Epsilon, Hadamard8x8};
use kifmm::traits::tree::{SingleFmmTree, SingleTree};
use kifmm::tree::helpers::points_fixture;
use kifmm::KiFmm;
use num::Float;
use rand_distr::uniform::SampleUniform;
use rlst::{rlst_dynamic_array2, MatrixSvd, RawAccess, RawAccessMut, RlstScalar};

fn benchmark_fft_m2l<
    T: RlstScalar<Real = T>
        + SampleUniform
        + Float
        + Epsilon
        + MatrixSvd
        + AsComplex
        + Dft<InputType = T, OutputType = <T as AsComplex>::ComplexType>
        + Default,
    M: Measurement,
>(
    group: &mut criterion::BenchmarkGroup<'_, M>,
    digits: usize,
    n_points: usize,
    e: usize,
    block_size: Option<usize>,
    depth: Option<u64>,
) where
    <T as RlstScalar>::Real: Epsilon + RlstScalar + Default,
    <T as Dft>::Plan: Sync,
    <T as AsComplex>::ComplexType: AlignedAllocable,
    <T as AsComplex>::ComplexType: Hadamard8x8<Scalar = <T as AsComplex>::ComplexType>,
    T: AlignedAllocable,
{
    // FFT based M2L for a vector of charges
    // FMM parameters
    let n_crit = None;
    let expansion_order = vec![e; depth.unwrap() as usize + 1];
    let prune_empty = true;
    let sources = points_fixture::<T>(n_points, None, None, Some(0));
    let targets = points_fixture::<T>(n_points, None, None, Some(1));
    let tmp = vec![T::one(); n_points];
    let mut charges = rlst_dynamic_array2!(T, [n_points, 1]);
    charges.data_mut().copy_from_slice(&tmp);

    let mut fmm_fft = SingleNodeBuilder::new(false)
        .tree(sources.data(), targets.data(), n_crit, depth, prune_empty)
        .unwrap()
        .parameters(
            charges.data(),
            &expansion_order,
            Laplace3dKernel::new(),
            GreenKernelEvalType::Value,
            FftFieldTranslation::<T>::new(block_size),
        )
        .unwrap()
        .build()
        .unwrap();

    group.bench_function(
        format!("M2L=FFT digits={} n_points={}", digits, n_points),
        |b| b.iter(|| fmm_fft.evaluate()),
    );

    group.bench_function(
        format!("M2L=FFT digits={} n_points={},  M2L ", digits, n_points),
        |b| {
            b.iter(|| {
                for level in 2..=fmm_fft.tree().target_tree().depth() {
                    fmm_fft.m2l(level).unwrap();
                }
            })
        },
    );

    group.bench_function(
        format!("M2L=FFT digits={} n_points={}, P2P ", digits, n_points),
        |b| b.iter(|| fmm_fft.p2p().unwrap()),
    );
}

fn benchmark_blas_m2l<
    T: RlstScalar<Real = T> + Epsilon + MatrixRsvd + Float + SampleUniform,
    M: Measurement,
>(
    group: &mut criterion::BenchmarkGroup<'_, M>,
    digits: usize,
    n_points: usize,
    n_vecs: usize,
    e: usize,
    surface_diff: Option<usize>,
    depth: Option<u64>,
    svd_mode: FmmSvdMode,
    svd_threshold: Option<T>,
) where
    <T as RlstScalar>::Real: Epsilon,
{
    let sources = points_fixture::<T>(n_points, None, None, Some(0));
    let targets = points_fixture::<T>(n_points, None, None, Some(1));

    let tmp = vec![T::one(); n_points * n_vecs];
    let mut charges = rlst_dynamic_array2!(T, [n_points, n_vecs]);
    charges.data_mut().copy_from_slice(&tmp);

    // BLAS based M2L for a vector of charges
    // FMM parameters
    let n_crit = None;
    let expansion_order = vec![e; depth.unwrap() as usize + 1];
    let prune_empty = true;

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

    group.bench_function(
        format!("M2L=BLAS digits={} n_points={} n_vecs={}", digits, n_points, n_vecs),
        |b| b.iter(|| fmm_blas.evaluate()),
    );

    group.bench_function(
        format!("M2L=BLAS digits={} n_points={} n_vecs={}, M2L", digits, n_points, n_vecs),
        |b| {
            b.iter(|| {
                for level in 2..=fmm_blas.tree().target_tree().depth() {
                    fmm_blas.m2l(level).unwrap();
                }
            })
        },
    );

    group.bench_function(
        format!("M2L=BLAS digits={} n_points={} n_vecs={}, P2P", digits, n_points, n_vecs),
        |b| b.iter(|| fmm_blas.p2p().unwrap()),
    );
}

fn laplace_potentials_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("F64 Potentials");

    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(10));

    // 1e6 points
    //benchmark_fft_m2l::<f64, WallTime>(&mut group, 6, 1e6 as usize, 6, Some(128), Some(5));
    //benchmark_fft_m2l::<f64, WallTime>(&mut group, 8, 1e6 as usize, 8, Some(32), Some(4));
    //benchmark_fft_m2l::<f64, WallTime>(&mut group, 10, 1e6 as usize, 10, Some(64), Some(4));

    // 8e6 points
    benchmark_fft_m2l::<f64, WallTime>(&mut group, 4, 8e6 as usize, 4, Some(128), Some(6));
    //benchmark_fft_m2l::<f64, WallTime>(&mut group, 6, 8e6 as usize, 6, Some(16), Some(5));
    //benchmark_fft_m2l::<f64, WallTime>(&mut group, 8, 8e6 as usize, 8, Some(64), Some(5));
    //benchmark_fft_m2l::<f64, WallTime>(&mut group, 10, 8e6 as usize, 10, Some(32), Some(5));


    for &n_vecs in [10].iter() {

        // 1e6 points
     //   benchmark_blas_m2l::<f64, WallTime>(
     //       &mut group,
     //       6,
     //       1e6 as usize,
     //       n_vecs,
     //       5,
     //       Some(1),
     //       Some(5),
     //       FmmSvdMode::new(true, None, None, Some(5), None),
     //       Some(0.001),
     //   );

     //   benchmark_blas_m2l::<f64, WallTime>(
     //       &mut group,
     //       8,
     //       1e6 as usize,
     //       n_vecs,
     //       8,
     //       Some(2),
     //       Some(5),
     //       FmmSvdMode::new(true, None, None, Some(10), None),
     //       Some(0.001),
     //   );

     //   benchmark_blas_m2l::<f64, WallTime>(
     //       &mut group,
     //       10,
     //       1e6 as usize,
     //       n_vecs,
     //       9,
     //       Some(2),
     //       Some(4),
     //       FmmSvdMode::new(true, None, None, Some(10), None),
     //       Some(1e-7),
     //   );

     //   // 8e6 points
     //   benchmark_blas_m2l::<f64, WallTime>(
     //       &mut group,
     //       4,
     //       8e6 as usize,
     //       n_vecs,
     //       3,
     //       Some(2),
     //       Some(6),
     //       FmmSvdMode::new(true, None, None, Some(10), None),
     //       Some(0.001),
     //   );

     //   benchmark_blas_m2l::<f64, WallTime>(
     //       &mut group,
     //       6,
     //       8e6 as usize,
     //       n_vecs,
     //       5,
     //       Some(1),
     //       Some(5),
     //       FmmSvdMode::new(true, None, None, Some(20), None),
     //       Some(0.001),
     //   );

        benchmark_blas_m2l::<f64, WallTime>(
            &mut group,
            8,
            8e6 as usize,
            n_vecs,
            7,
            Some(1),
            Some(5),
            FmmSvdMode::new(true, None, None, Some(20), None),
            Some(0.00001),
        );

        benchmark_blas_m2l::<f64, WallTime>(
            &mut group,
            10,
            8e6 as usize,
            n_vecs,
            9,
            Some(2),
            Some(5),
            FmmSvdMode::new(true, None, None, Some(20), None),
            Some(1e-7),
        );
    }
}

criterion_group!(laplace_p_f32, laplace_potentials_f64);
criterion_main!(laplace_p_f32);
