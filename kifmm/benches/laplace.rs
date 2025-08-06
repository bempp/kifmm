use std::{fs, path::PathBuf, time::Duration};

use criterion::{
    criterion_group, criterion_main,
    measurement::{Measurement, WallTime},
    Criterion,
};

use num::Float;
use rand_distr::uniform::SampleUniform;
use rlst::{rlst_dynamic_array2, MatrixSvd, RawAccess, RawAccessMut, RlstScalar};
use serde_yaml::Value;

use green_kernels::{laplace_3d::Laplace3dKernel, types::GreenKernelEvalType};

use kifmm::{
    fftw::array::AlignedAllocable,
    fmm::types::{BlasFieldTranslationSaRcmp, FftFieldTranslation, FmmSvdMode, SingleNodeBuilder},
    linalg::rsvd::MatrixRsvd,
    traits::{
        fftw::Dft,
        field::{SourceToTargetTranslation, TargetTranslation},
        fmm::{DataAccess, Evaluate},
        general::single_node::{AsComplex, Epsilon, Hadamard8x8},
        tree::{SingleFmmTree, SingleTree},
    },
    tree::helpers::points_fixture,
};

fn benchmark_fft_m2l<
    T: RlstScalar<Real = T>
        + SampleUniform
        + Float
        + Epsilon
        + MatrixSvd
        + AsComplex
        + Dft<InputType = T, OutputType = <T as AsComplex>::ComplexType>
        + Default
        + AlignedAllocable,
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
        format!("M2L=FFT digits={digits} n_points={n_points}"),
        |b| b.iter(|| fmm_fft.evaluate()),
    );

    group.bench_function(
        format!("M2L=FFT digits={digits} n_points={n_points},  M2L "),
        |b| {
            b.iter(|| {
                for level in 2..=fmm_fft.tree().target_tree().depth() {
                    fmm_fft.m2l(level).unwrap();
                }
            })
        },
    );

    group.bench_function(
        format!("M2L=FFT digits={digits} n_points={n_points}, P2P "),
        |b| b.iter(|| fmm_fft.p2p().unwrap()),
    );
}

#[allow(clippy::too_many_arguments)]
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
        format!("M2L=BLAS digits={digits} n_points={n_points} n_vecs={n_vecs}",),
        |b| b.iter(|| fmm_blas.evaluate()),
    );

    group.bench_function(
        format!("M2L=BLAS digits={digits} n_points={n_points} n_vecs={n_vecs}, M2L",),
        |b| {
            b.iter(|| {
                for level in 2..=fmm_blas.tree().target_tree().depth() {
                    fmm_blas.m2l(level).unwrap();
                }
            })
        },
    );

    group.bench_function(
        format!("M2L=BLAS digits={digits} n_points={n_points} n_vecs={n_vecs}, P2P",),
        |b| b.iter(|| fmm_blas.p2p().unwrap()),
    );
}

fn laplace_potentials(c: &mut Criterion) {
    let mut group = c.benchmark_group("Potentials");

    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(10));

    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("bench-conf-fmm.yaml");

    let yaml_str = fs::read_to_string(path).unwrap();
    let root: Value = serde_yaml::from_str(&yaml_str).unwrap();

    let arch = std::env::var("ARCH").unwrap_or_else(|_| {
        panic!("Please set ARCH in .cargo/config.toml or your environment");
    });

    let kernel = "laplace";
    println!("Testing kernel: {kernel}");
    println!("Using arch: {arch}");

    let arch = root
        .get("kernel")
        .and_then(Value::as_mapping)
        .expect(".")
        .get(kernel)
        .and_then(Value::as_mapping)
        .expect(".")
        .get("arch")
        .and_then(Value::as_mapping)
        .expect(".")
        .get(arch)
        .and_then(Value::as_mapping)
        .expect("Expected 'arch' to be a mapping");

    for precision in ["fp32", "fp64"] {
        let data = arch
            .get(precision)
            .and_then(Value::as_mapping)
            .expect("Expected 'fp32' to be a mapping");

        // Parse the FFT M2L parameters
        let fft_m2l = data
            .get("fft")
            .and_then(Value::as_mapping)
            .expect("Expected 'fft' to be a mapping");

        let n_points_map = fft_m2l
            .get("n_points")
            .and_then(Value::as_mapping)
            .expect("Expected 'n_points' to be a mapping");

        for (n_points_key, n_points_val) in n_points_map {
            let n_points = n_points_key.as_u64().unwrap_or(0);
            if let Some(digits_map) = n_points_val.get("digits").and_then(Value::as_mapping) {
                for (digit_key, params_val) in digits_map {
                    let digits = digit_key.as_u64().unwrap_or(1) as usize;

                    let e = params_val.get("order").and_then(Value::as_u64).unwrap_or(0);

                    let block_size = params_val
                        .get("block_size")
                        .and_then(Value::as_u64)
                        .unwrap_or(0);
                    let depth = params_val.get("depth").and_then(Value::as_u64).unwrap_or(0);

                    println!(
                        "precision: {precision}, m2l: fft, n_points: {n_points}, digits: {digits} "
                    );

                    if precision == "fp32" {
                        benchmark_fft_m2l::<f32, WallTime>(
                            &mut group,
                            digits,
                            n_points.try_into().unwrap(),
                            e.try_into().unwrap(),
                            Some(block_size.try_into().unwrap()),
                            Some(depth),
                        );
                    } else {
                        benchmark_fft_m2l::<f64, WallTime>(
                            &mut group,
                            digits,
                            n_points.try_into().unwrap(),
                            e.try_into().unwrap(),
                            Some(block_size.try_into().unwrap()),
                            Some(depth),
                        );
                    }
                }
            }
        }

        // Parse the BLAS M2L parameters
        let blas_m2l = data
            .get("blas")
            .and_then(Value::as_mapping)
            .expect("Expected 'blas' to be a mapping");

        let n_points_map = blas_m2l
            .get("n_points")
            .and_then(Value::as_mapping)
            .expect("Expected 'n_points' to be a mapping");

        for (n_points_key, n_points_val) in n_points_map {
            let n_points = n_points_key.as_u64().unwrap_or(0);
            if let Some(digits_map) = n_points_val.get("digits").and_then(Value::as_mapping) {
                for (digit_key, params_val) in digits_map {
                    let digits = digit_key.as_i64().unwrap_or(0) as usize;

                    let e = params_val.get("order").and_then(Value::as_u64).unwrap_or(0);

                    let surface_diff = params_val
                        .get("surface_diff")
                        .and_then(Value::as_u64)
                        .unwrap_or(0);
                    let svd_threshold = params_val
                        .get("svd_threshold")
                        .and_then(Value::as_f64)
                        .unwrap_or(0.);
                    let n_oversamples = params_val
                        .get("n_oversamples")
                        .and_then(Value::as_u64)
                        .unwrap_or(0);
                    let depth = params_val.get("depth").and_then(Value::as_u64).unwrap_or(0);

                    println!(
                        "precision: {precision}, m2l: blas, n_points: {n_points}, digits: {digits}"
                    );

                    if precision == "fp32" {
                        benchmark_blas_m2l::<f32, WallTime>(
                            &mut group,
                            digits,
                            n_points.try_into().unwrap(),
                            1,
                            e.try_into().unwrap(),
                            Some(surface_diff.try_into().unwrap()),
                            Some(depth),
                            FmmSvdMode::new(
                                true,
                                None,
                                None,
                                Some(n_oversamples.try_into().unwrap()),
                                None,
                            ),
                            Some(svd_threshold as f32),
                        );
                    } else {
                        benchmark_blas_m2l::<f64, WallTime>(
                            &mut group,
                            digits,
                            n_points.try_into().unwrap(),
                            1,
                            e.try_into().unwrap(),
                            Some(surface_diff.try_into().unwrap()),
                            Some(depth),
                            FmmSvdMode::new(
                                true,
                                None,
                                None,
                                Some(n_oversamples.try_into().unwrap()),
                                None,
                            ),
                            Some(svd_threshold),
                        );
                    }
                }
            }
        }
    }
}

criterion_group!(benches, laplace_potentials);
criterion_main!(benches);
