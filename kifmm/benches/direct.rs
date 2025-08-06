use std::{fs, path::PathBuf, time::Duration};

use criterion::{
    criterion_group, criterion_main,
    measurement::{Measurement, WallTime},
    Criterion,
};

use num::{Complex, Float, One, Zero};
use rand_distr::uniform::SampleUniform;
use rlst::{RawAccess, RlstScalar};
use serde_yaml::Value;

use green_kernels::{
    helmholtz_3d::Helmholtz3dKernel, laplace_3d::Laplace3dKernel, traits::Kernel,
    types::GreenKernelEvalType,
};

use kifmm::{traits::general::single_node::Epsilon, tree::helpers::points_fixture};

fn benchmark_direct_laplace<
    T: RlstScalar<Real = T> + SampleUniform + Float + Default,
    M: Measurement,
>(
    mode: &str,
    precision: &str,
    group: &mut criterion::BenchmarkGroup<'_, M>,
    n_points: usize,
) where
    <T as RlstScalar>::Real: Epsilon + RlstScalar + Default,
{
    let sources = points_fixture::<T>(n_points, None, None, Some(0));
    let charges = vec![T::one(); n_points];
    let mut result = vec![T::zero(); n_points];
    let kernel = Laplace3dKernel::<T>::new();

    group.bench_function(
        format!(
            "mode={}, precision={}, n_points={}",
            mode, precision, n_points
        ),
        |b| {
            b.iter(|| {
                if mode == "multi_threaded" {
                    kernel.evaluate_mt(
                        GreenKernelEvalType::Value,
                        sources.data(),
                        sources.data(),
                        &charges,
                        &mut result,
                    )
                } else if mode == "single_threaded" {
                    kernel.evaluate_st(
                        GreenKernelEvalType::Value,
                        sources.data(),
                        sources.data(),
                        &charges,
                        &mut result,
                    )
                }
            })
        },
    );
}

fn benchmark_direct_helmholtz<
    T: RlstScalar<Real = T> + SampleUniform + Float + Default,
    M: Measurement,
>(
    mode: &str,
    precision: &str,
    group: &mut criterion::BenchmarkGroup<'_, M>,
    n_points: usize,
) where
    <T as RlstScalar>::Real: Epsilon + RlstScalar + Default,
    Complex<T>: RlstScalar<Complex = Complex<T>>,
    <Complex<T> as RlstScalar>::Real: RlstScalar + SampleUniform,
{
    let sources = points_fixture::<<Complex<T> as RlstScalar>::Real>(n_points, None, None, Some(0));
    let charges = vec![Complex::<T>::one(); n_points];
    let mut result = vec![Complex::<T>::zero(); n_points];
    let kernel = Helmholtz3dKernel::<Complex<T>>::new(<Complex<T> as RlstScalar>::real(1.0));

    group.bench_function(
        format!(
            "mode={}, precision={}, n_points={}",
            mode, precision, n_points
        ),
        |b| {
            b.iter(|| {
                if mode == "multi_threaded" {
                    kernel.evaluate_mt(
                        GreenKernelEvalType::Value,
                        sources.data(),
                        sources.data(),
                        &charges,
                        &mut result,
                    )
                } else if mode == "single_threaded" {
                    kernel.evaluate_st(
                        GreenKernelEvalType::Value,
                        sources.data(),
                        sources.data(),
                        &charges,
                        &mut result,
                    )
                }
            })
        },
    );
}

fn direct(c: &mut Criterion) {
    let mut group = c.benchmark_group("Kernel Evaluations");

    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(10));

    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("bench-conf-direct.yaml");

    let yaml_str = fs::read_to_string(path).unwrap();
    let root: Value = serde_yaml::from_str(&yaml_str).unwrap();

    let arch = std::env::var("ARCH").unwrap_or_else(|_| {
        panic!("Please set ARCH in .cargo/config.toml or your environment");
    });

    let kernel = "helmholtz";
    println!("Testing kernel: {}", kernel);
    println!("Using arch: {}", arch);

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

    for (mode_key, mode_val) in arch
        .get("mode")
        .and_then(Value::as_mapping)
        .expect("Expected 'mode' to be a mapping")
    {
        let mode = mode_key.as_str().expect("Expected mode key to be a string");
        for precision in ["fp32", "fp64"] {
            let data = mode_val
                .get(precision)
                .and_then(Value::as_mapping)
                .expect("Expected 'fp32' to be a mapping");

            let n_points_list = data
                .get("n_points")
                .and_then(Value::as_sequence)
                .expect("Expected 'n_points' to be a sequence");

            for n_points in n_points_list.iter() {
                let n_points: usize = n_points.as_u64().unwrap_or(0).try_into().unwrap();

                if kernel == "helmholtz" {
                    if precision == "fp32" {
                        benchmark_direct_helmholtz::<f32, WallTime>(
                            mode, precision, &mut group, n_points,
                        );
                    } else {
                        benchmark_direct_helmholtz::<f64, WallTime>(
                            mode, precision, &mut group, n_points,
                        );
                    }
                }

                if kernel == "laplace" {
                    if precision == "fp32" {
                        benchmark_direct_laplace::<f32, WallTime>(
                            mode, precision, &mut group, n_points,
                        );
                    } else {
                        benchmark_direct_laplace::<f64, WallTime>(
                            mode, precision, &mut group, n_points,
                        );
                    }
                }
            }
        }
    }
}

criterion_group!(benches, direct);
criterion_main!(benches);
