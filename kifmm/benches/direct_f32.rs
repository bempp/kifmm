use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion};
use green_kernels::traits::Kernel;
use green_kernels::{laplace_3d::Laplace3dKernel, types::GreenKernelEvalType};
use kifmm::tree::helpers::points_fixture;
use rlst::{rlst_dynamic_array2, RawAccess, RawAccessMut};

fn multithreaded_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("Multi Threaded Direct f32");

    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(15));

    {
        let n_sources = 20000;
        let n_targets = 20000;
        let sources = points_fixture::<f32>(n_sources, None, None, Some(0));
        let targets = points_fixture::<f32>(n_targets, None, None, Some(1));

        // FFT based M2L for a vector of charges
        let nvecs = 1;
        let tmp = vec![1.0; n_sources * nvecs];
        let mut charges = rlst_dynamic_array2!(f32, [n_sources, nvecs]);
        charges.data_mut().copy_from_slice(&tmp);

        let kernel = Laplace3dKernel::new();

        let mut result = rlst_dynamic_array2!(f32, [n_sources, nvecs]);

        group.bench_function(format!("N={n_sources}"), |b| {
            b.iter(|| {
                kernel.evaluate_mt(
                    GreenKernelEvalType::Value,
                    sources.data(),
                    targets.data(),
                    charges.data(),
                    result.data_mut(),
                )
            })
        });
    }

    {
        let n_sources = 100000;
        let n_targets = 100000;
        let sources = points_fixture::<f32>(n_sources, None, None, Some(0));
        let targets = points_fixture::<f32>(n_targets, None, None, Some(1));

        // FFT based M2L for a vector of charges
        let nvecs = 1;
        let tmp = vec![1.0; n_sources * nvecs];
        let mut charges = rlst_dynamic_array2!(f32, [n_sources, nvecs]);
        charges.data_mut().copy_from_slice(&tmp);

        let kernel = Laplace3dKernel::new();

        let mut result = rlst_dynamic_array2!(f32, [n_sources, nvecs]);

        group.bench_function(format!("N={n_sources}"), |b| {
            b.iter(|| {
                kernel.evaluate_mt(
                    GreenKernelEvalType::Value,
                    sources.data(),
                    targets.data(),
                    charges.data(),
                    result.data_mut(),
                )
            })
        });
    }

    {
        let n_sources = 500000;
        let n_targets = 500000;
        let sources = points_fixture::<f32>(n_sources, None, None, Some(0));
        let targets = points_fixture::<f32>(n_targets, None, None, Some(1));

        // FFT based M2L for a vector of charges
        let nvecs = 1;
        let tmp = vec![1.0; n_sources * nvecs];
        let mut charges = rlst_dynamic_array2!(f32, [n_sources, nvecs]);
        charges.data_mut().copy_from_slice(&tmp);

        let kernel = Laplace3dKernel::new();

        let mut result = rlst_dynamic_array2!(f32, [n_sources, nvecs]);

        group.bench_function(format!("N={n_sources}"), |b| {
            b.iter(|| {
                kernel.evaluate_mt(
                    GreenKernelEvalType::Value,
                    sources.data(),
                    targets.data(),
                    charges.data(),
                    result.data_mut(),
                )
            })
        });
    }
}

fn singlethreaded_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("Single Threaded Direct f32");

    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(15));

    {
        let n_sources = 5000;
        let n_targets = 5000;
        let sources = points_fixture::<f32>(n_sources, None, None, Some(0));
        let targets = points_fixture::<f32>(n_targets, None, None, Some(1));

        // FFT based M2L for a vector of charges
        let nvecs = 1;
        let tmp = vec![1.0; n_sources * nvecs];
        let mut charges = rlst_dynamic_array2!(f32, [n_sources, nvecs]);
        charges.data_mut().copy_from_slice(&tmp);

        let kernel = Laplace3dKernel::new();

        let mut result = rlst_dynamic_array2!(f32, [n_sources, nvecs]);

        group.bench_function(format!("N={n_sources}"), |b| {
            b.iter(|| {
                kernel.evaluate_st(
                    GreenKernelEvalType::Value,
                    sources.data(),
                    targets.data(),
                    charges.data(),
                    result.data_mut(),
                )
            })
        });
    }

    {
        let n_sources = 20000;
        let n_targets = 20000;
        let sources = points_fixture::<f32>(n_sources, None, None, Some(0));
        let targets = points_fixture::<f32>(n_targets, None, None, Some(1));

        // FFT based M2L for a vector of charges
        let nvecs = 1;
        let tmp = vec![1.0; n_sources * nvecs];
        let mut charges = rlst_dynamic_array2!(f32, [n_sources, nvecs]);
        charges.data_mut().copy_from_slice(&tmp);

        let kernel = Laplace3dKernel::new();

        let mut result = rlst_dynamic_array2!(f32, [n_sources, nvecs]);

        group.bench_function(format!("N={n_sources}"), |b| {
            b.iter(|| {
                kernel.evaluate_st(
                    GreenKernelEvalType::Value,
                    sources.data(),
                    targets.data(),
                    charges.data(),
                    result.data_mut(),
                )
            })
        });
    }
}

criterion_group!(d_f32, multithreaded_f32, singlethreaded_f32);
criterion_main!(d_f32);
