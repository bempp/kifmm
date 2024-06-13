
use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion};
use green_kernels::{laplace_3d::Laplace3dKernel, types::EvalType};
use kifmm::fmm::types::{BlasFieldTranslationSaRcmp, FftFieldTranslation, SingleNodeBuilder};
use kifmm::traits::fmm::Fmm;
use kifmm::tree::helpers::points_fixture;
use rlst::{rlst_dynamic_array2, RawAccess, RawAccessMut};
use green_kernels::traits::Kernel;

extern crate blas_src;
extern crate lapack_src;


fn direct_f64(c: &mut Criterion) {
    // Setup random sources and targets
    let nsources = 100000;
    let ntargets = 100000;
    let sources = points_fixture::<f64>(nsources, None, None, Some(0));
    let targets = points_fixture::<f64>(ntargets, None, None, Some(1));
 
    let mut group = c.benchmark_group("Direct f64");
    
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(15));

    {
        let nsources = 20000;
        let ntargets = 20000;
        let sources = points_fixture::<f64>(nsources, None, None, Some(0));
        let targets = points_fixture::<f64>(ntargets, None, None, Some(1));

        // FFT based M2L for a vector of charges
        let nvecs = 1;
        let tmp = vec![1.0; nsources * nvecs];
        let mut charges = rlst_dynamic_array2!(f64, [nsources, nvecs]);
        charges.data_mut().copy_from_slice(&tmp);

        let kernel = Laplace3dKernel::new();
        
        let mut result = rlst_dynamic_array2!(f64, [nsources, nvecs]);


        group.bench_function(format!("M2L=BLAS, N={nsources}"), |b| {b.iter(|| kernel.evaluate_mt(EvalType::Value, sources.data(), targets.data(), charges.data(), result.data_mut()))});
    }
    
    {
        let nsources = 100000;
        let ntargets = 100000;
        let sources = points_fixture::<f64>(nsources, None, None, Some(0));
        let targets = points_fixture::<f64>(ntargets, None, None, Some(1));

        // FFT based M2L for a vector of charges
        let nvecs = 1;
        let tmp = vec![1.0; nsources * nvecs];
        let mut charges = rlst_dynamic_array2!(f64, [nsources, nvecs]);
        charges.data_mut().copy_from_slice(&tmp);

        let kernel = Laplace3dKernel::new();
        
        let mut result = rlst_dynamic_array2!(f64, [nsources, nvecs]);


        group.bench_function(format!("M2L=BLAS, N={nsources}"), |b| {b.iter(|| kernel.evaluate_mt(EvalType::Value, sources.data(), targets.data(), charges.data(), result.data_mut()))});
    }
    
    {
        let nsources = 500000;
        let ntargets = 500000;
        let sources = points_fixture::<f64>(nsources, None, None, Some(0));
        let targets = points_fixture::<f64>(ntargets, None, None, Some(1));

        // FFT based M2L for a vector of charges
        let nvecs = 1;
        let tmp = vec![1.0; nsources * nvecs];
        let mut charges = rlst_dynamic_array2!(f64, [nsources, nvecs]);
        charges.data_mut().copy_from_slice(&tmp);

        let kernel = Laplace3dKernel::new();
        
        let mut result = rlst_dynamic_array2!(f64, [nsources, nvecs]);


        group.bench_function(format!("M2L=BLAS, N={nsources}"), |b| {b.iter(|| kernel.evaluate_mt(EvalType::Value, sources.data(), targets.data(), charges.data(), result.data_mut()))});
    }
}

criterion_group!(d_f64, direct_f64);

criterion_main!(d_f64);
