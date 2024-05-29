//! C ABI
extern crate blas_src;
extern crate lapack_src;

use std::{collections::HashMap, time::Duration};

use green_kernels::{
    helmholtz_3d::Helmholtz3dKernel, laplace_3d::Laplace3dKernel, types::EvalType,
};
use rlst::{c32, c64, RlstScalar};

use crate::{
    fmm::KiFmm, BlasFieldTranslationIa, BlasFieldTranslationSaRcmp, FftFieldTranslation,
    SingleNodeBuilder,
    traits::fmm::Fmm,
};

#[no_mangle]
pub extern "C" fn evaluate_laplace_fft_f32(fmm: *const KiFmm<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>>, timed: bool) -> HashMap<String, Duration> {
    unsafe { fmm.as_ref().unwrap().evaluate(timed).unwrap() }
}

#[no_mangle]
pub extern "C" fn evaluate_laplace_fft_f64(fmm: *const KiFmm<f64, Laplace3dKernel<f64>, FftFieldTranslation<f64>>, timed: bool) {
    unsafe { fmm.as_ref().unwrap().evaluate(timed).unwrap() };
}

#[no_mangle]
pub extern "C" fn evaluate_laplace_blas_f32(fmm: *const KiFmm<f32, Laplace3dKernel<f32>, BlasFieldTranslationSaRcmp<f32>>, timed: bool) {
    unsafe { fmm.as_ref().unwrap().evaluate(timed).unwrap() };
}

#[no_mangle]
pub extern "C" fn evaluate_laplace_blas_f64(fmm: *const KiFmm<f64, Laplace3dKernel<f64>, BlasFieldTranslationSaRcmp<f64>>, timed: bool) {
    unsafe { fmm.as_ref().unwrap().evaluate(timed).unwrap() };
}

#[no_mangle]
pub extern "C" fn evaluate_helmholtz_fft_f32(fmm: *const KiFmm<c32, Helmholtz3dKernel<c32>, FftFieldTranslation<c32>>, timed: bool) {
    unsafe { fmm.as_ref().unwrap().evaluate(timed).unwrap() };
}

#[no_mangle]
pub extern "C" fn evaluate_helmholtz_fft_f64(fmm: *const KiFmm<c64, Helmholtz3dKernel<c64>, FftFieldTranslation<c64>>, timed: bool) {
    unsafe { fmm.as_ref().unwrap().evaluate(timed).unwrap() };
}

#[no_mangle]
pub extern "C" fn evaluate_helmholtz_blas_f32(fmm: *const KiFmm<c32, Helmholtz3dKernel<c32>, BlasFieldTranslationIa<c32>>, timed: bool) {
    unsafe { fmm.as_ref().unwrap().evaluate(timed).unwrap() };
}

#[no_mangle]
pub extern "C" fn evaluate_helmholtz_blas_f64(fmm: *const KiFmm<c64, Helmholtz3dKernel<c64>, BlasFieldTranslationIa<c64>>, timed: bool) {
    unsafe { fmm.as_ref().unwrap().evaluate(timed).unwrap() };
}

#[no_mangle]
pub extern "C" fn laplace_fft_f32(
    expansion_order: usize,
    charges: *const f32,
    sources: *const f32,
    nsources: usize,
    targets: *const f32,
    ntargets: usize,
    n_crit: u64,
    prune_empty: bool,
    kernel_eval_type: usize,
) -> *const KiFmm<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>> {
    let dim = 3;

    let kernel_eval_type = if kernel_eval_type == 0 {
        EvalType::Value
    } else if kernel_eval_type == 1 {
        EvalType::ValueDeriv
    } else {
        panic!("Invalid evaluation mode")
    };

    let source_to_target = FftFieldTranslation::new();
    let kernel = Laplace3dKernel::new();

    let sources_slice: &[f32] =
        unsafe { std::slice::from_raw_parts(sources, nsources * dim) };
    let targets_slice: &[f32] =
        unsafe { std::slice::from_raw_parts(targets, ntargets * dim) };
    let charges_slice: &[f32] = unsafe { std::slice::from_raw_parts(charges, nsources) };

    let b = Box::new(
        SingleNodeBuilder::new()
            .tree(&sources_slice, &targets_slice, Some(n_crit), prune_empty)
            .unwrap()
            .parameters(
                charges_slice,
                expansion_order,
                kernel,
                kernel_eval_type,
                source_to_target,
            )
            .unwrap()
            .build()
            .unwrap(),
    );

    Box::into_raw(b)
}

#[no_mangle]
pub extern "C" fn laplace_fft_f64(
    expansion_order: usize,
    charges: *const f64,
    sources: *const f64,
    nsources: usize,
    targets: *const f64,
    ntargets: usize,
    n_crit: u64,
    prune_empty: bool,
    kernel_eval_type: usize,
) -> *const KiFmm<f64, Laplace3dKernel<f64>, FftFieldTranslation<f64>> {
    let dim = 3;

    let kernel_eval_type = if kernel_eval_type == 0 {
        EvalType::Value
    } else if kernel_eval_type == 1 {
        EvalType::ValueDeriv
    } else {
        panic!("Invalid evaluation mode")
    };

    let source_to_target = FftFieldTranslation::new();
    let kernel = Laplace3dKernel::new();

    let sources_slice: &[f64] =
        unsafe { std::slice::from_raw_parts(sources, nsources * dim) };
    let targets_slice: &[f64] =
        unsafe { std::slice::from_raw_parts(targets, ntargets * dim) };
    let charges_slice: &[f64] = unsafe { std::slice::from_raw_parts(charges, nsources) };

    let b = Box::new(
        SingleNodeBuilder::new()
            .tree(&sources_slice, &targets_slice, Some(n_crit), prune_empty)
            .unwrap()
            .parameters(
                charges_slice,
                expansion_order,
                kernel,
                kernel_eval_type,
                source_to_target,
            )
            .unwrap()
            .build()
            .unwrap(),
    );

    Box::into_raw(b)
}

#[no_mangle]
pub extern "C" fn laplace_blas_f32(
    expansion_order: usize,
    charges: *const f32,
    sources: *const f32,
    nsources: usize,
    targets: *const f32,
    ntargets: usize,
    n_crit: u64,
    prune_empty: bool,
    kernel_eval_type: usize,
    svd_threshold: f32,
) -> *const KiFmm<f32, Laplace3dKernel<f32>, BlasFieldTranslationSaRcmp<f32>> {
    let dim = 3;

    let kernel_eval_type = if kernel_eval_type == 0 {
        EvalType::Value
    } else if kernel_eval_type == 1 {
        EvalType::ValueDeriv
    } else {
        panic!("Invalid evaluation mode")
    };

    let source_to_target = BlasFieldTranslationSaRcmp::<f32>::new(Some(svd_threshold));
    let kernel = Laplace3dKernel::<f32>::new();

    let sources_slice: &[f32] =
        unsafe { std::slice::from_raw_parts(sources, nsources * dim) };
    let targets_slice: &[f32] =
        unsafe { std::slice::from_raw_parts(targets, ntargets * dim) };
    let charges_slice: &[f32] = unsafe { std::slice::from_raw_parts(charges, nsources) };

    let b = Box::new(
        SingleNodeBuilder::new()
            .tree(&sources_slice, &targets_slice, Some(n_crit), prune_empty)
            .unwrap()
            .parameters(
                &charges_slice,
                expansion_order,
                kernel,
                kernel_eval_type,
                source_to_target,
            )
            .unwrap()
            .build()
            .unwrap(),
    );

    Box::into_raw(b)
}

#[no_mangle]
pub extern "C" fn laplace_blas_f64(
    expansion_order: usize,
    charges: *const f64,
    sources: *const f64,
    nsources: usize,
    targets: *const f64,
    ntargets: usize,
    n_crit: u64,
    prune_empty: bool,
    kernel_eval_type: usize,
    svd_threshold: f64,
) -> *const KiFmm<f64, Laplace3dKernel<f64>, BlasFieldTranslationSaRcmp<f64>> {
    let dim = 3;

    let kernel_eval_type = if kernel_eval_type == 0 {
        EvalType::Value
    } else if kernel_eval_type == 1 {
        EvalType::ValueDeriv
    } else {
        panic!("Invalid evaluation mode")
    };

    let source_to_target = BlasFieldTranslationSaRcmp::<f64>::new(Some(svd_threshold));
    let kernel = Laplace3dKernel::<f64>::new();

    let sources_slice: &[f64] =
        unsafe { std::slice::from_raw_parts(sources, nsources * dim) };
    let targets_slice: &[f64] =
        unsafe { std::slice::from_raw_parts(targets, ntargets * dim) };
    let charges_slice: &[f64] = unsafe { std::slice::from_raw_parts(charges, nsources) };

    let b = Box::new(
        SingleNodeBuilder::new()
            .tree(&sources_slice, &targets_slice, Some(n_crit), prune_empty)
            .unwrap()
            .parameters(
                &charges_slice,
                expansion_order,
                kernel,
                kernel_eval_type,
                source_to_target,
            )
            .unwrap()
            .build()
            .unwrap(),
    );

    Box::into_raw(b)
}

#[no_mangle]
pub extern "C" fn helmholtz_fft_f32(
    expansion_order: usize,
    charges: *const f32,
    sources: *const f32,
    nsources: usize,
    targets: *const f32,
    ntargets: usize,
    n_crit: u64,
    prune_empty: bool,
    kernel_eval_type: usize,
    wavenumber: f32,
) -> *const KiFmm<c32, Helmholtz3dKernel<c32>, FftFieldTranslation<c32>> {
    let dim = 3;

    let kernel_eval_type = if kernel_eval_type == 0 {
        EvalType::Value
    } else if kernel_eval_type == 1 {
        EvalType::ValueDeriv
    } else {
        panic!("Invalid evaluation mode")
    };

    let source_to_target = FftFieldTranslation::new();
    let kernel = Helmholtz3dKernel::new(wavenumber);

    let sources_slice: &[f32] =
        unsafe { std::slice::from_raw_parts(sources, nsources * dim) };
    let targets_slice: &[f32] =
        unsafe { std::slice::from_raw_parts(targets, ntargets * dim) };
    let charges_slice: &[c32] =
        unsafe { std::slice::from_raw_parts(charges as *const c32, nsources * 2) };

    let b = Box::new(
        SingleNodeBuilder::new()
            .tree(&sources_slice, &targets_slice, Some(n_crit), prune_empty)
            .unwrap()
            .parameters(
                charges_slice,
                expansion_order,
                kernel,
                kernel_eval_type,
                source_to_target,
            )
            .unwrap()
            .build()
            .unwrap(),
    );

    Box::into_raw(b)
}

#[no_mangle]
pub extern "C" fn helmholtz_fft_f64(
    expansion_order: usize,
    charges: *const f64,
    sources: *const f64,
    nsources: usize,
    targets: *const f64,
    ntargets: usize,
    n_crit: u64,
    prune_empty: bool,
    kernel_eval_type: usize,
    wavenumber: f64,
) -> *const KiFmm<c64, Helmholtz3dKernel<c64>, FftFieldTranslation<c64>> {
    let dim = 3;

    let kernel_eval_type = if kernel_eval_type == 0 {
        EvalType::Value
    } else if kernel_eval_type == 1 {
        EvalType::ValueDeriv
    } else {
        panic!("Invalid evaluation mode")
    };

    let source_to_target = FftFieldTranslation::new();
    let kernel = Helmholtz3dKernel::new(wavenumber);

    let sources_slice: &[f64] =
        unsafe { std::slice::from_raw_parts(sources, nsources * dim) };
    let targets_slice: &[f64] =
        unsafe { std::slice::from_raw_parts(targets, ntargets * dim) };
    let charges_slice: &[c64] =
        unsafe { std::slice::from_raw_parts(charges as *const c64, nsources * 2) };

    let b = Box::new(
        SingleNodeBuilder::new()
            .tree(&sources_slice, &targets_slice, Some(n_crit), prune_empty)
            .unwrap()
            .parameters(
                charges_slice,
                expansion_order,
                kernel,
                kernel_eval_type,
                source_to_target,
            )
            .unwrap()
            .build()
            .unwrap(),
    );

    Box::into_raw(b)
}


#[no_mangle]
pub extern "C" fn helmholtz_blas_f32(
    expansion_order: usize,
    charges: *const f32,
    sources: *const f32,
    nsources: usize,
    targets: *const f32,
    ntargets: usize,
    n_crit: u64,
    prune_empty: bool,
    kernel_eval_type: usize,
    wavenumber: f32,
    svd_threshold: f32,
) -> *const KiFmm<c32, Helmholtz3dKernel<c32>, BlasFieldTranslationIa<c32>> {
    let dim = 3;

    let kernel_eval_type = if kernel_eval_type == 0 {
        EvalType::Value
    } else if kernel_eval_type == 1 {
        EvalType::ValueDeriv
    } else {
        panic!("Invalid evaluation mode")
    };

    let source_to_target = BlasFieldTranslationIa::new(Some(svd_threshold));
    let kernel = Helmholtz3dKernel::new(wavenumber);

    let sources_slice: &[f32] =
        unsafe { std::slice::from_raw_parts(sources, nsources * dim) };
    let targets_slice: &[f32] =
        unsafe { std::slice::from_raw_parts(targets, ntargets * dim) };

    let charges_slice: &[c32] =
        unsafe { std::slice::from_raw_parts(charges as *const c32, nsources * 2) };

    let b = Box::new(
        SingleNodeBuilder::new()
            .tree(&sources_slice, &targets_slice, Some(n_crit), prune_empty)
            .unwrap()
            .parameters(
                charges_slice,
                expansion_order,
                kernel,
                kernel_eval_type,
                source_to_target,
            )
            .unwrap()
            .build()
            .unwrap(),
    );

    Box::into_raw(b)
}

#[no_mangle]
pub extern "C" fn helmholtz_blas_f64(
    expansion_order: usize,
    charges: *const f64,
    sources: *const f64,
    nsources: usize,
    targets: *const f64,
    ntargets: usize,
    n_crit: u64,
    prune_empty: bool,
    kernel_eval_type: usize,
    wavenumber: f64,
    svd_threshold: f64,
) -> *const KiFmm<c64, Helmholtz3dKernel<c64>, BlasFieldTranslationIa<c64>> {
    let dim = 3;

    let kernel_eval_type = if kernel_eval_type == 0 {
        EvalType::Value
    } else if kernel_eval_type == 1 {
        EvalType::ValueDeriv
    } else {
        panic!("Invalid evaluation mode")
    };

    let source_to_target = BlasFieldTranslationIa::new(Some(svd_threshold));
    let kernel = Helmholtz3dKernel::new(wavenumber);

    let sources_slice: &[f64] =
        unsafe { std::slice::from_raw_parts(sources, nsources * dim) };
    let targets_slice: &[f64] =
        unsafe { std::slice::from_raw_parts(targets, ntargets * dim) };

    let charges_slice: &[c64] =
        unsafe { std::slice::from_raw_parts(charges as *const c64, nsources * 2) };

    let b = Box::new(
        SingleNodeBuilder::new()
            .tree(&sources_slice, &targets_slice, Some(n_crit), prune_empty)
            .unwrap()
            .parameters(
                charges_slice,
                expansion_order,
                kernel,
                kernel_eval_type,
                source_to_target,
            )
            .unwrap()
            .build()
            .unwrap(),
    );

    Box::into_raw(b)
}


