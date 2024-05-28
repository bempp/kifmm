extern crate blas_src;
extern crate lapack_src;

use green_kernels::{
    helmholtz_3d::Helmholtz3dKernel, laplace_3d::Laplace3dKernel, types::EvalType,
};
use rlst::{c32, c64, RlstScalar};

use crate::{
    fmm::KiFmm, BlasFieldTranslationIa, BlasFieldTranslationSaRcmp, FftFieldTranslation,
    SingleNodeBuilder,
};

macro_rules! laplace_fft_constructors {
    ($name: ident, $type: ident) => {
        /// Constructor for Laplace FFT FMMs
        #[no_mangle]
        pub extern "C" fn $name(
            expansion_order: usize,
            charges: *const $type,
            sources: *const $type,
            nsources: usize,
            targets: *const $type,
            ntargets: usize,
            n_crit: u64,
            sparse: bool,
            kernel_eval_type: usize,
        ) -> *const KiFmm<$type, Laplace3dKernel<$type>, FftFieldTranslation<$type>> {
            let dim = 3;

            let kernel_eval_type = if kernel_eval_type == 0 {
                EvalType::Value
            } else if kernel_eval_type == 1 {
                EvalType::ValueDeriv
            } else {
                panic!("Invalid evaluation mode")
            };

            let source_to_target = FftFieldTranslation::<$type>::new();
            let kernel = Laplace3dKernel::<$type>::new();

            let sources_slice: &[$type] =
                unsafe { std::slice::from_raw_parts(sources, nsources * dim) };
            let targets_slice: &[$type] =
                unsafe { std::slice::from_raw_parts(targets, ntargets * dim) };
            let charges_slice: &[$type] = unsafe { std::slice::from_raw_parts(charges, nsources) };

            let b = Box::new(
                SingleNodeBuilder::new()
                    .tree(&sources_slice, &targets_slice, Some(n_crit), sparse)
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
    };
}

laplace_fft_constructors!(laplace_fft_f32, f32);
laplace_fft_constructors!(laplace_fft_f64, f64);

macro_rules! laplace_blas_constructors {
    ($name: ident, $type: ident) => {
        /// Constructor for Laplace BLAS FMMs
        #[no_mangle]
        pub extern "C" fn $name(
            expansion_order: usize,
            charges: *const $type,
            sources: *const $type,
            nsources: usize,
            targets: *const $type,
            ntargets: usize,
            n_crit: u64,
            sparse: bool,
            kernel_eval_type: usize,
            svd_threshold: <$type as RlstScalar>::Real,
        ) -> *const KiFmm<$type, Laplace3dKernel<$type>, BlasFieldTranslationSaRcmp<$type>> {
            let dim = 3;

            let kernel_eval_type = if kernel_eval_type == 0 {
                EvalType::Value
            } else if kernel_eval_type == 1 {
                EvalType::ValueDeriv
            } else {
                panic!("Invalid evaluation mode")
            };

            let source_to_target = BlasFieldTranslationSaRcmp::<$type>::new(Some(svd_threshold));
            let kernel = Laplace3dKernel::<$type>::new();

            let sources_slice: &[$type] =
                unsafe { std::slice::from_raw_parts(sources, nsources * dim) };
            let targets_slice: &[$type] =
                unsafe { std::slice::from_raw_parts(targets, ntargets * dim) };
            let charges_slice: &[$type] = unsafe { std::slice::from_raw_parts(charges, nsources) };

            let b = Box::new(
                SingleNodeBuilder::new()
                    .tree(&sources_slice, &targets_slice, Some(n_crit), sparse)
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
    };
}

laplace_blas_constructors!(laplace_blas_f32, f32);
laplace_blas_constructors!(laplace_blas_f64, f64);

macro_rules! helmholtz_fft_constructors {
    ($name: ident, $type: ident) => {
        /// Constructor for Laplace FFT FMMs
        #[no_mangle]
        pub extern "C" fn $name(
            expansion_order: usize,
            charges: *const $type,
            sources: *const <$type as RlstScalar>::Real,
            nsources: usize,
            targets: *const <$type as RlstScalar>::Real,
            ntargets: usize,
            n_crit: u64,
            sparse: bool,
            kernel_eval_type: usize,
            wavenumber: <$type as RlstScalar>::Real,
        ) -> *const KiFmm<$type, Helmholtz3dKernel<$type>, FftFieldTranslation<$type>> {
            let dim = 3;

            let kernel_eval_type = if kernel_eval_type == 0 {
                EvalType::Value
            } else if kernel_eval_type == 1 {
                EvalType::ValueDeriv
            } else {
                panic!("Invalid evaluation mode")
            };

            let source_to_target = FftFieldTranslation::<$type>::new();
            let kernel = Helmholtz3dKernel::<$type>::new(wavenumber);

            let sources_slice: &[<$type as RlstScalar>::Real] =
                unsafe { std::slice::from_raw_parts(sources, nsources * dim) };
            let targets_slice: &[<$type as RlstScalar>::Real] =
                unsafe { std::slice::from_raw_parts(targets, ntargets * dim) };
            let charges_slice: &[$type] = unsafe { std::slice::from_raw_parts(charges, nsources) };

            let b = Box::new(
                SingleNodeBuilder::new()
                    .tree(&sources_slice, &targets_slice, Some(n_crit), sparse)
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
    };
}

helmholtz_fft_constructors!(helmholtz_fft_f32, c32);
helmholtz_fft_constructors!(helmholtz_fft_f64, c64);

macro_rules! helmholtz_blas_constructors {
    ($name: ident, $type: ident) => {
        /// Constructor for Laplace FFT FMMs
        #[no_mangle]
        pub extern "C" fn $name(
            expansion_order: usize,
            charges: *const $type,
            sources: *const <$type as RlstScalar>::Real,
            nsources: usize,
            targets: *const <$type as RlstScalar>::Real,
            ntargets: usize,
            n_crit: u64,
            sparse: bool,
            kernel_eval_type: usize,
            wavenumber: <$type as RlstScalar>::Real,
            svd_threshold: <$type as RlstScalar>::Real,
        ) -> *const KiFmm<$type, Helmholtz3dKernel<$type>, BlasFieldTranslationIa<$type>> {
            let dim = 3;

            let kernel_eval_type = if kernel_eval_type == 0 {
                EvalType::Value
            } else if kernel_eval_type == 1 {
                EvalType::ValueDeriv
            } else {
                panic!("Invalid evaluation mode")
            };

            let source_to_target = BlasFieldTranslationIa::<$type>::new(Some(svd_threshold));
            let kernel = Helmholtz3dKernel::<$type>::new(wavenumber);

            let sources_slice: &[<$type as RlstScalar>::Real] =
                unsafe { std::slice::from_raw_parts(sources, nsources * dim) };
            let targets_slice: &[<$type as RlstScalar>::Real] =
                unsafe { std::slice::from_raw_parts(targets, ntargets * dim) };
            let charges_slice: &[$type] = unsafe { std::slice::from_raw_parts(charges, nsources) };

            let b = Box::new(
                SingleNodeBuilder::new()
                    .tree(&sources_slice, &targets_slice, Some(n_crit), sparse)
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
    };
}

helmholtz_blas_constructors!(helmholtz_blas_f32, c32);
helmholtz_blas_constructors!(helmholtz_blas_f64, c64);
