extern crate blas_src;
extern crate lapack_src;

use green_kernels::{laplace_3d::Laplace3dKernel, types::EvalType};
use rlst::{rlst_array_from_slice2, rlst_dynamic_array2, RawAccess, RawAccessMut};

use crate::{
    fmm::{types::Charges, KiFmm},
    FftFieldTranslation, SingleNodeBuilder,
};

macro_rules! laplace_fft_constructors {
    ($name: ident, $type: ident) => {
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

            let mut charges_arr = rlst_dynamic_array2!($type, [nsources, 1]);
            charges_arr.data_mut().copy_from_slice(charges_slice);

            let b = Box::new(
                SingleNodeBuilder::new()
                    .tree(&sources_slice, &targets_slice, Some(n_crit), sparse)
                    .unwrap()
                    .parameters(
                        &charges_arr,
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

            let source_to_target = FftFieldTranslation::<$type>::new(Some(svd_threshold));
            let kernel = Laplace3dKernel::<$type>::new();

            let sources_slice: &[$type] =
                unsafe { std::slice::from_raw_parts(sources, nsources * dim) };
            let targets_slice: &[$type] =
                unsafe { std::slice::from_raw_parts(targets, ntargets * dim) };
            let charges_slice: &[$type] = unsafe { std::slice::from_raw_parts(charges, nsources) };

            let mut charges_arr = rlst_dynamic_array2!($type, [nsources, 1]);
            charges_arr.data_mut().copy_from_slice(charges_slice);

            let b = Box::new(
                SingleNodeBuilder::new()
                    .tree(&sources_slice, &targets_slice, Some(n_crit), sparse)
                    .unwrap()
                    .parameters(
                        &charges_arr,
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
