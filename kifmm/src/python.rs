extern crate blas_src;
extern crate lapack_src;


use green_kernels::{laplace_3d::Laplace3dKernel, types::EvalType};
use rlst::{rlst_array_from_slice2, rlst_dynamic_array2, RawAccess, RawAccessMut};

use crate::{
    fmm::{types::Charges, KiFmm},
    FftFieldTranslation, SingleNodeBuilder,
};

/// Foo
#[no_mangle]
pub extern "C" fn laplace_builder(
    expansion_order: usize,
    charges: *const f32,
    sources: *const f32,
    nsources: usize,
    targets: *const f32,
    ntargets: usize,
    n_crit: u64,
    sparse: bool,
) -> *const KiFmm<f32, Laplace3dKernel<f32>, FftFieldTranslation<f32>> {
    let eval_type = EvalType::Value;
    let source_to_target = FftFieldTranslation::new();
    let kernel = Laplace3dKernel::new();

    let sources_slice: &[f32] = unsafe { std::slice::from_raw_parts(sources, nsources * 3) };

    let mut sources_arr = rlst_dynamic_array2!(f32, [nsources, 3]);
    sources_arr.data_mut().copy_from_slice(sources_slice);

    let targets_slice: &[f32] = unsafe { std::slice::from_raw_parts(targets, ntargets * 3) };
    let mut targets_arr = rlst_dynamic_array2!(f32, [ntargets, 3]);
    targets_arr.data_mut().copy_from_slice(targets_slice);

    let mut charges_arr = rlst_dynamic_array2!(f32, [nsources, 1]);
    let charges_slice: &[f32] = unsafe { std::slice::from_raw_parts(charges, nsources) };
    charges_arr.data_mut().copy_from_slice(charges_slice);

    let b = Box::new(
        SingleNodeBuilder::new()
            .parameters(
                &charges_arr,
                expansion_order,
                kernel,
                eval_type,
                source_to_target,
            )
            .unwrap()
            .tree(&sources_arr, &targets_arr, Some(n_crit), sparse)
            .unwrap()
            .build()
            .unwrap(),
    );

    Box::into_raw(b)
}
