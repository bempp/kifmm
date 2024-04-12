use green_kernels::{
    helmholtz_3d::Helmholtz3dKernel, laplace_3d::Laplace3dKernel, types::EvalType,
};
use kifmm::{
    new_fmm::types::{BlasFieldTranslation, FftFieldTranslation, SingleNodeBuilder},
    traits::fmm::SourceTranslation,
    tree::helpers::points_fixture,
};
use num::traits::One;
use rlst::{c64, rlst_dynamic_array2, RawAccessMut};

extern crate blas_src;
extern crate lapack_src;

fn main() {
    let nsources = 1000;
    let ntargets = 2000;
    let sources = points_fixture::<f64>(nsources, None, None, Some(0));
    let targets = points_fixture::<f64>(ntargets, None, None, Some(1));

    // FMM parameters
    let n_crit = Some(150);
    let expansion_order = 2;
    let sparse = true;

    // FFT based M2L for a vector of charges
    let nvecs = 1;

    let tmp = vec![1.0; nsources * nvecs];
    let mut charges = rlst_dynamic_array2!(f64, [nsources, nvecs]);
    charges.data_mut().copy_from_slice(&tmp);
    let kernel = Laplace3dKernel::<f64>::new();

    // let tmp = vec![c64::one(); nsources * nvecs];
    // let mut charges = rlst_dynamic_array2!(c64, [nsources, nvecs]);
    // charges.data_mut().copy_from_slice(&tmp);
    // let kernel = Helmholtz3dKernel::<c64>::new(1.0);

    let fmm = SingleNodeBuilder::new()
        .tree(&sources, &targets, n_crit, sparse)
        .unwrap()
        .parameters(
            &charges,
            expansion_order,
            kernel,
            EvalType::Value,
            FftFieldTranslation::new(),
        )
        .unwrap()
        .build()
        .unwrap();

    fmm.p2m();

    // println!("HERE {:?}", fmm.multipoles);
}
