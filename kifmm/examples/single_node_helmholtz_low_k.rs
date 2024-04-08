use green_kernels::{helmholtz_3d::Helmholtz3dKernel, types::EvalType};
use kifmm::{BlasFieldTranslation, FftFieldTranslation, Fmm, SingleNodeBuilder};

use kifmm::tree::helpers::points_fixture;
use rlst::{rlst_dynamic_array2, RawAccessMut, RlstScalar, c32};

extern crate blas_src;
extern crate lapack_src;

fn main() {
    // Setup random sources and targets
    let nsources = 1000;
    let ntargets = 2000;
    let sources = points_fixture::<f32>(nsources, None, None, Some(0));
    let targets = points_fixture::<f32>(ntargets, None, None, Some(1));

    // FMM parameters
    let n_crit = Some(150);
    let expansion_order = 5;
    let sparse = true;

    let a: c32 = RlstScalar::from_real(1.0);

    println!("A {:?}", a);

    // FFT based M2L for a vector of charges
    {
        let nvecs = 1;
        let tmp = vec![1.0; nsources * nvecs];
        let mut charges = rlst_dynamic_array2!(f32, [nsources, nvecs]);
        charges.data_mut().copy_from_slice(&tmp);

        // let fmm_fft = SingleNodeBuilder::new()
        //     .tree(&sources, &targets, n_crit, sparse)
        //     .unwrap()
        //     .parameters(
        //         &charges,
        //         expansion_order,
        //         Helmholtz3dKernel::new(),
        //         EvalType::Value,
        //         FftFieldTranslation::new(),
        //     )
        //     .unwrap()
        //     .build()
        //     .unwrap();
        // fmm_fft.evaluate();
    }


}
