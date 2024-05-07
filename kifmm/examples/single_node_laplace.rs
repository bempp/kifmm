use green_kernels::{laplace_3d::Laplace3dKernel, types::EvalType};
use kifmm::{BlasFieldTranslationSaRcmp, FftFieldTranslation, Fmm, SingleNodeBuilder};

use kifmm::tree::helpers::points_fixture;
use rlst::{rlst_dynamic_array2, RawAccessMut};

use std::time::Instant;

extern crate blas_src;
extern crate lapack_src;

fn main() {
    // Setup random sources and targets
    let nsources = 1000000;
    let ntargets = 1000000;
    let sources = points_fixture::<f32>(nsources, None, None, Some(0));
    let targets = points_fixture::<f32>(ntargets, None, None, Some(1));

    // FMM parameters
    let n_crit = Some(400);
    let expansion_order = 5;
    let sparse = true;

    // FFT based M2L for a vector of charges
    {
        let nvecs = 1;
        let tmp = vec![1.0; nsources * nvecs];
        let mut charges = rlst_dynamic_array2!(f32, [nsources, nvecs]);
        charges.data_mut().copy_from_slice(&tmp);

        let fmm_fft = SingleNodeBuilder::new()
            .tree(&sources, &targets, n_crit, sparse)
            .unwrap()
            .parameters(
                &charges,
                expansion_order,
                Laplace3dKernel::new(),
                EvalType::Value,
                FftFieldTranslation::new(),
            )
            .unwrap()
            .build()
            .unwrap();
        let s = Instant::now();
        fmm_fft.evaluate().unwrap();
        println!("fft {:?}", s.elapsed());
    }

    // BLAS based M2L
    {
        // Vector of charges
        let nvecs = 1;
        let mut charges = rlst_dynamic_array2!(f32, [nsources, nvecs]);
        charges
            .data_mut()
            .chunks_exact_mut(nsources)
            .enumerate()
            .for_each(|(i, chunk)| chunk.iter_mut().for_each(|elem| *elem += (1 + i) as f32));

        let singular_value_threshold = Some(1e-1);

        let fmm_vec = SingleNodeBuilder::new()
            .tree(&sources, &targets, n_crit, sparse)
            .unwrap()
            .parameters(
                &charges,
                expansion_order,
                Laplace3dKernel::new(),
                EvalType::Value,
                BlasFieldTranslationSaRcmp::new(singular_value_threshold),
            )
            .unwrap()
            .build()
            .unwrap();

        let s = Instant::now();
        fmm_vec.evaluate().unwrap();
        println!("blas {:?}", s.elapsed());
        // Matrix of charges
        let nvecs = 5;
        let mut charges = rlst_dynamic_array2!(f32, [nsources, nvecs]);
        charges
            .data_mut()
            .chunks_exact_mut(nsources)
            .enumerate()
            .for_each(|(i, chunk)| chunk.iter_mut().for_each(|elem| *elem += (1 + i) as f32));

        let fmm_mat = SingleNodeBuilder::new()
            .tree(&sources, &targets, n_crit, sparse)
            .unwrap()
            .parameters(
                &charges,
                expansion_order,
                Laplace3dKernel::new(),
                EvalType::Value,
                BlasFieldTranslationSaRcmp::new(singular_value_threshold),
            )
            .unwrap()
            .build()
            .unwrap();

        let s = Instant::now();
        fmm_mat.evaluate().unwrap();
        println!("blas mat {:?}", s.elapsed());
    }
}
