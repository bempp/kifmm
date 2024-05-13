use std::time::Instant;

use green_kernels::{laplace_3d::Laplace3dKernel, types::EvalType};
use kifmm::fmm::types::{BlasFieldTranslationSaRcmpLaplaceMetal, SingleNodeBuilderLaplaceMetal};
use kifmm::{BlasFieldTranslationSaRcmp, FftFieldTranslation, Fmm, SingleNodeBuilder};

use kifmm::tree::helpers::points_fixture;
use rlst::{rlst_dynamic_array2, RawAccessMut};

extern crate blas_src;
extern crate lapack_src;

fn main() {
    // Setup random sources and targets
    let nsources = 1000000;
    let ntargets = 1000000;
    let sources = points_fixture::<f32>(nsources, None, None, Some(0));
    let targets = points_fixture::<f32>(ntargets, None, None, Some(1));

    // FMM parameters
    let n_crit = Some(150);
    let expansion_order = 5;
    let sparse = true;

    // BLAS based M2L
    {
        // Vector of charges
        let nvecs = 10;
        let mut charges = rlst_dynamic_array2!(f32, [nsources, nvecs]);
        charges
            .data_mut()
            .chunks_exact_mut(nsources)
            .enumerate()
            .for_each(|(i, chunk)| chunk.iter_mut().for_each(|elem| *elem += (1 + i) as f32));

        let singular_value_threshold = Some(1e-5);
        let metal_level = 5;

        // let fmm = SingleNodeBuilder::new()
        //     .tree(&sources, &targets, n_crit, sparse)
        //     .unwrap()
        //     .parameters(
        //         &charges,
        //         expansion_order,
        //         Laplace3dKernel::new(),
        //         EvalType::Value,
        //         BlasFieldTranslationSaRcmp::new(singular_value_threshold),
        //     )
        //     .unwrap()
        //     .build()
        //     .unwrap();

        let s = Instant::now();
        let fmm = SingleNodeBuilderLaplaceMetal::new()
            .tree(&sources, &targets, n_crit, sparse)
            .unwrap()
            .parameters(
                &charges,
                expansion_order,
                Laplace3dKernel::new(),
                EvalType::Value,
                metal_level,
                BlasFieldTranslationSaRcmpLaplaceMetal::new(singular_value_threshold),
            )
            .unwrap()
            .build()
            .unwrap();

        fmm.evaluate(false).unwrap();
        println!("metal {:?}", s.elapsed());
        // println!("op times {:?}", )

    }
}
