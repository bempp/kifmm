use std::time::Instant;

use green_kernels::{laplace_3d::Laplace3dKernel, types::EvalType};
use kifmm::fmm::helpers::ncoeffs_kifmm;
use kifmm::fmm::types::{BlasFieldTranslationSaRcmpLaplaceMetal, SingleNodeBuilderLaplaceMetal};
use kifmm::traits::tree::Tree;
use kifmm::{BlasFieldTranslationSaRcmp, FftFieldTranslation, Fmm, SingleNodeBuilder};

use kifmm::tree::helpers::points_fixture;
use rlst::{rlst_dynamic_array2, RawAccessMut, Shape};

extern crate blas_src;
extern crate lapack_src;

fn main() {
    // Setup random sources and targets
    let nsources = 1000000;
    let ntargets = 1000000;
    let sources = points_fixture::<f32>(nsources, None, None, Some(0));
    let targets = points_fixture::<f32>(ntargets, None, None, Some(1));

    // FMM parameters
    // let n_crit = Some(150);
    let expansion_order = 5;
    let sparse = false;

    let singular_value_threshold = Some(1e-7);
    // let nvecs = 10;
    // let metal_level = 6;

    // Vector of charges

    // Runtimes
    // {
    //     println!("n_leaves,svd_threshold, nvecs, depth, metal_level, using_metal, runtime_per_vec, m2l_time_per_vec,flops");
    //     for n_crit in [15] {
    //         for nvecs in [1, 5, 10] {
    //             for metal_level in [5, 6, 7] {
    //                 let mut charges = rlst_dynamic_array2!(f32, [nsources, nvecs]);
    //                 charges
    //                     .data_mut()
    //                     .chunks_exact_mut(nsources)
    //                     .enumerate()
    //                     .for_each(|(i, chunk)| chunk.iter_mut().for_each(|elem| *elem += (1 + i) as f32));

    //                 let fmm = SingleNodeBuilderLaplaceMetal::new()
    //                 .tree(&sources, &targets, Some(n_crit), sparse)
    //                 .unwrap()
    //                 .parameters(
    //                     &charges,
    //                     expansion_order,
    //                     Laplace3dKernel::new(),
    //                     EvalType::Value,
    //                     metal_level,
    //                     BlasFieldTranslationSaRcmpLaplaceMetal::new(singular_value_threshold),
    //                 )
    //                 .unwrap()
    //                 .build()
    //                 .unwrap();

    //                 let s = Instant::now();
    //                 let times = fmm.evaluate(true).unwrap();
    //                 let runtime = s.elapsed().as_secs_f64() / (nvecs as f64);
    //                 let m2l_time = times.get("m2l").unwrap().as_secs_f64() /( nvecs as f64);

    //                 let mut flops = 0;
    //                 let svd_threshold = singular_value_threshold.unwrap();
    //                 let depth = fmm.tree.source_tree.depth;
    //                 let using_metal = metal_level <= depth;
    //                 let n_leaves = fmm.tree.source_tree.n_leaves().unwrap() * nvecs;

    //                 for mat in  fmm.source_to_target.metadata[0].c_metal.iter() {
    //                     flops += mat.shape().iter().product::<usize>() * n_leaves * ncoeffs_kifmm(fmm.expansion_order)
    //                 };
    //                 println!("{:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?}", n_leaves, svd_threshold, nvecs, depth, metal_level, using_metal, runtime, m2l_time, flops);
    //             }
    //         }
    //     }

        {
            println!("n_leaves,svd_threshold, nvecs, depth, metal_level, using_metal, flops, data_organisation_time, matmul_time, allocation_time, saving_time");
            for n_crit in [15] {
                for nvecs in [1, 1, 5, 10] {
                    for metal_level in [5, 6, 7] {
                        let mut charges = rlst_dynamic_array2!(f32, [nsources, nvecs]);
                        charges
                            .data_mut()
                            .chunks_exact_mut(nsources)
                            .enumerate()
                            .for_each(|(i, chunk)| chunk.iter_mut().for_each(|elem| *elem += (1 + i) as f32));

                        let fmm = SingleNodeBuilderLaplaceMetal::new()
                        .tree(&sources, &targets, Some(n_crit), sparse)
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

                        // let s = Instant::now();
                        let (times, flops) = fmm.evaluate(true).unwrap();
                        // let runtime = s.elapsed().as_secs_f64() / (nvecs as f64);
                        // // let m2l_time = times.get("m2l").unwrap().as_secs_f64() /( nvecs as f64);
                        let data_organisation_time = times.get("data_organisation").unwrap().as_secs_f64();
                        let allocation_time = times.get("allocation").unwrap().as_secs_f64();
                        let saving_time = times.get("saving").unwrap().as_secs_f64();
                        let matmul_time = times.get("matmul").unwrap().as_secs_f64();

                        let svd_threshold = singular_value_threshold.unwrap();
                        let depth = fmm.tree.source_tree.depth;
                        let using_metal = metal_level <= depth;
                        let n_leaves = fmm.tree.source_tree.n_leaves().unwrap() * nvecs;

                        // for mat in  fmm.source_to_target.metadata[0].c_metal.iter() {
                        //     flops += mat.shape().iter().product::<usize>() * n_leaves * ncoeffs_kifmm(fmm.expansion_order)
                        // };
                        println!("{:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?}", n_leaves, svd_threshold, nvecs, depth, metal_level, using_metal, flops as usize, data_organisation_time, matmul_time, allocation_time, saving_time);
                    }
                }
            }
        }

        // println!(""metal" {:?}", s.elapsed());

        // for level in 0..=fmm.tree().source_tree().depth()
        // println!("op times {:?}", )

}
