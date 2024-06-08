use std::{fs::File, ops::DerefMut, sync::Mutex, time::Instant};

use green_kernels::{laplace_3d::Laplace3dKernel, traits::Kernel, types::EvalType};
use itertools::{iproduct, Itertools};
use kifmm::{traits::{general::Epsilon, tree::{FmmTree, Tree}}, tree::helpers::points_fixture, BlasFieldTranslationSaRcmp, Fmm, SingleNodeBuilder};
use num::Float;
use rand::distributions::uniform::SampleUniform;
use rayon::prelude::*;
use rlst::{rlst_dynamic_array2, RawAccess, RawAccessMut, RlstScalar, Shape};
extern crate lapack_src;
extern crate blas_src;
use csv::Writer;



fn grid_search_blas()
{
    // Setup random sources and targets
    let nsources = 1000;
    let ntargets = 2000;
    let sources = points_fixture::<f32>(nsources, None, None, Some(0));
    let targets = points_fixture::<f32>(ntargets, None, None, Some(1));
    let nvecs = 1;
    let tmp = vec![0f32; nsources * nvecs];
    let mut charges = rlst_dynamic_array2!(f32, [nsources, nvecs]);
    charges.data_mut().copy_from_slice(&tmp);

    // FMM parameters
    let prune_empty = true;

    let surface_diff_vec = vec![0, 1, 2];
    let svd_threshold_vec = vec![None, Some(1e-7), Some(1e-6), Some(1e-5), Some(1e-4), Some(1e-3), Some(1e-2), Some(1e-1), Some(2e-2)];
    let depth_vec: Vec<u64> = vec![4, 5];
    let expansion_order_vec: Vec<usize> = vec![3, 4, 5, 6];

    let parameters  = iproduct!(
        surface_diff_vec.iter(),
        svd_threshold_vec.iter(),
        depth_vec.iter(),
        expansion_order_vec.iter()
    )
    .map(|(surface_diff, svd_threshold, depth, expansion_order)| {
        (*surface_diff, *svd_threshold, *depth, *expansion_order)
    })
    .collect_vec();

    let parameters_cloned = parameters.iter().cloned().collect_vec();

    let fmms = Mutex::new(Vec::new());
    let progress = Mutex::new(0usize);

    let n_params = parameters.len();

    let s = Instant::now();
    parameters.into_par_iter().enumerate().for_each(|(i, (surface_diff, svd_threshold, depth, expansion_order))| {

        let expansion_order = vec![expansion_order; (depth+1) as usize];
        let fmm = SingleNodeBuilder::new()
            .tree(sources.data(), targets.data(), None, Some(depth), prune_empty)
            .unwrap()
            .parameters(
                charges.data(),
                &expansion_order,
                Laplace3dKernel::new(),
                EvalType::Value,
                BlasFieldTranslationSaRcmp::new(svd_threshold, Some(surface_diff)),
            )
            .unwrap()
            .build()
            .unwrap();

        fmms.lock().unwrap().push((i, fmm));
        *progress.lock().unwrap().deref_mut() += 1;

        println!("i {:?}/{:?}", *progress.lock().unwrap(), n_params);
    });

    println!("Time Elapsed {:?}", s.elapsed().as_secs());

    for (i, fmm) in fmms.lock().unwrap().iter() {

        let (surface_diff, svd_threshold, depth, expansion_order) = parameters_cloned[*i];

        let svd_threshold = svd_threshold.unwrap_or(0.);

        let s = Instant::now();
        fmm.evaluate(false).unwrap();
        let time = s.elapsed().as_millis() as f32;

        let file = File::create("accuracy_single_precision_fft.csv").unwrap();
        let mut writer = Writer::from_writer(file);


        let leaf_idx = 0;
        let leaf = fmm.tree().target_tree().all_leaves().unwrap()[leaf_idx];
        let potential = fmm.potential(&leaf).unwrap()[0];
        let leaf_targets = fmm.tree().target_tree().coordinates(&leaf).unwrap();
        let ntargets = leaf_targets.len() / fmm.dim();
        let mut direct = vec![0f32; ntargets];
        fmm.kernel().evaluate_st(
            EvalType::Value,
            sources.data(),
            leaf_targets,
            charges.data(),
            &mut direct,
        );

        let rel_error = direct.iter().zip(potential).map(|(&d, &p)| {
            let abs_error = RlstScalar::abs(d - p);
            abs_error / p
        }).collect_vec();

        let mean_rel_err = rel_error.iter().sum::<f32>() / rel_error.len() as f32;
        let min_rel_err = rel_error.iter()
            .filter(|&&x| !x.is_nan())
            .cloned()
            .min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let max_rel_err = rel_error.iter()
            .filter(|&&x| !x.is_nan())
            .cloned()
            .max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();

        writer.write_record(&[
            depth.to_string(),
            surface_diff.to_string(),
            svd_threshold.to_string(),
            expansion_order.to_string(),
            time.to_string(),
            min_rel_err.to_string(),
            mean_rel_err.to_string(),
            max_rel_err.to_string(),
        ]).unwrap();
    }
}

fn main() {

    grid_search_blas();

}