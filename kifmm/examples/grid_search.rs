use std::{fs::File, ops::DerefMut, sync::Mutex, time::Instant};

use green_kernels::{laplace_3d::Laplace3dKernel, traits::Kernel, types::EvalType};
use itertools::{iproduct, Itertools};
use kifmm::{
    fmm::types::RandomSVDSettings, linalg::rsvd::MatrixRsvd, traits::{
        fftw::Dft,
        general::{AsComplex, Epsilon, Gemv8x8},
        tree::{FmmTree, Tree},
    }, tree::helpers::points_fixture, BlasFieldTranslationSaRcmp, FftFieldTranslation, Fmm, SingleNodeBuilder
};
use num::{zero, Float};
use rand::distributions::uniform::SampleUniform;
use rayon::prelude::*;
use rlst::{
    rlst_dynamic_array2, Array, BaseArray, MatrixSvd, RawAccess, RawAccessMut, RlstScalar, Shape,
    VectorContainer,
};
extern crate blas_src;
extern crate lapack_src;
use csv::Writer;

fn grid_search_laplace_blas<T>(
    filename: String,
    expansion_order_vec: &Vec<usize>,
    svd_threshold_vec: &Vec<Option<T>>,
    surface_diff_vec: &Vec<usize>,
    depth_vec: &Vec<u64>,
    rsvd_settings_vec: &Vec<Option<RandomSVDSettings>>
) where
    T: RlstScalar<Real = T> + Epsilon + Float + Default + SampleUniform + MatrixRsvd,
    Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>: MatrixSvd<Item = T>,
{
    // FMM parameters
    let prune_empty = true;

    let parameters = iproduct!(
        surface_diff_vec.iter(),
        svd_threshold_vec.iter(),
        depth_vec.iter(),
        expansion_order_vec.iter(),
        rsvd_settings_vec.iter()
    )
    .map(|(surface_diff, svd_threshold, depth, expansion_order, rsvd_settings)| {
        (*surface_diff, *svd_threshold, *depth, *expansion_order, rsvd_settings)
    })
    .collect_vec();

    let parameters_cloned = parameters.iter().cloned().collect_vec();

    let fmms = Mutex::new(Vec::new());
    let progress = Mutex::new(0usize);

    let n_params = parameters.len();

    // Setup random sources and targets
    let nsources = 1000000;
    let ntargets = 1000000;
    let sources = points_fixture::<T::Real>(nsources, None, None, Some(0));
    let targets = points_fixture::<T::Real>(ntargets, None, None, Some(1));
    let nvecs = 1;
    let tmp = vec![T::one(); nsources * nvecs];
    let mut charges = rlst_dynamic_array2!(T, [nsources, nvecs]);
    charges.data_mut().copy_from_slice(&tmp);

    let s = Instant::now();
    parameters.into_iter().enumerate().for_each(
        |(i, (surface_diff, svd_threshold, depth, expansion_order, rsvd_settings))| {
            let expansion_order = vec![expansion_order; (depth + 1) as usize];

            let s = Instant::now();
            let fmm = SingleNodeBuilder::new()
                .tree(
                    sources.data(),
                    targets.data(),
                    None,
                    Some(depth),
                    prune_empty,
                )
                .unwrap()
                .parameters(
                    charges.data(),
                    &expansion_order,
                    Laplace3dKernel::new(),
                    EvalType::Value,
                    BlasFieldTranslationSaRcmp::new(svd_threshold, Some(surface_diff), *rsvd_settings),
                )
                .unwrap()
                .build()
                .unwrap();
            let setup_time = s.elapsed();
            fmms.lock().unwrap().push((i, fmm, setup_time));
            *progress.lock().unwrap().deref_mut() += 1;

            println!(
                "BLAS Pre-computed {:?}/{:?}",
                *progress.lock().unwrap(),
                n_params
            );
        },
    );

    let file = File::create(format!("{filename}.csv")).unwrap();
    let mut writer = Writer::from_writer(file);
    writer
        .write_record(&[
            "depth".to_string(),
            "surface_diff".to_string(),
            "svd_threshold".to_string(),
            "expansion_order".to_string(),
            "runtime".to_string(),
            "min_rel_err".to_string(),
            "mean_rel_err".to_string(),
            "max_rel_err".to_string(),
            "n_iter".to_string(),
            "n_oversamples".to_string(),
            "setup_time".to_string()
        ])
        .unwrap();

    println!(
        "BLAS Pre-computation Time Elapsed {:?}",
        s.elapsed().as_secs()
    );

    // Setup random sources and targets
    let nsources = 1000000;
    let sources = points_fixture::<T::Real>(nsources, None, None, Some(0));
    let nvecs = 1;
    let tmp = vec![T::one(); nsources * nvecs];
    let mut charges = rlst_dynamic_array2!(T, [nsources, nvecs]);
    charges.data_mut().copy_from_slice(&tmp);

    let mut progress = 0;
    for (i, fmm, setup_time) in fmms.lock().unwrap().iter() {
        let (surface_diff, svd_threshold, depth, expansion_order, rsvd_settings) = parameters_cloned[*i];

        let svd_threshold = svd_threshold.unwrap_or(T::zero().re());

        let s = Instant::now();
        fmm.evaluate(false).unwrap();
        let time = s.elapsed().as_millis() as f32;
        progress += 1;
        println!("BLAS Evaluated {:?}/{:?}", progress, n_params);

        let leaf_idx = 1;
        let leaf = fmm.tree().target_tree().all_leaves().unwrap()[leaf_idx];
        let potential = fmm.potential(&leaf).unwrap()[0];
        let leaf_targets = fmm.tree().target_tree().coordinates(&leaf).unwrap();
        let ntargets = leaf_targets.len() / fmm.dim();
        let mut direct = vec![T::zero(); ntargets];
        fmm.kernel().evaluate_st(
            EvalType::Value,
            sources.data(),
            leaf_targets,
            charges.data(),
            &mut direct,
        );

        let rel_error = direct
            .iter()
            .zip(potential)
            .map(|(&d, &p)| {
                let abs_error = RlstScalar::abs(d - p);
                abs_error / p
            })
            .collect_vec();

        let min_rel_err = rel_error
            .iter()
            .filter(|&&x| !x.is_nan())
            .cloned()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let max_rel_err = rel_error
            .iter()
            .filter(|&&x| !x.is_nan())
            .cloned()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        let mean_rel_err: T = rel_error.into_iter().sum::<T>() / T::from(direct.len()).unwrap();

        let n_iter;
        let n_oversamples;
        if let Some(rsvd_settings) = rsvd_settings {
            n_iter = rsvd_settings.n_iter;
            n_oversamples = rsvd_settings.n_oversamples.unwrap()
        } else {
            n_iter = 0;
            n_oversamples = 0;
        }

        writer
            .write_record(&[
                depth.to_string(),
                surface_diff.to_string(),
                svd_threshold.to_string(),
                expansion_order.to_string(),
                time.to_string(),
                min_rel_err.to_string(),
                mean_rel_err.to_string(),
                max_rel_err.to_string(),
                n_iter.to_string(),
                n_oversamples.to_string(),
                (setup_time.as_millis() as f32 ).to_string()
            ])
            .unwrap();
    }
}

fn grid_search_laplace_fft<T>(
    filename: String,
    expansion_order_vec: &Vec<usize>,
    depth_vec: &Vec<u64>,
) where
    Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>: MatrixSvd<Item = T>,
    T: RlstScalar<Real = T>
        + AsComplex
        + Default
        + Dft<InputType = T, OutputType = <T as AsComplex>::ComplexType>
        + SampleUniform
        + Float
        + Epsilon,
    <T as AsComplex>::ComplexType: Gemv8x8<Scalar = <T as AsComplex>::ComplexType>,
{
    // FMM parameters
    let prune_empty = true;

    let parameters = iproduct!(depth_vec.iter(), expansion_order_vec.iter())
        .map(|(depth, expansion_order)| (*depth, *expansion_order))
        .collect_vec();

    let parameters_cloned = parameters.iter().cloned().collect_vec();

    let fmms = Mutex::new(Vec::new());
    let progress = Mutex::new(0usize);

    let n_params = parameters.len();

    let s = Instant::now();
    parameters
        .into_iter()
        .enumerate()
        .for_each(|(i, (depth, expansion_order))| {
            let expansion_order = vec![expansion_order; (depth + 1) as usize];
            // Setup random sources and targets
            let nsources = 1000000;
            let ntargets = 1000000;
            let sources = points_fixture::<T::Real>(nsources, None, None, Some(0));
            let targets = points_fixture::<T::Real>(ntargets, None, None, Some(1));
            let nvecs = 1;
            let tmp = vec![T::one(); nsources * nvecs];
            let mut charges = rlst_dynamic_array2!(T, [nsources, nvecs]);
            charges.data_mut().copy_from_slice(&tmp);

            let s = Instant::now();
            let fmm = SingleNodeBuilder::new()
                .tree(
                    sources.data(),
                    targets.data(),
                    None,
                    Some(depth),
                    prune_empty,
                )
                .unwrap()
                .parameters(
                    charges.data(),
                    &expansion_order,
                    Laplace3dKernel::new(),
                    EvalType::Value,
                    FftFieldTranslation::new(None),
                )
                .unwrap()
                .build()
                .unwrap();
            let setup_time = s.elapsed();

            fmms.lock().unwrap().push((i, fmm, setup_time));
            *progress.lock().unwrap().deref_mut() += 1;

            println!(
                "FFT Pre-computed {:?}/{:?}",
                *progress.lock().unwrap(),
                n_params
            );
        });

    println!(
        "FFT Pre-computation Time Elapsed {:?}",
        s.elapsed().as_secs()
    );
    let file = File::create(format!("{filename}.csv")).unwrap();
    let mut writer = Writer::from_writer(file);
    writer
        .write_record(&[
            "depth".to_string(),
            "expansion_order".to_string(),
            "time".to_string(),
            "min_rel_err".to_string(),
            "mean_rel_err".to_string(),
            "max_rel_err".to_string(),
            "setup_time".to_string(),
        ])
        .unwrap();

    let nsources = 1000000;
    let sources = points_fixture::<T::Real>(nsources, None, None, Some(0));
    let nvecs = 1;
    let tmp = vec![T::one(); nsources * nvecs];
    let mut charges = rlst_dynamic_array2!(T, [nsources, nvecs]);
    charges.data_mut().copy_from_slice(&tmp);

    let mut progress = 0;
    for (i, fmm, setup_time) in fmms.lock().unwrap().iter() {
        let (depth, expansion_order) = parameters_cloned[*i];

        let s = Instant::now();
        fmm.evaluate(false).unwrap();
        let time = s.elapsed().as_millis() as f32;
        progress += 1;
        println!("FFT Evaluated {:?}/{:?}", progress, n_params);

        let leaf_idx = 1;
        let leaf = fmm.tree().target_tree().all_leaves().unwrap()[leaf_idx];
        let potential = fmm.potential(&leaf).unwrap()[0];
        let leaf_targets = fmm.tree().target_tree().coordinates(&leaf).unwrap();
        let ntargets = leaf_targets.len() / fmm.dim();
        let mut direct = vec![T::zero(); ntargets];
        fmm.kernel().evaluate_st(
            EvalType::Value,
            sources.data(),
            leaf_targets,
            charges.data(),
            &mut direct,
        );

        let rel_error = direct
            .iter()
            .zip(potential)
            .map(|(&d, &p)| {
                let abs_error = RlstScalar::abs(d - p);
                abs_error / p
            })
            .collect_vec();

        let min_rel_err = rel_error
            .iter()
            .filter(|&&x| !x.is_nan())
            .cloned()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let max_rel_err = rel_error
            .iter()
            .filter(|&&x| !x.is_nan())
            .cloned()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        let mean_rel_err: T = rel_error.into_iter().sum::<T>() / T::from(direct.len()).unwrap();

        writer
            .write_record(&[
                depth.to_string(),
                expansion_order.to_string(),
                time.to_string(),
                min_rel_err.to_string(),
                mean_rel_err.to_string(),
                max_rel_err.to_string(),
                (setup_time.as_millis() as f32).to_string(),
            ])
            .unwrap();
    }
}

fn main() {
    let expansion_order_vec: Vec<usize> = vec![3, 4, 5, 6];
    let svd_threshold_vec = vec![
        None,
        Some(1e-7),
        Some(1e-6),
        Some(1e-5),
        Some(1e-4),
        Some(1e-3),
        Some(1e-2),
        Some(1e-1),
        Some(2e-2),
    ];
    let surface_diff_vec: Vec<usize> = vec![0, 1, 2];
    let depth_vec: Vec<u64> = vec![4, 5];

    // grid_search_laplace_fft::<f32>(
    //     "grid_search_laplace_fft_f32_m1".to_string(),
    //     &expansion_order_vec,
    //     &depth_vec,
    // );

    let rsvd_settings_vec = vec![
        None,
        // Some(RandomSVDSettings::new(1, None, Some(5), None)),
        // Some(RandomSVDSettings::new(2, None, Some(5), None)),
        // Some(RandomSVDSettings::new(4, None, Some(5), None)),
        // Some(RandomSVDSettings::new(1, None, Some(10), None)),
        // Some(RandomSVDSettings::new(2, None, Some(10), None)),
        // Some(RandomSVDSettings::new(4, None, Some(10), None)),
        // Some(RandomSVDSettings::new(1, None, Some(20), None)),
        // Some(RandomSVDSettings::new(2, None, Some(20), None)),
        // Some(RandomSVDSettings::new(4, None, Some(20), None)),
    ];

    grid_search_laplace_blas::<f32>(
        "foo".to_string(),
        &expansion_order_vec,
        &svd_threshold_vec,
        &surface_diff_vec,
        &depth_vec,
        &rsvd_settings_vec
    );

    let expansion_order_vec: Vec<usize> = vec![6, 7, 8, 9, 10];
    let svd_threshold_vec = vec![
        None,
        Some(1e-15),
        Some(1e-12),
        Some(1e-9),
        Some(1e-6),
        Some(1e-3),
        Some(1e-1),
    ];

    let surface_diff_vec: Vec<usize> = vec![0, 1, 2];
    let depth_vec: Vec<u64> = vec![4, 5];

    // grid_search_laplace_fft::<f64>(
    //     "grid_search_laplace_fft_f64_m1".to_string(),
    //     &expansion_order_vec,
    //     &depth_vec,
    // );

    // grid_search_laplace_blas::<f64>(
    //     "grid_search_laplace_blas_f64_m1_8".to_string(),
    //     &expansion_order_vec,
    //     &svd_threshold_vec,
    //     &surface_diff_vec,
    //     &depth_vec,
    // );
}