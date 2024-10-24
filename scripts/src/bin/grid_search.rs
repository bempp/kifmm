//! Optimal parameter search
use std::{fs::File, ops::DerefMut, sync::Mutex, time::Instant};

use csv::Writer;
use green_kernels::{laplace_3d::Laplace3dKernel, traits::Kernel, types::GreenKernelEvalType};
use itertools::{iproduct, Itertools};
use kifmm::{
    fftw::array::AlignedAllocable,
    fmm::types::FmmSvdMode,
    linalg::rsvd::{MatrixRsvd, Normaliser},
    traits::{
        fftw::Dft,
        fmm::{DataAccess, Evaluate},
        general::single_node::{AsComplex, Epsilon, Hadamard8x8},
        tree::{SingleFmmTree, SingleTree},
    },
    tree::helpers::points_fixture,
    BlasFieldTranslationSaRcmp, FftFieldTranslation, SingleNodeBuilder,
};
use num::Float;
use rand::distributions::uniform::SampleUniform;
use rlst::{rlst_dynamic_array2, MatrixSvd, RawAccess, RawAccessMut, RlstScalar};

fn grid_search_laplace_blas<T>(
    filename: String,
    n_points: usize,
    expansion_order_vec: &[usize],
    svd_threshold_vec: &[Option<T>],
    surface_diff_vec: &[usize],
    depth_vec: &[u64],
    rsvd_settings_vec: &[FmmSvdMode],
) where
    T: RlstScalar<Real = T> + Epsilon + Float + Default + SampleUniform + MatrixRsvd,
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
    .map(
        |(surface_diff, svd_threshold, depth, expansion_order, rsvd_settings)| {
            (
                *surface_diff,
                *svd_threshold,
                *depth,
                *expansion_order,
                rsvd_settings,
            )
        },
    )
    .collect_vec();

    let parameters_cloned = parameters.iter().cloned().collect_vec();

    let fmms = Mutex::new(Vec::new());
    let progress = Mutex::new(0usize);

    let n_params = parameters.len();

    // Setup random sources and targets
    let n_sources = n_points;
    let n_targets = n_points;
    let sources = points_fixture::<T::Real>(n_sources, None, None, Some(0));
    let targets = points_fixture::<T::Real>(n_targets, None, None, Some(1));
    let n_vecs = 1;
    let tmp = vec![T::one(); n_sources * n_vecs];
    let mut charges = rlst_dynamic_array2!(T, [n_sources, n_vecs]);
    charges.data_mut().copy_from_slice(&tmp);

    let s = Instant::now();
    parameters.into_iter().enumerate().for_each(
        |(i, (surface_diff, svd_threshold, depth, expansion_order, rsvd_settings))| {
            let expansion_order = vec![expansion_order; (depth + 1) as usize];

            let s = Instant::now();
            let fmm = SingleNodeBuilder::new(true)
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
                    GreenKernelEvalType::Value,
                    BlasFieldTranslationSaRcmp::new(
                        svd_threshold,
                        Some(surface_diff),
                        *rsvd_settings,
                    ),
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
            "setup_time".to_string(),
        ])
        .unwrap();

    println!(
        "BLAS Pre-computation Time Elapsed {:?}",
        s.elapsed().as_secs()
    );

    // Setup random sources and targets
    let n_sources = 1000000;
    let sources = points_fixture::<T::Real>(n_sources, None, None, Some(0));
    let n_vecs = 1;
    let tmp = vec![T::one(); n_sources * n_vecs];
    let mut charges = rlst_dynamic_array2!(T, [n_sources, n_vecs]);
    charges.data_mut().copy_from_slice(&tmp);

    let mut progress = 0;
    for (i, fmm, setup_time) in fmms.lock().unwrap().iter_mut() {
        let (surface_diff, svd_threshold, depth, expansion_order, rsvd_settings) =
            parameters_cloned[*i];

        let svd_threshold = svd_threshold.unwrap_or(T::zero().re());

        let s = Instant::now();
        fmm.evaluate().unwrap();
        let time = s.elapsed().as_millis() as f32;
        progress += 1;
        println!("BLAS Evaluated {:?}/{:?}", progress, n_params);

        let leaf_idx = 1;
        let leaf = fmm.tree().target_tree().all_leaves().unwrap()[leaf_idx];
        let potential = fmm.potential(&leaf).unwrap()[0];
        let leaf_targets = fmm.tree().target_tree().coordinates(&leaf).unwrap();
        let n_targets = leaf_targets.len() / fmm.dim();
        let mut direct = vec![T::zero(); n_targets];
        fmm.kernel().evaluate_st(
            GreenKernelEvalType::Value,
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

        let n_iter_;
        let n_oversamples_;
        match rsvd_settings {
            FmmSvdMode::Random {
                n_components: _,
                normaliser,
                n_oversamples,
                random_state: _,
            } => {
                if let Some(Normaliser::Qr(n)) = normaliser {
                    n_iter_ = *n
                } else {
                    n_iter_ = 0;
                }
                n_oversamples_ = n_oversamples.unwrap();
            }

            FmmSvdMode::Deterministic => {
                n_iter_ = 0usize;
                n_oversamples_ = 0usize;
            }
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
                n_iter_.to_string(),
                n_oversamples_.to_string(),
                (setup_time.as_millis() as f32).to_string(),
            ])
            .unwrap();
    }
}

fn grid_search_laplace_fft<T>(
    filename: String,
    n_points: usize,
    expansion_order_vec: &[usize],
    depth_vec: &[u64],
    block_size_vec: &[usize],
) where
    T: RlstScalar<Real = T>
        + AsComplex
        + Default
        + Dft<InputType = T, OutputType = <T as AsComplex>::ComplexType>
        + SampleUniform
        + Float
        + Epsilon
        + AlignedAllocable
        + MatrixSvd,
    <T as AsComplex>::ComplexType:
        Hadamard8x8<Scalar = <T as AsComplex>::ComplexType> + AlignedAllocable,
    <T as Dft>::Plan: Sync,
{
    // FMM parameters
    let prune_empty = true;

    let parameters = iproduct!(
        depth_vec.iter(),
        expansion_order_vec.iter(),
        block_size_vec.iter()
    )
    .map(|(depth, expansion_order, block_size)| (*depth, *expansion_order, *block_size))
    .collect_vec();

    let parameters_cloned = parameters.iter().cloned().collect_vec();

    let fmms = Mutex::new(Vec::new());
    let progress = Mutex::new(0usize);

    let n_params = parameters.len();

    let s = Instant::now();
    parameters
        .into_iter()
        .enumerate()
        .for_each(|(i, (depth, expansion_order, block_size))| {
            let expansion_order = vec![expansion_order; (depth + 1) as usize];
            // Setup random sources and targets
            let n_sources = n_points;
            let n_targets = n_points;
            let sources = points_fixture::<T::Real>(n_sources, None, None, Some(0));
            let targets = points_fixture::<T::Real>(n_targets, None, None, Some(1));
            let n_vecs = 1;
            let tmp = vec![T::one(); n_sources * n_vecs];
            let mut charges = rlst_dynamic_array2!(T, [n_sources, n_vecs]);
            charges.data_mut().copy_from_slice(&tmp);

            let s = Instant::now();
            let fmm = SingleNodeBuilder::new(true)
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
                    GreenKernelEvalType::Value,
                    FftFieldTranslation::new(Some(block_size)),
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
            "block_size".to_string(),
            "runtime".to_string(),
            "min_rel_err".to_string(),
            "mean_rel_err".to_string(),
            "max_rel_err".to_string(),
            "setup_time".to_string(),
        ])
        .unwrap();

    let n_sources = n_points;
    let sources = points_fixture::<T::Real>(n_sources, None, None, Some(0));
    let n_vecs = 1;
    let tmp = vec![T::one(); n_sources * n_vecs];
    let mut charges = rlst_dynamic_array2!(T, [n_sources, n_vecs]);
    charges.data_mut().copy_from_slice(&tmp);

    let mut progress = 0;
    for (i, fmm, setup_time) in fmms.lock().unwrap().iter_mut() {
        let (depth, expansion_order, block_size) = parameters_cloned[*i];

        let s = Instant::now();
        fmm.evaluate().unwrap();
        let time = s.elapsed().as_millis() as f32;
        progress += 1;
        println!("FFT Evaluated {:?}/{:?}", progress, n_params);

        let leaf_idx = 1;
        let leaf = fmm.tree().target_tree().all_leaves().unwrap()[leaf_idx];
        let potential = fmm.potential(&leaf).unwrap()[0];
        let leaf_targets = fmm.tree().target_tree().coordinates(&leaf).unwrap();
        let n_targets = leaf_targets.len() / fmm.dim();
        let mut direct = vec![T::zero(); n_targets];
        fmm.kernel().evaluate_st(
            GreenKernelEvalType::Value,
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
                block_size.to_string(),
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
    let max_m2l_fft_block_size_vec = vec![16, 32, 64, 128];
    let rsvd_settings_vec = vec![FmmSvdMode::new(false, None, None, None, None)];

    // Single Precision
    {
        let expansion_order_vec: Vec<usize> = vec![3];

        let svd_threshold_vec = vec![None, Some(1e-7), Some(1e-5), Some(1e-3), Some(1e-1)];

        let surface_diff_vec: Vec<usize> = vec![0, 1, 2];
        let depth_vec: Vec<u64> = vec![4, 5];

        let n_points = 10000;

        grid_search_laplace_fft::<f32>(
            "grid_search_laplace_fft_f32_m1".to_string(),
            n_points,
            &expansion_order_vec,
            &depth_vec,
            &max_m2l_fft_block_size_vec,
        );

        for (i, &rsvd_settings) in rsvd_settings_vec.iter().enumerate() {
            grid_search_laplace_blas::<f32>(
                format!("grid_search_laplace_blas_f32_m1_{i}").to_string(),
                n_points,
                &expansion_order_vec,
                &svd_threshold_vec,
                &surface_diff_vec,
                &depth_vec,
                &[rsvd_settings],
            );
        }
    }
}
