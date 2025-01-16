use core::panic;

use green_kernels::{laplace_3d::Laplace3dKernel, types::GreenKernelEvalType};
use kifmm::fmm::types::FmmSvdMode;
use kifmm::fmm::types::{BlasFieldTranslationSaRcmp, FftFieldTranslation, SingleNodeBuilder};
use kifmm::traits::field::{SourceToTargetTranslation, TargetTranslation};
use kifmm::traits::fmm::{DataAccess, Evaluate};
use kifmm::traits::tree::{SingleFmmTree, SingleTree};
use kifmm::tree::helpers::{points_fixture, points_fixture_oblate_spheroid, points_fixture_sphere};
use rlst::{rlst_dynamic_array2, RawAccess, RawAccessMut};


fn fft_f64() {

    let n_points = 1000000;
    let geometries = ["uniform", "sphere", "spheroid"];

    // Tree depth
    let depth_vec = vec![
        [5, 5, 5], // 6 digits, for each geometry
        [4, 5, 5], // 8 digits for each geometry
        [4, 5, 5], // 10 digits for each geometry
    ];

    // Expansion order
    let e_vec = vec![
        [6, 6, 7],    // 6 digits, for each geometry
        [8, 9, 9],   // 8 digits for each geometry
        [10, 10, 11], // 10 digits for each geometry
    ];

    // Block size
    let b_vec = vec![
        [32, 32, 32], // 6 digits, for each geometry
        [64, 64, 64], // 8 digits for each geometry
        [32, 64, 64],  // 10 digits for each geometry
    ];

    let experiments = [6, 8, 10]; // number of digits sought

    let prune_empty = true;

    for (i, digits) in experiments.iter().enumerate() {
        for (j, &geometry) in geometries.iter().enumerate() {
            let sources;
            let targets;
            if geometry == "uniform" {
                sources = points_fixture::<f64>(n_points, Some(0.), Some(1.), None);
                targets = points_fixture::<f64>(n_points, Some(0.), Some(1.), None);
            } else if geometry == "sphere" {
                sources = points_fixture_sphere(n_points);
                targets = points_fixture_sphere(n_points);
            } else if geometry == "spheroid" {
                sources = points_fixture_oblate_spheroid(n_points, 1.0, 0.5);
                targets = points_fixture_oblate_spheroid(n_points, 1.0, 0.5);
            } else {
                panic!("invalid geometry");
            }

            let charges = vec![1f64; n_points];

            let depth = Some(depth_vec[i][j]);
            let e = e_vec[i][j];
            let b = b_vec[i][j];
            let expansion_order = vec![e; depth.unwrap() as usize + 1];
            let block_size = Some(b);

            let mut fmm = SingleNodeBuilder::new(false)
                .tree(sources.data(), targets.data(), None, depth, prune_empty)
                .unwrap()
                .parameters(
                    &charges,
                    &expansion_order,
                    Laplace3dKernel::new(),
                    GreenKernelEvalType::Value,
                    FftFieldTranslation::new(block_size),
                )
                .unwrap()
                .build()
                .unwrap();

            fmm.evaluate();

            println!("FFT-M2L geometry {:?}, digits {:?}, flops {:?}", geometry, digits, fmm.nflops)

        }
    }
}

fn blas_f64() {

    let n_points = 1000000;
    let geometries = ["uniform", "sphere", "spheroid"];

    // Tree depth
    let depth_vec = vec![
        [5, 5, 5], // 6 digits, for each geometry
        [4, 5, 5], // 8 digits for each geometry
        [4, 5, 5], // 10 digits for each geometry
    ];

    // Expansion order
    let e_vec = vec![
        [5, 6,6],   // 6 digits, for each geometry
        [7, 8, 8],   // 8 digits for each geometry
        [9, 10, 10], // 10 digits for each geometry
    ];

    // SVD threshold
    let svd_threshold_vec = vec![
        [Some(1e-5), Some(1e-5), Some(1e-7)],  // 6 digits
        [Some(1e-5), Some(1e-7), Some(1e-7)],  // 8 digits
        [Some(1e-7), Some(1e-7), Some(1e-11)], // 10 digits
    ];

    let svd_mode_vec = vec![
        [
            FmmSvdMode::new(true, None, None, Some(5), None),
            FmmSvdMode::new(true, None, None, Some(5), None),
            FmmSvdMode::new(true, None, None, Some(20), None),
        ],
        [
            FmmSvdMode::new(true, None, None, Some(10), None),
            FmmSvdMode::new(true, None, None, Some(5), None),
            FmmSvdMode::new(true, None, None, Some(5), None),
        ],
        [
            FmmSvdMode::new(true, None, None, Some(10), None),
            FmmSvdMode::new(false, None, None, None, None),
            FmmSvdMode::new(true, None, None, Some(20), None),
        ],
    ];

    let surface_diff_vec = vec![
        [Some(1), Some(2), None],       // 6 digits
        [Some(2), Some(1), Some(1)], // 8 digits
        [Some(2), Some(2), None], // 10 digits
    ];

    let nvecs = vec![1];

    let experiments = [6, 8, 10]; // number of digits sought

    let prune_empty = true;

    for (i, digits) in experiments.iter().enumerate() {
        for (j, &geometry) in geometries.iter().enumerate() {
            for (k, nvec) in nvecs.iter().enumerate() {
                let sources;
                let targets;
                if geometry == "uniform" {
                    sources = points_fixture::<f64>(n_points, Some(0.), Some(1.), None);
                    targets = points_fixture::<f64>(n_points, Some(0.), Some(1.), None);
                } else if geometry == "sphere" {
                    sources = points_fixture_sphere(n_points);
                    targets = points_fixture_sphere(n_points);
                } else if geometry == "spheroid" {
                    sources = points_fixture_oblate_spheroid(n_points, 1.0, 0.5);
                    targets = points_fixture_oblate_spheroid(n_points, 1.0, 0.5);
                } else {
                    panic!("invalid geometry");
                }

                let charges = vec![1f64; n_points * nvec];

                let depth = Some(depth_vec[i][j]);
                let e = e_vec[i][j];
                let threshold = svd_threshold_vec[i][j];
                let surface_diff = surface_diff_vec[i][j];
                let svd_mode = svd_mode_vec[i][j];

                let expansion_order = vec![e; depth.unwrap() as usize + 1];

                let mut fmm = SingleNodeBuilder::new(false)
                    .tree(sources.data(), targets.data(), None, depth, prune_empty)
                    .unwrap()
                    .parameters(
                        &charges,
                        &expansion_order,
                        Laplace3dKernel::new(),
                        GreenKernelEvalType::Value,
                        BlasFieldTranslationSaRcmp::new(threshold, surface_diff, svd_mode),
                    )
                    .unwrap()
                    .build()
                    .unwrap();

            fmm.evaluate();

            println!("BLAS-M2L geometry {:?}, digits {:?}, flops {:?}", geometry, digits, fmm.nflops)
            }
        }
    }
}


fn main() {
    fft_f64();;
    blas_f64();
}
