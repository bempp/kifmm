use core::panic;
use std::time::Instant;

use green_kernels::{laplace_3d::Laplace3dKernel, types::GreenKernelEvalType};
use kifmm::fmm::types::FmmSvdMode;
use kifmm::fmm::types::{BlasFieldTranslationSaRcmp, FftFieldTranslation, SingleNodeBuilder};
use kifmm::traits::fftw::Dft;
use kifmm::traits::field::{SourceToTargetTranslation, TargetTranslation};
use kifmm::traits::fmm::{DataAccess, Evaluate};
use kifmm::traits::general::single_node::AsComplex;
use kifmm::traits::tree::{SingleFmmTree, SingleTree};
use kifmm::tree::helpers::{points_fixture, points_fixture_oblate_spheroid, points_fixture_sphere};
use kifmm::{fmm, KiFmm};
use num::Float;
use rlst::{rlst_dynamic_array2, RawAccess, RawAccessMut, RlstScalar, Shape};



fn fft_f32() {
    let n_points = 1000000;
    let geometries = ["uniform", "sphere", "spheroid"];

    // Tree depth
    let depth_vec = vec![
        [4, 5, 5], // 3 digits, for each geometry
        [4, 5, 5], // 4 digits for each geometry
    ];

    // Expansion order
    let e_vec = vec![
        [3, 3, 3], // 3 digits, for each geometry
        [4, 5, 5], // 4 digits for each geometry
    ];

    // Block size
    let b_vec = vec![
        [128, 64, 128], // 3 digits, for each geometry
        [16, 64, 64], // 4 digits for each geometry
    ];

    let experiments = [3, 4]; // number of digits sought

    let prune_empty = true;

    for (i, digits) in experiments.iter().enumerate() {
        for (j, &geometry) in geometries.iter().enumerate() {
            let sources;
            let targets;
            if geometry == "uniform" {
                sources = points_fixture::<f32>(n_points, Some(0.), Some(1.), None);
                targets = points_fixture::<f32>(n_points, Some(0.), Some(1.), None);
            } else if geometry == "sphere" {
                sources = points_fixture_sphere(n_points);
                targets = points_fixture_sphere(n_points);
            } else if geometry == "spheroid" {
                sources = points_fixture_oblate_spheroid(n_points, 1.0, 0.5);
                targets = points_fixture_oblate_spheroid(n_points, 1.0, 0.5);
            } else {
                panic!("invalid geometry");
            }

            let charges = vec![1f32; n_points];

            let depth = Some(depth_vec[i][j]);
            let e = e_vec[i][j];
            let b = b_vec[i][j];
            let expansion_order = vec![e; depth.unwrap() as usize + 1];
            let block_size = Some(b);

            println!("FFT M2L geometry {:?}, digits {:?}", geometry, digits);

            let s= Instant::now();
            let fmm = SingleNodeBuilder::new(false)
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
            println!("SETUP {:?}", s.elapsed().as_millis());

        }
    }
}


fn fft_f64() {
    let n_points = 1000000;
    let geometries = ["uniform", "sphere", "spheroid"];

    let depth_vec = vec![
        [4, 4, 4], // 6 digits, for each geometry
        [4, 5, 5], // 8 digits for each geometry
        [4, 4, 4], // 10 digits for each geometry
    ];

    // Expansion order
    let e_vec = vec![
        [6, 6, 6],    // 6 digits, for each geometry
        [8, 9, 8],    // 8 digits for each geometry
        [10, 10, 10], // 10 digits for each geometry
    ];

    // Block size
    let b_vec = vec![
        [32, 32, 32], // 6 digits, for each geometry
        [64, 64, 64],  // 8 digits for each geometry
        [32, 64, 128],  // 10 digits for each geometry
    ];

    let experiments = [6, 8, 10]; // number of digits sought

    let prune_empty = true;

    for (i, digits) in experiments.iter().enumerate() {
        for (j, &geometry) in geometries.iter().enumerate() {
            let sources;
            let targets;
            if geometry == "uniform" {
                sources = points_fixture(n_points, Some(0.), Some(1.), None);
                targets = points_fixture(n_points, Some(0.), Some(1.), None);
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

            println!("FFT M2L geometry {:?}, digits {:?}", geometry, digits);
            let s= Instant::now();
            let fmm = SingleNodeBuilder::new(false)
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
            println!("SETUP {:?}", s.elapsed().as_millis());
        }
    }
}

fn main() {
    // println!("FFT F32");
    // fft_f32();


    // println!("FFT F64");
    fft_f64();

}
