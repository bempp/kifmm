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

use std::mem;


fn calculate_compression<Scalar: RlstScalar + Float + AsComplex + Dft>(
    fmm: &KiFmm<Scalar, Laplace3dKernel<Scalar>, BlasFieldTranslationSaRcmp<Scalar>>,
    fmm_full: &KiFmm<Scalar, Laplace3dKernel<Scalar>, BlasFieldTranslationSaRcmp<Scalar>>,
) -> f64 {
    let element_size = mem::size_of::<Scalar>(); // Assuming f32 is the element type

    // M2L
    let data = &fmm.source_to_target.metadata;
    let mut size_m2l_compressed = 0;

    for metadata in data {
        let u = &metadata.u;
        let st = &metadata.st;
        let c_u = &metadata.c_u;
        let c_vt = &metadata.c_vt;

        let mut size: usize = u.shape()[0] * u.shape()[1] + st.shape()[0] + st.shape()[1];
        for (l, r) in c_u.iter().zip(c_vt.iter()) {
            size += l.shape()[0] + l.shape()[1] + r.shape()[0] * r.shape()[1];
        }
        size_m2l_compressed += size;
    }

    let data = &fmm_full.source_to_target.metadata;
    let mut size_m2l = 0;

    for metadata in data {
        let u = &metadata.u;
        let st = &metadata.st;
        let c_u = &metadata.c_u;
        let c_vt = &metadata.c_vt;

        let mut size: usize = u.shape()[0] * u.shape()[1] + st.shape()[0] + st.shape()[1];
        for (l, r) in c_u.iter().zip(c_vt.iter()) {
            size += l.shape()[0] + l.shape()[1] + r.shape()[0] * r.shape()[1];
        }
        size_m2l += size;
    }

    println!("HERE {:?} {:?}", size_m2l, size_m2l_compressed);

    let compression = 100. - 100. * ((size_m2l_compressed as f64) / (size_m2l as f64));

    compression
}


fn blas_f32() {
    let n_points = 1000000;
    let geometries = ["uniform", "sphere", "spheroid"];

    // TODO: 4 digits for sphere and spheroid are dummies for now

    // Tree depth
    let depth_vec = vec![
        [4, 5, 5], // 3 digits, for each geometry
        [4, 5, 5], // 4 digits for each geometry
    ];

    // Expansion order
    let e_vec = vec![
        [3, 2, 3], // 3 digits, for each geometry
        [3, 3, 4], // 4 digits for each geometry
    ];

    // SVD threshold
    let svd_threshold_vec = vec![
        [Some(0.00001), Some(0.00001), Some(0.001)],              // 3 digits
        [None, None, None], // 4 digits
    ];

    let svd_mode_vec = vec![
        [
            FmmSvdMode::new(true, None, None, Some(10), None),
            FmmSvdMode::new(true, None, None, Some(10), None),
            FmmSvdMode::new(false, None, None, None, None),
        ],
        [
            FmmSvdMode::new(false, None, None, None, None),
            FmmSvdMode::new(true, None, None, Some(20), None),
            FmmSvdMode::new(true, None, None, Some(5), None),
        ],
    ];

    let surface_diff_vec = vec![
        [None, Some(2), Some(2)], // 3 digits
        [Some(1), Some(2), Some(1)],       // 4 digits
    ];

    let nvecs = vec![1];

    let experiments = [3, 4]; // number of digits sought

    let prune_empty = true;

    for (i, &digits) in experiments.iter().enumerate() {
        for (j, &geometry) in geometries.iter().enumerate() {

            // if geometry == "spheroid" && digits == 3 {
                for (k, nvec) in nvecs.iter().enumerate() {
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

                    let charges = vec![1f32; n_points * nvec];

                    let depth = Some(depth_vec[i][j]);
                    let e = e_vec[i][j];
                    let threshold = svd_threshold_vec[i][j];
                    let surface_diff = surface_diff_vec[i][j];
                    let svd_mode = svd_mode_vec[i][j];

                    let expansion_order = vec![e; depth.unwrap() as usize + 1];

                    let s = Instant::now();
                    let fmm = SingleNodeBuilder::new(false)
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
                    println!("SETUP TIME {:?}", s.elapsed());

                    let fmm_full = SingleNodeBuilder::new(false)
                        .tree(sources.data(), targets.data(), None, depth, prune_empty)
                        .unwrap()
                        .parameters(
                            &charges,
                            &expansion_order,
                            Laplace3dKernel::new(),
                            GreenKernelEvalType::Value,
                            BlasFieldTranslationSaRcmp::new(None, surface_diff, svd_mode),
                        )
                        .unwrap()
                        .build()
                        .unwrap();

                    let compression = calculate_compression(&fmm, &fmm_full);

                    println!("BLAS M2L geometry {:?}, digits {:?}", geometry, digits);
                    println!("Compression {:?} %", compression);

                }
            // }
        }
    }
}


fn blas_f64() {
    let n_points = 1000000;
    let geometries = ["uniform", "sphere", "spheroid"];

    // Tree depth
    // Tree depth
    let depth_vec = vec![
        [5, 5, 4], // 6 digits, for each geometry
        [4, 5, 4], // 8 digits for each geometry
        [4, 5, 4], // 10 digits for each geometry
    ];

    // Expansion order
    let e_vec = vec![
        [5, 6, 6],    // 6 digits, for each geometry
        [7, 8, 8],    // 8 digits for each geometry
        [9, 10, 10], // 10 digits for each geometry
    ];

    // SVD threshold
    let svd_threshold_vec = vec![
        [Some(1e-5), Some(1e-5), Some(1e-5)],  // 6 digits
        [Some(1e-5), Some(1e-7), Some(1e-5)],  // 8 digits
        [Some(1e-7), Some(1e-7), Some(1e-7)], // 10 digits
    ];

    let svd_mode_vec = vec![
        [
            FmmSvdMode::new(true, None, None, Some(5), None),
            FmmSvdMode::new(true, None, None, Some(5), None),
            FmmSvdMode::new(true, None, None, Some(10), None),
        ],
        [
            FmmSvdMode::new(true, None, None, Some(10), None),
            FmmSvdMode::new(true, None, None, Some(5), None),
            FmmSvdMode::new(true, None, None, Some(20), None),
        ],
        [
            FmmSvdMode::new(true, None, None, Some(10), None),
            FmmSvdMode::new(false, None, None, None, None),
            FmmSvdMode::new(false, None, None, None, None),
        ],
    ];

    let surface_diff_vec = vec![
        [Some(1), Some(1), Some(1)],       // 6 digits
        [Some(2), Some(1), Some(1)], // 8 digits
        [Some(2), Some(2), Some(2)], // 10 digits
    ];

    let nvecs = vec![1];

    let experiments = [6, 8, 10]; // number of digits sought

    let prune_empty = true;

    for (i, &digits) in experiments.iter().enumerate() {
        for (j, &geometry) in geometries.iter().enumerate() {

            // if digits == 10 && geometry == "uniform" {
                for (k, nvec) in nvecs.iter().enumerate() {
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

                    let charges = vec![1f64; n_points * nvec];

                    let depth = Some(depth_vec[i][j]);
                    let e = e_vec[i][j];
                    let threshold = svd_threshold_vec[i][j];
                    let surface_diff = surface_diff_vec[i][j];
                    let svd_mode = svd_mode_vec[i][j];

                    let expansion_order = vec![e; depth.unwrap() as usize + 1];

                    let s = Instant::now();
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
                    let e = s.elapsed();
                    // let mut fmm_full = SingleNodeBuilder::new(false)
                    //     .tree(sources.data(), targets.data(), None, depth, prune_empty)
                    //     .unwrap()
                    //     .parameters(
                    //         &charges,
                    //         &expansion_order,
                    //         Laplace3dKernel::new(),
                    //         GreenKernelEvalType::Value,
                    //         BlasFieldTranslationSaRcmp::new(None, surface_diff, svd_mode),
                    //     )
                    //     .unwrap()
                    //     .build()
                    //     .unwrap();

                    // let compression = calculate_compression(&fmm, &fmm_full);
                    println!("BLAS M2L geometry {:?}, digits {:?}", geometry, digits);
                    println!("SETUP {:?} ms", e);
                    // println!("Compression {:?} %", compression);
                }
            // }
        }
    }
}

fn main() {

    // println!("BLAS F32");
    // blas_f32();

    // println!("BLAS F64");
    blas_f64();
}
