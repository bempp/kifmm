use core::panic;

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


fn calculate_fmm_storage_fft_m2l<Scalar: RlstScalar + Float + AsComplex + Dft>(
    fmm: &KiFmm<Scalar, Laplace3dKernel<Scalar>, FftFieldTranslation<Scalar>>,
) -> (f64, f64, f64) {
    let element_size = mem::size_of::<Scalar>(); // Assuming f32 is the element type

    // UC2E_INV_1
    let mut storage_uc2e_inv_1 = 0;
    let mut size_uc2e_inv_1 = 0;
    for mat in &fmm.uc2e_inv_1 {
        size_uc2e_inv_1 += mat.shape()[0] * mat.shape()[1];
    }
    storage_uc2e_inv_1 = size_uc2e_inv_1 * element_size;

    // UC2E_INV_2
    let mut storage_uc2e_inv_2 = 0;
    let mut size_uc2e_inv_2 = 0;
    for mat in &fmm.uc2e_inv_2 {
        size_uc2e_inv_2 += mat.shape()[0] * mat.shape()[1];
    }
    storage_uc2e_inv_2 = size_uc2e_inv_2 * element_size;

    // M2M
    let mut storage_m2m = 0;
    let mut size_m2m = 0;
    for mat in &fmm.source {
        size_m2m += mat.shape()[0] * mat.shape()[1];
    }
    storage_m2m = size_m2m * element_size;

    // L2L
    let mut storage_l2l = 0;
    let mut size_l2l = 0;
    for vec in &fmm.target_vec {
        for mat in vec.iter() {
            size_l2l += mat.shape()[0] * mat.shape()[1];
        }
    }
    storage_l2l = size_l2l * element_size;

    // M2L
    let kernel_data = &fmm.source_to_target.metadata[0].kernel_data;
    let mut size_m2l = 0;
    let mut storage_m2l = 0;
    for vec in kernel_data {
        size_m2l += vec.len();
    }
    size_m2l *= 2; // need it twice, freq re-ordered
    storage_m2l = size_m2l * element_size * 2; // complex numbers

    // Convert storage to megabytes
    (
        (storage_m2m + storage_uc2e_inv_1 + storage_uc2e_inv_2) as f64 / 1_048_576.0,
        (storage_l2l + storage_uc2e_inv_1 + storage_uc2e_inv_2) as f64 / 1_048_576.0,
        storage_m2l as f64 / 1_048_576.0,
    )
}

fn calculate_fmm_storage_blas_m2l<Scalar: RlstScalar + Float + AsComplex + Dft>(
    fmm: &KiFmm<Scalar, Laplace3dKernel<Scalar>, BlasFieldTranslationSaRcmp<Scalar>>,
) -> (f64, f64, f64) {
    let element_size = mem::size_of::<Scalar>(); // Assuming f32 is the element type

    // UC2E_INV_1
    let mut storage_uc2e_inv_1 = 0;
    let mut size_uc2e_inv_1 = 0;
    for mat in &fmm.uc2e_inv_1 {
        size_uc2e_inv_1 += mat.shape()[0] * mat.shape()[1];
    }
    storage_uc2e_inv_1 = size_uc2e_inv_1 * element_size;

    // UC2E_INV_2
    let mut storage_uc2e_inv_2 = 0;
    let mut size_uc2e_inv_2 = 0;
    for mat in &fmm.uc2e_inv_2 {
        size_uc2e_inv_2 += mat.shape()[0] * mat.shape()[1];
    }
    storage_uc2e_inv_2 = size_uc2e_inv_2 * element_size;

    // M2M
    let mut storage_m2m = 0;
    let mut size_m2m = 0;
    for mat in &fmm.source {
        size_m2m += mat.shape()[0] * mat.shape()[1];
    }
    storage_m2m = size_m2m * element_size;

    // L2L
    let mut storage_l2l = 0;
    let mut size_l2l = 0;
    for vec in &fmm.target_vec {
        for mat in vec.iter() {
            size_l2l += mat.shape()[0] * mat.shape()[1];
        }
    }
    storage_l2l = size_l2l * element_size;

    // M2L
    let data = &fmm.source_to_target.metadata;
    let mut size_m2l = 0;
    let mut storage_m2l = 0;

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

    storage_m2l = size_m2l * element_size;

    // Convert storage to megabytes
    (
        (storage_m2m + storage_uc2e_inv_1 + storage_uc2e_inv_2) as f64 / 1_048_576.0,
        (storage_l2l + storage_uc2e_inv_1 + storage_uc2e_inv_2) as f64 / 1_048_576.0,
        storage_m2l as f64 / 1_048_576.0,
    )
}

fn fft_f32() {
    let n_points = 1000000;
    let geometries = ["uniform", "sphere", "spheroid"];

    // Tree depth
    let depth_vec = vec![
        [5, 5, 5], // 3 digits, for each geometry
        [5, 5, 5], // 4 digits for each geometry
    ];

    // Expansion order
    let e_vec = vec![
        [3, 3, 4], // 3 digits, for each geometry
        [4, 5, 5], // 4 digits for each geometry
    ];

    // Block size
    let b_vec = vec![
        [64, 64, 128], // 3 digits, for each geometry
        [32, 128, 128],   // 4 digits for each geometry
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

            let (m2m, l2l, m2l) = calculate_fmm_storage_fft_m2l(&fmm);

            println!("FFT M2L geometry {:?}, digits {:?}", geometry, digits);
            println!("M2M {:?} mb  L2L {:?} mb M2L {:?} mb", m2m, l2l, m2l);
        }
    }
}

fn blas_f32() {
    let n_points = 1000000;
    let geometries = ["uniform", "sphere", "spheroid"];

    // TODO: 4 digits for sphere and spheroid are dummies for now

    // Tree depth
    let depth_vec = vec![
        [5, 5, 5], // 3 digits, for each geometry
        [5, 5, 5], // 4 digits for each geometry
    ];

    // Expansion order
    let e_vec = vec![
        [3, 3, 3], // 3 digits, for each geometry
        [3, 3, 5], // 4 digits for each geometry
    ];

    // SVD threshold
    let svd_threshold_vec = vec![
        [Some(0.1), Some(0.00001), Some(0.00001)], // 3 digits
        [Some(0.001), Some(0.001), None],  // 4 digits
    ];

    let svd_mode_vec = vec![
        [
            FmmSvdMode::new(true, None, None, Some(5), None),
            FmmSvdMode::new(false, None, None, None, None),
            FmmSvdMode::new(true, None, None, Some(20), None),
        ],
        [
            FmmSvdMode::new(true, None, None, Some(10), None),
            FmmSvdMode::new(false, None, None, None, None),
            FmmSvdMode::new(false, None, None, None, None),
        ],
    ];

    let surface_diff_vec = vec![
        [Some(1), None, None], // 3 digits
        [Some(1), Some(1), None],    // 4 digits
    ];

    let nvecs = vec![1];

    let experiments = [3, 4]; // number of digits sought

    let prune_empty = true;

    for (i, digits) in experiments.iter().enumerate() {
        for (j, &geometry) in geometries.iter().enumerate() {
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

                let (m2m, l2l, m2l) = calculate_fmm_storage_blas_m2l(&fmm);

                println!("BLAS M2L geometry {:?}, digits {:?}", geometry, digits);
                println!("M2M {:?} mb L2L {:?} mb M2L {:?} mb", m2m, l2l, m2l);

            }
        }
    }
}

fn fft_f64() {
    let n_points = 1000000;
    let geometries = ["uniform", "sphere", "spheroid"];

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

            let fmm_fft = SingleNodeBuilder::new(false)
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

            let (m2m, l2l, m2l) = calculate_fmm_storage_fft_m2l(&fmm_fft);

            println!("FFT M2L geometry {:?}, digits {:?}", geometry, digits);
            println!("M2M {:?} mb  L2L {:?} mb M2L {:?} mb", m2m, l2l, m2l);
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

                let (m2m, l2l, m2l) = calculate_fmm_storage_blas_m2l(&fmm);
                println!("BLAS M2L geometry {:?}, digits {:?}", geometry, digits);
                println!("M2M {:?} mb L2L {:?} mb M2L {:?} mb", m2m, l2l, m2l);
            }
        }
    }
}

fn main() {
    // println!("FFT F32");
    // fft_f32();

    // println!("BLAS F32");
    // blas_f32();

    println!("FFT F64");
    fft_f64();

    println!("BLAS F64");
    blas_f64();
}
