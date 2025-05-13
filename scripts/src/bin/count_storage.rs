use green_kernels::{laplace_3d::Laplace3dKernel, types::GreenKernelEvalType};
use kifmm::fmm::types::FmmSvdMode;
use kifmm::fmm::types::{BlasFieldTranslationSaRcmp, FftFieldTranslation, SingleNodeBuilder};
use kifmm::traits::fftw::Dft;
use kifmm::traits::general::single_node::AsComplex;
use kifmm::tree::helpers::points_fixture;
use kifmm::{KiFmm};
use num::Float;
use rlst::{RawAccess, RlstScalar, Shape};
use serde_yaml::Value;

use std::{fs, mem};
use std::path::PathBuf;

fn calculate_fmm_storage_fft_m2l<Scalar: RlstScalar + Float + AsComplex + Dft>(
    fmm: &KiFmm<Scalar, Laplace3dKernel<Scalar>, FftFieldTranslation<Scalar>>,
) -> (f64, f64, f64) {
    let element_size = mem::size_of::<Scalar>(); // Assuming f32 is the element type

    // UC2E_INV_1
    let mut size_uc2e_inv_1 = 0;
    for mat in &fmm.uc2e_inv_1 {
        size_uc2e_inv_1 += mat.shape()[0] * mat.shape()[1];
    }
    let storage_uc2e_inv_1 = size_uc2e_inv_1 * element_size;

    // UC2E_INV_2
    let mut size_uc2e_inv_2 = 0;
    for mat in &fmm.uc2e_inv_2 {
        size_uc2e_inv_2 += mat.shape()[0] * mat.shape()[1];
    }
    let storage_uc2e_inv_2 = size_uc2e_inv_2 * element_size;

    // M2M
    let mut size_m2m = 0;
    for mat in &fmm.source {
        size_m2m += mat.shape()[0] * mat.shape()[1];
    }
    let storage_m2m = size_m2m * element_size;

    // L2L
    let mut size_l2l = 0;
    for vec in &fmm.target_vec {
        for mat in vec.iter() {
            size_l2l += mat.shape()[0] * mat.shape()[1];
        }
    }
    let storage_l2l = size_l2l * element_size;

    // M2L
    let kernel_data = &fmm.source_to_target.metadata[0].kernel_data;
    let mut size_m2l = 0;
    for vec in kernel_data {
        size_m2l += vec.len();
    }
    size_m2l *= 2; // need it twice, freq re-ordered
    let storage_m2l = size_m2l * element_size * 2; // complex numbers

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
    let mut size_uc2e_inv_1 = 0;
    for mat in &fmm.uc2e_inv_1 {
        size_uc2e_inv_1 += mat.shape()[0] * mat.shape()[1];
    }
    let storage_uc2e_inv_1 = size_uc2e_inv_1 * element_size;

    // UC2E_INV_2
    let mut size_uc2e_inv_2 = 0;
    for mat in &fmm.uc2e_inv_2 {
        size_uc2e_inv_2 += mat.shape()[0] * mat.shape()[1];
    }
    let storage_uc2e_inv_2 = size_uc2e_inv_2 * element_size;

    // M2M
    let mut size_m2m = 0;
    for mat in &fmm.source {
        size_m2m += mat.shape()[0] * mat.shape()[1];
    }
    let storage_m2m = size_m2m * element_size;

    // L2L
    let mut size_l2l = 0;
    for vec in &fmm.target_vec {
        for mat in vec.iter() {
            size_l2l += mat.shape()[0] * mat.shape()[1];
        }
    }
    let storage_l2l = size_l2l * element_size;

    // M2L
    let data = &fmm.source_to_target.metadata;
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

    let storage_m2l = size_m2l * element_size;

    // Convert storage to megabytes
    (
        (storage_m2m + storage_uc2e_inv_1 + storage_uc2e_inv_2) as f64 / 1_048_576.0,
        (storage_l2l + storage_uc2e_inv_1 + storage_uc2e_inv_2) as f64 / 1_048_576.0,
        storage_m2l as f64 / 1_048_576.0,
    )
}

fn main() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("arch-conf.yaml");
    let yaml_str = fs::read_to_string(path).unwrap();
    let root: Value = serde_yaml::from_str(&yaml_str).unwrap();

    let arch = std::env::var("ARCH").unwrap_or_else(|_| {
        panic!("Please set ARCH in .cargo/config.toml or your environment");
    });

    println!("Using arch: {}", arch);

    let arch = root
        .get("arch")
        .and_then(Value::as_mapping)
        .expect(".")
        .get(arch)
        .and_then(Value::as_mapping)
        .expect("Expected 'arch' to be a mapping");

    for precision in ["fp32", "fp64"] {
        let data = arch
            .get(precision)
            .and_then(Value::as_mapping)
            .expect("Expected 'fp32' to be a mapping");

        // Parse the FFT M2L parameters
        let fft_m2l = data
            .get("fft")
            .and_then(Value::as_mapping)
            .expect("Expected 'fft' to be a mapping");

        let n_points_map = fft_m2l
            .get("n_points")
            .and_then(Value::as_mapping)
            .expect("Expected 'n_points' to be a mapping");

        for (n_points_key, n_points_val) in n_points_map {
            let n_points = n_points_key.as_u64().unwrap_or(0) as usize;
            if let Some(digits_map) = n_points_val.get("digits").and_then(Value::as_mapping) {
                for (digit_key, params_val) in digits_map {
                    let digits = digit_key.as_i64().unwrap_or(-1);

                    let e = params_val.get("order").and_then(Value::as_u64).unwrap_or(0);

                    let block_size = params_val
                        .get("block_size")
                        .and_then(Value::as_u64)
                        .unwrap_or(0) as usize;
                    let depth = params_val.get("depth").and_then(Value::as_u64).unwrap_or(0);

                    let expansion_order = vec![e as usize; depth as usize + 1];

                    let storage;
                    if precision == "fp32" {
                        let sources = points_fixture::<f32>(n_points, None, None, None);
                        let charges = vec![1.0f32; n_points];
                        let fmm = SingleNodeBuilder::new(false)
                            .tree(sources.data(), sources.data(), None, Some(depth), true)
                            .unwrap()
                            .parameters(
                                &charges,
                                &expansion_order,
                                Laplace3dKernel::new(),
                                GreenKernelEvalType::Value,
                                FftFieldTranslation::new(Some(block_size)),
                            )
                            .unwrap()
                            .build()
                            .unwrap();

                        storage = calculate_fmm_storage_fft_m2l(&fmm);
                    } else {
                        let sources = points_fixture::<f64>(n_points, None, None, None);
                        let charges = vec![1.0f64; n_points];
                        let fmm = SingleNodeBuilder::new(false)
                            .tree(sources.data(), sources.data(), None, Some(depth), true)
                            .unwrap()
                            .parameters(
                                &charges,
                                &expansion_order,
                                Laplace3dKernel::new(),
                                GreenKernelEvalType::Value,
                                FftFieldTranslation::new(Some(block_size)),
                            )
                            .unwrap()
                            .build()
                            .unwrap();
                        storage = calculate_fmm_storage_fft_m2l(&fmm);
                    }

                    let m2l_storage = storage.2;
                    println!("precision: {precision}, m2l: fft, n_points: {n_points}, digits: {digits} M2L storage: {m2l_storage} MB");
                }
            }
        }

        // Parse the BLAS M2L parameters
        let blas_m2l = data
            .get("blas")
            .and_then(Value::as_mapping)
            .expect("Expected 'blas' to be a mapping");

        let n_points_map = blas_m2l
            .get("n_points")
            .and_then(Value::as_mapping)
            .expect("Expected 'n_points' to be a mapping");

        for (n_points_key, n_points_val) in n_points_map {
            let n_points = n_points_key.as_u64().unwrap_or(0) as usize;
            if let Some(digits_map) = n_points_val.get("digits").and_then(Value::as_mapping) {
                for (digit_key, params_val) in digits_map {
                    let digits = digit_key.as_u64().unwrap_or(0);

                    let e = params_val.get("order").and_then(Value::as_u64).unwrap_or(0);

                    let surface_diff = params_val
                        .get("surface_diff")
                        .and_then(Value::as_u64)
                        .unwrap_or(0) as usize;
                    let svd_threshold = params_val
                        .get("svd_threshold")
                        .and_then(Value::as_f64)
                        .unwrap_or(0.);
                    let n_oversamples = params_val
                        .get("n_oversamples")
                        .and_then(Value::as_u64)
                        .unwrap_or(0) as usize;
                    let depth = params_val.get("depth").and_then(Value::as_u64).unwrap_or(0);
                    let expansion_order = vec![e as usize; depth as usize + 1];

                    let storage;
                    if precision == "fp32" {
                        let sources = points_fixture::<f32>(n_points, None, None, None);
                        let charges = vec![1.0f32; n_points];
                        let fmm = SingleNodeBuilder::new(false)
                            .tree(sources.data(), sources.data(), None, Some(depth), true)
                            .unwrap()
                            .parameters(
                                &charges,
                                &expansion_order,
                                Laplace3dKernel::new(),
                                GreenKernelEvalType::Value,
                                BlasFieldTranslationSaRcmp::new(
                                    Some(svd_threshold as f32),
                                    Some(surface_diff),
                                    FmmSvdMode::new(true, None, None, Some(n_oversamples), None),
                                ),
                            )
                            .unwrap()
                            .build()
                            .unwrap();

                        storage = calculate_fmm_storage_blas_m2l(&fmm);
                    } else {
                        let sources = points_fixture::<f64>(n_points, None, None, None);
                        let charges = vec![1.0f64; n_points];
                        let fmm = SingleNodeBuilder::new(false)
                            .tree(sources.data(), sources.data(), None, Some(depth), true)
                            .unwrap()
                            .parameters(
                                &charges,
                                &expansion_order,
                                Laplace3dKernel::new(),
                                GreenKernelEvalType::Value,
                                BlasFieldTranslationSaRcmp::new(
                                    Some(svd_threshold),
                                    Some(surface_diff),
                                    FmmSvdMode::new(true, None, None, Some(n_oversamples), None),
                                ),
                            )
                            .unwrap()
                            .build()
                            .unwrap();

                        storage = calculate_fmm_storage_blas_m2l(&fmm);
                    }

                    let m2l_storage = storage.2;
                    println!("precision: {precision}, m2l: fft, n_points: {n_points}, digits: {digits} M2L storage: {m2l_storage} MB");
                }
            }
        }
    }
}
