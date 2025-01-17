use core::panic;

use green_kernels::helmholtz_3d::Helmholtz3dKernel;
use green_kernels::{laplace_3d::Laplace3dKernel, types::GreenKernelEvalType};
use itertools::{izip, Itertools};
use kifmm::fmm::types::FmmSvdMode;
use kifmm::fmm::types::{BlasFieldTranslationSaRcmp, FftFieldTranslation, SingleNodeBuilder};
use kifmm::traits::fftw::Dft;
use kifmm::traits::field::{SourceToTargetTranslation, TargetTranslation};
use kifmm::traits::fmm::{DataAccess, Evaluate};
use kifmm::traits::general::single_node::AsComplex;
use kifmm::traits::tree::{SingleFmmTree, SingleTree};
use kifmm::tree::helpers::{points_fixture, points_fixture_oblate_spheroid, points_fixture_sphere};
use kifmm::{fmm, BlasFieldTranslationIa, KiFmm};
use num::Float;
use rlst::{c32, c64, rlst_dynamic_array2, RawAccess, RawAccessMut, RlstScalar, Shape};

use std::mem;


fn p_vec(p_leaf: usize, p_scale: f64) -> Vec<usize> {

    let mut res = Vec::new();

    let mut curr = p_leaf;
    for l in 0..5 {
        res.push(curr as usize);
        curr = ((curr as f64) * p_scale) as usize;
    }

    res = res.into_iter().rev().collect_vec();
    res[0] = 0;
    res[1] = 0;
    res
}

fn calculate_fmm_storage_fft_m2l<Scalar>(
    fmm: &KiFmm<Scalar, Helmholtz3dKernel<Scalar>, FftFieldTranslation<Scalar>>,
) -> (f64, f64, f64)
where
    Scalar: RlstScalar<Complex = Scalar> + AsComplex + Dft,

{
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
    storage_m2l = size_m2l * element_size; // complex numbers

    // Convert storage to megabytes
    (
        (storage_m2m + storage_uc2e_inv_1 + storage_uc2e_inv_2) as f64 / 1_048_576.0,
        (storage_l2l + storage_uc2e_inv_1 + storage_uc2e_inv_2) as f64 / 1_048_576.0,
        storage_m2l as f64 / 1_048_576.0,
    )
}

fn calculate_fmm_storage_blas_m2l<Scalar>(
    fmm: &KiFmm<Scalar, Helmholtz3dKernel<Scalar>, BlasFieldTranslationIa<Scalar>>,
) -> (f64, f64, f64)
where
    Scalar: RlstScalar<Complex = Scalar>
{
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
        let vt = &metadata.vt;

        let mut size = 0;
        for (l, r) in u.iter().zip(vt.iter()) {
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

fn fft_f32(arch: String) {
    let n_points = 1000;
    let geometries = ["uniform", "sphere", "spheroid"];

    let depth = 4;

    let p_leaf_vec;
    let p_scale_vec;
    if arch == "amd" {
        p_leaf_vec = vec![
            [4, 5, 6, 8, 11], // uniform
            [4, 5, 6, 8, 11], // sphere
            [4, 5, 6, 9, 11], //speroid
        ];
        p_scale_vec = vec![
            [1.0, 1.3, 1.3, 1.3, 1.3], // uniform
            [1., 1.3, 1.3, 1.5, 1.5], //sphere
            [1., 1.5, 1.3, 1.3, 1.3],
        ];
    } else if arch == "m1" {
        p_leaf_vec = vec![
            [4, 5, 6, 8, 11], // uniform
            [4, 5, 6, 8, 11], // sphere
            [4, 5, 7, 8, 11], //speroid
        ];
        p_scale_vec = vec![
            [1.0, 1.3, 1.3, 1.3, 1.3], // uniform
            [1., 1.3, 1.3, 1.3, 1.3], //sphere
            [1., 1.3, 1.3, 1.5, 1.3],
        ];
    } else {
        panic!("")
    }

    let wavenumber_vec = vec![
        [5., 25., 50., 75., 100.],
        [5., 25., 50., 75., 100.],
        [5., 25., 50., 75., 100.],
        ];


    let prune_empty = true;

    for (i, &geometry) in geometries.iter().enumerate() {
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

            let charges = vec![c32::ONE; n_points];

        for (wavenumber, p_scale, p_leaf) in izip!(wavenumber_vec[i], p_scale_vec[i], p_leaf_vec[i]) {

            let expansion_order = p_vec(p_leaf, p_scale);
            let mut fmm = SingleNodeBuilder::new(false)
                .tree(sources.data(), targets.data(), None, Some(depth), prune_empty)
                .unwrap()
                .parameters(
                    &charges,
                    &expansion_order,
                    Helmholtz3dKernel::new(wavenumber),
                    GreenKernelEvalType::Value,
                    FftFieldTranslation::new(None),
                )
                .unwrap()
                .build()
                .unwrap();

            let (m2m, l2l, m2l) = calculate_fmm_storage_fft_m2l::<c32>(&fmm);

            println!("FFT M2L geometry {:?}, wave number {:?}", geometry, wavenumber);
            println!("M2M {:?} mb L2L {:?} mb M2L {:?} mb", m2m, l2l, m2l);
        }
    }
}


fn blas_f32() {
    let n_points = 1000000;
    let geometries = ["uniform", "sphere", "spheroid"];

    let depth = 4;

    let p_leaf_vec = vec![
        [3, 5, 6], // uniform
        [4, 5, 6], // sphere
        [4, 5, 6], //speroid
    ];

    let p_scale_vec = vec![
        [1.5, 1.3, 1.5], // uniform
        [1.5, 1.5, 1.3], //sphere
        [1., 1.3, 1.5],
    ];

    let wavenumber_vec = vec![
        [5., 25., 50.],
        [5., 25., 50.],
        [5., 25., 50.],
    ];

    let surface_diff_vec = vec![
        [1, 0, 0],
        [0, 0, 2],
        [0, 0, 2]
    ];

    let prune_empty = true;

    for (i, &geometry) in geometries.iter().enumerate() {
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

            let charges = vec![c32::ONE; n_points];

        for (wavenumber, p_scale, p_leaf, surface_diff) in izip!(wavenumber_vec[i], p_scale_vec[i], p_leaf_vec[i], surface_diff_vec[i]) {

            let expansion_order = p_vec(p_leaf, p_scale);
            let mut fmm = SingleNodeBuilder::new(false)
                .tree(sources.data(), targets.data(), None, Some(depth), prune_empty)
                .unwrap()
                .parameters(
                    &charges,
                    &expansion_order,
                    Helmholtz3dKernel::new(wavenumber),
                    GreenKernelEvalType::Value,
                    BlasFieldTranslationIa::new(None, Some(surface_diff), FmmSvdMode::Deterministic),
                )
                .unwrap()
                .build()
                .unwrap();

            let (m2m, l2l, m2l) = calculate_fmm_storage_blas_m2l::<c32>(&fmm);

            println!("BLAS M2L geometry {:?}, wave number {:?}", geometry, wavenumber);
            println!("M2M {:?} mb L2L {:?} mb M2L {:?} mb", m2m, l2l, m2l);

        }
    }
}


fn main() {
    // fft_f32("m1".to_string());

    blas_f32();
}
