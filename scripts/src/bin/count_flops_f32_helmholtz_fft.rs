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


fn fft_f32(arch: String) {
    let n_points = 1000000;
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

            fmm.evaluate();


            println!("FFT M2L geometry {:?}, wave number {:?} flops {:?}", geometry, wavenumber, fmm.nflops);
        }
    }
}



fn main() {
    println!("M1");
    fft_f32("m1".to_string());
    println!("AMD");
    fft_f32("amd".to_string());
}
