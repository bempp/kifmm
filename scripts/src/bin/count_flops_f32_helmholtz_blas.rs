use core::panic;

use green_kernels::helmholtz_3d::Helmholtz3dKernel;
use green_kernels::{laplace_3d::Laplace3dKernel, types::GreenKernelEvalType};
use itertools::{izip, Itertools};
use kifmm::fmm::types::{BlasMetadataIa, FmmSvdMode};
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

            fmm.evaluate();


            println!("BLAS M2L geometry {:?}, wave number {:?} flops {:?}", geometry, wavenumber, fmm.nflops);
        }
    }
}


fn main() {
    blas_f32();
}
