//! ACA implementation for kernels supported by this library
use std::collections::HashSet;

use green_kernels::traits::Kernel as KernelTrait;
use itertools::Itertools;
use num::Zero;
use rand::{rngs, Rng};
use rlst::{c32, c64, rlst_dynamic_array2, Array, RawAccessMut, RlstScalar};

use crate::traits::general::single_node::Epsilon;

/// Trait that abstracts over real/complex numbers for their magnitude
pub(crate) trait ArgmaxValue<Scalar: RlstScalar> {
    fn argmax_value(&self) -> <Scalar as RlstScalar>::Real;
}

macro_rules! impl_argmax_value {
    // For real numbers, argmax defined by value
    ($t:ty) => {
        impl ArgmaxValue<$t> for $t {
            fn argmax_value(&self) -> $t {
                *self
            }
        }
    };

    // For complex numbers, argmax defined by magnitude
    ($t:ty, $r:ty) => {
        impl ArgmaxValue<$t> for $t {
            fn argmax_value(&self) -> $r {
                self.abs()
            }
        }
    };
}

impl_argmax_value!(f32);
impl_argmax_value!(f64);
impl_argmax_value!(c32, f32);
impl_argmax_value!(c64, f64);

/// Returns the index of the maximum value in a slice
fn argmax<Scalar>(v: &[Scalar]) -> Option<usize>
where
    Scalar: RlstScalar + ArgmaxValue<Scalar>,
{
    if v.is_empty() {
        return None;
    }

    let mut max_index = 0;
    let mut max_value = v[0].argmax_value();

    for (i, val) in v.iter().enumerate() {
        let cmp_val = val.argmax_value();
        if cmp_val > max_value {
            max_value = cmp_val;
            max_index = i
        }
    }

    Some(max_index)
}

/// Sorts an array of floats by magnitude of elements, return the indices
/// that sort them
///
/// # Safety
/// Doesn't handle NANs
fn argsort<Scalar>(v: &[Scalar]) -> Vec<usize>
where
    Scalar: RlstScalar + ArgmaxValue<Scalar>,
{
    let mut indices = (0..v.len()).collect_vec();

    indices.sort_by(
        |&i, &j| match v[i].argmax_value().partial_cmp(&v[j].argmax_value()) {
            Some(ord) => ord,
            None => std::cmp::Ordering::Greater,
        },
    );
    indices
}

/// Return index of maximum value in array that isn't explicitly disallowed
fn exclusive_argmax<Scalar>(arr: &[Scalar], disallowed: &HashSet<usize>) -> Option<usize>
where
    Scalar: RlstScalar + ArgmaxValue<Scalar>,
    <Scalar as RlstScalar>::Real: ArgmaxValue<Scalar::Real>,
{
    // Sort indices by magnitude
    let order = argsort(arr);

    // Traverse in descending order of magnitude
    for &idx in order.iter().rev() {
        if !disallowed.contains(&idx) {
            return Some(idx);
        }
    }

    // Fallback: all disallowed or empty
    argmax(arr)
}

/// Reset the current reference row index (i_ref), and return
/// updated reference row after subtracting current approximation terms (us, vs).
#[allow(clippy::too_many_arguments)]
fn reset_reference_row<Scalar, Kernel>(
    sources: &[Scalar::Real],
    targets: &[Scalar::Real],
    kernel: &Kernel,
    n_targets: usize,
    rng: &mut rngs::ThreadRng,
    i_ref: &mut usize,
    i_star: Option<usize>,
    local_radius_rows: Option<usize>,
    prev_i_star: &HashSet<usize>,
    us: &[Vec<Scalar>],
    vs: &[Vec<Scalar>],
) -> Vec<Scalar>
where
    Scalar: RlstScalar,
    Kernel: KernelTrait<T = Scalar>,
{
    // Local reset near last pivot row if available
    if let Some(local_radius_rows) = local_radius_rows {
        let base = if let Some(i_star) = i_star {
            i_star
        } else {
            *i_ref
        };

        let mut cand;
        for _ in 0..20 {
            let offset: isize =
                rng.gen_range(-(local_radius_rows as isize)..(local_radius_rows as isize));
            cand = (base as isize + offset).rem_euclid(n_targets as isize) as usize;
            if !prev_i_star.contains(&cand) {
                *i_ref = cand;
                break;
            }
        }
    } else {
        let mut cand;
        for _ in 0..20 {
            cand = rng.gen_range(0..n_targets);
            if !prev_i_star.contains(&cand) {
                *i_ref = cand;
                break;
            }
        }
    }

    calc_residual_rows(sources, targets, kernel, *i_ref, *i_ref + 1, us, vs)
}

/// Reset the current reference col index (j_ref), and return
/// updated reference col after subtracting current approximation terms (us, vs).
#[allow(clippy::too_many_arguments)]
fn reset_reference_col<Scalar, Kernel>(
    sources: &[Scalar::Real],
    targets: &[Scalar::Real],
    kernel: &Kernel,
    n_sources: usize,
    rng: &mut rngs::ThreadRng,
    j_ref: &mut usize,
    j_star: Option<usize>,
    local_radius_cols: Option<usize>,
    prev_j_star: &HashSet<usize>,
    us: &[Vec<Scalar>],
    vs: &[Vec<Scalar>],
) -> Vec<Scalar>
where
    Scalar: RlstScalar,
    Kernel: KernelTrait<T = Scalar>,
{
    if let Some(local_radius_cols) = local_radius_cols {
        let base = if let Some(j_star) = j_star {
            j_star
        } else {
            *j_ref
        };

        let mut cand;
        for _ in 0..20 {
            let offset: isize =
                rng.gen_range(-(local_radius_cols as isize)..(local_radius_cols as isize));
            cand = (base as isize + offset).rem_euclid(n_sources as isize) as usize;
            if !prev_j_star.contains(&cand) {
                *j_ref = cand;
                break;
            }
        }
    } else {
        let mut cand;
        for _ in 0..20 {
            cand = rng.gen_range(0..n_sources);
            if !prev_j_star.contains(&cand) {
                *j_ref = cand;
                break;
            }
        }
    }

    calc_residual_cols(sources, targets, kernel, *j_ref, *j_ref + 1, us, vs)
}

/// Update a given row with current approximation terms (us, vs)
fn calc_residual_rows<Scalar, Kernel>(
    sources: &[Scalar::Real],
    targets: &[Scalar::Real],
    kernel: &Kernel,
    i_start: usize,
    i_end: usize,
    us: &[Vec<Scalar>],
    vs: &[Vec<Scalar>],
) -> Vec<Scalar>
where
    Scalar: RlstScalar,
    Kernel: KernelTrait<T = Scalar>,
{
    let mut out = calc_rows(kernel, sources, targets, i_start, i_end);

    for i in 0..us.len() {
        let scale = us[i][i_start];
        out.iter_mut().zip(vs[i].iter()).for_each(|(o, &v)| {
            *o -= v * scale;
        });
    }

    out
}

/// Update a given col with current approximation terms (us, vs)
fn calc_residual_cols<Scalar, Kernel>(
    sources: &[Scalar::Real],
    targets: &[Scalar::Real],
    kernel: &Kernel,
    j_start: usize,
    j_end: usize,
    us: &[Vec<Scalar>],
    vs: &[Vec<Scalar>],
) -> Vec<Scalar>
where
    Scalar: RlstScalar,
    Kernel: KernelTrait<T = Scalar>,
{
    let mut out = calc_cols(kernel, sources, targets, j_start, j_end);
    for i in 0..us.len() {
        let scale = vs[i][j_start];
        out.iter_mut().zip(us[i].iter()).for_each(|(o, &u)| {
            *o -= u * scale;
        });
    }

    out
}

/// Calculate the columns of a kernel matrix generated between a set of sources and targets
fn calc_cols<Scalar, Kernel>(
    kernel: &Kernel,
    sources: &[Scalar::Real],
    targets: &[Scalar::Real],
    j_start: usize,
    j_end: usize,
) -> Vec<Scalar>
where
    Scalar: RlstScalar,
    Kernel: KernelTrait<T = Scalar>,
{
    let dim = 3;
    let n_cols = j_end - j_start;
    let n_targets = targets.len() / dim;

    let mut cols = vec![Scalar::default(); n_cols * n_targets];
    let charges = vec![Scalar::one(); j_end - j_start];
    kernel.evaluate_mt(
        green_kernels::types::GreenKernelEvalType::Value,
        &sources[j_start * dim..j_end * dim],
        targets,
        &charges,
        &mut cols,
    );

    cols
}

/// Calculate the rows of a kernel matrix generated between a set of sources and targets
fn calc_rows<Scalar, Kernel>(
    kernel: &Kernel,
    sources: &[Scalar::Real],
    targets: &[Scalar::Real],
    i_start: usize,
    i_end: usize,
) -> Vec<Scalar>
where
    Scalar: RlstScalar,
    Kernel: KernelTrait<T = Scalar>,
{
    let dim = 3;
    let n_rows = i_end - i_start;
    let n_sources = sources.len() / dim;

    let mut rows = vec![Scalar::default(); n_rows * n_sources];
    let charges = vec![Scalar::one(); i_end - i_start];
    kernel.evaluate_mt(
        green_kernels::types::GreenKernelEvalType::Value,
        &targets[i_start * dim..i_end * dim],
        sources,
        &charges,
        &mut rows,
    );

    rows
}

#[allow(dead_code, clippy::too_many_arguments, clippy::type_complexity)]
/// ACA+ algorithm specified for interaction matrix between a set of sources/targets which are
/// points in 3D
pub(crate) fn aca_plus<Kernel, Scalar>(
    sources: &[Scalar::Real],
    targets: &[Scalar::Real],
    kernel: Kernel,
    eps: Option<Scalar::Real>,
    max_iter: Option<usize>,
    local_radius_rows: Option<usize>,
    local_radius_cols: Option<usize>,
    verbose: bool,
) -> (
    Array<Scalar, rlst::BaseArray<Scalar, rlst::VectorContainer<Scalar>, 2>, 2>,
    Array<Scalar, rlst::BaseArray<Scalar, rlst::VectorContainer<Scalar>, 2>, 2>,
)
where
    Scalar: ArgmaxValue<Scalar> + Epsilon,
    <Scalar as RlstScalar>::Real: ArgmaxValue<Scalar::Real>,
    Kernel: KernelTrait<T = Scalar>,
{
    let dim = 3;
    let n_sources = sources.len() / dim;
    let n_targets = targets.len() / dim;

    // let charges_s = vec![Scalar::one(); n_sources];
    // let charges_t = vec![Scalar::one(); n_targets];

    // Set convergence threshold
    let eps = if let Some(eps) = eps {
        eps
    } else {
        Scalar::epsilon()
    };

    // Set maximum number of iterations
    let max_iter = if let Some(max_iter) = max_iter {
        max_iter
    } else {
        std::cmp::min(n_sources, n_targets)
    };

    // Set random number generator
    let mut rng = rand::thread_rng();

    let mut us = Vec::new();
    let mut vs = Vec::new();

    let mut prev_i_star = HashSet::new();
    let mut prev_j_star = HashSet::new();

    // Initial references for rows and columns, pick randomly
    let mut i_ref = rng.gen_range(0..n_targets);
    let mut j_ref = rng.gen_range(0..n_sources);
    // Calculate initial residual values, mutates i_ref
    let mut r_iref = reset_reference_row(
        sources,
        targets,
        &kernel,
        n_targets,
        &mut rng,
        &mut i_ref,
        None,
        local_radius_rows,
        &prev_i_star,
        &us,
        &vs,
    );

    let mut r_jref = reset_reference_col(
        sources,
        targets,
        &kernel,
        n_sources,
        &mut rng,
        &mut j_ref,
        None,
        local_radius_cols,
        &prev_j_star,
        &us,
        &vs,
    );

    let mut i_star;
    let mut j_star;
    let mut r_jstar;
    let mut r_istar;

    let mut pivot;

    for k in 0..max_iter {
        j_star = exclusive_argmax(&r_iref, &prev_j_star).unwrap();
        i_star = exclusive_argmax(&r_jref, &prev_i_star).unwrap();

        // decide path based on larger reference entry
        if r_iref[j_star].abs() >= r_jref[i_star].abs() {
            // build residual column at j_star, find i_star
            r_jstar = calc_residual_cols(sources, targets, &kernel, j_star, j_star + 1, &us, &vs);
            let r_jstar_mag = r_jstar.iter().clone().map(|x| x.abs()).collect_vec();
            i_star = argmax(&r_jstar_mag).unwrap();
            r_istar = calc_residual_rows(sources, targets, &kernel, i_star, i_star + 1, &us, &vs);
            pivot = r_istar[j_star];
        } else {
            // build residual row at i_star, find j_star
            r_istar = calc_residual_rows(sources, targets, &kernel, i_star, i_star + 1, &us, &vs);
            let r_istar_mag = r_istar.iter().clone().map(|x| x.abs()).collect_vec();
            j_star = argmax(&r_istar_mag).unwrap();
            r_jstar = calc_residual_cols(sources, targets, &kernel, j_star, j_star + 1, &us, &vs);
            pivot = r_istar[j_star];
        }

        if pivot.abs() <= eps {
            // we've converged
            if verbose {
                println!("[stop] k={:?} |pivot|={:?} <= {:?}", k, pivot.abs(), eps);
            }
            break;
        }

        // Otherwise update current approximation and reference rows/columns
        let v_new = r_istar.clone().into_iter().map(|x| x / pivot).collect_vec();
        let u_new = r_jstar.clone();

        // Update references (outer product)
        // RIref := RIref - u[Iref]*v
        // RJref := RJref - u * v[Jref]
        r_iref
            .iter_mut()
            .zip(v_new.iter())
            .for_each(|(r, &v)| *r -= v * u_new[i_ref]);

        r_jref
            .iter_mut()
            .zip(u_new.iter())
            .for_each(|(r, &u)| *r -= u * v_new[j_ref]);

        us.push(u_new.clone());
        vs.push(v_new.clone());

        prev_i_star.insert(i_star);
        prev_j_star.insert(j_star);

        // if reference coincided with a pivot, pick a new (local) one
        if i_ref == i_star {
            r_iref = reset_reference_row(
                sources,
                targets,
                &kernel,
                n_targets,
                &mut rng,
                &mut i_ref,
                Some(i_star),
                local_radius_rows,
                &prev_i_star,
                &us,
                &vs,
            )
        }

        if j_ref == j_star {
            r_jref = reset_reference_col(
                sources,
                targets,
                &kernel,
                n_sources,
                &mut rng,
                &mut j_ref,
                Some(j_star),
                local_radius_cols,
                &prev_j_star,
                &us,
                &vs,
            )
        }

        if verbose {
            let uu = u_new.iter().fold(Scalar::Real::zero(), |acc, x| {
                acc + RlstScalar::pow(x.abs(), Scalar::real(2.0))
            });
            let vv = v_new.iter().fold(Scalar::Real::zero(), |acc, x| {
                acc + RlstScalar::pow(x.abs(), Scalar::real(2.0))
            });
            let step_size = (uu * vv).sqrt();

            println!(
                "k={:?} pivot(row={:?}, col={:?}) |pivot|={:?} step={:?}",
                k,
                i_star,
                j_star,
                pivot.abs(),
                step_size
            );
        }
    }

    // Return compressed basis vectors
    let m = us[0].len();
    let k = us.len();
    let n = vs[0].len();

    // RLST arrays are column major by default
    let mut u_aca = rlst_dynamic_array2!(Scalar, [m, k]);
    let mut v_aca = rlst_dynamic_array2!(Scalar, [n, k]);
    let mut v_aca_t = rlst_dynamic_array2!(Scalar, [k, n]);

    for (j, u) in us.iter().enumerate() {
        // copy in us -> column vectors
        u_aca.data_mut()[j * m..(j + 1) * m].copy_from_slice(u);
    }

    for (i, v) in vs.iter().enumerate() {
        // copy in vs -> row vectors
        v_aca.data_mut()[i * n..(i + 1) * n].copy_from_slice(v);
    }

    // Have to account for RLST memory ordering
    v_aca_t.fill_from(v_aca.transpose());

    (u_aca, v_aca_t)
}

#[cfg(test)]
mod test {

    use green_kernels::{helmholtz_3d::Helmholtz3dKernel, laplace_3d::Laplace3dKernel};

    use num::One;
    use rand::thread_rng;
    use rlst::{c32, empty_array, DefaultIteratorMut, MultIntoResize, RawAccess, RawAccessMut};

    use crate::{fmm::helpers::single_node::l2_error, tree::helpers::points_fixture};

    use super::*;

    #[test]
    fn test_argmax() {
        // Test real
        let n = 10;
        let mut arr = vec![0f32; n];
        arr.iter_mut()
            .enumerate()
            .for_each(|(i, elem)| *elem = i as f32);

        let expected = 9usize;
        let found = argmax(&arr).unwrap();
        assert_eq!(expected, found);

        // Test complex
        let n = 10;
        let mut arr = vec![c32::zero(); n];
        arr.iter_mut()
            .enumerate()
            .for_each(|(i, elem)| *elem = c32::from_real(i as f32));

        let expected = 9usize;
        let found = argmax(&arr).unwrap();
        assert_eq!(expected, found);

        // Test empty
        let arr: Vec<f32> = vec![];
        assert!(argmax(&arr).is_none());
    }

    #[test]
    fn test_argsort() {
        let mut rng = thread_rng();
        let n = 100;

        // Test real
        let mut arr: Vec<f32> = vec![0f32; n];
        arr.iter_mut().for_each(|e| *e = rng.gen());
        let found = argsort(&arr);
        let sorted = found.iter().map(|&i| arr[i]).collect_vec();
        let mut curr = sorted[0];
        for &item in sorted.iter().take(sorted.len() - 1).skip(1) {
            assert!(curr <= item);
            curr = item;
        }

        // Test complex
        let mut arr: Vec<c32> = vec![c32::zero(); n];
        arr.iter_mut().for_each(|e| *e = rng.gen());
        let found = argsort(&arr);
        let sorted = found.iter().map(|&i| arr[i]).collect_vec();
        let mut curr = sorted[0];
        for &item in sorted.iter().take(sorted.len() - 1).skip(1) {
            assert!(curr.abs() <= item.abs());
            curr = item;
        }
    }

    #[test]
    fn test_exclusive_argmax() {}

    #[test]
    fn test_aca_plus() {
        // Test f32
        let n_sources = 100;
        let n_targets = 100;
        let sources = points_fixture::<f32>(n_sources, None, None, None);
        let mut targets = points_fixture::<f32>(n_targets, None, None, None);
        // Displace targets by a fixed amount to ensure that interaction is low rank


        let displacement = 2.0;
        targets.iter_mut().for_each(|t| *t += displacement);

        let n = 100;
        let sources = points_fixture::<f32>(n, Some(0.1), Some(0.9), None);
        let targets = points_fixture::<f32>(n, Some(1.7), Some(2.5), None);
        // targets.iter_mut().for_each(|t| *t += 1.5);


        let eps = 1e-6;

        // Test Laplace
        {
            let kernel = Laplace3dKernel::<f32>::new();

            let (u, v) = aca_plus(
                sources.data(),
                targets.data(),
                kernel.clone(),
                Some(eps),
                None,
                None,
                None,
                false,
            );

            // generate a random vector
            let mut rng = rand::thread_rng();
            let mut x = rlst_dynamic_array2![f32, [n_sources, 1]];
            x.data_mut().iter_mut().for_each(|e| *e = rng.gen());

            // Apply matrix to a random vector
            let mut b_true = vec![0f32; n_targets];

            kernel.evaluate_st(
                green_kernels::types::GreenKernelEvalType::Value,
                sources.data(),
                targets.data(),
                x.data(),
                &mut b_true,
            );

            let b_aca = empty_array::<f32, 2>().simple_mult_into_resize(
                u.r(),
                empty_array::<f32, 2>().simple_mult_into_resize(v.r(), x),
            );

            let l2_error = l2_error(b_aca.data(), &b_true);

            assert!(l2_error < eps * 10.);
        }

        // Test Helmholtz (low wavenumber)
        {
            let wavenumber = 5.;
            let kernel = Helmholtz3dKernel::<c32>::new(wavenumber);

            let (u, v) = aca_plus(
                sources.data(),
                targets.data(),
                kernel.clone(),
                Some(eps),
                None,
                None,
                None,
                false,
            );

            // generate a test vector
            let mut x = rlst_dynamic_array2![c32, [n_sources, 1]];
            x.data_mut().iter_mut().for_each(|e| *e = c32::one());

            // Apply matrix to test vector
            let mut b_true = vec![c32::zero(); n_targets];

            kernel.evaluate_st(
                green_kernels::types::GreenKernelEvalType::Value,
                sources.data(),
                targets.data(),
                x.data(),
                &mut b_true,
            );

            let b_aca = empty_array::<c32, 2>().simple_mult_into_resize(
                u.r(),
                empty_array::<c32, 2>().simple_mult_into_resize(v.r(), x),
            );

            let l2_error = l2_error(b_aca.data(), &b_true);

            assert!(l2_error < eps * 10.);

            println!("FOO {:?}", l2_error);
            assert!(false);
        }
    }
}
