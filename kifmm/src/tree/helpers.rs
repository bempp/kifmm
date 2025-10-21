//! Helper functions used in testing tree implementations, specifically test point generators,
//! as well as helpers for handling surfaces that discretise a box corresponding to a Morton key.

use itertools::Itertools;
use num::Float;
use rand::{distributions::Uniform, prelude::*, rngs::StdRng, SeedableRng};
use rlst::{rlst_dynamic_array2, Array, BaseArray, RlstScalar, VectorContainer};

/// Alias for an rlst container for point data, expected with shape [n_points, 3];
pub type PointsMat<T> = Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>;

/// Points fixture for testing, uniformly samples in each axis from min to max.
///
/// # Arguments
/// * `n_points` - The number of points to sample.
/// * `min` - The minimum coordinate value along each axis, defaults to 1
/// * `max` - The maximum coordinate value along each axis, defaults to 0.
/// * `seed` - Random seed, defaults to 0.
pub fn points_fixture<T: Float + RlstScalar + rand::distributions::uniform::SampleUniform>(
    n_points: usize,
    min: Option<T>,
    max: Option<T>,
    seed: Option<u64>,
) -> PointsMat<T> {
    // Generate a set of randomly distributed points
    let seed = seed.unwrap_or(0);
    let mut range = StdRng::seed_from_u64(seed);

    let between;
    if let (Some(min), Some(max)) = (min, max) {
        between = rand::distributions::Uniform::from(min..max);
    } else {
        between = rand::distributions::Uniform::from(T::zero()..T::one());
    }

    let mut points = rlst_dynamic_array2!(T, [3, n_points]);

    for i in 0..n_points {
        points[[0, i]] = between.sample(&mut range);
        points[[1, i]] = between.sample(&mut range);
        points[[2, i]] = between.sample(&mut range);
    }

    points
}

/// Points fixture for testing, uniformly samples on surface of a sphere of diameter 1.
///
/// # Arguments
/// * `n_points` - The number of points to sample.
/// * `min` - The minimum coordinate value along each axis.
/// * `max` - The maximum coordinate value along each axis.
pub fn points_fixture_sphere<T: RlstScalar + rand::distributions::uniform::SampleUniform>(
    n_points: usize,
    seed: Option<u64>
) -> PointsMat<T> {

    let seed = seed.unwrap_or(0);
    // Seeded random number generator for reproducibility
    let mut rng = StdRng::seed_from_u64(seed);
    let pi = T::from(std::f64::consts::PI).unwrap();
    let two = T::from(2.0).unwrap();

    // Uniform distributions for phi and z = cos(theta)
    let phi_dist = Uniform::from(T::zero()..(two * pi));
    let z_dist = Uniform::from(T::from(-1.0).unwrap()..T::from(1.0).unwrap());

    // Initialize points array
    let mut points = rlst_dynamic_array2!(T, [3, n_points]);

    for i in 0..n_points {
        // Generate random phi and theta
        let phi = phi_dist.sample(&mut rng);
        let z = z_dist.sample(&mut rng);
        let r_xy = (T::one() - z * z).sqrt();

        // Compute Cartesian coordinates
        let x = r_xy * phi.cos();
        let y = r_xy * phi.sin();

        // Assign to points array
        points[[0, i]] = x;
        points[[1, i]] = y;
        points[[2, i]] = z;
    }

    points
}

/// Points fixture for testing, uniformly samples on surface of an oblate-spheroid
///
/// # Arguments
/// * `n_points` - The number of points to sample.
/// * `a` - Semi-axis length along x- and y-axes.
/// * `c` -  Semi-axis length along the z-axis.
pub fn points_fixture_oblate_spheroid<
    T: RlstScalar + rand::distributions::uniform::SampleUniform,
>(
    n_points: usize,
    a: T,
    c: T,
) -> PointsMat<T> {
    let mut rng = StdRng::seed_from_u64(0); // Use a fixed seed for reproducibility
    let pi = T::from(std::f64::consts::PI).unwrap();
    let two = T::from(2.0).unwrap();
    let one = T::one();

    let phi_dist = Uniform::from(T::zero()..(two * pi)); // Azimuthal angle
    let cos_theta_dist = Uniform::from(-one..one); // Cosine of polar angle

    let mut points = rlst_dynamic_array2!(T, [3, n_points]);

    for i in 0..n_points {
        let phi = phi_dist.sample(&mut rng); // Random azimuthal angle
        let cos_theta = cos_theta_dist.sample(&mut rng); // Random cosine of polar angle
        let theta = cos_theta.acos(); // Compute polar angle

        // Parametric equations for the oblate spheroid
        points[[0, i]] = a * theta.sin() * phi.cos(); // x-axis
        points[[1, i]] = a * theta.sin() * phi.sin(); // y-axis
        points[[2, i]] = c * theta.cos(); // z-axis
    }

    points
}

/// Points fixture for testing, uniformly samples in the bounds [[0, 1), [0, 1), [0, 500)] for the x, y, and z
/// axes respectively.
///
/// # Arguments
/// * `n_points` - The number of points to sample.
pub fn points_fixture_col<T: Float + RlstScalar + rand::distributions::uniform::SampleUniform>(
    n_points: usize,
) -> PointsMat<T> {
    // Generate a set of randomly distributed points
    let mut range = StdRng::seed_from_u64(0);

    let between1 = rand::distributions::Uniform::from(T::zero()..T::from(0.1).unwrap());
    let between2 = rand::distributions::Uniform::from(T::zero()..T::from(500).unwrap());

    let mut points = rlst_dynamic_array2!(T, [3, n_points]);

    for i in 0..n_points {
        // One axis has a different sampling
        points[[0, i]] = between1.sample(&mut range);
        points[[1, i]] = between1.sample(&mut range);
        points[[2, i]] = between2.sample(&mut range);
    }

    points
}

/// Find the corners of a box discretising the surface of a box described by a Morton Key. The coordinates
/// are expected in row major order [x_1, y_1, z_1,...x_N, y_N, z_N]
///
/// # Arguments:
/// * `coordinates` - points on the surface of a box.
pub fn find_corners<T: Float>(coordinates: &[T]) -> Vec<T> {
    let xs = coordinates.iter().step_by(3).cloned().collect_vec();
    let ys = coordinates.iter().skip(1).step_by(3).cloned().collect_vec();
    let zs = coordinates.iter().skip(2).step_by(3).cloned().collect_vec();

    let x_min = *xs
        .iter()
        .min_by(|&a, &b| a.partial_cmp(b).unwrap())
        .unwrap();

    let x_max = *xs
        .iter()
        .max_by(|&a, &b| a.partial_cmp(b).unwrap())
        .unwrap();

    let y_min = *ys
        .iter()
        .min_by(|&a, &b| a.partial_cmp(b).unwrap())
        .unwrap();

    let y_max = *ys
        .iter()
        .max_by(|&a, &b| a.partial_cmp(b).unwrap())
        .unwrap();

    let z_min = *zs
        .iter()
        .min_by(|&a, &b| a.partial_cmp(b).unwrap())
        .unwrap();

    let z_max = *zs
        .iter()
        .max_by(|&a, &b| a.partial_cmp(b).unwrap())
        .unwrap();

    // Returned in row major order
    let corners = vec![
        x_min, y_min, z_min, x_max, y_min, z_min, x_min, y_max, z_min, x_max, y_max, z_min, x_min,
        y_min, z_max, x_max, y_min, z_max, x_min, y_max, z_max, x_max, y_max, z_max,
    ];

    corners
}

#[cfg(test)]
mod test {

    use super::*;
    use crate::tree::morton::surface_grid;

    #[test]
    fn test_find_corners() {
        let expansion_order = 5;
        let grid_1: Vec<f64> = surface_grid(expansion_order);

        let expansion_order = 2;
        let grid_2: Vec<f64> = surface_grid(expansion_order);

        let corners_1 = find_corners(&grid_1);
        let corners_2 = find_corners(&grid_2);

        // Test the corners are invariant by order of grid
        for (&c1, c2) in corners_1.iter().zip(corners_2) {
            assert!(c1 == c2);
        }

        // Test that the corners are the ones expected
        for (&c1, g2) in corners_1.iter().zip(grid_2) {
            assert!(c1 == g2);
        }
    }
}
