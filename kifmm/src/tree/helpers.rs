//! Helper functions used in testing tree implementations, specifically test point generators,
//! as well as helpers for handling surfaces that discretise a box corresponding to a Morton key.

use itertools::Itertools;
use num::Float;
use rand::prelude::*;
use rlst::RlstScalar;
use rlst::{rlst_dynamic_array2, Array, BaseArray, VectorContainer};

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
) -> PointsMat<T> {
    // Generate a set of randomly distributed points
    let mut range = StdRng::seed_from_u64(0);
    let pi = T::from(3.134159).unwrap();
    let two = T::from(2.0).unwrap();
    let half = T::from(0.5).unwrap();

    let between = rand::distributions::Uniform::from(T::zero()..T::one());

    let mut points = rlst_dynamic_array2!(T, [3, n_points]);
    let mut phi = rlst_dynamic_array2!(T, [n_points, 1]);
    let mut theta = rlst_dynamic_array2!(T, [n_points, 1]);

    for i in 0..n_points {
        phi[[i, 0]] = between.sample(&mut range) * two * pi;
        theta[[i, 0]] = ((between.sample(&mut range) - half) * two).acos();
    }

    for i in 0..n_points {
        points[[0, i]] = half * theta[[i, 0]].sin() * phi[[i, 0]].cos() + half;
        points[[1, i]] = half * theta[[i, 0]].sin() * phi[[i, 0]].sin() + half;
        points[[2, i]] = half * theta[[i, 0]].cos() + half;
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
