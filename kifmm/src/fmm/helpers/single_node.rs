//! Helper Functions
use std::collections::HashMap;

use itertools::Itertools;
use num::traits::Float;
use rlst::{
    rlst_dynamic_array3, Array, BaseArray, RandomAccessByRef, RandomAccessMut, RlstScalar, Shape,
    VectorContainer,
};

use crate::{
    fmm::types::SendPtrMut,
    traits::tree::{FmmTreeNode, SingleTree},
    tree::types::{MortonKey, SingleNodeTree},
};

/// Number of coefficients for multipole and local expansions for the kernel independent FMM
/// for a given expansion order. Coefficients correspond to points on the equivalent surface.
///
/// # Arguments
/// * `expansion_order` - Expansion order of the FMM
pub fn ncoeffs_kifmm(expansion_order: usize) -> usize {
    6 * (expansion_order - 1).pow(2) + 2
}

/// Euclidean algorithm to find greatest divisor of `n` less than or equal to `max`
///
/// # Arguments
/// * `max` - The maximum chunk size
pub fn chunk_size(n: usize, max: usize) -> usize {
    let max_divisor = max;
    for divisor in (1..=max_divisor).rev() {
        if n % divisor == 0 {
            return divisor;
        }
    }
    1 // If no divisor is found greater than 1, return 1 as the GCD
}

/// Scaling to apply to homogenous scale invariant kernels at a given octree level.
///
/// # Arguments
/// * `level` - The octree level
pub fn homogenous_kernel_scale<T: RlstScalar>(level: u64) -> T {
    let numerator = T::from(1).unwrap();
    let denominator = T::from(2.).unwrap();
    let power = T::from(level).unwrap().re();
    let denominator = <T as RlstScalar>::powf(denominator, power);
    numerator / denominator
}

/// Scaling to apply to M2L operators calculated using homogenous scale invariant kernels at a given octree level.
///
/// # Arguments
/// * `level` - The octree level
pub fn m2l_scale<T: RlstScalar>(level: u64) -> Result<T, std::io::Error> {
    if level < 2 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "M2L only performed on level 2 and below",
        ));
    }

    if level == 2 {
        Ok(T::from(1. / 2.).unwrap())
    } else {
        let two = T::from(2.0).unwrap();
        Ok(<T as RlstScalar>::powf(
            two,
            T::from(level - 3).unwrap().re(),
        ))
    }
}

/// Compute the scaling for each leaf box in a tree
///
/// # Arguments
/// * `tree`- Single node tree
/// * `n_coeffs`- Number of interpolation points on leaf box
pub fn leaf_scales_single_node<T>(tree: &SingleNodeTree<T::Real>, n_coeffs_leaf: usize) -> Vec<T>
where
    T: RlstScalar + Default,
{
    let mut result = vec![T::default(); tree.n_leaves().unwrap() * n_coeffs_leaf];

    for (i, leaf) in tree.all_leaves().unwrap().iter().enumerate() {
        // Assign scales
        let l = i * n_coeffs_leaf;
        let r = l + n_coeffs_leaf;

        result[l..r]
            .copy_from_slice(vec![homogenous_kernel_scale(leaf.level()); n_coeffs_leaf].as_slice());
    }
    result
}

/// Compute the surfaces for each leaf box
///
/// # Arguments
/// * `tree`- Single node tree
/// * `n_coeffs`- Number of interpolation points on leaf box
/// * `alpha` - The multiplier being used to modify the diameter of the surface grid uniformly along each coordinate axis.
/// * `expansion_order` - Expansion order of the FMM
pub fn leaf_surfaces_single_node<T>(
    tree: &SingleNodeTree<T>,
    n_coeffs_leaf: usize,
    alpha: T,
    expansion_order_leaf: usize,
) -> Vec<T>
where
    T: RlstScalar + Float + Default,
{
    let dim = 3;
    let n_keys = tree.n_leaves().unwrap();
    let mut result = vec![T::default(); n_coeffs_leaf * dim * n_keys];

    for (i, leaf) in tree.all_leaves().unwrap().iter().enumerate() {
        let l = i * n_coeffs_leaf * dim;
        let r = l + n_coeffs_leaf * dim;
        let surface = leaf.surface_grid(expansion_order_leaf, tree.domain(), alpha);

        result[l..r].copy_from_slice(&surface);
    }

    result
}

/// Create an index pointer for the coordinates in a source and a target tree
/// between the local indices for each leaf and their associated charges
pub fn coordinate_index_pointer_single_node<T>(tree: &SingleNodeTree<T>) -> Vec<(usize, usize)>
where
    T: RlstScalar + Float,
{
    let mut index_pointer = 0;

    let mut result = vec![(0usize, 0usize); tree.n_leaves().unwrap()];

    for (i, leaf) in tree.all_leaves().unwrap().iter().enumerate() {
        let n_points = tree.n_coordinates(leaf).unwrap_or_default();

        // Update charge index pointer
        result[i] = (index_pointer, index_pointer + n_points);
        index_pointer += n_points;
    }

    result
}

/// Create index pointers for each key at each level of an octree
pub fn level_index_pointer_single_node<T>(
    tree: &SingleNodeTree<T>,
) -> Vec<HashMap<MortonKey<T>, usize>>
where
    T: RlstScalar + Float,
{
    let mut result = vec![HashMap::new(); (tree.depth() + 1).try_into().unwrap()];

    for level in 0..=tree.depth() {
        if let Some(keys) = tree.keys(level) {
            for (level_idx, key) in keys.iter().enumerate() {
                result[level as usize].insert(*key, level_idx);
            }
        }
    }
    result
}

/// Create mutable pointers corresponding to each multipole expansion at each level of an octree
pub fn level_expansion_pointers_single_node<T>(
    tree: &SingleNodeTree<T::Real>,
    n_coeffs: &[usize],
    n_matvecs: usize,
    expansions: &[T],
) -> Vec<Vec<Vec<SendPtrMut<T>>>>
where
    T: RlstScalar,
{
    let mut result = vec![Vec::new(); (tree.depth() + 1).try_into().unwrap()];

    let mut level_displacement = 0;
    for level in 0..=tree.depth() {
        let mut tmp_multipoles = Vec::new();

        let n_coeffs = if n_coeffs.len() > 1 {
            n_coeffs[level as usize]
        } else {
            n_coeffs[0]
        };

        if let Some(keys) = tree.keys(level) {
            let n_keys_level = keys.len();

            for key_idx in 0..n_keys_level {
                let key_displacement = level_displacement + n_coeffs * n_matvecs * key_idx;
                let mut key_multipoles = Vec::new();
                for eval_idx in 0..n_matvecs {
                    let eval_displacement = n_coeffs * eval_idx;
                    let raw = unsafe {
                        expansions
                            .as_ptr()
                            .add(key_displacement + eval_displacement)
                            as *mut T
                    };
                    key_multipoles.push(SendPtrMut { raw });
                }
                tmp_multipoles.push(key_multipoles)
            }
            result[level as usize] = tmp_multipoles;

            level_displacement += n_keys_level * n_coeffs * n_matvecs;
        }
    }

    result
}

/// Create mutable pointers for leaf expansions in a tree
pub fn leaf_expansion_pointers_single_node<T>(
    tree: &SingleNodeTree<T::Real>,
    n_coeffs: &[usize],
    n_matvecs: usize,
    expansions: &[T],
) -> Vec<Vec<SendPtrMut<T>>>
where
    T: RlstScalar,
{
    let n_leaves = tree.n_leaves().unwrap();

    let mut result = vec![Vec::new(); n_leaves];

    let iterator = if n_coeffs.len() > 1 {
        (0..tree.depth()).zip(n_coeffs.to_vec()).collect_vec()
    } else {
        (0..tree.depth())
            .zip(vec![*n_coeffs.last().unwrap(); tree.depth() as usize])
            .collect_vec()
    };

    let level_displacement = iterator.iter().fold(0usize, |acc, &(level, n_coeffs)| {
        if let Some(n_keys) = tree.n_keys(level) {
            acc + n_keys * n_coeffs * n_matvecs
        } else {
            acc
        }
    });

    let &n_coeffs_leaf = n_coeffs.last().unwrap();

    for (leaf_idx, result_i) in result.iter_mut().enumerate().take(n_leaves) {
        let key_displacement = level_displacement + (leaf_idx * n_coeffs_leaf) * n_matvecs;

        for eval_idx in 0..n_matvecs {
            let eval_displacement = n_coeffs_leaf * eval_idx;
            let raw = unsafe {
                expansions
                    .as_ptr()
                    .add(eval_displacement + key_displacement) as *mut T
            };

            result_i.push(SendPtrMut { raw });
        }
    }

    result
}

/// Create mutable pointers for potentials in a tree
pub fn potential_pointers_single_node<T>(
    tree: &SingleNodeTree<T::Real>,
    n_matvecs: usize,
    kernel_eval_size: usize,
    potentials: &[T],
) -> Vec<SendPtrMut<T>>
where
    T: RlstScalar,
{
    let n_points = tree.n_coordinates_tot().unwrap();
    let n_leaves = tree.n_leaves().unwrap();
    let mut result = vec![SendPtrMut::default(); n_leaves * n_matvecs];

    let mut raw_pointers = Vec::new();
    for eval_idx in 0..n_matvecs {
        let ptr = unsafe {
            potentials
                .as_ptr()
                .add(eval_idx * n_points * kernel_eval_size) as *mut T
        };
        raw_pointers.push(ptr)
    }

    for (i, leaf) in tree.all_leaves().unwrap().iter().enumerate() {
        let n_evals;

        if let Some(n_points) = tree.n_coordinates(leaf) {
            n_evals = n_points * kernel_eval_size;
        } else {
            n_evals = 0;
        }

        for j in 0..n_matvecs {
            result[n_leaves * j + i] = SendPtrMut {
                raw: raw_pointers[j],
            }
        }

        // Update raw pointer with number of points at this leaf
        for ptr in raw_pointers.iter_mut() {
            *ptr = unsafe { ptr.add(n_evals) }
        }
    }

    result
}

/// Map charges to map global indices
pub fn map_charges<T: RlstScalar>(
    global_indices: &[usize],
    charges: &[T],
    n_matvecs: usize,
) -> Vec<T> {
    let ncharges = charges.len() / n_matvecs;
    let mut reordered_charges = vec![T::zero(); charges.len()];

    for eval_idx in 0..n_matvecs {
        let eval_displacement = eval_idx * ncharges;
        for (new_idx, old_idx) in global_indices.iter().enumerate() {
            reordered_charges[new_idx + eval_displacement] = charges[old_idx + eval_displacement];
        }
    }

    reordered_charges
}

/// Pad an Array3D from a given `pad_index` with an amount of zeros specified by `pad_size` to the right of each axis.
///
/// # Arguments
/// * `arr` - An array to be padded.
/// * `pad_size` - The amount of padding to be added along each axis.
/// * `pad_index` - The position in the array to start the padding from.
pub fn pad3<T>(
    arr: &Array<T, BaseArray<T, VectorContainer<T>, 3>, 3>,
    pad_size: (usize, usize, usize),
    pad_index: (usize, usize, usize),
) -> Array<T, BaseArray<T, VectorContainer<T>, 3>, 3>
where
    T: Clone + Copy + RlstScalar,
{
    let [m, n, o] = arr.shape();

    let (x, y, z) = pad_index;
    let (p, q, r) = pad_size;

    // Check that there is enough space for pad
    assert!(x + p <= m + p && y + q <= n + q && z + r <= o + r);

    let mut padded = rlst_dynamic_array3!(T, [p + m, q + n, r + o]);

    for i in 0..m {
        for j in 0..n {
            for k in 0..o {
                *padded.get_mut([x + i, y + j, z + k]).unwrap() = *arr.get([i, j, k]).unwrap();
            }
        }
    }

    padded
}

/// Flip an Array3D along each axis, returns a new array.
///
/// # Arguments
/// * `arr` - An array to be flipped.
pub fn flip3<T>(
    arr: &Array<T, BaseArray<T, VectorContainer<T>, 3>, 3>,
) -> Array<T, BaseArray<T, VectorContainer<T>, 3>, 3>
where
    T: Clone + Copy + RlstScalar,
{
    let mut flipped = rlst_dynamic_array3!(T, arr.shape());

    let [m, n, o] = arr.shape();

    for i in 0..m {
        for j in 0..n {
            for k in 0..o {
                *flipped.get_mut([i, j, k]).unwrap() =
                    *arr.get([m - i - 1, n - j - 1, o - k - 1]).unwrap();
            }
        }
    }

    flipped
}

#[cfg(test)]
mod test {

    use super::*;
    use approx::*;

    #[test]
    fn test_flip3() {
        let n = 2;
        let mut arr = rlst_dynamic_array3!(f64, [n, n, n]);
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    *arr.get_mut([i, j, k]).unwrap() = (i + j * n + k * n * n) as f64;
                }
            }
        }
        let result = flip3(&arr);
        assert_relative_eq!(*result.get([0, 0, 0]).unwrap(), 7.0);
        assert_relative_eq!(*result.get([0, 0, 1]).unwrap(), 3.0);
        assert_relative_eq!(*result.get([0, 1, 0]).unwrap(), 5.0);
        assert_relative_eq!(*result.get([0, 1, 1]).unwrap(), 1.0);
        assert_relative_eq!(*result.get([1, 0, 0]).unwrap(), 6.0);
        assert_relative_eq!(*result.get([1, 0, 1]).unwrap(), 2.0);
        assert_relative_eq!(*result.get([1, 1, 0]).unwrap(), 4.0);
        assert_relative_eq!(*result.get([1, 1, 1]).unwrap(), 0.0);
    }

    #[test]
    fn test_pad3() {
        let dim = 3;
        // Initialise input data
        let mut input = rlst_dynamic_array3!(f64, [dim, dim, dim]);
        for i in 0..dim {
            for j in 0..dim {
                for k in 0..dim {
                    *input.get_mut([i, j, k]).unwrap() = (i + j * dim + k * dim * dim + 1) as f64
                }
            }
        }

        // Test padding at edge of each axis
        let pad_size = (2, 3, 4);
        let pad_index = (0, 0, 0);
        let padded = pad3(&input, pad_size, pad_index);

        let [m, n, o] = padded.shape();

        // Check dimension
        assert_eq!(m, dim + pad_size.0);
        assert_eq!(n, dim + pad_size.1);
        assert_eq!(o, dim + pad_size.2);

        // Check that padding has been correctly applied
        for i in dim..m {
            for j in dim..n {
                for k in dim..o {
                    assert_eq!(*padded.get([i, j, k]).unwrap(), 0f64)
                }
            }
        }

        for i in 0..dim {
            for j in 0..dim {
                for k in 0..dim {
                    assert_eq!(
                        *padded.get([i, j, k]).unwrap(),
                        *input.get([i, j, k]).unwrap()
                    )
                }
            }
        }

        // Test padding at the start of each axis
        let pad_index = (2, 2, 2);

        let padded = pad3(&input, pad_size, pad_index);

        // Check that padding has been correctly applied
        for i in 0..pad_index.0 {
            for j in 0..pad_index.1 {
                for k in 0..pad_index.2 {
                    assert_eq!(*padded.get([i, j, k]).unwrap(), 0f64)
                }
            }
        }

        for i in 0..dim {
            for j in 0..dim {
                for k in 0..dim {
                    assert_eq!(
                        *padded
                            .get([i + pad_index.0, j + pad_index.1, k + pad_index.2])
                            .unwrap(),
                        *input.get([i, j, k]).unwrap()
                    );
                }
            }
        }
    }
}
