//! Helper functions for MPI setting
use std::collections::HashMap;

use itertools::Itertools;
use mpi::{topology::SimpleCommunicator, traits::Equivalence};
use num::Float;
use rlst::RlstScalar;

use crate::{
    fmm::{helpers::single_node::homogenous_kernel_scale, types::SendPtrMut},
    traits::tree::{FmmTreeNode, MultiTree},
    tree::{types::MortonKey, MultiNodeTree},
};

/// Create index pointers for each key at each level of an octree
pub(crate) fn level_index_pointer_multi_node<T>(
    tree: &MultiNodeTree<T, SimpleCommunicator>,
) -> Vec<HashMap<MortonKey<T>, usize>>
where
    T: RlstScalar + Float + Equivalence,
{
    let mut result = vec![HashMap::new(); (tree.total_depth() + 1).try_into().unwrap()];

    for level in 0..=tree.total_depth() {
        if let Some(keys) = tree.keys(level) {
            for (level_idx, key) in keys.iter().enumerate() {
                result[level as usize].insert(*key, level_idx);
            }
        }
    }
    result
}

/// Compute surfaces for each leaf box
pub(crate) fn leaf_surfaces_multi_node<T>(
    tree: &MultiNodeTree<T, SimpleCommunicator>,
    n_coeffs_leaf: usize,
    alpha: T,
    expansion_order_leaf: usize,
) -> Vec<T>
where
    T: RlstScalar + Equivalence + Float + Default,
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

/// Create mutable pointers corresponding to each multipole expansion at each level of an octree
pub(crate) fn level_expansion_pointers_multi_node<T>(
    tree: &MultiNodeTree<T::Real, SimpleCommunicator>,
    n_coeffs: &[usize],
    _n_matvecs: usize,
    expansions: &[T],
) -> Vec<Vec<SendPtrMut<T>>>
where
    T: RlstScalar + Equivalence + Float,
    <T as RlstScalar>::Real: Float + Equivalence,
{
    let mut result = vec![Vec::new(); (tree.total_depth() + 1).try_into().unwrap()];

    let mut level_displacement = 0;
    for level in 0..=tree.total_depth() {
        let mut tmp_multipoles = Vec::new();

        let n_coeffs = if n_coeffs.len() > 1 {
            n_coeffs[level as usize]
        } else {
            n_coeffs[0]
        };

        if let Some(keys) = tree.keys(level) {
            let n_keys_level = keys.len();

            for key_idx in 0..n_keys_level {
                let key_displacement = level_displacement + n_coeffs * key_idx;
                let raw = unsafe { expansions.as_ptr().add(key_displacement) as *mut T };
                tmp_multipoles.push(SendPtrMut { raw })
            }

            result[level as usize] = tmp_multipoles;
            level_displacement += n_keys_level * n_coeffs;
        }
    }

    result
}

/// Create mutable pointers for leaf expansions in a tree
pub(crate) fn leaf_expansion_pointers_multi_node<T>(
    tree: &MultiNodeTree<T::Real, SimpleCommunicator>,
    n_coeffs: &[usize],
    _n_matvecs: usize,
    expansions: &[T],
) -> Vec<SendPtrMut<T>>
where
    T: RlstScalar + Equivalence + Float,
    <T as RlstScalar>::Real: RlstScalar + Equivalence + Float,
{
    let n_leaves = tree.n_leaves().unwrap();
    let mut result = vec![SendPtrMut::default(); n_leaves];

    let iterator = if n_coeffs.len() > 1 {
        (0..tree.total_depth()).zip(n_coeffs.to_vec()).collect_vec()
    } else {
        (0..tree.total_depth())
            .zip(vec![*n_coeffs.last().unwrap(); tree.total_depth() as usize])
            .collect_vec()
    };

    let level_displacement = iterator.iter().fold(0usize, |acc, &(level, ncoeffs)| {
        if let Some(n_keys) = tree.n_keys(level) {
            acc + n_keys * ncoeffs
        } else {
            acc
        }
    });

    let &n_coeffs_leaf = n_coeffs.last().unwrap();
    for (leaf_idx, result_i) in result.iter_mut().enumerate().take(n_leaves) {
        let key_displacement = level_displacement + (leaf_idx * n_coeffs_leaf);
        let raw = unsafe { expansions.as_ptr().add(key_displacement) as *mut T };
        *result_i = SendPtrMut { raw };
    }

    result
}

/// Create mutable pointers for potentials in a tree
pub(crate) fn potential_pointers_multi_node<T>(
    tree: &MultiNodeTree<T::Real, SimpleCommunicator>,
    kernel_eval_size: usize,
    potentials: &[T],
) -> Vec<SendPtrMut<T>>
where
    T: RlstScalar + Equivalence + Float,
    <T as RlstScalar>::Real: RlstScalar + Equivalence + Float,
{
    let n_leaves = tree.n_leaves().unwrap();
    let mut result = vec![SendPtrMut::default(); n_leaves];

    let mut raw_pointer = potentials.as_ptr() as *mut T;

    for (i, leaf) in tree.all_leaves().unwrap().iter().enumerate() {
        let n_evals;

        if let Some(n_points) = tree.n_coordinates(leaf) {
            n_evals = n_points * kernel_eval_size;
        } else {
            n_evals = 0;
        }

        result[i] = SendPtrMut { raw: raw_pointer };

        // Update raw pointer with number of points at this leaf
        raw_pointer = unsafe { raw_pointer.add(n_evals) };
    }

    result
}

/// Create an index pointer for the coordinates in a source and a target tree
/// between the local indices for each leaf and their associated charges
pub fn coordinate_index_pointer_multi_node<T>(
    tree: &MultiNodeTree<T, SimpleCommunicator>,
) -> Vec<(usize, usize)>
where
    T: RlstScalar + Equivalence + Float,
    <T as RlstScalar>::Real: RlstScalar + Equivalence + Float,
{
    let mut index_pointer = 0;

    let mut result = Vec::new();

    if let Some(n_leaves) = tree.n_leaves() {
        result = vec![(0usize, 0usize); n_leaves];

        for (i, leaf) in tree.all_leaves().unwrap().iter().enumerate() {
            let n_points = tree.n_coordinates(leaf).unwrap_or_default();

            // Update charge index pointer
            result[i] = (index_pointer, index_pointer + n_points);
            index_pointer += n_points;
        }
    }

    result
}
