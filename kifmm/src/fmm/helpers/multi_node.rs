//! Helper functions for MPI setting
use std::collections::HashMap;

use bytemuck::{cast_slice, Pod};
use itertools::Itertools;
use mpi::{
    datatype::PartitionMut,
    topology::SimpleCommunicator,
    traits::{Communicator, CommunicatorCollectives, Equivalence},
    Count,
};
use num::Float;
use rlst::{
    rlst_dynamic_array2, Array, BaseArray, RawAccess, RawAccessMut, RlstScalar, Shape,
    VectorContainer,
};

use crate::{
    fmm::{
        constants::LEN_BYTES,
        types::{BlasMetadataSaRcmp, FftMetadata, SendPtrMut},
    },
    traits::tree::{FmmTreeNode, MultiTree},
    tree::{types::MortonKey, MultiNodeTree},
};

pub(crate) fn serialise_vec<T: Pod>(input: &[T]) -> Vec<u8> {
    let mut buffer = Vec::new();
    buffer.extend_from_slice(&(input.len() as u64).to_le_bytes());

    if !input.is_empty() {
        buffer.extend_from_slice(cast_slice(input));
    }

    buffer
}

pub(crate) fn deserialise_vec<T: Pod>(input: &[u8]) -> (&[T], &[u8]) {
    let (len_bytes, rest) = input.split_at(LEN_BYTES);
    let len = u64::from_le_bytes(len_bytes.try_into().unwrap()) as usize;
    let total_bytes = len * std::mem::size_of::<T>();
    let (data_bytes, remaining) = rest.split_at(total_bytes);
    let data = cast_slice::<u8, T>(data_bytes);
    (data, remaining)
}

pub(crate) fn serialise_nested_vec<T: Pod>(input: &[Vec<T>]) -> Vec<u8> {
    let mut buffer = Vec::new();
    buffer.extend_from_slice(&(input.len() as u64).to_le_bytes());

    if !input.is_empty() {
        for vec in input.iter() {
            buffer.extend_from_slice(&serialise_vec(vec));
        }
    }

    buffer
}

pub(crate) fn deserialise_nested_vec<T: Pod>(input: &[u8]) -> (Vec<Vec<T>>, &[u8]) {
    let (len_bytes, mut rest) = input.split_at(LEN_BYTES);
    let len = u64::from_le_bytes(len_bytes.try_into().unwrap()) as usize;
    let mut buffer = Vec::new();
    if len > 0 {
        for _ in 0..len {
            let (t1, t2) = deserialise_vec::<T>(rest);
            buffer.push(t1.to_vec());
            rest = t2;
        }
    }

    (buffer, rest)
}

pub(crate) fn serialise_array<T: RlstScalar + Pod>(
    input: &Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>,
) -> Vec<u8> {
    let mut buffer = Vec::new();
    let shape = input.shape();
    let rows = &(shape[0] as u64).to_le_bytes();
    let cols = &(shape[1] as u64).to_le_bytes();
    buffer.extend_from_slice(rows);
    buffer.extend_from_slice(cols);

    if !input.is_empty() {
        buffer.extend_from_slice(cast_slice(input.data()));
    }

    buffer
}

#[allow(clippy::type_complexity)]
pub(crate) fn deserialise_array<T: RlstScalar + Pod>(
    input: &[u8],
) -> (Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>, &[u8]) {
    let (rows_bytes, rest) = input.split_at(LEN_BYTES);
    let rows = u64::from_le_bytes(rows_bytes.try_into().unwrap()) as usize;
    let (cols_bytes, rest) = rest.split_at(LEN_BYTES);
    let cols = u64::from_le_bytes(cols_bytes.try_into().unwrap()) as usize;

    let expected_size = rows * cols;
    let total_bytes = std::mem::size_of::<T>() * expected_size;

    let (data_bytes, remaining) = rest.split_at(total_bytes);
    let data = cast_slice::<u8, T>(data_bytes);

    let mut array = rlst_dynamic_array2!(T, [rows, cols]);
    array.data_mut().copy_from_slice(data);

    (array, remaining)
}

pub(crate) fn serialise_nested_array<T: RlstScalar + Pod>(
    input: &[Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>],
) -> Vec<u8> {
    let mut buffer = Vec::new();
    buffer.extend_from_slice(&(input.len() as u64).to_le_bytes());

    if !input.is_empty() {
        for vec in input.iter() {
            buffer.extend_from_slice(&serialise_array(vec));
        }
    }
    buffer
}

#[allow(clippy::type_complexity)]
pub(crate) fn deserialise_nested_array<T: RlstScalar + Pod>(
    input: &[u8],
) -> (Vec<Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>>, &[u8]) {
    let (len_bytes, mut rest) = input.split_at(LEN_BYTES);
    let len = u64::from_le_bytes(len_bytes.try_into().unwrap()) as usize;
    let mut buffer = Vec::new();
    if len > 0 {
        for _ in 0..len {
            let (t1, t2) = deserialise_array::<T>(rest);
            buffer.push(t1);
            rest = t2;
        }
    }

    (buffer, rest)
}

pub(crate) fn serialise_blas_metadata_sarcmp<T: RlstScalar + Pod>(
    input: &BlasMetadataSaRcmp<T>,
) -> Vec<u8> {
    let mut buffer: Vec<u8> = Vec::new();
    buffer.extend_from_slice(&serialise_array(&input.u));
    buffer.extend_from_slice(&serialise_array(&input.st));
    buffer.extend_from_slice(&serialise_nested_array(&input.c_u));
    buffer.extend_from_slice(&serialise_nested_array(&input.c_vt));
    buffer
}

pub(crate) fn deserialise_blas_metadata_sarcmp<T: RlstScalar + Pod>(
    input: &[u8],
) -> (BlasMetadataSaRcmp<T>, &[u8]) {
    let (u, rest) = deserialise_array(input);
    let (st, rest) = deserialise_array(rest);
    let (c_u, rest) = deserialise_nested_array(rest);
    let (c_vt, rest) = deserialise_nested_array(rest);
    (BlasMetadataSaRcmp { u, st, c_u, c_vt }, rest)
}

pub(crate) fn serialise_vec_blas_metadata_sarcmp<T: RlstScalar + Pod>(
    input: &Vec<BlasMetadataSaRcmp<T>>,
) -> Vec<u8> {
    let len = input.len() as u64;
    let mut buffer = Vec::new();
    buffer.extend_from_slice(&len.to_le_bytes());

    for data in input {
        buffer.extend_from_slice(&serialise_blas_metadata_sarcmp(data));
    }

    buffer
}

pub(crate) fn deserialise_vec_blas_metadata_sarcmp<T: RlstScalar + Pod>(
    input: &[u8],
) -> (Vec<BlasMetadataSaRcmp<T>>, &[u8]) {
    let (len_bytes, mut rest) = input.split_at(LEN_BYTES);
    let len = u64::from_le_bytes(len_bytes.try_into().unwrap()) as usize;

    let mut buffer = Vec::new();

    for _ in 0..len {
        let (data, t1) = deserialise_blas_metadata_sarcmp::<T>(rest);
        rest = t1;
        buffer.push(data);
    }

    (buffer, rest)
}

pub(crate) fn serialise_fft_metadata<T: RlstScalar + Pod>(input: &FftMetadata<T>) -> Vec<u8> {
    let mut buffer = Vec::new();
    buffer.extend_from_slice(&serialise_nested_vec(&input.kernel_data));
    buffer.extend_from_slice(&serialise_nested_vec(&input.kernel_data_f));
    buffer
}

pub(crate) fn deserialise_fft_metadata<T: RlstScalar + Pod>(
    input: &[u8],
) -> (FftMetadata<T>, &[u8]) {
    let (kernel_data, rest) = deserialise_nested_vec(input);
    let (kernel_data_f, rest) = deserialise_nested_vec(rest);

    (
        FftMetadata {
            kernel_data,
            kernel_data_f,
        },
        rest,
    )
}

pub(crate) fn serialise_vec_fft_metadata<T: RlstScalar + Pod>(
    input: &Vec<FftMetadata<T>>,
) -> Vec<u8> {
    let len = input.len() as u64;
    let mut buffer = Vec::new();
    buffer.extend_from_slice(&len.to_le_bytes());

    for data in input {
        buffer.extend_from_slice(&serialise_fft_metadata(data));
    }

    buffer
}

pub(crate) fn deserialise_vec_fft_metadata<T: RlstScalar + Pod>(
    input: &[u8],
) -> (Vec<FftMetadata<T>>, &[u8]) {
    let (len_bytes, mut rest) = input.split_at(LEN_BYTES);
    let len = u64::from_le_bytes(len_bytes.try_into().unwrap()) as usize;

    let mut buffer = Vec::new();

    for _ in 0..len {
        let (data, t1) = deserialise_fft_metadata::<T>(rest);
        rest = t1;
        buffer.push(data);
    }

    (buffer, rest)
}

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

// Communicate all to all with serialised data from each process
// of variable length
pub(crate) fn all_gather_v_serialised(
    input_r: &[u8],
    communicator: &SimpleCommunicator,
) -> Vec<u8> {
    let size = communicator.size();
    let mut counts = vec![0i32; size as usize];
    let input_r_count = input_r.len() as i32;
    communicator.all_gather_into(&input_r_count, &mut counts);

    let displacements = counts
        .iter()
        .scan(0, |acc, &x| {
            let tmp = *acc;
            *acc += x;
            Some(tmp)
        })
        .collect_vec();

    let buffer_size = counts.iter().sum::<i32>() as usize;
    let mut output = vec![0u8; buffer_size];

    // Communicate data
    {
        let mut partition = PartitionMut::new(&mut output, &counts[..], &displacements[..]);

        communicator.all_gather_varcount_into(input_r, &mut partition);
    }

    output
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

/// Calculate load for precomputation based on a block distribution strategy
pub(crate) fn calculate_precomputation_load(
    n_precomputations: i32,
    size: i32,
) -> Option<(Vec<Count>, Vec<Count>)> {
    if n_precomputations > 0 {
        let mut counts;

        if n_precomputations > 1 {
            // Distributed pre-computation
            let q = n_precomputations / size; // Base number of calculations per processor
            let r = n_precomputations % size; // Extra calculations to distribute evenly among ranks

            // Block distribution strategy
            counts = (0..size)
                .map(|i| if i < r { q + 1 } else { q })
                .collect_vec();
        } else {
            // If only have one pre-computation, carry out on a single rank (root rank)
            counts = vec![0; size as usize];
            counts[0] = n_precomputations;
        }

        let mut curr = 0;
        let mut displacements = Vec::new();
        for &count in counts.iter() {
            displacements.push(curr);
            curr += count;
        }

        Some((counts, displacements))
    } else {
        None
    }
}
