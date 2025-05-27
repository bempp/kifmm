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
    rlst_dynamic_array2, rlst_dynamic_array3, Array, BaseArray, RawAccess, RawAccessMut,
    RlstScalar, Shape, VectorContainer,
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

pub(crate) fn serialise_array_2x2<T: RlstScalar + Pod>(
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

pub(crate) fn serialise_array_3x3<T: RlstScalar + Pod>(
    input: &Array<T, BaseArray<T, VectorContainer<T>, 3>, 3>,
) -> Vec<u8> {
    let mut buffer = Vec::new();
    let shape = input.shape();
    let rows = &(shape[0] as u64).to_le_bytes();
    let cols = &(shape[1] as u64).to_le_bytes();
    let depth = &(shape[2] as u64).to_le_bytes();

    buffer.extend_from_slice(rows);
    buffer.extend_from_slice(cols);
    buffer.extend_from_slice(depth);

    if !input.is_empty() {
        buffer.extend_from_slice(cast_slice(input.data()));
    }

    buffer
}

#[allow(clippy::type_complexity)]
pub(crate) fn deserialise_array_2x2<T: RlstScalar + Pod>(
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

#[allow(clippy::type_complexity)]
pub(crate) fn deserialise_array_3x3<T: RlstScalar + Pod>(
    input: &[u8],
) -> (Array<T, BaseArray<T, VectorContainer<T>, 3>, 3>, &[u8]) {
    let (rows_bytes, rest) = input.split_at(LEN_BYTES);
    let rows = u64::from_le_bytes(rows_bytes.try_into().unwrap()) as usize;
    let (cols_bytes, rest) = rest.split_at(LEN_BYTES);
    let cols = u64::from_le_bytes(cols_bytes.try_into().unwrap()) as usize;
    let (depth_bytes, rest) = rest.split_at(LEN_BYTES);
    let depth = u64::from_le_bytes(depth_bytes.try_into().unwrap()) as usize;

    let expected_size = rows * cols * depth;
    let total_bytes = std::mem::size_of::<T>() * expected_size;

    let (data_bytes, remaining) = rest.split_at(total_bytes);
    let data = cast_slice::<u8, T>(data_bytes);

    let mut array = rlst_dynamic_array3!(T, [rows, cols, depth]);
    array.data_mut().copy_from_slice(data);

    (array, remaining)
}

pub(crate) fn serialise_nested_array_2x2<T: RlstScalar + Pod>(
    input: &[Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>],
) -> Vec<u8> {
    let mut buffer = Vec::new();
    buffer.extend_from_slice(&(input.len() as u64).to_le_bytes());

    if !input.is_empty() {
        for vec in input.iter() {
            buffer.extend_from_slice(&serialise_array_2x2(vec));
        }
    }
    buffer
}

pub(crate) fn serialise_nested_array_3x3<T: RlstScalar + Pod>(
    input: &[Array<T, BaseArray<T, VectorContainer<T>, 3>, 3>],
) -> Vec<u8> {
    let mut buffer = Vec::new();
    buffer.extend_from_slice(&(input.len() as u64).to_le_bytes());

    if !input.is_empty() {
        for vec in input.iter() {
            buffer.extend_from_slice(&serialise_array_3x3(vec));
        }
    }
    buffer
}

#[allow(clippy::type_complexity)]
pub(crate) fn deserialise_nested_array_2x2<T: RlstScalar + Pod>(
    input: &[u8],
) -> (Vec<Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>>, &[u8]) {
    let (len_bytes, mut rest) = input.split_at(LEN_BYTES);
    let len = u64::from_le_bytes(len_bytes.try_into().unwrap()) as usize;
    let mut buffer = Vec::new();
    if len > 0 {
        for _ in 0..len {
            let (t1, t2) = deserialise_array_2x2::<T>(rest);
            buffer.push(t1);
            rest = t2;
        }
    }

    (buffer, rest)
}

#[allow(clippy::type_complexity)]
pub(crate) fn deserialise_nested_array_3x3<T: RlstScalar + Pod>(
    input: &[u8],
) -> (Vec<Array<T, BaseArray<T, VectorContainer<T>, 3>, 3>>, &[u8]) {
    let (len_bytes, mut rest) = input.split_at(LEN_BYTES);
    let len = u64::from_le_bytes(len_bytes.try_into().unwrap()) as usize;
    let mut buffer = Vec::new();
    if len > 0 {
        for _ in 0..len {
            let (t1, t2) = deserialise_array_3x3::<T>(rest);
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
    buffer.extend_from_slice(&serialise_array_2x2(&input.u));
    buffer.extend_from_slice(&serialise_array_2x2(&input.st));
    buffer.extend_from_slice(&serialise_nested_array_2x2(&input.c_u));
    buffer.extend_from_slice(&serialise_nested_array_2x2(&input.c_vt));
    buffer
}

pub(crate) fn deserialise_blas_metadata_sarcmp<T: RlstScalar + Pod>(
    input: &[u8],
) -> (BlasMetadataSaRcmp<T>, &[u8]) {
    let (u, rest) = deserialise_array_2x2(input);
    let (st, rest) = deserialise_array_2x2(rest);
    let (c_u, rest) = deserialise_nested_array_2x2(rest);
    let (c_vt, rest) = deserialise_nested_array_2x2(rest);
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
    T: RlstScalar + Equivalence,
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
    T: RlstScalar + Equivalence,
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
    T: RlstScalar + Equivalence,
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

#[cfg(test)]
mod test {
    use super::*;
    use num::Complex;
    use rand::distributions::{Distribution, Uniform};
    use rlst::{dense::tools::RandScalar, DefaultIteratorMut};

    fn test_array_real<T: RlstScalar + PartialOrd + RandScalar>()
    where
        T: rand::distributions::uniform::SampleUniform,
    {
        let m = 5;
        let n = 4;
        let mut expected = rlst_dynamic_array2!(T, [m, n]);
        let mut rng = rand::thread_rng();
        let between = Uniform::try_from(T::from(0.).unwrap()..T::from(1.0).unwrap()).unwrap();
        expected
            .iter_mut()
            .for_each(|e| *e = between.sample(&mut rng));

        let serialised = serialise_array_2x2(&expected);
        let found = deserialise_array_2x2::<T>(&serialised).0;

        (found.data().iter())
            .zip(expected.data().iter())
            .for_each(|(&e, &f)| {
                assert!(RlstScalar::abs(e - f) <= T::from(1e-6).unwrap().re());
            });
    }

    fn test_array_complex<T: RlstScalar<Real = T> + RandScalar + PartialOrd>()
    where
        T: rand::distributions::uniform::SampleUniform,
        Complex<T>: RlstScalar,
    {
        let m = 5;
        let n = 4;
        let mut expected = rlst_dynamic_array2!(Complex<T>, [m, n]);

        let mut rng = rand::thread_rng();
        let between = Uniform::try_from(T::from(0.).unwrap()..T::from(1.0).unwrap()).unwrap();

        expected.iter_mut().for_each(|e| {
            *e = Complex {
                re: between.sample(&mut rng),
                im: between.sample(&mut rng),
            }
        });

        let serialised = serialise_array_2x2(&expected);
        let found = deserialise_array_2x2::<Complex<T>>(&serialised).0;

        (found.data().iter())
            .zip(expected.data().iter())
            .for_each(|(&e, &f)| {
                let diff = RlstScalar::powf(
                    T::from(e.re()).unwrap() - T::from(f.re()).unwrap(),
                    T::from(2.0).unwrap(),
                ) + RlstScalar::powf(
                    T::from(e.im()).unwrap() - T::from(f.im()).unwrap(),
                    T::from(2.0).unwrap(),
                );
                let err = RlstScalar::powf(diff, T::from(0.5).unwrap().re());
                println!("HERE {:?} {:?}", e, f);
                assert!(err <= T::from(1e-6).unwrap().re());
            });
    }

    fn test_array_real_empty<T: RlstScalar + PartialOrd + RandScalar>()
    where
        T: rand::distributions::uniform::SampleUniform,
    {
        let expected = rlst_dynamic_array2!(T, [0, 0]);
        let serialised = serialise_array_2x2(&expected);
        let found = deserialise_array_2x2::<T>(&serialised).0;
        assert!(found.shape()[0] == 0);
        assert!(found.shape()[1] == 0);
        assert!(found.data().len() == 0);
    }

    fn test_array_complex_empty<T: RlstScalar + PartialOrd + RandScalar>()
    where
        T: rand::distributions::uniform::SampleUniform,
        Complex<T>: RlstScalar,
    {
        let expected = rlst_dynamic_array2!(Complex<T>, [0, 0]);
        let serialised = serialise_array_2x2(&expected);
        let found = deserialise_array_2x2::<Complex<T>>(&serialised).0;
        assert!(found.shape()[0] == 0);
        assert!(found.shape()[1] == 0);
        assert!(found.data().len() == 0);
    }

    fn test_vector_real<T: RlstScalar + PartialOrd>() {
        let mut expected = vec![T::from(0.0).unwrap(); 10];
        expected.iter_mut().for_each(|x| *x += T::one());
        let serialised = serialise_vec(&expected);
        let found = deserialise_vec::<T>(&serialised).0;

        (expected.iter()).zip(found).for_each(|(&e, &f)| {
            assert!(RlstScalar::abs(e - f) <= T::from(1e-6).unwrap().re());
        });
    }

    fn test_nested_vector_real<T: RlstScalar + PartialOrd>() {
        let n = 3;
        let mut expected = vec![vec![T::from(0.0).unwrap(); 10]; n];

        for i in 0..n {
            expected[i]
                .iter_mut()
                .for_each(|x| *x += T::from(i as f32).unwrap());
        }

        // Insert an empty vector at the end
        expected.push(Vec::new());

        let serialised = serialise_nested_vec(&expected);
        let (found, _rest) = deserialise_nested_vec::<T>(&serialised);

        for i in 0..n {
            (expected[i].iter())
                .zip(found[i].iter())
                .for_each(|(&e, &f)| {
                    assert!(RlstScalar::abs(e - f) <= T::from(1e-6).unwrap().re());
                });
        }

        assert!(found.len() == n + 1);
        assert!(found.last().unwrap().is_empty());

        // Now test with empty vectors in the middle, at even indices
        let n = 5;
        let mut expected = vec![Vec::new(); n];
        for i in 0..n {
            if n % 2 == 0 {
                expected[i] = vec![T::one() * (T::from(i).unwrap())];
            }
        }

        let serialised = serialise_nested_vec(&expected);
        let (found, _) = deserialise_nested_vec::<T>(&serialised);

        for i in 0..n {
            if n % 2 == 0 {
                (expected[i].iter())
                    .zip(found[i].iter())
                    .for_each(|(&e, &f)| {
                        assert!(RlstScalar::abs(e - f) <= T::from(1e-6).unwrap().re());
                    });
            } else {
                assert!(found[i].is_empty());
            }
        }

        assert!(found.len() == n);
    }

    fn test_vector_complex<T: RlstScalar + Pod>()
    where
        <T as RlstScalar>::Real: PartialOrd,
    {
        let mut expected = vec![T::from(0.0).unwrap(); 10];
        expected.iter_mut().for_each(|x| *x += T::one());
        let serialised = serialise_vec(&expected);
        let found = deserialise_vec::<T>(&serialised).0;

        (expected.iter()).zip(found).for_each(|(&e, f)| {
            let err = RlstScalar::powf(e.re() - f.re(), T::from(2.0).unwrap().re())
                + RlstScalar::powf(e.im() - f.im(), T::from(2.0).unwrap().re());
            let err = RlstScalar::powf(err, T::from(0.5).unwrap().re());
            assert!(err <= T::from(1e-6).unwrap().re());
        });
    }

    fn test_nested_vector_complex<T: RlstScalar + Pod>()
    where
        <T as RlstScalar>::Real: PartialOrd,
    {
        let n = 3;
        let mut expected = vec![vec![T::from(0.0).unwrap(); 10]; n];

        for i in 0..n {
            expected[i]
                .iter_mut()
                .for_each(|x| *x += T::from(i as f32).unwrap());
        }

        // Insert an empty vector at the end
        expected.push(Vec::new());

        let serialised = serialise_nested_vec(&expected);
        let (found, _) = deserialise_nested_vec::<T>(&serialised);

        for i in 0..n {
            (expected[i].iter())
                .zip(found[i].iter())
                .for_each(|(&e, &f)| {
                    assert!(RlstScalar::abs(e - f) <= T::from(1e-6).unwrap().re());
                });
        }

        assert!(found.len() == n + 1);
        assert!(found.last().unwrap().is_empty());

        // Now test with empty vectors in the middle, at even indices
        let n = 5;
        let mut expected = vec![Vec::new(); n];
        for i in 0..n {
            if n % 2 == 0 {
                expected[i] = vec![T::one() * (T::from(i).unwrap())];
            }
        }

        let serialised = serialise_nested_vec(&expected);
        let (found, _) = deserialise_nested_vec::<T>(&serialised);

        for i in 0..n {
            if n % 2 == 0 {
                (expected[i].iter())
                    .zip(found[i].iter())
                    .for_each(|(&e, &f)| {
                        let err = RlstScalar::powf(e.re() - f.re(), T::from(2.0).unwrap().re())
                            + RlstScalar::powf(e.im() - f.im(), T::from(2.0).unwrap().re());
                        let err = RlstScalar::powf(err, T::from(0.5).unwrap().re());
                        assert!(err <= T::from(1e-6).unwrap().re());
                    });
            } else {
                assert!(found[i].is_empty());
            }
        }

        assert!(found.len() == n);
    }

    fn test_empty_vector_real<T: Pod + RlstScalar + PartialOrd>() {
        let expected: Vec<T> = Vec::new();
        let serialised = serialise_vec(&expected);
        let found = deserialise_vec::<T>(&serialised).0;
        assert!(found.is_empty());
    }

    fn test_empty_vector_complex<T: RlstScalar>()
    where
        <T as RlstScalar>::Real: PartialOrd,
    {
        let expected: Vec<T> = Vec::new();
        let serialised = serialise_vec(&expected);
        let found = deserialise_vec::<T>(&serialised).0;
        assert!(found.is_empty());
    }

    fn test_blas_metadata_sarcmp<T: Pod + RlstScalar + PartialOrd>() {
        // test case where all elements are full
        let m = 5;
        let n = 4;
        let mut u = rlst_dynamic_array2!(T, [m, n]);
        u.data_mut().iter_mut().for_each(|e| *e += T::one());

        let mut st = rlst_dynamic_array2!(T, [n, m]);
        st.data_mut()
            .iter_mut()
            .for_each(|e| *e += T::from(2.0).unwrap());

        let mut c_u = Vec::new();
        let mut c_vt = Vec::new();

        for _ in 0..316 {
            let mut tmp = rlst_dynamic_array2!(T, [m, n]);
            tmp.data_mut()
                .iter_mut()
                .for_each(|e| *e += T::from(3.0).unwrap());
            c_u.push(tmp);

            let mut tmp = rlst_dynamic_array2!(T, [m, n]);
            tmp.data_mut()
                .iter_mut()
                .for_each(|e| *e += T::from(4.0).unwrap());
            c_vt.push(tmp);
        }

        let expected = BlasMetadataSaRcmp { u, st, c_u, c_vt };

        let serialised = serialise_blas_metadata_sarcmp(&expected);
        let (found, _) = deserialise_blas_metadata_sarcmp::<T>(&serialised);

        // Test u
        {
            // test shape
            assert!(found.u.shape()[0] == expected.u.shape()[0]);
            assert!(found.u.shape()[1] == expected.u.shape()[1]);

            // test data
            found
                .u
                .data()
                .iter()
                .zip(expected.u.data().iter())
                .for_each(|(&f, &e)| {
                    assert!(RlstScalar::abs(f - e) < T::from(1e-6).unwrap().re());
                });
        }

        // Test st
        {
            // test shape
            assert!(found.st.shape()[0] == expected.st.shape()[0]);
            assert!(found.st.shape()[1] == expected.st.shape()[1]);

            // test data
            found
                .st
                .data()
                .iter()
                .zip(expected.st.data().iter())
                .for_each(|(&f, &e)| {
                    assert!(RlstScalar::abs(f - e) < T::from(1e-6).unwrap().re());
                });
        }

        // test c_u
        {
            // test shape
            assert!(found.c_u.len() == expected.c_u.len());

            // test data
            found
                .c_u
                .iter()
                .zip(expected.c_u.iter())
                .for_each(|(f, e)| {
                    f.data().iter().zip(e.data().iter()).for_each(|(&f, &e)| {
                        assert!(RlstScalar::abs(f - e) < T::from(1e-6).unwrap().re());
                    });
                });
        }

        // test c_vt
        {
            // test shape
            assert!(found.c_vt.len() == expected.c_vt.len());

            // test data
            found
                .c_vt
                .iter()
                .zip(expected.c_vt.iter())
                .for_each(|(f, e)| {
                    f.data().iter().zip(e.data().iter()).for_each(|(&f, &e)| {
                        assert!(RlstScalar::abs(f - e) < T::from(1e-6).unwrap().re());
                    });
                });
        }
    }

    fn test_blas_metadata_sarcmp_empty<T: Pod + RlstScalar + PartialOrd>() {
        // test case where some elements are empty
        let m = 0;
        let n = 0;
        let u = rlst_dynamic_array2!(T, [m, n]);

        let mut st = rlst_dynamic_array2!(T, [n, m]);
        st.data_mut()
            .iter_mut()
            .for_each(|e| *e += T::from(2.0).unwrap());

        let mut c_u = Vec::new();
        let mut c_vt = Vec::new();

        for i in 0..316 {
            if i % 2 == 0 {
                let mut tmp = rlst_dynamic_array2!(T, [m, n]);
                tmp.data_mut()
                    .iter_mut()
                    .for_each(|e| *e += T::from(3.0).unwrap());
                c_u.push(tmp);

                let mut tmp = rlst_dynamic_array2!(T, [m, n]);
                tmp.data_mut()
                    .iter_mut()
                    .for_each(|e| *e += T::from(4.0).unwrap());
                c_vt.push(tmp);
            } else {
                c_u.push(rlst_dynamic_array2!(T, [0, 0]));
                c_vt.push(rlst_dynamic_array2!(T, [0, 0]));
            }
        }

        let expected = BlasMetadataSaRcmp { u, st, c_u, c_vt };

        let serialised = serialise_blas_metadata_sarcmp(&expected);
        let (found, _) = deserialise_blas_metadata_sarcmp::<T>(&serialised);

        // Test u
        {
            // test shape
            assert!(found.u.shape()[0] == expected.u.shape()[0]);
            assert!(found.u.shape()[1] == expected.u.shape()[1]);

            // test data
            assert!(found.u.is_empty() && expected.u.is_empty());
        }

        // Test st
        {
            // test shape
            assert!(found.st.shape()[0] == expected.st.shape()[0]);
            assert!(found.st.shape()[1] == expected.st.shape()[1]);

            // test data
            found
                .st
                .data()
                .iter()
                .zip(expected.st.data().iter())
                .for_each(|(&f, &e)| {
                    assert!(RlstScalar::abs(f - e) < T::from(1e-6).unwrap().re());
                });
        }

        // test c_u
        {
            // test shape
            assert!(found.c_u.len() == expected.c_u.len());

            // test data
            found
                .c_u
                .iter()
                .zip(expected.c_u.iter())
                .enumerate()
                .for_each(|(i, (f, e))| {
                    if i % 2 == 0 {
                        f.data().iter().zip(e.data().iter()).for_each(|(&f, &e)| {
                            assert!(RlstScalar::abs(f - e) < T::from(1e-6).unwrap().re());
                        });
                    } else {
                        assert!(f.is_empty() && e.is_empty());
                    }
                });
        }

        // test c_vt
        {
            // test shape
            assert!(found.c_vt.len() == expected.c_vt.len());

            // test data
            found
                .c_vt
                .iter()
                .zip(expected.c_vt.iter())
                .enumerate()
                .for_each(|(i, (f, e))| {
                    if i % 2 == 0 {
                        f.data().iter().zip(e.data().iter()).for_each(|(&f, &e)| {
                            assert!(RlstScalar::abs(f - e) < T::from(1e-6).unwrap().re());
                        });
                    } else {
                        assert!(f.is_empty() && e.is_empty());
                    }
                });
        }
    }

    fn test_nested_blas_metadata_sarcmp<T: Pod + RlstScalar + PartialOrd>() {
        let e = 10;

        let mut expected = Vec::new();
        for _ in 0..e {
            // test case where some elements are empty
            let m = 0;
            let n = 0;
            let u = rlst_dynamic_array2!(T, [m, n]);

            let mut st = rlst_dynamic_array2!(T, [n, m]);
            st.data_mut()
                .iter_mut()
                .for_each(|e| *e += T::from(2.0).unwrap());

            let mut c_u = Vec::new();
            let mut c_vt = Vec::new();

            for i in 0..316 {
                if i % 2 == 0 {
                    let mut tmp = rlst_dynamic_array2!(T, [m, n]);
                    tmp.data_mut()
                        .iter_mut()
                        .for_each(|e| *e += T::from(3.0).unwrap());
                    c_u.push(tmp);

                    let mut tmp = rlst_dynamic_array2!(T, [m, n]);
                    tmp.data_mut()
                        .iter_mut()
                        .for_each(|e| *e += T::from(4.0).unwrap());
                    c_vt.push(tmp);
                } else {
                    c_u.push(rlst_dynamic_array2!(T, [0, 0]));
                    c_vt.push(rlst_dynamic_array2!(T, [0, 0]));
                }
            }

            expected.push(BlasMetadataSaRcmp { u, st, c_u, c_vt });
        }

        let serialised = serialise_vec_blas_metadata_sarcmp(&expected);
        let (found, _rest) = deserialise_vec_blas_metadata_sarcmp::<T>(&serialised);

        found.iter().zip(expected.iter()).for_each(|(f, e)| {
            // Test u
            {
                // test shape
                assert!(f.u.shape()[0] == e.u.shape()[0]);
                assert!(f.u.shape()[1] == e.u.shape()[1]);

                // test data
                assert!(f.u.is_empty() && e.u.is_empty());
            }

            // Test st
            {
                // test shape
                assert!(f.st.shape()[0] == e.st.shape()[0]);
                assert!(f.st.shape()[1] == e.st.shape()[1]);

                // test data
                f.st.data()
                    .iter()
                    .zip(e.st.data().iter())
                    .for_each(|(&f, &e)| {
                        assert!(RlstScalar::abs(f - e) < T::from(1e-6).unwrap().re());
                    });
            }

            // test c_u
            {
                // test shape
                assert!(f.c_u.len() == e.c_u.len());

                // test data
                f.c_u
                    .iter()
                    .zip(e.c_u.iter())
                    .enumerate()
                    .for_each(|(i, (f, e))| {
                        if i % 2 == 0 {
                            f.data().iter().zip(e.data().iter()).for_each(|(&f, &e)| {
                                assert!(RlstScalar::abs(f - e) < T::from(1e-6).unwrap().re());
                            });
                        } else {
                            assert!(f.is_empty() && e.is_empty());
                        }
                    });
            }

            // test c_vt
            {
                // test shape
                assert!(f.c_vt.len() == e.c_vt.len());

                // test data
                f.c_vt
                    .iter()
                    .zip(e.c_vt.iter())
                    .enumerate()
                    .for_each(|(i, (f, e))| {
                        if i % 2 == 0 {
                            f.data().iter().zip(e.data().iter()).for_each(|(&f, &e)| {
                                assert!(RlstScalar::abs(f - e) < T::from(1e-6).unwrap().re());
                            });
                        } else {
                            assert!(f.is_empty() && e.is_empty());
                        }
                    });
            }
        });
    }

    #[test]
    fn test_serialisation() {
        test_vector_real::<f32>();
        test_vector_real::<f64>();
        test_vector_complex::<Complex<f32>>();
        test_vector_complex::<Complex<f64>>();

        test_empty_vector_real::<f32>();
        test_empty_vector_real::<f64>();
        test_empty_vector_complex::<Complex<f32>>();
        test_empty_vector_complex::<Complex<f64>>();
    }

    #[test]
    fn test_nested_serialisation() {
        test_nested_vector_real::<f32>();
        test_nested_vector_real::<f64>();
        test_nested_vector_complex::<f64>();
        test_nested_vector_complex::<f64>();
    }

    #[test]
    fn test_serialisation_array() {
        test_array_real::<f32>();
        test_array_real::<f64>();
        test_array_complex::<f32>();
        test_array_complex::<f64>();

        test_array_real_empty::<f32>();
        test_array_real_empty::<f64>();
        test_array_complex_empty::<f32>();
        test_array_complex_empty::<f64>();
    }

    #[test]
    fn test_blas_metadata() {
        test_blas_metadata_sarcmp::<f32>();
        test_blas_metadata_sarcmp::<f64>();

        test_blas_metadata_sarcmp_empty::<f32>();
        test_blas_metadata_sarcmp_empty::<f64>();
    }

    #[test]
    fn test_nested_blas_metadata() {
        test_nested_blas_metadata_sarcmp::<f32>();
    }
}
