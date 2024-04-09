//! Functions for handling transfer vectors
use crate::fmm::types::TransferVector;
use crate::tree::types::{Domain, MortonKey};
use crate::RlstScalarFloat;
use itertools::Itertools;
use rlst::RlstScalar;
use std::collections::HashSet;

/// Unique M2L interactions for homogenous, translationally invariant kernel functions (e.g. Laplace/Helmholtz).
/// There are at most 316 such interactions, corresponding to unique `transfer vectors'. Here we compute all of them
/// with respect to level 3 of an associated octree.
pub fn compute_transfer_vectors<T>() -> Vec<TransferVector<T>>
where
    T: RlstScalarFloat,
    <T as RlstScalar>::Real: RlstScalarFloat,
{
    let half = T::from(0.5).unwrap().re();
    let zero = T::zero().re();
    let one = T::one().re();
    let point = [half, half, half];
    let domain = Domain::<T::Real>::new(&[zero, zero, zero], &[one, one, one]);

    // Encode point in centre of domain
    let key = MortonKey::<T::Real>::from_point(&point, &domain, 3);

    // Add neighbours, and their resp. siblings to v list.
    let mut neighbours = key.neighbors();
    let mut keys = Vec::new();
    keys.push(key);
    keys.append(&mut neighbours);

    for key in neighbours.iter() {
        let mut siblings = key.siblings();
        keys.append(&mut siblings);
    }

    // Keep only unique keys
    let keys = keys.iter().unique().collect_vec();

    let mut transfer_vectors: Vec<usize> = Vec::new();
    let mut targets = Vec::new();
    let mut sources = Vec::new();

    for key in keys.iter() {
        // Dense v_list
        let v_list = key
            .parent()
            .neighbors()
            .iter()
            .flat_map(|pn| pn.children())
            .filter(|pnc| !key.is_adjacent(pnc))
            .collect_vec();

        // Find transfer vectors for everything in dense v list of each key
        let tmp: Vec<usize> = v_list
            .iter()
            .map(|v| key.find_transfer_vector(v).unwrap())
            .collect_vec();

        transfer_vectors.extend(&mut tmp.iter().cloned());
        sources.extend(&mut v_list.iter().cloned());

        let tmp_targets = vec![**key; tmp.len()];
        targets.extend(&mut tmp_targets.iter().cloned());
    }

    // Filter for unique transfer vectors, and their corresponding index
    let mut unique_transfer_vectors = Vec::new();
    let mut unique_indices = HashSet::new();

    for (i, vec) in transfer_vectors.iter().enumerate() {
        if !unique_transfer_vectors.contains(vec) {
            unique_transfer_vectors.push(*vec);
            unique_indices.insert(i);
        }
    }

    // Identify sources/targets which correspond to unique transfer vectors.
    let unique_sources = sources
        .iter()
        .enumerate()
        .filter(|(i, _)| unique_indices.contains(i))
        .map(|(_, x)| *x)
        .collect_vec();

    let unique_targets = targets
        .iter()
        .enumerate()
        .filter(|(i, _)| unique_indices.contains(i))
        .map(|(_, x)| *x)
        .collect_vec();

    let mut result = Vec::<TransferVector<T>>::new();

    for ((t, s), v) in unique_targets
        .into_iter()
        .zip(unique_sources)
        .zip(unique_transfer_vectors)
    {
        let components = t.find_transfer_vector_components(&s).unwrap();

        result.push(TransferVector {
            components,
            hash: v,
            source: s,
            target: t,
        });
    }

    result
}
