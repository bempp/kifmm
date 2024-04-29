//! Functions for handling transfer vectors
use std::collections::{HashMap, HashSet};

use itertools::Itertools;
use num::traits::Float;
use rlst::RlstScalar;

use crate::{
    fmm::types::TransferVector,
    tree::{
        constants::{NHALO, NSIBLINGS_SQUARED},
        types::{Domain, MortonKey},
    },
};

/// Unique M2L interactions for homogenous, translationally invariant kernel functions (e.g. Laplace/Helmholtz).
/// There are at most 316 such interactions, corresponding to unique `transfer vectors'. Here we compute all of them
/// with respect to level 3 of an associated octree.
pub fn compute_transfer_vectors<T>() -> Vec<TransferVector<T>>
where
    T: RlstScalar + Float,
{
    let half = T::from(0.5).unwrap();
    let zero = T::zero();
    let one = T::one();
    let point = [half, half, half];
    let domain = Domain::<T>::new(&[zero, zero, zero], &[one, one, one]);

    // Encode point in centre of domain
    let key = MortonKey::<T>::from_point(&point, &domain, 3);

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

pub fn compute_transfer_vectors_at_level<T: RlstScalar + Float>(
    level: u64,
) -> Result<Vec<TransferVector<T>>, std::io::Error> {
    let mut result = Vec::new();

    let mut parent_key = MortonKey::<T>::root();

    for _ in 0..(level - 1) {
        parent_key = parent_key.first_child()
    }
    let parent_cluster = parent_key.siblings();

    let child_clusters = parent_cluster.iter().map(|k| k.children()).collect_vec();

    let mut seen = HashMap::new();
    for (i, source_cluster) in child_clusters.iter().enumerate() {
        for (j, target_cluster) in child_clusters.iter().enumerate() {
            if i != j {
                // calculate transfer vectors between different clusters
                for (_, source) in source_cluster.iter().enumerate() {
                    for (_, target) in target_cluster.iter().enumerate() {
                        let tv = source.find_transfer_vector(target).unwrap();
                        let tv_components = source.find_transfer_vector_components(target).unwrap();
                        if !source.is_adjacent(target) && !seen.keys().contains(&tv) {
                            seen.insert(tv, (source, target, tv_components));
                        }
                    }
                }
            }
        }
    }

    // Impose ordering on transfer vectors
    let mut seen = seen.iter().collect_vec();
    seen.sort_by_key(|k| k.0);

    for (&hash, (&source, &target, components)) in seen.iter() {
        result.push(TransferVector {
            components: *components,
            hash,
            source,
            target,
        });
    }
    Ok(result)
}

#[cfg(test)]
mod test {
    use std::collections::HashSet;

    use super::{compute_transfer_vectors, compute_transfer_vectors_at_level};

    #[test]
    fn test_compute_transfer_vectors_at_level() {
        for level in 2..4 {
            let tvs = compute_transfer_vectors_at_level::<f64>(level).unwrap();
            println!("len {:?}", tvs.len());
            assert!(tvs.len() == 316);

            for tv in tvs.iter() {
                assert!(tv.source.level() == level);
                assert!(tv.target.level() == level);
            }
        }

        let tvs_2 = compute_transfer_vectors_at_level::<f64>(2).unwrap();
        let tvs_2_hashes: HashSet<_> = tvs_2.iter().map(|t| t.hash).collect();
        let tvs_3 = compute_transfer_vectors_at_level::<f64>(3).unwrap();
        let tvs_3_hashes: HashSet<_> = tvs_3.iter().map(|t| t.hash).collect();

        for tv in tvs_2.iter() {
            assert!(tvs_3_hashes.contains(&tv.hash))
        }

        for tv in tvs_3.iter() {
            assert!(tvs_2_hashes.contains(&tv.hash))
        }

        assert!(tvs_2_hashes.len() == 316);

        let call_1 = compute_transfer_vectors_at_level::<f64>(2).unwrap();
        let call_2 = compute_transfer_vectors_at_level::<f64>(2).unwrap();

        call_1.iter().zip(call_2).for_each(|(c1, c2)| {
            println!("c1 {:?} c2 {:?}", c1, c2);
            assert!(c1.hash == c2.hash);
            assert!(c1.source == c2.source);
            assert!(c1.target == c2.target)
        })
    }
}
