//! Functions for handling transfer vectors
use std::collections::HashMap;

use itertools::Itertools;
use num::traits::Float;
use rlst::RlstScalar;

use crate::{fmm::types::TransferVector, tree::types::MortonKey};

/// Compute all unique transfer vectors at a given level.
pub fn compute_transfer_vectors_at_level<T: RlstScalar + Float>(
    level: u64,
) -> Result<Vec<TransferVector<T>>, std::io::Error> {
    let mut result = Vec::new();

    let mut parent_key = MortonKey::<T>::root(None);

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
                for source in source_cluster.iter() {
                    for target in target_cluster.iter() {
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

    use super::compute_transfer_vectors_at_level;

    #[test]
    fn test_compute_transfer_vectors_at_level() {
        // Test that there are 316 transfer vectors (for trans. inv. kernels) at each level
        for level in 2..4 {
            let tvs = compute_transfer_vectors_at_level::<f64>(level).unwrap();
            assert!(tvs.len() == 316);

            for tv in tvs.iter() {
                assert!(tv.source.level() == level);
                assert!(tv.target.level() == level);
            }
        }

        // Test that transfer vectors found at each level are equivalent.
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

        // Test that the ordering is consistent between calls.
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
