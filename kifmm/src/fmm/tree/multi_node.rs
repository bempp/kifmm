use std::collections::{HashMap, HashSet};

use itertools::Itertools;

use mpi::{
    datatype::{PartitionMut, Partitioned},
    traits::{Communicator, CommunicatorCollectives, Equivalence},
    Count, Rank,
};
use num::Float;
use rlst::RlstScalar;

use crate::{
    fmm::{
        helpers::single_node::optionally_time,
        types::{Layout, Query},
    },
    traits::{
        tree::{MultiFmmTree, MultiTree, SingleTree},
        types::{MPICollectiveType, OperatorTime},
    },
    tree::{MortonKey, MultiNodeTree},
    MultiNodeFmmTree,
};

impl<T, C> MultiFmmTree for MultiNodeFmmTree<T, C>
where
    T: RlstScalar + Default + Float + Equivalence,
    C: Communicator,
{
    type Tree = MultiNodeTree<T, C>;

    fn domain(&self) -> &<<Self::Tree as MultiTree>::SingleTree as SingleTree>::Domain {
        &self.domain
    }

    fn n_source_trees(&self) -> usize {
        self.source_tree().n_trees()
    }

    fn n_target_trees(&self) -> usize {
        self.target_tree().n_trees()
    }

    fn source_tree(&self) -> &Self::Tree {
        &self.source_tree
    }

    fn target_tree(&self) -> &Self::Tree {
        &self.target_tree
    }
}

impl<T, C> MultiNodeFmmTree<T, C>
where
    T: RlstScalar + Default + Float + Equivalence,
    C: Communicator,
{
    /// Configure queries to send from this rank for either U or V list data
    pub fn set_queries(&mut self, admissible: bool) {
        let mut queries = HashSet::new();

        if admissible {
            // V list queries
            for key in self.target_tree.all_keys().unwrap() {
                // Compute interaction list
                let interaction_list = key
                    .parent()
                    .neighbors()
                    .iter()
                    .flat_map(|pn| pn.children())
                    .filter(|pnc| !key.is_adjacent(pnc))
                    .collect_vec();

                // Filter for those contained on foreign ranks
                let interaction_list = interaction_list
                    .into_iter()
                    .filter(|key| {
                        // Check if the rank is not equal to this rank
                        if let Some(&rank) = self.source_layout.rank_from_key(key) {
                            rank != self.source_tree.rank()
                        } else {
                            false
                        }
                        // !self.source_tree.keys_set.contains(key)
                    })
                    .collect_vec();

                queries.extend(interaction_list);
            }
        } else {
            // U list queries

            for leaf in self.target_tree.all_leaves().unwrap() {
                let interaction_list = leaf.neighbors();

                // Filter for those contained on foreign ranks
                let interaction_list = interaction_list
                    .into_iter()
                    .filter(|key| {
                        // Check if the rank is not equal to this rank
                        if let Some(&rank) = self.source_layout.rank_from_key(key) {
                            rank != self.source_tree.rank()
                        } else {
                            false
                        }
                    })
                    .collect_vec();

                queries.extend(interaction_list);
            }
        }

        // Compute the send ranks, counts, and mark each global process involved in query
        // communication
        let queries = queries.into_iter().collect_vec();
        let mut ranks = Vec::new();
        let mut send_counts = vec![0 as Count; self.source_tree.communicator.size() as usize];
        let mut send_marker = vec![0 as Rank; self.source_tree.communicator.size() as usize];

        for query in queries.iter() {
            let rank = *self.source_layout.rank_from_key(query).unwrap();

            ranks.push(rank);
            send_counts[rank as usize] += 1;
        }

        for (rank, &send_count) in send_counts.iter().enumerate() {
            if send_count > 0 {
                send_marker[rank] = 1;
            }
        }

        // Sort queries by destination rank
        let queries = {
            let mut indices = (0..queries.len()).collect_vec();
            indices.sort_by_key(|&i| ranks[i]);

            let mut sorted_queries_ = Vec::with_capacity(queries.len());
            for i in indices {
                sorted_queries_.push(queries[i].morton)
            }
            sorted_queries_
        };

        // Sort ranks of queries into rank order
        ranks.sort();

        // Compute the receive counts, and mark again processes involved
        let mut receive_counts = vec![0i32; self.source_tree().communicator.size() as usize];
        let mut receive_marker = vec![0i32; self.source_tree().communicator.size() as usize];

        let (_, duration) = optionally_time(true, {
            || {
                self.source_tree
                    .communicator
                    .all_to_all_into(&send_counts, &mut receive_counts);
            }
        });

        if let Some(d) = duration {
            self.mpi_times
                .entry(MPICollectiveType::AlltoAll)
                .and_modify(|t| t.time += d.as_millis() as u64)
                .or_insert(OperatorTime {
                    time: d.as_millis() as u64,
                });
        };

        for (rank, &receive_count) in receive_counts.iter().enumerate() {
            if receive_count > 0 {
                receive_marker[rank] = 1
            }
        }

        let query = Query {
            queries,
            ranks,
            send_counts,
            send_marker,
            receive_marker,
            receive_counts,
        };

        if admissible {
            self.v_list_query = query
        } else {
            self.u_list_query = query
        }
    }

    /// Gather source box ranges controlled by each local rank
    pub fn set_source_layout(&mut self) {
        let size = self.source_tree.communicator.size();

        // Gather ranges on all processes, define by roots they own
        let mut roots = Vec::new();
        for i in 0..self.source_tree.n_trees {
            roots.push(self.source_tree.trees[i].root());
        }

        let n_roots = roots.len() as i32;
        let mut counts_ = vec![0i32; size as usize];

        // All gather to calculate the counts of roots on each processor

        // TODO cleanup timing
        let (_, duration) = optionally_time(true, || {
            self.source_tree
                .communicator
                .all_gather_into(&n_roots, &mut counts_)
        });

        if let Some(d) = duration {
            self.mpi_times
                .entry(MPICollectiveType::AllGather)
                .and_modify(|t| t.time += d.as_millis() as u64)
                .or_insert(OperatorTime {
                    time: d.as_millis() as u64,
                });
        };

        // Calculate displacements from the counts on each processor
        let mut displacements_ = Vec::new();
        let mut displacement = 0;
        for &count in counts_.iter() {
            displacements_.push(displacement);
            displacement += count
        }

        let n_roots_global = counts_.iter().sum::<i32>();

        // Allocate buffer to store layouts
        let mut raw = vec![MortonKey::<T>::default(); n_roots_global as usize];

        // Store a copy of counts and displacements

        // TODO: tidy timing
        let counts;
        let displacements;
        let mut partition = PartitionMut::new(&mut raw, counts_, &displacements_[..]);

        let (_, duration) = optionally_time(true, {
            || {
                self.source_tree
                    .communicator
                    .all_gather_varcount_into(&roots, &mut partition);
            }
        });

        counts = partition.counts().to_vec();
        displacements = partition.displs().to_vec();

        if let Some(d) = duration {
            self.mpi_times
                .entry(MPICollectiveType::AllGatherV)
                .and_modify(|t| t.time += d.as_millis() as u64)
                .or_insert(OperatorTime {
                    time: d.as_millis() as u64,
                });
        };

        // Store as a set for easy lookup
        let raw_set = raw.iter().cloned().collect();

        // Buffer of length total count, where index is matched to the roots
        let mut ranks: Vec<i32> = Vec::new();

        for (rank, &count) in counts.iter().enumerate() {
            ranks.extend_from_slice(&vec![rank as i32; count as usize]);
        }

        // Map between a root and its associated rank
        let mut range_to_rank = HashMap::new();

        for (&range, &rank) in raw.iter().zip(ranks.iter()) {
            range_to_rank.insert(range, rank);
        }

        self.source_layout = Layout {
            raw,
            raw_set,
            counts,
            displacements,
            ranks,
            range_to_rank,
        };
    }
}
