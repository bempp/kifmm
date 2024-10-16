//! Used for discovering global layout
use itertools::Itertools;
use num::Float;
use rlst::RlstScalar;

use crate::tree::types::MortonKey;

use super::types::Layout;

impl<T: RlstScalar + Float> Layout<T> {
    /// rank associated with this rank
    pub fn rank_from_key(&self, key: &MortonKey<T>) -> Option<&i32> {
        let ancestors = key.ancestors();
        let intersection = ancestors.intersection(&self.raw_set).collect_vec();
        // Any valid key has to be from a given rank, so this should always be of size 1
        if intersection.len() == 1 {
            self.range_to_rank.get(intersection[0])
        } else {
            None
        }
    }
}
