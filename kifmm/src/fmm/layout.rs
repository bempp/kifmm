//! Used for discovering global layout
use itertools::Itertools;
use num::Float;
use rlst::RlstScalar;

use crate::tree::types::MortonKey;

use super::types::Layout;

impl<T: RlstScalar + Float> Layout<T> {
    /// rank associated with this rank
    pub fn rank_from_key(&self, key: &MortonKey<T::Real>) -> Option<&i32> {
        let ancestors = key.ancestors();
        // assuming of length 1
        let intersection = ancestors.intersection(&self.raw_set).collect_vec();
        self.range_to_rank.get(intersection[0])
    }
}
