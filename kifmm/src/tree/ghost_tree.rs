use std::collections::HashSet;

use num::Float;
use rlst::RlstScalar;

use super::types::{GhostTreeU, GhostTreeV, MortonKey};

impl<T> GhostTreeU<T>
where
    T: RlstScalar + Float,
{
    pub fn from_ghost_data(
        octants: &[Vec<MortonKey<T>>],
        index_pointers: &[Vec<i32>],
        coordinates: &[Vec<T>],
    ) -> Result<Self, std::io::Error> {
        // let

        Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "unimplemented",
        ))
    }
}

impl<T> GhostTreeV<T>
where
    T: RlstScalar + Float,
{
    pub fn from_ghost_data(
        octants: &[Vec<MortonKey<T::Real>>],
        multipoles: &[Vec<T>],
    ) -> Result<Self, std::io::Error> {
        let mut result = Self::default();

        // Find all sibling octants and insert into octants
        let mut keys = Vec::new();

        for packet in octants.iter() {
            for &key in packet.iter() {
                keys.extend(key.siblings().clone())
            }
        }

        keys.sort();
        let keys_set: HashSet<MortonKey<_>> = keys.iter().cloned().collect();

        result.keys = keys.into();
        // result.keys_set = keys_set;

        Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "unimplemented",
        ))
    }
}
