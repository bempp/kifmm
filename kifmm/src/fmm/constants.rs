//! Crate wide constants

/// Maximum chunk size to use to process leaf boxes during P2M kernel.
pub const P2M_MAX_CHUNK_SIZE: usize = 1;

/// Maximum chunk size to use to process boxes by level during M2M kernel.
pub const M2M_MAX_CHUNK_SIZE: usize = 1;

/// Maximum chunk size to use to process boxes by level during L2L kernel.
pub const L2L_MAX_CHUNK_SIZE: usize = 1;

/// Default value chosen for maximum number of particles per leaf.
pub const DEFAULT_NCRIT: u64 = 150;
