//! Crate wide constants

/// Maximum chunk size to use to process leaf boxes during P2M kernel.
pub const P2M_MAX_CHUNK_SIZE: usize = 256;

/// Maximum chunk size to use to process boxes by level during M2M kernel.
pub const M2M_MAX_CHUNK_SIZE: usize = 256;

/// Maximum chunk size to use to process boxes by level during L2L kernel.
pub const L2L_MAX_CHUNK_SIZE: usize = 256;

/// Default value chosen for maximum number of particles per leaf.
pub const DEFAULT_NCRIT: u64 = 150;

/// Default threshold for SVD singular value cutoff in SVD compressed field translations
pub const DEFAULT_SVD_THRESHOLD: f64 = 1e-16;
