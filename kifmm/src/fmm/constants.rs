//! Crate wide constants

/// Maximum block size to use to process leaf boxes during P2M kernel.
pub const P2M_MAX_BLOCK_SIZE: usize = 32;

/// Maximum block size to use to process boxes by level during M2M kernel.
pub const M2M_MAX_BLOCK_SIZE: usize = 16;

/// Maximum block size to use to process boxes by level during L2L kernel.
pub const L2L_MAX_BLOCK_SIZE: usize = 16;

/// Default value chosen for maximum number of particles per leaf.
pub const DEFAULT_NCRIT: u64 = 150;

/// Default maximum block size to use to process multiple child clusters during FFT M2L
pub const DEFAULT_M2L_FFT_BLOCK_SIZE: usize = 128;
