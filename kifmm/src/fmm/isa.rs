//! Implementation for ISA handlers
use crate::fmm::types::Isa;

impl Isa {
    /// Constructor
    #[allow(unreachable_code)]
    pub fn new() -> Self {
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            return Isa::Neon(pulp::aarch64::NeonFcma::try_new().unwrap());
        }

        #[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
        {
            return Isa::Avx(pulp::x86::V3::try_new().unwrap());
        }

        Isa::Default
    }

    /// Getter
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    pub fn isa(&self) -> Option<&pulp::aarch64::NeonFcma> {
        if let Isa::Neon(ref neon_fcma) = self {
            Some(neon_fcma)
        } else {
            None
        }
    }

    /// Getter
    #[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
    pub fn isa(&self) -> Option<&pulp::x86::V3> {
        if let Isa::Avx(ref avx) = self {
            Some(avx)
        } else {
            None
        }
    }

    /// Getter
    #[cfg(not(any(
        all(target_arch = "aarch64", target_feature = "neon"),
        all(target_arch = "x86_64", target_feature = "avx")
    )))]
    pub fn isa(&self) -> Option<()> {
        None
    }
}
