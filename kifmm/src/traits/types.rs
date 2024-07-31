//! Utility types for trait definitions.
use std::{fmt, time::Instant};

/// Type to handle FMM related errors
#[derive(Debug)]
pub enum FmmError {
    /// Failure to run some business logic
    Failed(String),

    /// Unimplemented section
    Unimplemented(String),

    /// I/O failure
    Io(std::io::Error),
}

/// Enumeration of operator types for timing
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum FmmOperatorType {
    /// particle to multipole
    P2M,

    /// multipole to multipole (level)
    M2M(u64),

    /// multipole to local (level)
    M2L(u64),

    /// local to local (level)
    L2L(u64),

    /// local to particle
    L2P,

    /// particle to particle
    P2P,
}

/// C compatible struct for operator timing
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct FmmOperatorTime {
    /// Operator name
    pub operator: FmmOperatorType,

    /// Time in milliseconds
    pub time: u64,
}

impl FmmOperatorTime {
    /// Constructor
    pub fn new(operator: FmmOperatorType, time: u64) -> Self {
        Self { operator, time }
    }

    /// Constructor from instant
    pub fn from_instant(operator: FmmOperatorType, time: Instant) -> Self {
        let time = time.elapsed().as_millis() as u64;
        Self { operator, time }
    }
}

impl fmt::Display for FmmOperatorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FmmOperatorType::P2M => write!(f, "P2M"),
            FmmOperatorType::M2M(level) => write!(f, "M2M({})", level),
            FmmOperatorType::M2L(level) => write!(f, "M2L({})", level),
            FmmOperatorType::L2L(level) => write!(f, "L2L({})", level),
            FmmOperatorType::L2P => write!(f, "L2P"),
            FmmOperatorType::P2P => write!(f, "P2P"),
        }
    }
}

impl std::fmt::Display for FmmOperatorTime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Operator: {}, Time: {} ms", self.operator, self.time)
    }
}

impl std::fmt::Display for FmmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FmmError::Failed(e) => write!(f, "Failed: {}", e),
            FmmError::Unimplemented(e) => write!(f, "Unimplemented: {}", e),
            FmmError::Io(e) => write!(f, "I/O error: {}", e),
        }
    }
}

impl std::error::Error for FmmError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            FmmError::Io(e) => Some(e),
            FmmError::Failed(_e) => None,
            FmmError::Unimplemented(_e) => None,
        }
    }
}
