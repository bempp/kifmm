//! Utility types for trait definitions.
use std::{collections::HashMap, fmt, time::Duration};

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

/// Type to handle FMM related errors
pub type FmmTime = HashMap<String, Duration>;

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
