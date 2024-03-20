//! General type definitions
pub use rlst::{c32, c64, RlstScalar};

/// Locality of an entity
pub enum Locality {
    /// Owned by the local process
    Local,
    /// A ghost of an entity owned by another process
    Ghost,
    /// Owned by another process
    Remote,
}

/// Generic error type
#[derive(thiserror::Error, Debug)]
pub enum Error {
    /// A generic error
    #[error("Solver Error: {0}")]
    Generic(String),
}

/// Result Type
pub type Result<T> = std::result::Result<T, Error>;

/// Evaluation Mode
#[derive(Clone, Copy)]
pub enum EvalType {
    /// Only values required
    Value,
    /// Both values and derivatives required
    ValueDeriv,
}
