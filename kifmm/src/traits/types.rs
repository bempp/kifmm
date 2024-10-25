//! Utility types for trait definitions.
use std::{
    fmt,
    time::{Duration, Instant},
};

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

/// Enumeration of communication types for timing
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum CommunicationType {
    /// Tree construction
    SourceTree,

    /// Tree construction
    TargetTree,

    /// Domain exchange
    TargetDomain,

    /// Domain exchange
    SourceDomain,

    /// Layout
    Layout,

    /// V list ghost exchange
    GhostExchangeV,

    /// V list ghost exchange at runtime
    GhostExchangeVRuntime,

    /// U list ghost exchange
    GhostExchangeU,

    /// Gather global FMM
    GatherGlobalFmm,

    /// Scatter global FMM
    ScatterGlobalFmm,
}

/// Enumeration of metadata construction for timing
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum MetadataType {
    /// Field translation data
    SourceToTargetData,

    /// Source tree translations
    SourceData,

    /// Target tree translations
    TargetData,

    /// Global FMM
    GlobalFmm,

    /// Ghost FMM V
    GhostFmmV,

    /// Ghost FMM U
    GhostFmmU,
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

/// C compatible struct for operator timing
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct CommunicationTime {
    /// Operator name
    pub operator: CommunicationType,

    /// Time in milliseconds
    pub time: u64,
}

/// C compatible struct for operator timing
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct MetadataTime {
    /// Operator name
    pub operator: MetadataType,

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

    /// Constructor from duration
    pub fn from_duration(operator: FmmOperatorType, time: Duration) -> Self {
        Self {
            operator,
            time: time.as_millis() as u64,
        }
    }
}

impl CommunicationTime {
    /// Constructor
    pub fn new(operator: CommunicationType, time: u64) -> Self {
        Self { operator, time }
    }

    /// Constructor from instant
    pub fn from_instant(operator: CommunicationType, time: Instant) -> Self {
        let time = time.elapsed().as_millis() as u64;
        Self { operator, time }
    }

    /// Constructor from duration
    pub fn from_duration(operator: CommunicationType, time: Duration) -> Self {
        Self {
            operator,
            time: time.as_millis() as u64,
        }
    }
}

impl MetadataTime {
    /// Constructor
    pub fn new(operator: MetadataType, time: u64) -> Self {
        Self { operator, time }
    }

    /// Constructor from instant
    pub fn from_instant(operator: MetadataType, time: Instant) -> Self {
        let time = time.elapsed().as_millis() as u64;
        Self { operator, time }
    }

    /// Constructor from duration
    pub fn from_duration(operator: MetadataType, time: Duration) -> Self {
        Self {
            operator,
            time: time.as_millis() as u64,
        }
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

impl fmt::Display for CommunicationType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CommunicationType::SourceTree => write!(f, "Source Tree"),
            CommunicationType::TargetTree => write!(f, "Target Tree"),
            CommunicationType::SourceDomain => write!(f, "Source Domain"),
            CommunicationType::TargetDomain => write!(f, "Target Domain"),
            CommunicationType::Layout => write!(f, "Layout"),
            CommunicationType::GhostExchangeU => write!(f, "Ghost Exchange U"),
            CommunicationType::GhostExchangeV => write!(f, "Ghost Exchange V"),
            CommunicationType::GhostExchangeVRuntime => write!(f, "Ghost Exchange V Runtime"),
            CommunicationType::GatherGlobalFmm => write!(f, "Gather Global FMM"),
            CommunicationType::ScatterGlobalFmm => write!(f, "Scatter Global FMM"),
        }
    }
}

impl fmt::Display for MetadataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MetadataType::SourceToTargetData => write!(f, "Source To Target Data"),
            MetadataType::SourceData => write!(f, "Source Data"),
            MetadataType::TargetData => write!(f, "Target Data"),
            MetadataType::GlobalFmm => write!(f, "Global FMM"),
            MetadataType::GhostFmmV => write!(f, "Ghost FMM V"),
            MetadataType::GhostFmmU => write!(f, "Ghost FMM U"),
        }
    }
}

impl std::fmt::Display for FmmOperatorTime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Operator: {}, Time: {} ms", self.operator, self.time)
    }
}

impl std::fmt::Display for CommunicationTime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Communication Type: {}, Time: {} ms",
            self.operator, self.time
        )
    }
}

impl std::fmt::Display for MetadataTime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Metadata Type: {}, Time: {} ms",
            self.operator, self.time
        )
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
