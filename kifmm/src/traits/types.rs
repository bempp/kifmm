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
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
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

/// Enumeration of MPI collective types for timing
#[repr(C)]
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum MPICollectiveType {
    /// All to All
    AlltoAll,

    /// All to All V
    AlltoAllV,

    /// Neighbour All to All
    NeighbourAlltoAll,

    /// Neighbour All to All V
    NeighbourAlltoAllv,

    /// Gather
    Gather,

    /// Scatter
    Scatter,

    /// Gather
    GatherV,

    /// Scatter
    ScatterV,

    /// All Gather
    AllGather,

    /// All Gather V
    AllGatherV,

    /// Cart dist-graph create
    DistGraphCreate,

    /// Parallel sort
    Sort,
}

/// Enumeration of communication types for timing
#[repr(C)]
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
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
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
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

    /// Pointer and Buffer Creationmp
    MetadataCreation,

    /// Displacement Map Creation
    DisplacementMap,
}
/// C compatible struct for timing
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct OperatorTime {
    /// Time in milliseconds
    pub time: u64,
}

impl OperatorTime {
    /// Constructor
    pub fn new(time: u64) -> Self {
        Self { time }
    }

    /// Construct from instant
    pub fn from_instant(time: Instant) -> Self {
        Self {
            time: time.elapsed().as_millis() as u64,
        }
    }

    /// Constructor from duration
    pub fn from_duration(time: Duration) -> Self {
        Self {
            time: time.as_millis() as u64,
        }
    }
}

impl fmt::Display for FmmOperatorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FmmOperatorType::P2M => write!(f, "P2M"),
            FmmOperatorType::M2M(level) => write!(f, "M2M({level})"),
            FmmOperatorType::M2L(level) => write!(f, "M2L({level})"),
            FmmOperatorType::L2L(level) => write!(f, "L2L({level})"),
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
            MetadataType::DisplacementMap => write!(f, "Displacement Map"),
            MetadataType::MetadataCreation => write!(f, "Pointer and Buffer Creation"),
        }
    }
}

impl std::fmt::Display for FmmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FmmError::Failed(e) => write!(f, "Failed: {e}"),
            FmmError::Unimplemented(e) => write!(f, "Unimplemented: {e}"),
            FmmError::Io(e) => write!(f, "I/O error: {e}"),
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
