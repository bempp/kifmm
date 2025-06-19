//! Functionality for attaching/removing/changing charges attached to FMM objects after
//! initialisation.
mod single_node;

#[cfg(feature = "mpi")]
mod multi_node;
