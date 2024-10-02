//! Neighbourhood communicator wrapper
use itertools::Itertools;
use mpi::{
    raw::{AsRaw, FromRaw},
    topology::{Communicator, SimpleCommunicator},
    traits::{Buffer, BufferMut, PartitionedBuffer, PartitionedBufferMut},
};
use mpi_sys;

use super::types::NeighbourhoodCommunicator;

impl NeighbourhoodCommunicator {
    /// Number of associated ranks
    pub fn size(&self) -> i32 {
        self.raw.size()
    }

    /// Forward send a buffer on the neighbourhood communicator
    pub fn all_to_all_into<S, R>(&self, sendbuf: &S, recvbuf: &mut R)
    where
        S: ?Sized + Buffer,
        R: ?Sized + BufferMut,
    {
        let c_size = self.raw.size();

        unsafe {
            mpi_sys::MPI_Neighbor_alltoall(
                sendbuf.pointer(),
                sendbuf.count() / c_size,
                sendbuf.as_datatype().as_raw(),
                recvbuf.pointer_mut(),
                recvbuf.count() / c_size,
                recvbuf.as_datatype().as_raw(),
                self.raw.as_raw(),
            );
        }
    }

    /// Forward send a buffer on the neighbourhood communicator
    pub fn all_to_all_varcount_into<S, R>(&self, sendbuf: &S, recvbuf: &mut R)
    where
        S: ?Sized + PartitionedBuffer,
        R: ?Sized + PartitionedBufferMut,
    {
        unsafe {
            mpi_sys::MPI_Neighbor_alltoallv(
                sendbuf.pointer(),
                sendbuf.counts().as_ptr(),
                sendbuf.displs().as_ptr(),
                sendbuf.as_datatype().as_raw(),
                recvbuf.pointer_mut(),
                recvbuf.counts().as_ptr(),
                recvbuf.displs().as_ptr(),
                recvbuf.as_datatype().as_raw(),
                self.raw.as_raw(),
            );
        }
    }

    /// Barrier synchronisation for all processes in neighbourhoood
    pub fn barrier(&self) {
        unsafe { mpi::ffi::MPI_Barrier(self.raw.as_raw()) };
    }

    /// Constructor from locations to send to
    pub fn new(
        world_comm: &SimpleCommunicator,
        send_marker: &[i32],
        receive_marker: &[i32],
    ) -> Self {
        let size = world_comm.size();

        // Now create neighbours, with send and receive displacements
        let mut neighbours = Vec::new();

        for world_rank in 0..size as usize {
            let world_rank_i32 = world_rank as i32;
            if send_marker[world_rank] != 0 || receive_marker[world_rank] != 0 {
                neighbours.push(world_rank_i32);
            }
        }

        // Can create neighbourhood communicators
        let raw = unsafe {
            let mut raw_comm = mpi_sys::RSMPI_COMM_NULL;
            mpi_sys::MPI_Dist_graph_create_adjacent(
                world_comm.as_raw(),
                neighbours.len() as i32,
                neighbours.as_ptr(),
                mpi_sys::RSMPI_UNWEIGHTED(),
                neighbours.len() as i32,
                neighbours.as_ptr(),
                mpi_sys::RSMPI_UNWEIGHTED(),
                mpi_sys::RSMPI_INFO_NULL,
                0,
                &mut raw_comm,
            );
            mpi::topology::SimpleCommunicator::from_raw(raw_comm)
        };

        Self { neighbours, raw }
    }

    /// Simple constructor
    pub fn from_comm(comm: &SimpleCommunicator) -> Self {
        let neighbours = (0..comm.size()).collect_vec();
        Self {
            neighbours,
            raw: comm.duplicate(),
        }
    }
}
