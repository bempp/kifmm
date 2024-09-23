//! Neighbourhood communicator wrapper
use std::mem::MaybeUninit;

use itertools::Itertools;
use mpi::{
    collective::CommunicatorCollectives, raw::{AsRaw, FromRaw}, topology::{Communicator, SimpleCommunicator, UserGroup}, traits::{Buffer, BufferMut, PartitionedBuffer, PartitionedBufferMut}, Count, Rank
};
use mpi_sys;

use super::types::NeighbourhoodCommunicator;

unsafe fn with_uninitialized<F, U, R>(f: F) -> (R, U)
where
    F: FnOnce(*mut U) -> R,
{
    let mut uninitialized = MaybeUninit::uninit();
    let res = f(uninitialized.as_mut_ptr());
    (res, uninitialized.assume_init())
}


impl NeighbourhoodCommunicator {
    /// Number of associated ranks
    pub fn size(&self) -> i32 {
        self.raw.size()
    }

    /// Forward send a buffer on the neighbourhood communicator
    pub fn all_to_all_into<S: ?Sized, R: ?Sized>(&self, sendbuf: &S, recvbuf: &mut R)
    where
        S: Buffer,
        R: BufferMut,
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
    pub fn all_to_all_varcount_into<S: ?Sized, R: ?Sized>(&self, sendbuf: &S, recvbuf: &mut R)
    where
        S: PartitionedBuffer,
        R: PartitionedBufferMut,
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

    pub fn translate_ranks(&self, global_comm: &SimpleCommunicator, ranks: &[Rank]) -> Vec<Rank>{

        let n_ranks = ranks.len();
        let mut result = vec![0 as Rank; n_ranks];
        unsafe {
            let global_comm_group = global_comm.group().as_raw();
            let neighbour_comm_group = with_uninitialized(|group| mpi::ffi::MPI_Comm_group(self.raw.as_raw(), group)).1;
            mpi::ffi::MPI_Group_translate_ranks(global_comm_group, n_ranks as Count, ranks.as_ptr(), neighbour_comm_group, result.as_mut_ptr());
        }

        result
    }

    /// Map from the global rank to the local rank
    pub fn global_to_local_rank(&self, global_rank: i32) -> Option<i32> {
        if let Some(idx) = self.neighbours.iter().position(|&g| global_rank == g) {
            Some(self.neighbours[idx])
        } else {
            None
        }
    }

    /// Barrier synchronisation for all processes in neighbourhoood
    pub fn barrier(&self) {
        unsafe {
            mpi::ffi::MPI_Barrier(self.raw.as_raw())
        };
    }

    /// Constructor from locations to send to
    pub fn new(world_comm: &SimpleCommunicator, send_marker: &[i32], receive_marker: &[i32]) -> Self {
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
