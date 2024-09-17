//! Neighbourhood communicator wrapper
use itertools::Itertools;
use mpi::{topology::{SimpleCommunicator, Communicator}, raw::{FromRaw, AsRaw}, traits::{Buffer, BufferMut, PartitionedBuffer, PartitionedBufferMut}, collective::CommunicatorCollectives};
use mpi_sys;

use super::types::NeighbourhoodCommunicator;

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

    /// Map from local rank to the global rank
    pub fn local_to_global_rank(&self, local_rank: i32) -> Option<i32> {
        if local_rank < (self.neighbours.len() - 1) as i32 {
            Some(self.neighbours[local_rank as usize])
        } else {
            None
        }
    }

    /// Map from the global rank to the local rank
    pub fn global_to_local_rank(&self, global_rank: i32) -> Option<i32> {
        if let Some(idx) = self.neighbours.iter().position(|&g| global_rank == g) {
            Some(self.neighbours[idx])
        } else {
            None
        }
    }

    /// Constructor from locations to send to
    pub fn new(world_comm: &SimpleCommunicator, to_send: &[i32]) -> Self {
        let size = world_comm.size();
        let rank: i32 = world_comm.rank();

        // Communicate whether to expect to be involved in send/receive with these ranks
        let mut to_receive = vec![0i32; size as usize];
        world_comm.all_to_all_into(to_send, &mut to_receive);

        // Now create neighbours, with send and receive displacements
        let mut neighbours = Vec::new();

        for world_rank in 0..size as usize {
            let world_rank_i32 = world_rank as i32;
            if to_send[world_rank] != 0 || to_receive[world_rank] != 0 {
                neighbours.push(world_rank_i32);
            } else if (world_rank as i32) == rank {
                neighbours.push(world_rank_i32)
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