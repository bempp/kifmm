- Need to implement clone for the metadata so that in blas displacements function don't have to call compute_transfer_vectors, can instead clone this in setup of the ghost FMM

- weak scaling script

- Need to test threading controls i.e. trade-off between threading choices on MPI and rayon threads

- Need to figure out how to pin MPI processes to nodes for performance on Archer 2, NUMA aware deployment etc

- Not sure about scaling in P2M, why does it need extra now?
