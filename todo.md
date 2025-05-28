3. helmholtz afterwards should be a copy and paste job

- BLAS-M2L not working for distributed helmholtz
    - test that metadata is same on multi and single node at each level
    - read logic, to check that it's the same

- confirmed that ghost multipoles are the same in blas and fft, which is expected as the ghost comm strategy is generic over this
- now will confirm the local multipoles are the same for both, they are confirmed the same, so the difference must be coming from the convolution step
- if they are, the only reasonable difference is the actual convolution step
    - can check the convolution output at each level to confirm that this is where the difference is coming, as if the multipoles are the same at level 4 (first of local trees convolutions)
    - then the problem is coming from the convolution step

- software engineering (code-reorg, documentation, clippy)
- add a feature to set the size of the communicator group used for precomputations
- so user can choose size <= total size
- Proper helmholtz config
- Ensure that counts are all u64 rather than i32, to avoid overflow in large FMM data buffers
- check API documentation for expansion order, which is now a slice in multinode