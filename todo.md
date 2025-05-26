3. helmholtz afterwards should be a copy and paste job

- software engineering (code-reorg, documentation, clippy)
- add test for helmholtz upward pass and FMM
- add experimental config code, re-add MPI scripts
- Proper helmholtz config
- Ensure that counts are all u64 rather than i32, to avoid overflow in large FMM data buffers
- check API documentation for expansion order, which is now a slice in multinode