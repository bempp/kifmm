- Ensure that counts are all u64 rather than i32, to avoid overflow in large FMM data buffers

- check API documentation for expansion order, which is now a slice in multinode

- software engineering (code-reorg, documentation)

- add a feature to set the size of the communicator group used for precomputations
    - so user can choose size <= total size

- Proper helmholtz config [save for future work], crate ticket
- create ticket for helmholtz single node config/benchmarking
