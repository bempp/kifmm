# Running `kifmm-rs` on HPC


## Experiments

### 1. Weak Scaling

Runtime represented by wall-clack over all involved processes

#### 1.i Adjust the grain size in different weak scaling runs

- Grain size is the number of local trees per process
- Larger grain size leads to greater bandwidth use in local comms


#### 1.ii Measure the scaling of each operation in runtime and setup




We summarise suggestions for successfully building and achieving performance with `kifmm` HPC platforms.

## ARCHER2

- ARCHER2 is a HPE CRAY EX system hosted at the Edinburgh Parallel Computing Centre.
- We provide the following suggestions for compiling `kifmm-rs`.

1. Download and install Rust using the rustup script
2. We recommend using PrgEnv-AOCC.
3. Ensure that symlinks are created for CRAY LibSci such that RLST is able to find BLAS and LAPACK which it expects to be called `libblas` or similar.
4. RLST may struggle to build the [sleef](https://github.com/shibatch/sleef) dependency, this is a known issue. For now we recommend simply removing the sleef dependency.



