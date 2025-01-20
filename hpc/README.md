# Running `kifmm-rs` on HPC


We summarise suggestions for successfully building and achieving performance with `kifmm` HPC platforms.

## ARCHER2

- ARCHER2 is a HPE CRAY EX system hosted at the Edinburgh Parallel Computing Centre.
- We provide the following suggestions for compiling `kifmm-rs`.

1. Download and install Rust using the rustup script
2. We recommend using PrgEnv-AOCC.
3. Ensure that symlinks are created for CRAY LibSci such that RLST is able to find BLAS and LAPACK which it expects to be called `libblas` or similar.
4. RLST may struggle to build the [sleef](https://github.com/shibatch/sleef) dependency, this is a known issue. For now we recommend simply removing the sleef dependency.



