# kifmm-rs: A Kernel-Independent Fast Multipole Framework in Rust

## Install


We currently only support Unix distributions. The current head can be built from source by adding the following to your `Cargo.toml` file.

```toml

# Clone the repo
# git clone git@github.com:bempp/kifmm.git
# git clone https://github.com/bempp/kifmm.git

# Build in release mode (optional)
cd kifmm && cargo build --release

# Build with MPI support (optional)
cd kifmm && cargo build --release --features mpi
```

The current release version can be installed using the deployment on `crates.io`, and can be added to your `Cargo.toml` file with

```bash
cargo add kifmm
```

Though we note that MPI functionality is not supported in the current release, and users must install from source.

## Dependencies

The main external dependencies of this package are

- FFTW
- BLAS
- LAPACK
- MPI (Optional)

FFTW is automatically downloaded, built and linked with optimal settings as a part of the provided Cargo build. For linear algebra and matrix computations we use the [RLST](https://github.com/linalg-rs/rlst/tree/main) crate, which itself detects and compiles with respect to the BLAS and LAPACK installed on your system, this must be exposed to your `LD_LIBRARY_PATH`. MPI, if used, is also detected from your system MPI and similarly must be exposed to your `LD_LIBRARY_PATH`.

## Quickstart

A quickstart example is available in the `kifmm/` directory's README, alongside instructions for building Python bindings or using the C bindings with CMake.

## Documentation
The latest documentation of the main branch of this repo is available at [bempp.github.io/kifmm/kifmm/index.html](https://bempp.github.io/kifmm/kifmm/index.html).

Build docs with Latex enabled

```bash
RUSTDOCFLAGS="--html-in-header kifmm/src/docs-header.html" cargo doc --no-deps
```

## Testing

The functionality of the shared memory library can be tested with

```bash
# Run tests in release mode (optional)
cargo test --release
```

There are also numerous MPI tests and examples in the `examples` folder. Tests are marked with the `mpi_test_*` prefix. For these we recommend the [`cargo mpirun` utility library](https://github.com/AndrewGaspar/cargo-mpirun). With this, distributed memory tests can be run with.

```bash
# Run tests in release mode (optional)
cargo mpirun -n <num_processes> --example mpi_test_fmm --release
```

### Library

This project uses Rust workspaces to organise the associated crates.

The `fftw-src` and `fftw-sys` are responsible for downloading, compiling and linking FFTW 3.9 with custom bindings and build steps. This is currently only available for UNIX systems, and therefore the entire library is also limited to UNIX systems at present.

The `kifmm` crate contains the source code of the library. This crate contains benchmarks, Rust, Python and C examples for library usage and installation.

The `scripts` crate contains various useful scripts for examining the performance of the software, used in single node and distributed benchmarks.

Additionally, there are a few directories containing metadata for the project. The `paper` directory contains our JOSS paper, which should be cited for any derivative works based on `kifmm-rs`.

The `hpc` directory contains example slurm scripts, and build hints for successfully compiling `kifmm-rs` on HPC platforms.

## Contributing

We welcome contributions to this project, and are grateful for your interest. Contributions can take many forms, including bug reports, feature requests code contributions and documentation.

### Reporting Bugs and Issues

Errors in the library should be added to the [GitHub issue tracker](https://github.com/bempp/kifmm/issues).

### Feature Requests

We're always open to suggestions for new features, please check the [GitHub issue tracker](https://github.com/bempp/kifmm/issues) to see if your concern has been discussed. If not, please open a new issue with the 'enhancement' label.


### Submitting Code or Documentation Changes

Your pull request should include a description of changes, any relevant issue numbers, and tests to cover the proposed changes.

### Questions and Discussion

Questions about the library and its use can be asked on the [Bempp Discourse](https://bempp.discourse.group).

## Licence
KiFMM is licensed under a BSD 3-Clause licence. Full text of the licence can be found [here](LICENSE.md).
