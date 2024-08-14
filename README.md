# kifmm-rs: A Kernel-Independent Fast Multipole Framework in Rust

## Install

We currently only support Unix distributions.

The current head can be built from source by adding the following to your `Cargo.toml` file.


```toml
kifmm = { git = "https://github.com/bempp/kifmm" }
```

The current release version can be installed using the deployment on `crates.io`, and can be added to your `Cargo.toml` file with

```bash
cargo add kifmm
```

## Quickstart

A quickstart example is available in the `kifmm/` directory's README, alongside instructions for building Python bindings or using the C bindings with CMake.

## Documentation
The latest documentation of the main branch of this repo is available at [bempp.github.io/kifmm/kifmm/index.html](https://bempp.github.io/kifmm/kifmm/index.html).

Build docs with Latex enabled

```bash
RUSTDOCFLAGS="--html-in-header kifmm/src/docs-header.html" cargo doc --no-deps
```

## Testing
The functionality of the library can be tested by running:
```bash
cargo test
```

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
