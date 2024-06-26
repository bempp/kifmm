# KiFMM: Kernel-independent FMM

## Install

Currently only supported on Unix.

```toml
kifmm = { git = "https://github.com/bempp/kifmm" }
```

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

## Getting help
Errors in the library should be added to the [GitHub issue tracker](https://github.com/bempp/kifmm/issues).

Questions about the library and its use can be asked on the [Bempp Discourse](https://bempp.discourse.group).

## Licence
KiFMM is licensed under a BSD 3-Clause licence. Full text of the licence can be found [here](LICENSE.md).
