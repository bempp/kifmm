# Build Python Bindings

We use Maturin for Python bindings, and provide an example of installation using the `uv` package manager for Python below.

In order to enable plotting with MayaVi only Python 3.10.* is currently supported,

1. Begin by installing Maturin (and pip) in a new virtual environment.

```bash
uv venv --python=3.10 && source .venv/bin/activate && uv pip install maturin pip
```

2. Use the Maturin CLI to install the Python bindings into this virtual environment.

Note that Maturin must be run from the `kifmm` crate root, not the workspace root.

<!-- ```bash
# Building bindings in release mode, enable Python binding
cd /path/to/kifmm/crate && maturin develop --release --features python
``` -->

```bash
maturin develop --release -b cffi
```

We provide example usage of the Python API, as well as visualisation, in the `python/examples` directory.

## Note

You must deactivate your virtual environment before building/testing the Rust library with `cargo` commands
as maturin wraps Cargo, and on MacOS and certain other platforms leads to [linker errors](https://pyo3.rs/v0.14.4/faq.html#i-cant-run-cargo-test-im-having-linker-issues-like-symbol-not-found-or-undefined-reference-to-_pyexc_systemerror).