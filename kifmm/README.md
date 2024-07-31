# Build Python Bindings

We use Maturin for Python bindings, and provide an example of installation using the `uv` package manager for Python below.

In order to enable plotting with MayaVi only Python 3.10.* is currently supported,

1. Begin by installing Maturin (and pip) in a new virtual environment.

```bash
uv venv --python=3.10 && source .venv/bin/activate && uv pip install maturin pip
```

2. Use the Maturin CLI to install the Python bindings into this virtual environment.

Note that Maturin must be run from the `kifmm` crate root, not the workspace root.

```bash
maturin develop --release
```

We provide example usage of the Python API, as well as visualisation, in the `python/examples` directory.
