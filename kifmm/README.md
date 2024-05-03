# Build Python Bindings

We use Maturin for Python bindings, and provide an example of installation using the `uv` package manager for Python below.

1. Begin by installing Maturin (and pip) in a new virtual environment.

```bash
uv venv && source .venv/bin/activate && uv pip install maturin pip
```

2. Use the Maturin CLI to install the Python bindings into this virtual environment.

Note that Maturin must be run from the `kifmm` crate root, not the workspace root.

```bash
cd /path/to/kifmm/crate && maturin develop --release # Building bindings in release mode
```

We provide example usage of the Python API in the `python/examples` directory.