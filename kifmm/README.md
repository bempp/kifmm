# Quick Start

Our Rust APIs are simple, with the requirement for no temporary metadata files, or setup of ancillary data structures such as hierarchical trees, required by the user. FMMs are simply parameterised using the builder pattern, with operator chaining to modulate the type of the runtime object. At its simplest, a user only specifies buffers associated with source and target point coordinates, and associated source densities. Trait interfaces implemented for FMM objects allows users to access the associated objects such as kernels and data such as multipole expansions.

Indeed, the full API is more extensive, including features that enable for variable expansion orders by tree level useful for oscillatory problems, accelerated pre-computations for the BLAS based field translations based on randomised SVDs and alternative field translation implementations. Both [Python](https://github.com/bempp/kifmm/tree/main/kifmm/python/examples) and [Rust](https://github.com/bempp/kifmm/tree/main/kifmm/examples) examples can be found in the repository.


```rust
use green_kernels::{laplace_3d::Laplace3dKernel, types::GreenKernelEvalType};
use kifmm::{BlasFieldTranslationSaRcmp, SingleNodeBuilder, FmmSvdMode, DataAccess, Evaluate,};
use kifmm::traits::tree::{SingleTree, SingleFmmTree};
use kifmm::tree::helpers::points_fixture;
use rlst::{RawAccess};

fn main() {
    // Generate some random source/target/charge data
    let n_sources = 1000000;
    let n_targets = 2000000;

    // The number of vectors of source densities, FMM is configured from data
    let sources =  points_fixture(n_sources, None, None, None);
    let targets = points_fixture(n_targets, None, None, None);
    let charges = vec![1f32; n_sources];

    // Set tree parameters
    // Library refines tree till fewer than 'n_crit' points per leaf box
    let n_crit = Some(150);
    // Alternatively, users can specify the tree depth they require
    let depth = None;
    // Choose to branches associated with empty leaves from constructed tree
    let prune_empty = true;

    let timed = true; // Optionally time the operators

    // Set FMM Parameters
    // Can either set globally for whole tree, or level-by-level
    let expansion_order = &[6];
    // Parameters which control speed and accuracy of BLAS based field translation
    let singular_value_threshold = Some(1e-5);
    let check_surface_diff = Some(2);

    // Create an FMM
    let svd_mode = FmmSvdMode::Deterministic; // Choose SVD compression mode, random or deterministic

    let mut fmm = SingleNodeBuilder::new(timed)
        .tree(sources.data(), targets.data(), n_crit, depth, prune_empty) // Create tree
        .unwrap()
        .parameters(
            &charges,
            expansion_order, // Set expansion order, by tree level or globally
            Laplace3dKernel::new(), // Choose kernel,
            GreenKernelEvalType::Value, // Choose potential or potential + deriv evaluation
            BlasFieldTranslationSaRcmp::new(
              singular_value_threshold,
              check_surface_diff,
              svd_mode), // Choose field translation
        )
        .unwrap()
        .build()
        .unwrap();

    // Run FMM
    let _  = fmm.evaluate();

    // Lookup potentials by leaf from target leaf boxes
    let leaf_idx = 0;
    let leaf = fmm.tree().target_tree().all_leaves().unwrap()[leaf_idx];
    let leaf_potential = fmm.potential(&leaf);
}
```

# Python Bindings

We use Maturin for Python bindings, and provide an example of installation using the `uv` package manager for Python below.

In order to enable plotting with MayaVi only Python 3.10.* is currently supported.

Dependencies are provided in the associated `pyproject.toml` file.

1. Begin by installing Maturin and pip in a new virtual environment.

```bash
uv venv --python=3.10 && source .venv/bin/activate && uv pip install maturin pip
```

2. Use the Maturin CLI to install the Python bindings into this virtual environment, which additionally will install all required dependencies.

```bash
maturin develop --release
```

Note that Maturin must be run from the `kifmm` crate root, not the workspace root, and that setuptools must be set to a version compatible with Mayavi.

```bash
pip install 'setuptools<69' .
```

We provide example usage of the Python API, as well as visualisation, in the `python/examples` directory.

# C Bindings

We provide a minimal C example using CMake in the `c` folder to use the C bindings directly.
