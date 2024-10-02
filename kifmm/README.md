# Quick Start

Our Rust APIs are simple, with the requirement for no temporary metadata files, or setup of ancillary data structures such as hierarchical trees, required by the user. FMMs are simply parameterised using the builder pattern, with operator chaining to modulate the type of the runtime object. At its simplest, a user only specifies buffers associated with source and target point coordinates, and associated source densities. Trait interfaces implemented for FMM objects allows users to access the associated objects such as kernels and data such as multipole expansions.

Indeed, the full API is more extensive, including features that enable for variable expansion orders by tree level useful for oscillatory problems, accelerated pre-computations for the BLAS based field translations based on randomised SVDs and alternative field translation implementations. Both [Python](https://github.com/bempp/kifmm/tree/main/kifmm/python/examples) and [Rust](https://github.com/bempp/kifmm/tree/main/kifmm/examples) examples can be found in the repository.


```rust
use rand::{thread_rng, Rng};
use green_kernels::{laplace_3d::Laplace3dKernel, types::GreenKernelEvalType};
use kifmm::{BlasFieldTranslationSaRcmp, SingleNodeBuilder, FmmSvdMode};
use kifmm::traits::tree::{FmmTree, Tree};
use kifmm::traits::fmm::Fmm;


fn main() {
    // Generate some random source/target/charge data
    let dim = 3;
    let n_sources = 1000000;
    let n_targets = 2000000;

    // The number of vectors of source densities, FMM is configured from data
    let n = 1;
    let mut rng = thread_rng();
    let mut sources = vec![0f32; n_sources * dim * n];
    let mut targets = vec![0f32; n_targets * dim * n];
    let mut charges = vec![0f32; n_sources * n];

    sources.iter_mut().for_each(|s| *s = rng.gen());
    targets.iter_mut().for_each(|t| *t = rng.gen());
    charges.iter_mut().for_each(|c| *c = rng.gen());

    // Set tree parameters
    // Library refines tree till fewer than 'n_crit' points per leaf box
    let n_crit = Some(150);
    // Alternatively, users can specify the tree depth they require
    let depth = None;
    // Choose to branches associated with empty leaves from constructed tree
    let prune_empty = true;

    // Set FMM Parameters
    // Can either set globally for whole tree, or level-by-level
    let expansion_order = &[6];
    // Parameters which control speed and accuracy of BLAS based field translation
    let singular_value_threshold = Some(1e-5);
    let check_surface_diff = Some(2);

    // Create an FMM
    let svd_mode = FmmSvdMode::Deterministic; // Choose SVD compression mode, random or deterministic

    let mut fmm = SingleNodeBuilder::new()
        .tree(&sources, &targets, n_crit, depth, prune_empty) // Create tree
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
    fmm.evaluate(true); // Optionally time the operators

    // Lookup potentials by leaf from target leaf boxes
    let leaf_idx = 0;
    let leaf = fmm.tree().target_tree().all_leaves().unwrap()[leaf_idx];
    let leaf_potential = fmm.potential(&leaf);
}
```

# Python Bindings

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


# C Bindings

We provide a minimal C example using CMake in the `c` folder to use the C bindings directly.
