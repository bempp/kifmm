---
title: 'kifmm-rs: A Kernel-Independent Fast Multipole Framework in Rust'
tags:
  - Rust
  - FMM
authors:
  - name: Srinath Kailasa
    orcid: 0000-0001-9734-8318
    equal-contrib: false
    affiliation: "1" # (Multiple affiliations must be quoted)
affiliations:
 - name: Department of Mathematics, University College London, UK
   index: 1
date: 5 April 2024
bibliography: paper.bib

---

# Summary

We present `kifmm-rs` a Rust based implementation of the kernel independent Fast Multipole Method (kiFMM), with C bindings, that serves as a implementation framework for kiFMMs [@Ying2004; @Greengard1987]. The FMM is a key algorithm for scientific computing, commonly cited as one of the top algorithmic advances of the twentieth century [@cipra2000best] due to its acceleration of the computation of $N$-body potential evaluation problems of the form,

\begin{equation}
    \phi(x_i) = \sum_{j=1}^N K(x_i, y_j) q(y_j)
    \label{eq:sec:summary:potential}
\end{equation}

from $O(N^2)$ to $O(N)$ or $O(N \log(N))$, where the potential $\phi$ is evaluated at a set of target points, $\{x_i\}_{i=1}^M$, due to a set of densities, $\{ q_j \}_{j=1}^N$ and $K(.,.)$ is the interaction kernel. Compatible kernels commonly arise from second-order elliptic partial differential equations, such as the Laplace kernel which models the electrostatic or gravitational potentials corresponding to a set of source points on a set of target points,

\begin{equation}
    K(x, y) = \begin{cases}
	\frac{1}{2 \pi} \log(\frac{1}{\|x-y\|}),  \> \> (2D) \\
	\frac{1}{4 \pi \|x-y\|}, \> \> (3D)
    \end{cases}
    \label{eq:sec:summary:laplace_kernel}
\end{equation}

The FMM also finds a major application in the acceleration of Boundary Element Methods (BEM) for elliptic boundary value problems [@steinbach2007numerical], which can be used to model a wide range of natural phenomena. Kernel independent variants of the FMM (kiFMMs) replace the analytical series approximations used to compress far-field interactions between clusters of source and target points, with approximation schemes that are based on kernel evaluations and extensions of the algorithm can also handle oscillatory problems specified by the Helmholtz kernel [@Ying2004; @engquist2010fast], with a common underlying algorithmic and software machinery that can be optimised in a kernel-independent manner.

The FMM splits (\ref{eq:sec:summary:potential}) for a given target cluster into \textit{near} and \textit{far} components, the latter of which are taken to be _admissable_, i.e. amenable to approximation via an expansion scheme or alternative interpolation method such as [@Ying2004; @Fong2009],

\begin{equation}
    \phi(x_i) = \sum_{y_j \in \text{Near}(x_i)} K(x_i, y_j) q_j + \sum_{y_j \in \text{Far}(x_i)} K(x_i, y_j) q_j
    \label{eq:sec:summary:near_far_split}
\end{equation}

the near component evaluated directly using the kernel function $K(.,.)$, and the far-field compressed via a _field translation_, referred to as the multipole to local (M2L) translation. This split, in addition to a recursive loop through a hierarchical data structure, commonly an octree for three dimensional problems, gives rise to the linear/log-linear complexity of the FMM, as the number of far-field interactions which are admissable is limited by a constant depending on the problem dimension. The evaluation of the near field component commonly referred to as the particle to particle (P2P) operation. These two operations conspire to dominate runtimes in practical implementations. An approximate rule of thumb being that the P2P is compute bound, and the M2L is memory bound, acceleration attempts for FMM softwares often focus on reformulations that ensure the M2L has a high arithmetic intensity.

# Statement of need

Previous high-performance codes for computing kiFMMs include [@Malhotra2015; @wang2021exafmm]. However, both of these efforts are provided as templated C++ libraries with optimisations specialised for x86 architectures, for the M2L and P2P operations that make it complex for users or new developers to exchange or experiment with new algorithmic or implementation ideas that improve runtime performance. Notably, neither softwares support building to Arm targets which are becoming more common as both commodity and HPC platforms. In both softwares, sub-components such as the octree data structures and kernel implementations are not readily re-usable for related algorithmic work by downstream users, and underlying software used in compute kernels such as libraries for BLAS, LAPACK, or the FFT are not readily exchangeable.

Our principle contributions with `kifmm-rs` are:

- A _highly portable_ Rust-based data-oriented software design that allows us to easily test the impact of different algorithmic approaches and computational backends, such as BLAS libraries, for critical algorithmic sub-components such as the M2L and P2P operations as well as deploy to different CPU targets. We present the software for shared memory, with plans for distributed memory extension.
- _Competitive_ single-node performance, especially in single precision, enabled by the optimisation of BLAS based M2L field translation, based entirely on level 3 operations with high arithmetic intensity that are well suited to current and future hardware architectures that prioritise minimal memory movement per flop.
- The ability to _process multiple sets of source densities_ corresponding to the same particle distribution using (\ref{eq:sec:summary:potential}), a common application in BEM.
- _A C API_, using Rust's C ABI compatibility allowing for the construction of bindings into other languages, with full Python bindings for non-specialist users. For basic usage all users need to specify are source and target coordinates, and associated source densities, with no temporary files.

`kifmm-rs` is a core dependency for the BEM library `bempp-rs` [@bempp_rs], and we present a detailed exposition behind the algorithmic and implementation approach in [@Kailasa2024].

# Software design

## Rust

As a platform for scientific computing, Rust's principal benefits are its build system `Cargo` enabling builds with as little as one terminal command from a user's perspective with dependencies specified in a modern TOML style, and its single centrally supported LLVM based compiler `rustc` that ensures consistent cross-platform performance. Compiled Rust code is compatible with the C application binary interface (ABI), which enables linking the extensive existing scientific ecosystem. This also makes it easy to create language bindings for Rust projects from Python, C++ and other languages.

## Data oriented design with traits

Rust's 'trait' system effectively allows us to write our code towards the 'data oriented design' paradigm. Which places optimal memory movement through cache hierarchies at the centre of a software's design. Resultantly, 'structs of arrays' are preferred to 'arrays of structs' - i.e. simple objects that wrap contiguously stored buffers in comparison to complex objects storing complex objects.

Traits are contracts between types, and types can implement multiple traits. Therefore we are able to compose complex polymorphic behaviour for our data structures, which consist of simple structs of arrays, by writing all interfaces using traits. In this way, we are able to easily compose sub-components of our software, such as field translation algorithms, explicit SIMD vectorisation strategies for different architectures (via the Pulp library [@pulp]), single node and MPI distributed octrees, interaction kernels and underlying BLAS or LAPACK implementations (via the RLST library [@rlst]). This makes our software more akin to a framework for developing kiFMMs, which can take different flavours, and be used to explore the efficacy of different FMM approaches across hardware targets and software backends.

## API

Our Rust APIs are simple, with the requirement for no temporary metadata files [@wang2021exafmm], or setup of ancillary data structures such as hierarchical trees [@Malhotra2015], required by the user. FMMs are simply parameterised using the builder pattern, with operator chaining to modulate the type of the runtime object. At its simplest, a user only specifies buffers associated with source and target particle coordinates, and associated source densities. Trait interfaces implemented for FMM objects allows users to access the associated objects such as kernels and data such as multipole expansions.

```rust
use rand::{thread_rng, Rng};
use green_kernels::{laplace_3d::Laplace3dKernel, types::EvalType};
use kifmm::{BlasFieldTranslationSaRcmp, SingleNodeBuilder};
use kifmm::traits::tree::{FmmTree, Tree};
use kifmm::traits::fmm::Fmm;

fn main() {
    // Generate some random source/target/charge data
    let dim = 3;
    let nsources = 1000000;
    let ntargets = 2000000;

    // The number of vectors of source densities, FMM is configured from data
    let n = 1;
    let mut rng = thread_rng();
    let mut sources = vec![0f32; nsources * dim * n];
    let mut targets = vec![0f32; ntargets * dim * n];
    let mut charges = vec![0f32; nsources * n];

    sources.iter_mut().for_each(|s| *s = rng.gen());
    targets.iter_mut().for_each(|t| *t = rng.gen());
    charges.iter_mut().for_each(|c| *c = rng.gen());

    // Set tree parameters
    // Library refines tree till fewer than 'n_crit' particles per leaf box
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
    let mut fmm = SingleNodeBuilder::new()
        .tree(&sources, &targets, n_crit, depth, prune_empty) // Create tree
        .unwrap()
        .parameters(
            &charges,
            expansion_order, // Set expansion order, by tree level or globally
            Laplace3dKernel::new(), // Choose kernel,
            EvalType::Value, // Choose potential or potential + deriv evaluation
            BlasFieldTranslationSaRcmp::new(
              singular_value_threshold,
              check_surface_diff
              ), // Choose field translation
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

Indeed, the full API is more extensive, including features that enable for variable expansion orders by tree level useful for oscillatory problems, accelerated pre-computations for the BLAS based field translations based on randomised SVDs and alternative field translation implementations. Both [Python](https://github.com/bempp/kifmm/tree/main/kifmm/python/examples) and [Rust](https://github.com/bempp/kifmm/tree/main/kifmm/examples) examples can be found in the repository.

# Benchmarks

We benchmark our software against other leading implementations on a single node [@Malhotra2015; @wang2021exafmm] in Figure (1) for the high performance x86 architecture in Table (\ref{tab:hardware_and_software}) for achieving relative errors, $\epsilon$, of $1 \times 10^{-11}$ in double precision and $1 \times 10^{-4}$ in single precision with respect to the direct evaluation of potential for particles contained in a given box for a benchmark problem of computing (\ref{eq:sec:summary:potential}) for the three dimensional Laplace kernel (\ref{eq:sec:summary:laplace_kernel}) for problem sizes between 100,000 and 1,000,000 uniformly distributed source and target points, which are taken to be the same set. Optimal parameters were calculated for this setting using a grid search, the results of which can be found in the Appendix of [@Kailasa2024]. We illustrate our software's performance using our BLAS based field translation method, which can handle multiple sets of source densities for a given set of source and target particles. This is particularly effective in single precision, where the required data is smaller and therefore results in fewer cache invalidations. We repeat the benchmark for the Arm architecture for `kifmm-rs` in Figure (2), presented without comparison to competing software due to lack of support, we see that the BLAS based field translation approach is effective for handling multiple sets of source densities in single precision due to the large cache sizes available on this architecture.

![X86 benchmarks against leading kiFMM software for achieving relative error $\epsilon$, for `kifmm-rs` the number of sets of source densities being processed is given in brackets, and runtimes are then reported per FMM call.](./images/joss.jpg)


![Arm benchmarks for achieving relative error $\epsilon$, for `kifmm-rs` the number of sets of source densities being processed is given in brackets, and runtimes are then reported per FMM call.](./images/joss2.jpg)

Table: Hardware and software used in our benchmarks, for the Apple M1 Pro we report only the specifications of its 'performance' CPU cores. We report per core cache sizes for L1/L2 and total cache size for L3. \label{tab:hardware_and_software}

|  | **Apple M1 Pro** | **AMD 3790X** |
|----------|----------|----------|
| **Cache Line Size**    | 128 B| 64 B   |
| **L1i/L1d**    | 192/128 KB   | 32/32 KB   |
| **L2**    | 12 MB   | 512 KB   |
| **L3**    | 12 MB   | 134 MB  |
| **Memory**    | 16 GB   | 252 GB   |
| **Max Clock Speed**    | 3.2 GhZ  | 3.7 GhZ   |
| **Sockets/Cores/Threads**    | 1/8/8   | 1/32/64   |
| **Architecture**    | Arm V8.5   | x86   |
| **BLAS**    | Apple Accelerate   | Open BLAS   |
| **LAPACK**    | Apple Accelerate   | Open BLAS  |
| **FFT**    | FFTW   | FFTW  |
| **Threading**    | Rayon   | Rayon |
| **SIMD Extensions** | Neon | SSE, SSE2, AVX, AVX2|

# Acknowledgements

Srinath Kailasa is supported by EPSRC Studentship 2417009

# References