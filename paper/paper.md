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

We present `kifmm-rs` a Rust based implementation of the kernel independent Fast Multipole Method (kiFMM), with Python bindings, that serves as a implementation framework for implementing the kiFMMs [@Ying2004; @Greengard1987]. The FMM is a core algorithm for scientific computing, commonly cited as one of the top algorithmic advances of the twentieth century [@cipra2000best] due to its acceleration of the computation of $N$-body potential evaluation problems of the form,

\begin{equation}
    \phi(x_i) = \sum_{j=1}^N K(x_i, y_j) q(y_j)
    \label{eq:sec:summary:potential}
\end{equation}

to $O(N)$ or $O(N \log(N))$, where the potential $\phi$ is evaluated at a set of target points, $\{x_i\}_{i=1}^M$, due to a set of densities, $\{ q_j \}_{j=1}^N$. Compatible kernels, which commonly arise from second-order elliptic partial differential equations, such as the Laplace kernel which models the electrostatic or gravitational potentials corresponding to a set of source points on a set of target points,

\begin{equation}
    K(x, y) = \begin{cases}
	\frac{1}{2 \pi} \log(\frac{1}{\|x-y\|}),  \> \> (2D) \\
	\frac{1}{4 \pi \|x-y\|}, \> \> (3D)
    \end{cases}
    \label{eq:sec:summary:laplace_kernel}
\end{equation}

The FMM finds a major application in the acceleration of Boundary Element Methods (BEM) for elliptic boundary value problems [@steinbach2007numerical], which can be used to model a wide range of natural phenomena.

Kernel independent variants of the FMM (kiFMMs) replace the analytical series approximations used to compress far-field interactions between clusters of source and target points, with approximation schemes that are based on kernel evaluations and extensions of the algorithm can also handle oscillatory problems specified by the Helmholtz kernel [@Ying2004; @engquist2010fast], with a common underlying algorithmic and software machinery that can be optimised in a kernel-independent manner.

The FMM splits (\ref{eq:sec:summary:potential}) for a given target cluster into \textit{near} and \textit{far} components, the latter of which are taken to be _admissable_, i.e. amenable to approximation via an expansion scheme or alternative interpolation method such as [@Ying2004; @Fong2009],

\begin{equation}
    \phi(x_i) = \sum_{y_j \in \text{Near}(x_i)} K(x_i, y_j) q_j + \sum_{y_j \in \text{Far}(x_i)} K(x_i, y_j) q_j
    \label{eq:sec:summary:near_far_split}
\end{equation}

the near component evaluated directly using the kernel function $K(.,.)$, and the far-field compressed via a _field translation_, referred to as the multipole to local (M2L) translation. This split, in addition to a recursive procedure through a hierarchical data structure, commonly an octree in 3D, used to discretise the problem domain gives rise to the linear/log-linear complexity of the FMM, as the number of far-field interactions which are admissable is limited by a constant depending on the problem dimension. The evaluation of the near field component commonly referred to as the particle to particle (P2P) operation. These two operations conspire to dominate runtimes in practical implementations. An approximate rule of thumb being that the P2P is compute bound, and the M2L is memory bound, acceleration attempts for FMM softwares often focus on reformulations that ensure the M2L has a high arithmetic intensity.

# Statement of need

Previous high-performance codes for computing kiFMMs include [@Malhotra2015; @wang2021exafmm]. However, both of these efforts are packaged as opaque heavily templated C++ libraries, with brittle optimisations for the M2L and P2P operations that make it complex for users or new developers to exchange or experiment with new algorithmic or implementation ideas that improve runtime performance. Furthermore, it is not possible to readily deploy the software on new hardware platforms due to reliance on hand written CMake based builds. Notably, neither softwares support building to Arm targets which are becoming more common as both commodity and HPC platforms.

Novice users are provided with Python bindings in [@wang2021exafmm], however the state of the art distributed code is only accessible via C++ [@Malhotra2015]. Software sub-components such as the hierarchical tree data structures and kernel implementations are not readily re-usable for related algorithmic work by downstream users, and underlying software used in compute kernels such as libraries for BLAS, LAPACK, or the FFT are not readily exchangeable by users to experiment with performance differences across hardware variations.

Our principle contributions with `kifmm-rs` that extend beyond current state of the art implementations are:

- A _highly portable_ Rust-based data-oriented software design that allows us to easily test the impact of different algorithmic approaches, and backends for computational kernels such as BLAS libraries, for critical algorithmic sub-components such as the M2L and P2P operations.
- _State of the art_ single-node performance enabled by the optimisation of BLAS based M2L field translation, based on level 3 operations with high arithmetic intensity that are well suited to current and future hardware architectures that prioritise minimal memory movement per flop.
- The ability to _process multiple right hand sides_ corresponding to the same particle distribution using (\ref{eq:sec:summary:potential}), a common application in BEM.
- _Simple API_, with full Python bindings for non-specialist users. For basic usage all users need to specify are source and target coordinates, and associated source densities, with no temporary files.

`kifmm-rs` is a core dependency for the BEM library `bempp-rs` [@bempp_rs], and we present a detailed exposition behind the algorithmic and implementation approach in [@Kailasa2024]. Currently limited to shared memory systems, distributed memory extensions are an area of active development.

# Software design

## Rust

As a platform for scientific computing, Rust's principal benefits are its build system `Cargo` enabling builds with as little as one terminal command from a user's perspective with dependencies specified in a modern TOML style, and its single centrally supported LLVM based compiler `rustc` that ensures consistent cross-platform performance. Compiled Rust code is compatible with the C application binary interface (ABI), which enables linking the extensive existing scientific ecosystem, and is indeed supported through `Cargo`'s `build.rs` files. This also makes it easy to create language bindings for Rust projects from Python, C++, Fortran and other languages.

## Data oriented design with traits

Rust's 'trait' system effectively allows us to write our code towards the 'data oriented design' paradigm. Which places optimal memory movement through cache hierarchies at the centre of a software's design. Resultantly, 'structs of arrays' are preferred to 'arrays of structs' - i.e. simple objects that wrap contiguously stored buffers in comparison to complex objects storing complex objects.

Traits are contracts between types, and types can implement multiple traits. Therefore we are able to compose complex polymorphic behaviour for our data structures, which consist of simple structs of arrays, by writing all interfaces using traits. In this way, we are able to easily compose sub-components of our software, such as field translation algorithms, explicit SIMD vectorisation strategies for different architectures, single node and MPI distributed octrees, interaction kernels and underlying BLAS or LAPACK implementations. This makes our software more akin to a framework for developing kiFMMs, which can take different flavours, and be used to explore the efficacy of different approaches on different hardware targets.

## API

Our Rust APIs are simple in comparison to other leading codes, with the requirement for no temporary metadata files [@wang2021exafmm], or setup of ancillary data structures such as hierarchical trees [@Malhotra2015], required by the user. FMMs are simply parametrised by performance specific parameters controlling the field translations, or tolerable tree depths, which if left unspecified leaves the software to make a best guess based on available hardware resources. At its simplest, a user only specifies buffers associated with source and target particle coordinates, and associated source densities. They can access potentials, and potential gradients, either by leaf box or indeed the underlying buffer.

```rust
// Rust API for creating an FMM
fn main() {

  let fmm = ...

  fmm.clear(&charges) // reset and run again with new charge data
}
```

The Python API mirrors Rust, and we provide a full set of API usage, file input/output and plotting examples [in the repository](https://github.com/bempp/kifmm/tree/main/kifmm/python/examples).

# Benchmarks

We benchmark our software against other leading implementations on a single node [@Malhotra2015; @wang2021exafmm] for the target architectures in Table (\ref{tab:hardware_and_software}) for achieving a given precision for a common benchmark problem of computing (\ref{eq:sec:summary:potential}) for the three dimensional Laplace kernel (\ref{eq:sec:summary:laplace_kernel}) for a million uniformly distributed source and target points. Optimal parameters were calculated for this setting using a grid search, the results of which can be found in Appendix A of [@Kailasa2024]. We illustrate our software performance using two common acceleration schemes for the field translation, FFT and BLAS level 3 operations, only the former of which are supported by [@Malhotra2015; @wang2021exafmm].


[Space for Plot]


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
| **BLAS**    | Apple Accelerate   | BLIS   |
| **LAPACK**    | Apple Accelerate   | Netlib   |
| **FFT**    | FFTW   | FFTW  |
| **Threading**    | Rayon   | Rayon |


# Figures

<!-- Figures can be included like this: -->
<!-- ![Caption for example figure.\label{fig:example}](fig.png)
and referenced from text using \autoref{fig:example}. -->

<!-- Figure sizes can be customized by adding an optional second parameter: -->
<!-- ![Caption for example figure.](figure.png){ width=20% } -->

# Acknowledgements

Srinath Kailasa is supported by EPSRC Studentship 2417009

# References