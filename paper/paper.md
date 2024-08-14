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

We present `kifmm-rs` a Rust based implementation of the kernel independent Fast Multipole Method (kiFMM) with C bindings, that serves as a implementation framework for kiFMMs [@Ying2004; @Greengard1987]. The FMM is a key algorithm for scientific computing due to its acceleration of the computation of $N$-body potential evaluation problems of the form,

\begin{equation}
    \phi(x_i) = \sum_{j=1}^N K(x_i, y_j) q(y_j)
    \label{eq:sec:summary:potential}
\end{equation}

from $O(N^2)$ to $O(N)$ or $O(N \log(N))$, where the potential $\phi$ is evaluated at a set of target points, $\{x_i\}_{i=1}^M$, due to a set of densities, $\{ q_j \}_{j=1}^N$ and $K(.,.)$ is the interaction kernel. Compatible kernels commonly arise in science and engineering, such as the Laplace kernel which models the electrostatic or gravitational potentials corresponding to a set of source points on a set of target points,

\begin{equation}
    K(x, y) = \begin{cases}
	\frac{1}{2 \pi} \log(\frac{1}{\|x-y\|}),  \> \> (2D) \\
	\frac{1}{4 \pi \|x-y\|}, \> \> (3D)
    \end{cases}
    \label{eq:sec:summary:laplace_kernel}
\end{equation}

FMMs split (\ref{eq:sec:summary:potential}) for a given target cluster into \textit{near} and \textit{far} components, the latter of which are taken to be amenable to approximation,

\begin{equation}
    \phi(x_i) = \sum_{y_j \in \text{Near}(x_i)} K(x_i, y_j) q_j + \sum_{y_j \in \text{Far}(x_i)} K(x_i, y_j) q_j
    \label{eq:sec:summary:near_far_split}
\end{equation}

The evaluation of the near field component is done directly and referred to as the point to point (P2P) operation. The far-field component is compressed via a _field translation_, referred to as the multipole to local (M2L) operation. This split, in addition to a recursive loop through a hierarchical data structure used to discretise the problem, gives rise to the complexity of the FMM. These two operations dominate runtimes in practical implementations, and commonly the focus of implementation optimisations.

# Statement of need

Previous high-performance codes for computing kiFMMs include [@Malhotra2015; @wang2021exafmm]. Both of these efforts are provided as templated C++ libraries with optimisations specialised for x86 architectures. Notably, neither softwares support building to Arm targets which are becoming more common as both commodity and HPC platforms.

Our principle contributions with `kifmm-rs` are:

- A _highly portable_ Rust-based data-oriented software design that allows us to easily test the impact of different algorithmic approaches and computational backends, such as BLAS libraries, for critical algorithmic sub-components as well as deploy to different architectures enabled with Rust's LLVM based compiler. We present the software for shared memory, with plans for distributed memory extension.
- _Competitive_ single-node performance, especially in single precision, enabled by the optimisation of BLAS based M2L field translation, based entirely on level 3 operations with high arithmetic intensity that are well suited to modern hardware architectures that prioritise minimal memory movement per flop.
- The ability to _process multiple sets of source densities_ corresponding to the same point distribution using (\ref{eq:sec:summary:potential}), a common application in the Boundary Element Method.
- _A C API_, using Rust's C ABI compatibility allowing for the construction of bindings into other languages, with full Python bindings for non-specialist users.

A full API description is available as a part of published [documentation](https://bempp.github.io/kifmm/kifmm/index.html). Both [Python](https://github.com/bempp/kifmm/tree/main/kifmm/python/examples) and [Rust](https://github.com/bempp/kifmm/tree/main/kifmm/examples) examples can be found in the repository.

# Software design

Rust traits are contracts between types, and types can implement multiple traits. We are able to compose complex polymorphic behaviour for our data structures, which consist of simple structs of arrays, by writing all interfaces using traits. Enabling us to compose sub-components of our software, such as field translation algorithms, explicit SIMD vectorisation strategies for different architectures (via the Pulp library [@pulp]), tree datastructures, interaction kernels and underlying BLAS or LAPACK implementations (via the RLST library [@rlst]). This makes our software more akin to a framework for developing kiFMMs, which can take different flavours, and be used to explore the efficacy of different FMM approaches across hardware targets and software backends.

# Benchmarks

We benchmark our software against leading implementations on a single node [@Malhotra2015; @wang2021exafmm] in Figure (1) for the high performance x86 architecture in Table (\ref{tab:hardware_and_software}) for achieving relative errors, $\epsilon$, of $1 \times 10^{-11}$ in double precision and $1 \times 10^{-4}$ in single precision with respect to the direct evaluation of potential for points contained in a given box for a benchmark problem of computing (\ref{eq:sec:summary:potential}) for the three dimensional Laplace kernel (\ref{eq:sec:summary:laplace_kernel}) for problem sizes between 100,000 and 1,000,000 uniformly distributed source and target points, which are taken to be the same set. Best parameter settings are described in the Appendix of [@Kailasa2024]. We repeat the benchmark for the Arm architecture for `kifmm-rs` in Figure (2), presented without comparison to competing software due to lack of support.

![X86 benchmarks against leading kiFMM software for achieving relative error $\epsilon$, for `kifmm-rs` the number of sets of source densities being processed is given in brackets, and runtimes are then reported per FMM call.](./images/joss.png)

![Arm benchmarks for achieving relative error $\epsilon$, for `kifmm-rs` the number of sets of source densities being processed is given in brackets, and runtimes are then reported per FMM call.](./images/joss2.png)

Table: Hardware and software used in our benchmarks. We report per core cache sizes for L1/L2 and total cache size for L3. \label{tab:hardware_and_software}

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