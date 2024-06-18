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

where the potential $\phi$ is evaluated at a set of targets, $\{x_i\}_{i=1}^M$, due to a set of densities, $\{ q_j \}_{j=1}^N$, corresponding to a set of sources, $\{y_j\}_{j=1}^N$, and $K(., .)$ is the interaction kernel from $O(N^2)$ to $O(N)$ or $O(N \log (N))$ for interaction kernels for second order elliptic partial differential equations. Such kernels commonly arise in scientific computations, such as the Laplace kernel which models the electrostatic or gravitational potentials corresponding to a set of source points on a set of target points,

\begin{equation}
    K(x, y) = \begin{cases}
	\frac{1}{2 \pi} \log(\frac{1}{\|x-y\|}),  \> \> (2D) \\
	\frac{1}{4 \pi \|x-y\|}, \> \> (3D)
    \end{cases}
    \label{eq:sec:summary:laplace_kernel}
\end{equation}

Indeed the FMM finds a major application in the acceleration of Boundary Element Methods (BEM) for elliptic boundary value problems [@steinbach2007numerical], which can be used to model a wide range of natural phenomena.

Kernel independent variants of the FMM (kiFMMs) replace the analytical series approximations used to compress far-field interactions between clusters of source and target points, with approximation schemes that are based on kernel evaluations and extensions of the algorithm can also handle oscillatory problems specified by the Helmholtz kernel [@Ying2004; @engquist2010fast], with a common underlying algorithmic and software machinery that can be optimised in a kernel-independent manner.

The power of FMMs is derived from degenerate approximations of the kernel function such that (\ref{eq:sec:summary:potential}) when evaluated between distant clusters of distant source and target points can be expressed a short sum,

\begin{equation}
    \phi(x_i) = \sum_{p=1}^{P} A(x_i)B(y_j)q(y_j)
    \label{eq:sec:summary:degenerate_kernel}
\end{equation}

where $P$, which we call the `expansion order', is chosen such that $P \ll M$, $P \ll N$, and the functions $A$ and $B$ are determined by the particular approximation scheme of an FMM method. In the original presentation the calculation of,

$$ \hat{q} = B(y_j)q(y_j) $$

corresponds to the construction of analytical expansions of the kernel function representing the potential due to the source densities at a set of source points, and the calculation,

$$ \phi = A(x_i)\hat{q} $$

represents the evaluation of this potential at a set of target points. The approximation (\ref{eq:sec:summary:degenerate_kernel}) can be formed when the distance between clusters of sources and targets is sufficient, or \textit{admissable}. By splitting (\ref{eq:sec:summary:potential}) for a given target cluster into \textit{near} and \textit{far} components, the latter of which are taken to be admissable and can be approximated by (\ref{eq:sec:summary:degenerate_kernel}),

\begin{equation}
    \phi(x_i) = \sum_{y_j \in \text{Near}(x_i)} K(x_i, y_j) q_j + \sum_{y_j \in \text{Far}(x_i)} K(x_i, y_j) q_j
    \label{eq:sec:summary:near_far_split}
\end{equation}

with the near component evaluated directly using the kernel function $K(.,.)$. This split, in addition to a recursive procedure through a hierarchical data structure, commonly an octree in 3D, used to discretise the problem domain gives rise to the linear/log-linear complexity of the FMM, as the number of far-field interactions which are admissable is limited by a constant depending on the problem dimension, commonly referred to as the multipole to local field translation (M2L) operation. The near field operations are computed directly using the kernel function $K(.,.)$, commonly referred to as the particle to particle (P2P) operation. These two operations conspire to dominate runtimes in practical implementations.

# Statement of need

Previous high-performance codes for computing kiFMMs include [@Malhotra2015; @wang2021exafmm]. However, both of these efforts are packaged as opaque heavily templated C++ libraries, with brittle optimisations for the M2L and P2P operations that make it complex for users or new developers to exchange or experiment with new algorithmic or implementation ideas that improve runtime performance. Furthermore, it is not possible to easily compare whether performance has been attained by a specific algorithmic approach, or readily deploy the software on new hardware platforms due to reliance on hand written CMake based builds. Novice users are provided with Python bindings in [@wang2021exafmm], however the state of the art distributed code is only accessible via C++ [@Malhotra2015]. Software sub-components such as the hierarchical tree data structures and kernel implementations are not readily re-usable for related algorithmic work by downstream users, and underlying software used in compute kernels such as libraries for BLAS, LAPACK, or the FFT are not readily exchangeable by users to experiment with performance differences across hardware variations.

Our principle contributions with `kifmm-rs` that extend beyond current state of the art implementations are:

- A _highly portable_ Rust-based data-oriented software design that allows us to easily test the impact of different algorithmic approaches, and backends for computational kernels such as BLAS libraries, for critical algorithmic sub-components such as the M2L and P2P operations.
- _State of the art_ single-node performance enabled by the optimisation of BLAS based M2L field translation, based on level 3 operations with high arithmetic intensity that are well suited to current and future hardware architectures that prioritise minimal memory movement per flop.
- The ability to process multiple right hand sides corresponding to the same particle distribution using (\ref{eq:sec:summary:potential}), a common application in BEM.
- Full Python bindings for non-specialist users.

`kifmm-rs` is a core dependency for the BEM library `bempp-rs` [@bempp_rs], and we present a detailed exposition behind the algorithmic and implementation approach in [@Kailasa2024]. Currently limited to shared memory systems, distributed memory extensions are an area of active development.

# Software design

## Rust as a platform for scientific computing

Rust's build system

- Simple cross platform builds.
- Standardised autovectoriser, i.e. single compiler, based on LLVM
- SIMD via Pulp.
- C ABI compatibility, enables language bindings, as well as access to open source.

## Data oriented design with traits

Rust's trait system and data oriented design.

- What is data oriented design?
- How do Rust's traits enable us to write in structs of arrays.

## Extensibility and portability

- How do traits enable us to create an extensible software that is ready for new kernel implementations, distriuted memory implementations, and re-use of subcomponents in related algorithms?



# Benchmarks

We benchmark our software against other leading implementations on a single node [@Malhotra2015, @wang2021exafmm] for the target architectures in Table (\ref{tab:hardware_and_software}) for achieving a given precision for a common benchmark problem of computing (\ref{eq:sec:summary:potential}) for the three dimensional Laplace kernel (\ref{eq:sec:summary:laplace_kernel}) for a million uniformly distributed source and target points.

Table: Hardware and software used in our benchmarks, for the Apple M1 Pro we report only the specifications of its 'performance' CPU cores. We report per core cache sizes for L1/L2 and total cache size for L3. We note that the Apple M series of processors are designed with unusually large cache sizes, as well as `unified memory' architectures enabling rapid data access across specialised hardware units such as the performance CPU cores and the specialised matrix coprocessor used for BLAS operations when run with Apple's Accelerate framework \label{tab:hardware_and_software}

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


<!-- # Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you canc do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

<!-- Figures can be included like this: -->
<!-- ![Caption for example figure.\label{fig:example}](fig.png)
and referenced from text using \autoref{fig:example}. -->

<!-- Figure sizes can be customized by adding an optional second parameter: -->
<!-- ![Caption for example figure.](figure.png){ width=20% } -->

# Acknowledgements

Srinath Kailasa is supported by EPSRC Studentship 2417009

# References