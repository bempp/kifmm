---
title: 'kifmm-rs: A Kernel-Independent Fast Multipole Framework in Rust'
tags:
  - Rust
  - FMM
authors:
  - name: Srinath Kailasa
    orcid: 0000-0001-9734-8318
    equal-contrib: false
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
affiliations:
 - name: Department of Mathematics, University College London, UK
   index: 1
 - name: Department of Engineering, University of Cambridge, UK
   index: 2
date: 22 January 2025
bibliography: paper.bib
---

# Summary

`kifmm-rs` is an open-source framework for implementing Fast Multipole Methods (FMMs) in shared and distributed memory in three dimensions. FMMs accelerate the calculation of $N$-body potential evaluation problems that arise in computational science and engineering of the form,

\begin{equation}
    \phi(x_i) = \sum_{j=1}^N K(x_i, y_j) q(y_j)
    \label{eq:sec:summary:potential}
\end{equation}

where the potential $\phi$ is evaluated at a set of target particles, $\{x_i\}_{i=1}^M$, due to a set of densities, $\{ q_j \}_{j=1}^N$ collocated at a set of source particles $\{y_j\}_{j=1}^N$. This evaluation can be interpreted as a matrix-vector multiplication,

\begin{equation}
    \symbfup{\phi} = \mathbf{K} \mathbf{q}
    \label{eq:sec:summary:matvec}
\end{equation}

where $\symbfup{\phi}$ is an $M$ dimensional vector, $\mathbf{K}$ is a dense $M \times N$ matrix and $\mathbf{q}$ is an $N$ dimensional vector. As $\mathbf{K}$ is dense, the naive evaluation of \eqref{eq:sec:summary:matvec} is of $O(M \cdot N)$ complexity. FMMs provide a way of evaluating \eqref{eq:sec:summary:matvec} in $O(P(M+N))$ where $P$ is a user defined parameter $P \ll M, N$ called the _expansion order_, for a range of interaction kernels $K(\cdot)$ that commonly arise from elliptic partial differential equations (PDEs). The prototypical example for which the FMM was first presented is the Laplace kernel [@Greengard1987], which describes particles interacting electrostatically or gravitationally,

\begin{equation}
    K(x, y) = \begin{cases}
	\frac{1}{2 \pi} \log(\frac{1}{\|x-y\|}),  \> \> \text{, 2D} \\
	\frac{1}{4 \pi \|x-y\|}, \> \> \text{, 3D}
    \end{cases}
å    \label{eq:sec:summary:laplace_kernel}
\end{equation}

Since their initial introduction FMMs have been developed for a wide variety of kernel functions such as the Helmholtz kernel which arises from the time-independent wave equation [@rokhlin1993diagonal],

\begin{equation}
    K(x, y) = \begin{cases}
      \frac{i}{4} H_0^{(1)}(k |\mathbf{x-y}|)  \text{, 2D}\\
        \frac{e^{ik |\mathbf{x-y}|}}{4\pi |\mathbf{x-y}|}  \text{, 3D}
  \end{cases}
  \label{eq:sec:summary:helmholtz_kernel}
\end{equation}

As well as the Stokes kernel describing fluid flow [@tornberg2008fast], and the Navier kernel used in elastostatics [@fu1998fast] among others. The major application of FMMs is found in the accelerated application of dense operator matrices that arise from the integral formulation of partial differential equations, and as a result FMM algorithms are a crucial component across simulations in many domains, from medical imaging to geophysics.

FMMs accelerate the evaluation of \eqref{eq:sec:summary:matvec} by decomposing the problem domain using a hierarchical tree, a quadtree in 2D and an octree in 3D, the algorithm consisting of a series of operations collectively referred to as _field translations_. Evaluations of the off-diagonal blocks of the matrix $\mathbf{K}$ in \eqref{eq:sec:summary:matvec} correspond to the `multipole to local' or M2L field translation which summarise the long-range interactions of a local cluster of target points, and the evaluation of the diagonal blocks correspond to unavoidable direct evaluations of the kernel function for evaluating near-range interactions. M2L is the most challenging optimisation bottleneck in FMMs due to its memory-bound nature.

Many parts of an implementation are common across FMM algorithms, such as the tree setup and the kernel function implementation. As a result, our framework is presented as a set of modular, re-usable, sub-components of each of the key algorithmic sub-components: (i) the tree in shared and distributed memory, (ii) the field translation operations and (iii) the kernel evaluation. We use our framework to develop an implementation of the so-called kernel independent Fast Multiple Method (kiFMM) [@Ying2004], compatible with a wide variety of elliptic PDE kernels with an implementation for the Laplace and Helmholtz kernels provided currently. The kiFMM has favourable implementation properties due to its formulation in terms of high operational-intensity BLAS operations.

# Statement of need

Other notable software efforts for the FMM include PVFMM [@Malhotra2015], ExaFMM [@wang2021exafmm], ScalFMM [@blanchard2015scalfmm] and TBFMM [@bramas2020tbfmm]. PVFMM, ScalFMM and TBFMM are fully distributed implementations of the black box FMM [@fong2009black] and kiFMM respectively. The latter two are notable for being distributed using a task-based runtime, with the former using a more traditional MPI based approach. ExaFMM offers a shared memory implementation of the kiFMM, with Python interfaces and a simple template based design. A commonality of previous implementations is the coupling of algorithmic optimisation with implementation optimisation. For example, ExaFMM and PVFMM both offer field translations that are highly optimised for x64 architectures and lack ARM support, with ScalFMM and TBFMM being tailored to the runtime systems they are designed for.

Our design is data oriented, with complex behaviour composed over simple linear data structures using Rust's trait system. Traits offer a way of specifying behaviour defining contracts between types that are enforced by Rust's compiler. This enables the exposure of performance critical sections in a manner that is easy to optimise in isolation. In contrast to previous software efforts, our design enables a decoupling of the underlying algorithmic implementation and the software optimisation. This has enabled the comparative analysis of the implementation of the critical M2L field translation [@Kailasa2024], and the future iterative refinement of field translations in response to algorithmic and hardware advances.

Our principle contributions with `kifmm-rs` can therefore be summarised as:

- _A modular data-oriented design_ which enables field translations to be implementd over simple linear data structures, allowing us to easily examine the impact of different algorithmic approaches and computational backends, such as BLAS libraries, for critical algorithmic sub-components.
- _Optimisations for ARM and x86 targets_. ARM targets are increasingly common in high-performance computing systems, with portability enabled by Rust's LLVM based compiler.
- _Competitive shared and distributed memory performance_. With state of the art M2L performance [@Kailasa2024], as well as a communication-optimal distributed memory implementation inspired by [@Ibeid2016].
- _A C API_, using Rust's C ABI compatibility allowing for the construction of bindings into other languages, with full Python bindings for non-specialist users.
- _A moderate frequency Helmholtz FMM_. Helmholtz FMMs are often presented for the low-frequency case [@wang2021exafmm,@Malhotra2015], due to the challenging data sizes involved. We present an extension of the kiFMM to the Helmholtz problem which has proven so far to be effective in single precision for relatively high wave numbers of up to $k \sim 100$.

`kifmm-rs` is currently a core library used as a part of the Bempp boundary element project [@bempp_rs]. A full API description is available as a part of published [documentation](https://bempp.github.io/kifmm/kifmm/index.html). [Python](https://github.com/bempp/kifmm/tree/main/kifmm/python/examples), [Rust](https://github.com/bempp/kifmm/tree/main/kifmm/examples) and [C](https://github.com/bempp/kifmm/tree/main/kifmm/c) examples can be found in the repository.

# Acknowledgements

Srinath Kailasa is supported by EPSRC Studentship 2417009

# References