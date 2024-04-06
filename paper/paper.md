---
title: 'kifmm-rs: A Kernel-Independent Fast Multipole Method in Rust'
tags:
  - Rust
  - FMM
authors:
  - name: Srinath Kailasa
    orcid: 0000-0001-9734-8318
    equal-contrib: true
    affiliation: "1" # (Multiple affiliations must be quoted)
affiliations:
 - name: Department of Mathematics, University College London, UK
   index: 1
date: 5 April 2024
bibliography: paper.bib

---

# Summary

The Kernel Independent Fast Multipole Method (kiFMM) is ...

# Statement of need

`kifmm-rs` is an

- Rust package build for speed and flexibility
- API designed to be user friendly, and easy to bind
- Simple trait based design, allow for separation of concerns and interface
- Can
  - evaluate potentials, potential gradients, for a range of compatible kernels
  - heterogenous support for critical operations
  - multi-platform deployment with Rust
  - state of the art performance on a single node.
  - design flexible, can easily extend to multi-node problems in a future release.

Combination of
  - speed + design + extensibility to new functionality (related algorithms)


Past and ongoing research projects

- where does this software fit in?
- Older FMM efforts
- Embedded within new Bempp-rs


# Mathematics

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
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

<!-- Figures can be included like this: -->
![Caption for example figure.\label{fig:example}](fig.png)
and referenced from text using \autoref{fig:example}.

<!-- Figure sizes can be customized by adding an optional second parameter: -->
<!-- ![Caption for example figure.](figure.png){ width=20% } -->

# Acknowledgements

Srinath Kailasa is supported by EPSRC Studentship 2417009

# References