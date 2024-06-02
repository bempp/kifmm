Code
----
3. Think about how to add different expansion orders for check/equivalent surfaces
7. Merge, and fixup SIMD branch - probably easier to just create a new branch?
8. Test SIMD branch for Helmholtz.

HF Extensions
--------------
1. Directional compression scheme (8 directions), to speed up Helmholtz pre-computation
in double precision.
2. Variable expansion order + c/e adjustable (exists).

Paper (i)
---------
Write up questions about MFS to ask.
    1. Figure out where in the paper it talks about C vs E surfaces?
    2. Would we also expect exponential convergence in 3D?
    2.i. Is it dependent on geometry (box?)

- What is M2L expensive, regardless of sparsification approach? Is it an inherently memory bound calculation? Why?

- What makes acceleration of the M2L hard depending on the approach?
    - already a lot of good content there, about how good implementations rely on the properties of the kernel
    - also rely on reducing memory accesses as well as flops, and finding a way to keep the arithmetic intensity high (even if the calculation remains memory bound)

SIMD benchmarks
    - What are good benchmarks? Where are we comparable to PVFMM?

Paper (ii)
----------
- New framework for developing kernel independent FMMs, with flexible design to mix and match field translation, kernel implementations.
- Rust enables easy deployment to multiple architectures.
- Python bindings with PyO3, enabled by C ABI compatiblity of Rust
- Designed in Rust, demonstrating the versatility of Rust for scientific computing.
- How have we implemented data oriented design, and used traits to do so?
- Demonstrate benchmark performance with respect to key softwares on a single node
- x86 and arm experiments vs other codes.
- Functionality that exists over existing codes:
    - variable expansion orders.
    - BLAS based field translations (flexible backend implementation), demonstrate state of the art performance and ability to handle multiple FMMs simulateneously over same set of coordiantes.
    - Scope to easily extend to multi-node setting due to trait design.
    - moderate frequency Helmholtz problems (variable expansion order).

