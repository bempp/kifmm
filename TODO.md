1. Ivestigate adaptor pattern instead of impls on raw kernel types
    - this way can implement a scale invariant FMM despite Rust's limitations on overloading.
    - Implement as adaptor pattern
2. eval type can derive default, and would simplify KiFMM default
3. Scales to apply in kernels are features of green's fucntion properties, should be inferred to maintain genericity. Same with level index of operators
4. potentially incorrect scaling on target helmholtz operators.