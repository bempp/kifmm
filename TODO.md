- weak scaling script
    - how best to fix points per leaf box?
        - ie my points are genearted over an MPI process, but the number of leaves is propoertional to this
        - BUT, I don't control exactly how they are ditributed, or how many to setup. i.e. the optimal depth
            - TO FIND OUT - what kind of depths can the system handle for a single node?
            - Need a new parameter search script.
    - how to measure weak scaling with increasing problem size?
        - not as simple as increasing core count - NUMA features important too
            - thread/process pinning.
            - numa aware pinning specialised for archer 2.

- repeat for BLAS translation
- check budget


- Docs and tests

- Linting

- Merge

- with optimum parameters for single node, have to try and run the same thing on 2 or more nodes.

- run weak scaling per MPI process, as it's not actually useful in my code to do any other way as there is no load balancing at the moment.

- Not sure about scaling in P2M, why does it need extra now?

