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



- Not sure about scaling in P2M, why does it need extra now?
