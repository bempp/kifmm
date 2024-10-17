- Need to test end-to-end
    - upward pass is working
        - wasn't working, found a bug in multi-node tree construction
        - tested that all multipoles in both single and multi-node are the same
    - found a few bugs in ghost exchange.
    - downward pass broken
        - trying, single MPI process, global depth = local depth = 1, 1 mpi process
            - still not converging, but in a different way
        - need to check
            - what multipoles are compared to regular single node FMM
            - fundamentally something off about the depths of the trees for distributed FMM
            total depth is calculated as local depth + global depth
            - but local depth 1 implies 2 levels
            - i.e. the depth is exclusive of the first level, so I'm fucking up anything to do with total depth, for one.
                - need to see how this plays into upward pass working somehow. Probably part of the reason P2M needs more scaling.
            - the depths seem fine, especially now after fixed evaluation of downward pass for multinode

        - to check:
            - what are locals at each level with respect to single node locals, are these expected?
                - first check right after global FMM, are these as expected?
                - then check for first level after

            - the p2p ones are still also off.

- Need to test threading controls i.e. trade-off between threading choices on MPI and rayon threads
- Need to figure out how to pin MPI processes to nodes on Kathleen


- Not sure about scaling in P2M, why does it need extra now?
