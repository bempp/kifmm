- multipole data exchange implementation
    - Test GhostTreeU creation
    - Metadata for MultiNode Tree needs to be implemented too.

    - Need to save origin of all local global roots so that I can broadcast back the evalauted locals after the root FMM has completed.
    - Need to complete all downward pass kernels for local downward pass

    - Load balancing (low priority)



- In multinode builder, root/depth/surfaces must set correctly for the global FMM, and need to incldue this in metadata calculations.

- Downward pass kernels harder on local trees
    - have to run sequentially over available local source trees and then the ghost tree for U and V list
    - shallow local trees will mean that these are relatively fast, e.g. depth 4 local trees.

        - M2L with FFT will require FFT of all multipole data at a given level, this can be done separateley for each source tree and the ghosts
        -
        - e.g. for M2L need to for each source tree:
        Sketch: (g - Ghost tree)

            for each s in source tree:
                for each t in target tree:
                    M2L(t, s)

            for each t target tree:
                M2L(g, t)

        - Either this, or need to flatten data structure with local source trees. Need to think about this tomorrow.

    - P2P sketch is the same
        for each s in source tree:
            for each t in target tree:
                P2P(s, t)
        for each t in target tree:
            M2L(g, t)

