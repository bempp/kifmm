1. make builder generic over homog/inhomog kernels
2. make operators generic over the same kernels
3. tests/cleanup.

4. Need to make a choice, at which level will the FMM be generic?

The KiFMM struct, and associated data is currently specific only to homogenous kernels
- This has to be made explicit

- The builder has to be matched to

