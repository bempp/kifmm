- test tree index pointer creation
    - charge index pointers
- finish metadata function with flat layout
- re-add ghost exchange
- add kernels
    - Local upward pass
    - global up/down
    - local downward pass

- it's easier not to have an internal FMM object, just store ghost data in the original MultiFMM struct
- can make this global FMM a part of the kernel code on the nominated rank, or add another trait for performing the
global part of the FMM

- Remember, the reason for doing everything flat is so that in M2L kernels only need a single pass through in terms of displacements
- Then another single pass through for Ghost data

- Upward pass kernels can also run completely without a loop like before, all I need to do is ensure that sibling data is contained which it should be.

- Not sure about scaling in P2M, why does it need extra now?
