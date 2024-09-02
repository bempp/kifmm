- multipole data exchange implementation
    - Change buffer communication so that packets are implicit rather than explicit, at least for v list
    - need to communicate index pointers for coordinate data, somehow accounting for displacement from different packets
    - Ghost tree creation

- node election (easy)
    - remainder of upward pass on elected node.

- local data exchange to each local root (also easy)
    - independent downward pass

- Downward pass kernels harder on local trees
    - have to run sequentially over available local source trees and then the ghost tree for U and V list
    - shallow local trees will mean that these are relatively fast, e.g. depth 4 local trees.

