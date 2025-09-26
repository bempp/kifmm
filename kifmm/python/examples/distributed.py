import numpy as np
from mpi4py import MPI

from kifmm_py import (
    KiFmmMulti,
    LaplaceKernel,
    HelmholtzKernel,
    MultiNodeTree,
    EvalType,
    BlasFieldTranslation,
    FftFieldTranslation,
    SortKind,
)

world = MPI.COMM_WORLD.Dup()

np.random.seed(0)

dim = 3
dtype = np.float32

# Set FMM Parameters
expansion_order = np.array([6], np.uint64)  # Single expansion order as using n_crit
n_vec = 1
n_crit = 150
n_sources = 10000
n_targets = 10000
prune_empty = True  # Optionally remove empty leaf boxes, their siblings, and ancestors, from the Tree

# Setup source/target/charge data in Fortran order
sources = np.random.rand(n_sources * dim).astype(dtype)
targets = np.random.rand(n_targets * dim).astype(dtype)
charges = np.random.rand(n_sources * n_vec).astype(dtype)

eval_type = EvalType.Value

# EvalType computes either potentials (EvalType.Value) or potentials + derivatives (EvalType.ValueDeriv)
kernel = LaplaceKernel(dtype, eval_type)

local_depth = 2
global_depth = 1
sort_kind = SortKind.SampleSort
n_samples = 100

tree = MultiNodeTree(
    world,
    sources,
    targets,
    charges,
    local_depth,
    global_depth,
    sort_kind,
    prune_empty,
    n_samples,
)

field_translation = FftFieldTranslation(kernel, block_size=32)

print("Rank", world.Get_rank(), "entering FMM")

fmm = KiFmmMulti(expansion_order, tree, field_translation, timed=True)

fmm.evaluate()


print(f"rank {world.rank}, Number of potentials {fmm.all_potentials.shape}")
