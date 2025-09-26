from mpi4py import MPI
import numpy as np

print(MPI.get_vendor())


world = MPI.COMM_WORLD.Dup()

print(world.rank)

from kifmm_py import KiFmmMulti
