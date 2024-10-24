# Running kifmm-rs on HPC

Here we document how to run this software on HPC systems, tried and tested configurations and optimal parameter search and setting.

## Archer 2

As there are 8 NUMA regions per node, 4 in each processor, each with 16 physical cores. We should aim to run 8 MPI processes per node, each with 16 threads, to fully utilise the nodes of Archer 2.

In kifmm-rs each MPI process is responsible for its portion of the global domain, and performs the recursive FMM loop on the data it controls, communicating with the nearest neighbours only. As a result, each MPI process needs to own a NUMA region.

This means we need to find out just how much a single node can handle, say 10e6 points per NUMA region, leads to approximately 80e6 over the whole node. There are up 1024 nodes available in standard QOS, so if we get expected scalability should be able to get to 80e9 quite easily.

### Compilation

- PrgEnv-aocc
- set RUSTFLAGS as well as LD lib path to find BLAS/LAPACK
- BLAS/LAPACK symlinked with libblas.so and liblapack.so
- Same for MPI?

### Dependencies

- Clang - because of bindgen
- MPI
- BLAS/LAPACK installation

Other dependencies such as fftw are built and linked by kifmm-rs as they contain custom bindings.

### Hardware

HPE Cray EX, 5860 compute nodes. Each with 128 cores, dual AMD EPYC 7742 64-core 2.25 GHz processors. total of 750,080 cores.

Details

- Memory per node: 256 GB standard, 512 GB, high memory
- Memory per core: 2GB standard, 4GB high memory
- L1i & L1d/L2 per core: 32kB/512kB
- L3 per 4 cores: 16MB
- AVX2 and below
- Network: 2x100 Gb/s ports per node.

L3 sizes are small per CCX compared to AMD 3290X workstation, actually smaller than M1 Pro. L1 and L2 sizes are same as on AMD 3290X.

This has implications for FFT based M2L performance, and block sizes, which should be relatively small compared to these machines.


Three types of node available to users
- login
- data analysis
- compute


Nodes are grouped into partition, specified with the `--partition` option in the slurm jobscript.

The relevant nodes for kifmm-rs are:

- standard - CPU nodes, 2x AMD EPYC 7742 64 Core processor 256/512 GB memory, 5860 nodes available.

### Software

- As of 2024 Rust should be installed by the user, the easiest way to avoid building from source is to use the [rustup script](https://www.rust-lang.org/tools/install).

May have to look into running multiple jobs to fully utilise node in order to get more use out of allocation.

The standard approach recommended for archer is to place processes sequentially on nodes until the maximum number of tasks is reached. `xthi` can be used to verify process placement.

AMD Optimising Compiler Collection (AOCC)

- Clang based, includes flang based fortran compiler

MPI:

Cray MPICH has two different low level protocols to transfer data across the network. UCX and OFI - OFI is the default.

- UCX suggested as a faster alternative for programs with a lot of collectives, this doesn't require recompilation simply a loading of the appropriate fabric in the slurm script.

```bash
module load PrgEnv-aocc # Required in slurm script if not using cce env
module load craype-network-ucx
module load cray-mpich-ucx
```

Note, cray library paths. When updating LD_LIBRARY_PATH, you must include CRAY_LD_LIBRARY_PATH to include the contents of cray modules.

```
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH:/whatever
```

At runtime, i.e. typically in the job script, need to repeat environment setup such as loading modules, programming environments, fabric configuration and library paths. Can check with ldd in the job submission script.

#### Performance Tuning

No choice of MPI on archer, must use cray MPI. This is because of the Slingshot network, which is tailored to cray.

Couple of significant environment variables for MPI on archer, `MPICH_OFI_STARTUP_CONNECT` and `MPICH_OFI_RMA_STARTUP_CONNECT`.

When using OFI, the connections between ranks are setup as-required. However, for jobs using all-to-all and other collectives it may be bettter to generate connections at the start of an application run in a coordinated way using

```bash
export MPICH_OFI_STARTUP_CONNECT=1
```

The other one is for one-sided communications between all processes in a communicator.

Even if profilers show that the code is spending a lot of time on collectives, not necessarily due to the collective being the bottleneck. Could also be a symptom of load imbalance, as collective is waiting to be called on all processes at the same time.

NUMA effects for multithreading in shared memory.

8 NUMA regions associated with each node, each with 16 associated CPU cores on archer. Default policy is to place data in NUMA region which first accesses it. This can be the worst option if data is initialised by a master thread and used in other threads in separate NUMA regions.

Care must therefore be taken to ensure that each thread block is local to a NUMA region.

On archer it is possible for just two threads to saturate available bandwidth in a NUMA region. So may not get much additional performance using more threads.

This might significantly effect performance of the FFT based M2L with more than two threads.

If a multithreaded code is not using all cores on a node, by default, Slurm spreads the threads out across NUMA regions to maximise available bandwidth. Another source of resource contention could be shared cache. every 4 cores shared 16MB of L3.

This might mean the level of process granularity should match each shared block L3 rather than a NUMA region. To test out.

Consequences:

- running at least one MPI process per NUMA region will be beneficial.

- the number of MPI processes per node should be a power of two, so that all threads run in the same NUMA regionas their parent MPI process.

- For applications where a process has a small memory footprint (e.g. P2P operator for the FMM) more than 4 OpenMP threads per MPI process may be beneficial so that all the threads share an L3.


#### File Systems

Home and work filesystems at

- `/home/[project id]/[project id]/[user id]`
- `/work/[project id]/[project id]/[user id]`

Home and work are available on the login nodes and data analysis nodes, the compute nodes only have access to work filesystem

All three node types have access to solid state NVMe filesystem.

- Note Rust binaries cannot be built in the HPE CCE which includes a specialised version of Clang, one must use the AMD-aocc compiler environment.

- One should use the HPE Cray MPICH library, for optimised usage of slingshot.

- One should use the HPE Cray numerical libraries, LibSci.



### Scheduling with Slurm

There are example slurm scheduling scripts in the `/hpc` directory for the weak scaling and parameter search scripts in this crate.

Archer job limits defined by Quality of Service, specified by `--qos` directive. There are numerous listed on the site.

The relevant ones are `standard`, which allows for a maximum of 1024 nodes per job with a max walltime of 24 hours. `short` may also be useful, with a max of 32 nodes and 20 minutes walltime.

Potentially, if results are going well, `largescale` may be useful, which allows you to use the whole system but requires a minimum of 1025 nodes in use, max wall time of 12 hours.

- optionally set a max wall clock time to help scheduler, and provide a custom name via the `--time` and `--job-name` directives

To prevent batch scripts from being dependent on the user environment in the login node you have to specify

`--export=none`

This also means that any environment variables, programming environments, or linking, must be specified again in the job submission script.

For parallel jobs need to specify the number of nodes `--nodes`, the tasks per node i.e. the number of parallel processes per compute node `--ntasks-per-node`, for parallel jobs that also use threading need to control the number of CPUs per task `--cpus-per-task`. Note that this is extra to also specifying the number of threads used by parallel threading modules like Rayon on BLAS.

For jobs using less than 128 cores per node the cpus per task setting is set to the stride between parallel processes.

E.g. 64 (mpi) processes per node and leave an empty core between each process, you'd set cpus-per-task to 2 and ntasks-per-node to 64.


`sbatch`
- used to submit a jobscript, typically contains one or more `srun` commands.

`srun`
- for submitting MPI jobs. Generally want to add options `--distribution=block:block` and `--hint=no-multithread` to get appropriate thread pinning. Without this the default process placement may lead to a drop in performance with Archer 2.

No multithread disables hyperthreading, and distribution block-block, the first block means use a block distribution of processes across nodes - i.e. fill up nodes before moving to the next one, and the second block means use a block distribution of processes across sockets within a node. A socket doesn't mean physical CPU socket in Archer 2, it means a 4 core CCX (core Complex), i.e. a 4 core unit that shares L3 cache.

Can check submission script with the checkScript tool. Using `--test-only` option provides an estimate of when the job could expect to be scheduled given current jobs.

Example jobscript using 4 nodes and 128 ranks per node for 20 minutes

```bash
#!/bin/bash

# Slurm job options (job-name, compute nodes, job time)
#SBATCH --job-name=Example_MPI_Job
#SBATCH --time=0:20:0
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1

# Replace [budget code] below with your budget code (e.g. t01)
#SBATCH --account=[budget code]
#SBATCH --partition=standard
#SBATCH --qos=standard

# Set the number of threads to 1
#   This prevents any threaded system libraries from automatically
#   using threading.
export OMP_NUM_THREADS=1

# Propagate the cpus-per-task setting from script to srun commands
#    By default, Slurm does not propagate this setting from the sbatch
#    options to srun commands in the job script. If this is not done,
#    process/thread pinning may be incorrect leading to poor performance
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

# Launch the parallel job
#   Using 512 MPI processes and 128 MPI processes per node
#   srun picks up the distribution from the sbatch options

# Disable hyperthreading
srun --distribution=block:block --hint=nomultithread ./my_mpi_executable.x
```

Other common sbatch settings that are not propagated to srun include, ntasks-per-node, mem-per-cpu and gpus-per-task

When using MPI and threading you have to ensure that the shared memory portion doesn't span more than one NUMA region.

On Archer2, nodes are made up of 2 sockets each containing 4 NUMA regions of 16 cores. Therefore 8 NUMA regions in total. Therefore the total number of threads (per NUMA region) should be no more than 16, otherwise you will get cache misses.

the cpus-per-task setting needs to match the number of OpenMP threads. The number of tasks per node should be set to ensure that the entire node is filled with MPI tasks.

Example job script for 4 nodes with 8 MPI processes per node and 16 threads per MPI process.

```bash
#!/bin/bash

# Slurm job options (job-name, compute nodes, job time)
#SBATCH --job-name=Example_MPI_Job
#SBATCH --time=0:20:0
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16

# Replace [budget code] below with your project code (e.g. t01)
#SBATCH --account=[budget code]
#SBATCH --partition=standard
#SBATCH --qos=standard

# Propagate the cpus-per-task setting from script to srun commands
#    By default, Slurm does not propagate this setting from the sbatch
#    options to srun commands in the job script. If this is not done,
#    process/thread pinning may be incorrect leading to poor performance
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

# Set the number of threads to 16 and specify placement
#   There are 16 OpenMP threads per MPI process
#   We want one thread per physical core
export OMP_NUM_THREADS=16
export OMP_PLACES=cores

# Launch the parallel job
#   Using 32 MPI processes
#   8 MPI processes per node
#   16 OpenMP threads per MPI process
#   Additional srun options to pin one thread per physical core
srun --hint=nomultithread --distribution=block:block ./my_mixed_executable.x arg1 arg2
```

