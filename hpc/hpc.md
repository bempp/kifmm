# Running kifmm-rs on HPC

Here we document how to run this software on HPC systems, tried and tested configurations and optimal parameter search and setting.

## Archer 2

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

Three types of node available to users
- login
- data analysis
- compute

Nodes are grouped into partition, specified with the `--partition` option in the slurm jobscript.

The relevant nodes for kifmm-rs are:

- standard - CPU nodes, 2x AMD EPYC 7742 64 Core processor 256/512 GB memory, 5860 nodes available.

### Software

- As of 2024 Rust should be installed by the user, the easiest way to avoid building from source is to use the [rustup script](https://www.rust-lang.org/tools/install).



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