#!/bin/bash
# Here, we perform a weak scaling run where the number of MPI processes is set equal to the number of CCX units, each of which
# share an L3 cache, conist of 4 processors. There are 16 CCX regions per processor, and 32 per node

# Running on just a single node, we try and examine the scaling performance as the number of tasks is increased
# just to see what a single node is capable of (in the best parameter settings found)

# This should give a good baseline of what to expect when communication costs are minimal

#SBATCH --job-name=weak_scaling_fft
#SBATCH --time=00:30:00
#SBATCH --nodes=128
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=4
#SBATCH --contiguous

#SBATCH --account=e738
#SBATCH --partition=standard
#SBATCH --qos=standard
# Development environment for KiFMM

# Restore AMD compiler env
module load PrgEnv-aocc
module load craype-network-ucx
module load cray-mpich-ucx

# Home and work directories
export HOME="/home/e738/e738/skailasa"
export WORK="/work/e738/e738/skailasa"

# Create a scratch directory for this run
export SCRATCH=${WORK}/weak_${SLURM_JOBID}

# Load Spack
source $HOME/spack/share/spack/setup-env.sh
. "$HOME/.cargo/env"

# Load BLAS
spack load openblas

# Ensure Rust can find the Cray libraries
# export RUSTFLAGS="-L $(echo $CRAY_LD_LIBRARY_PATH)"
# export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
export RUSTFLAGS="-L $(spack location -i openblas)/lib"
export LD_LIBRARY_PATH=$(spack location -i openblas)/lib:$LD_LIBRARY_PATH

mkdir -p ${SCRATCH}
cd ${SCRATCH}

# Pass variable to SRUN from SBATCH
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

# Script being run, expected in the work directory
script_name="fmm_m2l_fft_mpi_f32"

# Set simulation parameters for FMM
expansion_order=3
n_points=250000 # points per MPI process (n_tasks)
global_depth=(4 4 4) # Number of local roots
local_depth=4
n_samples=5000
block_size=128

# Set parameters for weak scaling run
n_tasks=(1024 2048 4096)
n_nodes=(32 64 128)
n_threads=4 # See if bandwidth saturates with different threading parameters for Rayon thread pool
cpus_per_task=4
export OMP_NUM_THREADS=1 # Need to set to 1 to avoid oversubsciption between Rayon and OpenMP

# Create a CSV output file for analysis
export OUTPUT=${SCRATCH}/weak_blas_${SLURM_JOBID}.csv
touch ${OUTPUT}
echo "
experiment_id,rank,runtime,p2m,m2m,l2l,m2l,p2p,\
source_tree,target_tree,source_domain,target_domain,layout,\
ghost_exchange_v,ghost_exchange_v_runtime,ghost_exchange_u,gather_global_fmm,scatter_global_fmm,\
source_to_target_data,source_data,target_data,global_fmm,ghost_fmm_v,ghost_fmm_u,\
expansion_order,n_points,local_depth,global_depth,block_size,n_threads,n_samples" >> ${OUTPUT}

# Perform weak scaling
for i in ${!n_tasks[@]}; do
    experiment_id="${i}"
    srun --nodes=${n_nodes[$i]} --ntasks=${n_tasks[$i]} --cpus-per-task=$cpus_per_task --distribution=block:block --hint=nomultithread \
        "${WORK}/${script_name}" --id $experiment_id --n-points $n_points \
        --expansion-order $expansion_order \
        --prune-empty \
        --global-depth ${global_depth[$i]} \
        --local-depth $local_depth \
        --n-samples $n_samples \
        --block-size $block_size \
        --n-threads $n_threads >> ${OUTPUT}
done

