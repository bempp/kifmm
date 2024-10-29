#!/bin/bash
# Here we perform a naive run

#SBATCH --job-name=grid_search
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128

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
export SCRATCH=${WORK}/grid_search_${SLURM_JOBID}

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

# Set simulation parameters
n_points=8000000 # points per MPI process
n_tasks=1
cpus_per_task=128
global_depth=1 # Number of local roots matches the number of MPI processes, therefore the number of NUMA regions
local_depth=(5)
n_samples=(5000)
block_size=(128)
n_threads=(128) # See if bandwidth saturates with different threading parameters for Rayon thread pool
export OMP_NUM_THREADS=1 # Need to set to 1 to avoid oversubsciption between Rayon and OpenMP

# Create a CSV output file for analysis
export OUTPUT=${SCRATCH}/grid_search_fft_${SLURM_JOBID}.csv
touch ${OUTPUT}
echo "
experiment_id,rank,runtime,p2m,m2m,l2l,m2l,p2p,\
source_tree,target_tree,source_domain,target_domain,layout,\
ghost_exchange_v,ghost_exchange_u,gather_global_fmm,scatter_global_fmm,\
source_to_target_data,source_data,target_data,global_fmm,ghost_fmm_v,ghost_fmm_u,\
expansion_order,n_points,local_depth, global_depth, block_size, n_threads, n_samples" >> ${OUTPUT}

# Perform grid search
for i in ${!local_depth[@]}; do
    for j in ${!n_samples[@]}; do
        for k in ${!block_size[@]}; do
            for l in ${!n_threads[@]}; do
                experiment_id="${i}_${j}_${k}_${l}"
                srun --ntasks=$n_tasks --cpus-per-task=$cpus_per_task --distribution=block:block --hint=nomultithread \
                    ${WORK}/fmm_m2l_fft_mpi_f32 --id $experiment_id --n-points $n_points \
                    --expansion-order 3 \
                    --prune-empty \
                    --global-depth $global_depth \
                    --local-depth ${local_depth[$i]} \
                    --n-samples ${n_samples[$j]} \
                    --block-size ${block_size[$k]} \
                    --n-threads ${n_threads[$l]} >> ${OUTPUT}
            done
        done
    done
done
