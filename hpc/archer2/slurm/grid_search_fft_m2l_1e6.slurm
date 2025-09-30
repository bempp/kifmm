#!/bin/bash
# Here, we perform grid search where the number of MPI processes is set equal to the number of CCX units, each of which
# share an L3 cache, conist of 4 processors. There are 16 CCX regions per processor, and 32 per node

#SBATCH --job-name=grid_search
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=4

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

# Script being run, expected in the work directory
script_name="fmm_m2l_fft_mpi_f32"

# Set simulation parameters
n_points=(250000 500000 1000000) # points per MPI process
n_tasks=32
cpus_per_task=4
global_depth=2 # Number of local roots matches the number of MPI processes, therefore the number of NUMA regions
local_depth=(4 5)
n_samples=(500)
block_size=(128)
n_threads=(4) # See if bandwidth saturates with different threading parameters for Rayon thread pool
export OMP_NUM_THREADS=1 # Need to set to 1 to avoid oversubsciption between Rayon and OpenMP

# Create a CSV output file for analysis
export OUTPUT=${SCRATCH}/grid_search_fft_${SLURM_JOBID}.csv
touch ${OUTPUT}
echo "
experiment_id,rank,runtime,p2m,m2m,l2l,m2l,p2p,\
source_tree,target_tree,source_domain,target_domain,layout,\
ghost_exchange_v,ghost_exchange_v_runtime,ghost_exchange_u,gather_global_fmm,scatter_global_fmm,\
source_to_target_data,source_data,target_data,global_fmm,ghost_fmm_v,ghost_fmm_u,\
displacement_map,metadata_creation,\
expansion_order,n_points,local_depth,global_depth,block_size,n_threads,n_samples" >> ${OUTPUT}

# Perform grid search
for i in ${!local_depth[@]}; do
    for j in ${!n_samples[@]}; do
        for k in ${!block_size[@]}; do
            for l in ${!n_threads[@]}; do
                for m in ${!n_points[@]}; do
                    experiment_id="${i}_${j}_${k}_${l}_${m}"
                    srun --ntasks=$n_tasks --cpus-per-task=$cpus_per_task --distribution=block:block --hint=nomultithread \
                        "${WORK}/${script_name}" --id $experiment_id --n-points ${n_points[$m]} \
                        --expansion-order 3 \
                        --prune-empty \
                        --global-depth $global_depth \
                        --local-depth ${local_depth[$i]} \
                        --n-samples ${n_samples[$j]} \
                        --n-threads ${n_threads[$l]} >> ${OUTPUT}
                done
            done
        done
    done
done
