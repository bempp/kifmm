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
export SCRATCH=${WORK}/grid_search_helmholtz_f64_${SLURM_JOBID}

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
script_name="helmholtz_m2l_fft_mpi_f64"

# Set simulation parameters
n_points=(50000) # points per MPI process
wavenumber=10
n_tasks=32
cpus_per_task=4
global_depth=2 # Number of local roots matches the number of MPI processes, therefore the number of NUMA regions
local_depth=(3 4 5)
n_samples=100
block_size=128
n_threads=4 # See if bandwidth saturates with different threading parameters for Rayon thread pool
leaf_expansion_order=(3 4 5 6 7 8)
expansion_order_multiplier=(1.0 1.2 1.5 1.7 2.0)
export OMP_NUM_THREADS=1 # Need to set to 1 to avoid oversubsciption between Rayon and OpenMP

# Create a CSV output file for analysis
export OUTPUT=${SCRATCH}/grid_search_helmholtz_fft_k_${wavenumber}.csv
touch ${OUTPUT}
echo "experiment_id,rank,runtime,p2m,m2m,l2l,m2l,p2p,\
source_tree,target_tree,source_domain,target_domain,layout,\
ghost_exchange_v,ghost_exchange_v_runtime,ghost_exchange_u,gather_global_fmm,scatter_global_fmm,\
source_to_target_data,source_data,target_data,global_fmm,ghost_fmm_v,ghost_fmm_u,\
displacement_map,metadata_creation,leaf_expansion_order,n_points,local_depth,global_depth,\
block_size,n_threads,n_samples,expansion_order_multiplier,l2_error" >> ${OUTPUT}

# Perform grid search
b=${#leaf_expansion_order[@]}
c=${#expansion_order_multiplier[@]}
d=${#n_points[@]}

for i in "${!local_depth[@]}"; do
  for j in "${!leaf_expansion_order[@]}"; do
    for k in "${!expansion_order_multiplier[@]}"; do
      for l in "${!n_points[@]}"; do
        experiment_id=$(( l + k*d + j*d*c + i*c*d*b ))
        srun --ntasks=$n_tasks --cpus-per-task=$cpus_per_task --distribution=block:block --hint=nomultithread \
          "${WORK}/${script_name}" \
          --id $experiment_id \
          --leaf-expansion-order ${leaf_expansion_order[$j]} \
          --expansion-order-multiplier ${expansion_order_multiplier[$k]} \
          --wavenumber $wavenumber \
          --prune-empty \
          --n-points ${n_points[$l]} \
          --local-depth ${local_depth[$i]} \
          --global-depth $global_depth \
          --n-threads $n_threads \
          --n-samples $n_samples >> ${OUTPUT}
      done
    done
  done
done