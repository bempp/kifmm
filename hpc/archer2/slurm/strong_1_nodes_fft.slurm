#!/bin/bash
# Here, we perform a strong scaling run where the number of MPI processes are tied to CCX units, each of which
# share an L3 cache, consist of 4 cores. There are 16 CCX regions per processor, and 32 per node (i.e. two AMD EPYC Rome units per node)
# We start with a large (>2.5B) point problem on a single fully utilised node, and gradually double the number of nodes to identify
# the saturation point (in terms of nodes)

#SBATCH --job-name=strong_scaling_fft
#SBATCH --time=01:00:00
#SBATCH --nodes=128
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=8
#SBATCH --contiguous

#SBATCH --account=e738
#SBATCH --partition=standard
#SBATCH --qos=standard

# Restore AMD compiler env
module load PrgEnv-aocc
module load craype-network-ucx
module load cray-mpich-ucx

# Home and work directories
export HOME="/home/e738/e738/skailasa"
export WORK="/work/e738/e738/skailasa"
TOTAL_POINTS=153600000             # 150M points

# Script being run, expected in the work directory
script_name="fmm_m2l_fft_mpi_f32" # must exist in $WORK

# Set simulation parameters for FMM
expansion_order=3
global_depth=(2 2 2 3 3 3 4 4) # Number of local roots
n_samples=500
block_size=128

# Set parameters for strong scaling run
n_points=(9600000 4800000 2400000 1200000  600000  300000  150000   75000)
local_depth=(5 4 4 4 3 3 3 3)
n_tasks=(16   32   64  128  256  512 1024 2048) # Equal to the number of local trees
n_nodes=(1 2 4 8 16 32 64 128)
n_threads=8 # See if bandwidth saturates with different threading parameters for Rayon thread pool
cpus_per_task=8

# Create a scratch directory for this run
last_nodes=${n_nodes[@]: -1}
export SCRATCH=${WORK}/strong_fft_n=${TOTAL_POINTS}_p=${last_nodes}_${SLURM_JOBID}

mkdir -p ${SCRATCH}
cd ${SCRATCH}

# Pass variable to SRUN from SBATCH
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

export OMP_NUM_THREADS=1 # Need to set to 1 to avoid oversubsciption between Rayon and OpenMP

# Create a CSV output file for analysis
export OUTPUT=${SCRATCH}/strong_fft_p=${TOTAL_POINTS}_n=${last_nodes}_${SLURM_JOBID}.csv

# Create a CSV output file for analysis
touch ${OUTPUT}
echo "
experiment_id,rank,runtime,p2m,m2m,l2l,m2l,p2p,\
source_tree,target_tree,source_domain,target_domain,layout,\
ghost_exchange_v,ghost_exchange_v_runtime,ghost_exchange_u,gather_global_fmm,scatter_global_fmm,\
source_to_target_data,source_data,target_data,global_fmm,ghost_fmm_v,ghost_fmm_u,\
displacement_map,metadata_creation,\
expansion_order,n_points,local_depth,global_depth,block_size,n_threads,n_samples" >> ${OUTPUT}

for i in ${!n_tasks[@]}; do

    experiment_id="${i}"
    srun --nodes=${n_nodes[$i]} --ntasks=${n_tasks[$i]} --cpus-per-task=$cpus_per_task --distribution=block:block --hint=nomultithread \
        "${WORK}/${script_name}" --id $experiment_id --n-points ${n_points[$i]} \
        --expansion-order $expansion_order \
        --prune-empty \
        --global-depth ${global_depth[$i]} \
        --local-depth ${local_depth[$i]} \
        --n-samples $n_samples \
        --block-size $block_size \
        --n-threads $n_threads >> ${OUTPUT} 2> ${SCRATCH}/err_run_${i}.log
done
