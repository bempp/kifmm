#!/bin/bash
#SBATCH --job-name=weak_scaling_instrumented
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=4
#SBATCH --contiguous
#SBATCH --account=e738
#SBATCH --partition=standard
#SBATCH --qos=standard

# Environment setup
module load PrgEnv-aocc
module load craype-network-ucx
module load cray-mpich-ucx
# module load perftools
module load perftools-preload

# Home/work dirs
export HOME="/home/e738/e738/skailasa"
export WORK="/work/e738/e738/skailasa"
export SCRATCH=${WORK}/weak_${SLURM_JOBID}
mkdir -p ${SCRATCH}
cd ${SCRATCH}

script_name="weak_scaling"
config_name="weak_scaling.conf"

# Executable
cp ${WORK}/${script_name} ${SCRATCH}/

# Config
cp ${WORK}/${config_name} ${SCRATCH}/

# Create CSV output file for analysis
export OUTPUT=${script_name}_${SLURM_JOBID}.csv
touch ${OUTPUT}
echo "
experiment_id,rank,runtime,p2m,m2m,l2l,m2l,p2p,\
source_tree,target_tree,source_domain,target_domain,layout,\
ghost_exchange_v,ghost_exchange_v_runtime,ghost_exchange_u,gather_global_fmm,scatter_global_fmm,\
source_to_target_data,source_data,target_data,global_fmm,ghost_fmm_v,ghost_fmm_u,\
expansion_order,n_points,local_depth,global_depth,block_size,n_threads,n_samples" >> ${OUTPUT}

# split into blocks that are separated by blank lines
csplit -s -f cfg_* "$config_name" '/^$/' '{*}'

# Avoid thread oversubscription with Rayon
export OMP_NUM_THREADS=1

for cfg in cfg_*; do
    # Run under CrayPAT
    export RAYON_NUM_THREADS=$nthreads

    srun --nodes=$nodes --ntasks=$ntasks --cpus-per-task=$cpus_per_task \
         --distribution=block:block --hint=nomultithread \
        pat_run -g mpi,blas,pthreads,numa,math ./${script_name} \
        --prune_empty=$prune_empty \
        --local_depth=$local_depth \
        --global_depth=$global_depth \
        --sort_kind=$sort_kind \
        --n_samples=$n_samples \
        --eval_type=$eval_type \
        --timed=$timed \
        --n_sources=$n_sources \
        --n_targets=$n_targets \
        --nexpansion_order=$nexpansion_order \
        --expansion_order=$expansion_order
done

