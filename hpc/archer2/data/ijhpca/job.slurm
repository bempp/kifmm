#!/bin/bash
#SBATCH --job-name=weak_scaling_fft
#SBATCH --time=00:30:00
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --account=e738
#SBATCH --partition=standard
#SBATCH --qos=standard

module load PrgEnv-aocc
module load craype-network-ucx
module load cray-mpich-ucx

export HOME="/home/e738/e738/skailasa"
export WORK="/work/e738/e738/skailasa"

script_name="fmm_m2l_fft_mpi_f32"

export SCRATCH=$WORK/weak_fft_n=64000000_p=64_${SLURM_JOBID}
mkdir -p $SCRATCH
cd $SCRATCH

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=1

export OUTPUT=$SCRATCH/weak_fft_n=64000000_p=64_${SLURM_JOBID}.csv
touch $OUTPUT
echo "
experiment_id,rank,runtime,p2m,m2m,l2l,m2l,p2p,source_tree,target_tree,source_domain,target_domain,layout,ghost_exchange_v,ghost_exchange_v_runtime,ghost_exchange_u,gather_global_fmm,scatter_global_fmm,source_to_target_data,source_data,target_data,global_fmm,ghost_fmm_v,ghost_fmm_u,displacement_map,metadata_creation,expansion_order,n_points,local_depth,global_depth,block_size,n_threads,n_samples,source_local_trees_per_rank,target_local_trees_per_rank" >> ${OUTPUT}

# Perform weak scaling

srun --nodes=1 --ntasks=4 --cpus-per-task=32 \
     --distribution=block:block --hint=nomultithread \
     $WORK/$script_name --id 0 --n-points 250000 \
     --expansion-order 3 --prune-empty \
     --global-depth 1 --local-depth 3 \
     --n-samples 500 --block-size 128 --n-threads 32 \
     >> $OUTPUT 2> $SCRATCH/err_run_0.log

srun --nodes=8 --ntasks=32 --cpus-per-task=32 \
     --distribution=block:block --hint=nomultithread \
     $WORK/$script_name --id 1 --n-points 250000 \
     --expansion-order 3 --prune-empty \
     --global-depth 2 --local-depth 3 \
     --n-samples 500 --block-size 128 --n-threads 32 \
     >> $OUTPUT 2> $SCRATCH/err_run_1.log

srun --nodes=64 --ntasks=256 --cpus-per-task=32 \
     --distribution=block:block --hint=nomultithread \
     $WORK/$script_name --id 2 --n-points 250000 \
     --expansion-order 3 --prune-empty \
     --global-depth 3 --local-depth 3 \
     --n-samples 500 --block-size 128 --n-threads 32 \
     >> $OUTPUT 2> $SCRATCH/err_run_2.log
