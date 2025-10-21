#!/bin/bash
#SBATCH --job-name=strong_scaling_fft
#SBATCH --time=00:30:00
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=8
#SBATCH --account=e738
#SBATCH --partition=standard
#SBATCH --qos=standard

#SBATCH --contiguous

module load PrgEnv-aocc
module load craype-network-ucx
module load cray-mpich-ucx

export HOME="/home/e738/e738/skailasa"
export WORK="/work/e738/e738/skailasa"

script_name="fmm_m2l_fft_mpi_f32"

export SCRATCH=$WORK/strong_fft_n=128000000_p=64_points_per_rank=125000_distribution=uniform_${SLURM_JOBID}
mkdir -p $SCRATCH
cd $SCRATCH

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=1

export OUTPUT=$SCRATCH/strong_fft_n=128000000_p=64_points_per_rank=125000_distribution=uniform_${SLURM_JOBID}.csv
touch $OUTPUT
echo "
experiment_id,rank,runtime,p2m,m2m,l2l,m2l,p2p,source_tree,target_tree,source_domain,target_domain,layout,ghost_exchange_v,ghost_exchange_v_runtime,ghost_exchange_u,gather_global_fmm,scatter_global_fmm,source_to_target_data,source_data,target_data,global_fmm,ghost_fmm_v,ghost_fmm_u,displacement_map,metadata_creation,expansion_order,n_points,local_depth,global_depth,block_size,n_threads,n_samples,source_local_trees_per_rank,target_local_trees_per_rank" >> ${OUTPUT}

# Perform strong scaling

srun --nodes=1 --ntasks=16 --cpus-per-task=8 \
     --distribution=block:block --hint=nomultithread \
     $WORK/$script_name --id 0 --n-points 8000000 \
     --expansion-order 3 --prune-empty \
     --global-depth 2 --local-depth 5 \
     --n-samples 500 --block-size 128 --n-threads 8 --distribution 0 \
     >> $OUTPUT 2> $SCRATCH/err_run_0.log

srun --nodes=8 --ntasks=128 --cpus-per-task=8 \
     --distribution=block:block --hint=nomultithread \
     $WORK/$script_name --id 1 --n-points 1000000 \
     --expansion-order 3 --prune-empty \
     --global-depth 3 --local-depth 4 \
     --n-samples 500 --block-size 128 --n-threads 8 --distribution 0 \
     >> $OUTPUT 2> $SCRATCH/err_run_1.log

srun --nodes=64 --ntasks=1024 --cpus-per-task=8 \
     --distribution=block:block --hint=nomultithread \
     $WORK/$script_name --id 2 --n-points 125000 \
     --expansion-order 3 --prune-empty \
     --global-depth 4 --local-depth 3 \
     --n-samples 500 --block-size 128 --n-threads 8 --distribution 0 \
     >> $OUTPUT 2> $SCRATCH/err_run_2.log
