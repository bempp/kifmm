#!/bin/bash
#SBATCH --job-name=weak_scaling_fft_n=4096000000_p=512_points_per_rank=500000_distribution=uniform
#SBATCH --time=0:30:00
#SBATCH --nodes=512
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=8
#SBATCH --account=e738
#SBATCH --partition=standard
#SBATCH --qos=standard

module load PrgEnv-aocc
module load craype-network-ucx
module load cray-mpich-ucx

export UCX_UD_TIMEOUT=20m

script_name="fmm_m2l_fft_mpi_f32"

export SCRATCH=$WORK/weak_fft_n=4096000000_p=512_points_per_rank=500000_distribution=uniform_${SLURM_JOBID}
mkdir -p $SCRATCH
cd $SCRATCH

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=1

export OUTPUT=$SCRATCH/weak_fft_n=4096000000_p=512_points_per_rank=500000_distribution=uniform_${SLURM_JOBID}.csv
touch $OUTPUT
echo "experiment_id,rank,runtime,p2m,m2m,l2l,m2l,p2p,source_tree,target_tree,source_domain,target_domain,layout,ghost_exchange_v,ghost_exchange_v_runtime,ghost_exchange_u,gather_global_fmm,scatter_global_fmm,source_to_target_data,source_data,target_data,global_fmm,ghost_fmm_v,ghost_fmm_u,displacement_map,metadata_creation,expansion_order,n_points,local_depth,global_depth,block_size,n_threads,n_samples,source_local_trees_per_rank,target_local_trees_per_rank,all_to_all,all_to_all_v,neighbour_all_to_all,neighbour_all_to_all_v,neighbour_all_to_all_v_runtime,gather,scatter,gather_v,scatter_v,gather_v_runtime,scatter_v_runtime,all_gather,all_gather_v,dist_graph_create,sort,tree_all_to_all,tree_all_to_all_v,tree_neighbour_all_to_all,tree_neighbour_all_to_all_v,tree_neighbour_all_to_all_v_runtime,tree_gather,tree_scatter,tree_gather_v,tree_scatter_v,tree_gather_v_runtime,tree_scatter_v_runtime,tree_all_gather,tree_all_gather_v,tree_dist_graph_create,tree_sort" >> ${OUTPUT}

# Perform weak scaling

srun --nodes=1 --ntasks=16 --cpus-per-task=8 \
     --distribution=block:block --hint=nomultithread \
     $WORK/$script_name --id 0 --n-points 500000 \
     --expansion-order 3 --prune-empty \
     --global-depth 2 --local-depth 4 \
     --n-samples 500 --block-size 128 --n-threads 8 --distribution 0 \
     >> $OUTPUT 2> $SCRATCH/err_run_0.log

srun --nodes=8 --ntasks=128 --cpus-per-task=8 \
     --distribution=block:block --hint=nomultithread \
     $WORK/$script_name --id 1 --n-points 500000 \
     --expansion-order 3 --prune-empty \
     --global-depth 3 --local-depth 4 \
     --n-samples 500 --block-size 128 --n-threads 8 --distribution 0 \
     >> $OUTPUT 2> $SCRATCH/err_run_1.log

srun --nodes=64 --ntasks=1024 --cpus-per-task=8 \
     --distribution=block:block --hint=nomultithread \
     $WORK/$script_name --id 2 --n-points 500000 \
     --expansion-order 3 --prune-empty \
     --global-depth 4 --local-depth 4 \
     --n-samples 500 --block-size 128 --n-threads 8 --distribution 0 \
     >> $OUTPUT 2> $SCRATCH/err_run_2.log

srun --nodes=512 --ntasks=8192 --cpus-per-task=8 \
     --distribution=block:block --hint=nomultithread \
     $WORK/$script_name --id 3 --n-points 500000 \
     --expansion-order 3 --prune-empty \
     --global-depth 5 --local-depth 4 \
     --n-samples 500 --block-size 128 --n-threads 8 --distribution 0 \
     >> $OUTPUT 2> $SCRATCH/err_run_3.log
