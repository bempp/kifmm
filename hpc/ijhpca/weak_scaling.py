#!/usr/bin/env python3
import numpy as np
import argparse
from pathlib import Path


def global_depth(rank):
    """
    Return the first power of 8 >= rank, the exponent (tree level).
    """
    level = 1
    while True:
        if rank <= 8**level:
            return level
        level += 1


v_global_depth = np.vectorize(global_depth)


def pow2(x):
    return np.log2(x).is_integer()


def pow8(x):
    return np.emath.logn(8, x).is_integer()


def experiment_parameters(min_nodes, max_nodes, ranks_per_node, points_per_rank, local_depth=4, scaling_func=pow8):

    max_cpus=128 # AMD EPYC rome

    total_ranks = ranks_per_node * max_nodes

    n = points_per_rank*total_ranks # total problem size
    print(f"Total problem size n={n/1e9}B")

    points_per_node = points_per_rank * ranks_per_node
    print(f"points per rank {points_per_rank/1e6}M")
    print(f"points per node {points_per_node/1e6}M")

    # Double number of nodes used until we reach max_nodes
    n_nodes = np.array(list(filter(scaling_func, [i for i in range(min_nodes, max_nodes+1)])))

    # The global depth is a function of the ranks per node (as will need enough global leaves to cover this many ranks (local trees))
    global_depth = v_global_depth(n_nodes*ranks_per_node)
    print(f"global depth {global_depth}")
    print(f"local depth {local_depth}")

    # local depth is a constant, chosen to balance M2L and P2P
    points_per_leaf = points_per_rank / 8**local_depth
    print(f"points per leaf {points_per_leaf}")

    n_tasks = n_nodes*ranks_per_node
    print(f"MPI tasks {n_tasks}")

    print(f"number of nodes {n_nodes}")

    local_trees_per_rank = (8**global_depth)/(n_tasks)
    print(f"local trees per rank {local_trees_per_rank}")

    max_threads_per_rank = max_cpus/ranks_per_node
    print(f"max threads per rank {max_threads_per_rank}")

    return n_nodes.tolist(), n_tasks.tolist(), global_depth.tolist(), max_threads_per_rank



def write_slurm(script_path, n_nodes, n_tasks, global_depths, max_threads, points_per_rank, local_depth, script_name="fmm_m2l_fft_mpi_f32"):
    expansion_order = 3
    n_points = points_per_rank
    n_samples = 500
    block_size = 128
    cpus_per_task = int(max_threads)


    last_nodes = n_nodes[-1]
    last_tasks = n_tasks[-1]
    max_points = last_tasks * n_points

    slurm = f"""#!/bin/bash
#SBATCH --job-name=weak_scaling_fft
#SBATCH --time=00:30:00
#SBATCH --nodes={last_nodes}
#SBATCH --ntasks-per-node={int(n_tasks[-1] // n_nodes[-1])}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --account=e738
#SBATCH --partition=standard
#SBATCH --qos=standard

module load PrgEnv-aocc
module load craype-network-ucx
module load cray-mpich-ucx

export HOME="/home/e738/e738/skailasa"
export WORK="/work/e738/e738/skailasa"

script_name="{script_name}"

export SCRATCH=$WORK/weak_fft_n={max_points}_p={last_nodes}_${{SLURM_JOBID}}
mkdir -p $SCRATCH
cd $SCRATCH

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=1

export OUTPUT=$SCRATCH/weak_fft_n={max_points}_p={last_nodes}_${{SLURM_JOBID}}.csv
touch $OUTPUT
echo "
experiment_id,rank,runtime,p2m,m2m,l2l,m2l,p2p,\
source_tree,target_tree,source_domain,target_domain,layout,\
ghost_exchange_v,ghost_exchange_v_runtime,ghost_exchange_u,gather_global_fmm,scatter_global_fmm,\
source_to_target_data,source_data,target_data,global_fmm,ghost_fmm_v,ghost_fmm_u,\
displacement_map,metadata_creation,\
expansion_order,n_points,local_depth,global_depth,block_size,n_threads,n_samples,source_local_trees_per_rank,target_local_trees_per_rank" >> ${{OUTPUT}}
"""

    # loop of runs
    slurm += "\n# Perform weak scaling\n"
    for i, (nn, nt, gd) in enumerate(zip(n_nodes, n_tasks, global_depths)):
        slurm += f"""
srun --nodes={nn} --ntasks={nt} --cpus-per-task={cpus_per_task} \\
     --distribution=block:block --hint=nomultithread \\
     $WORK/$script_name --id {i} --n-points {n_points} \\
     --expansion-order {expansion_order} --prune-empty \\
     --global-depth {gd} --local-depth {local_depth} \\
     --n-samples {n_samples} --block-size {block_size} --n-threads {int(max_threads)} \\
     >> $OUTPUT 2> $SCRATCH/err_run_{i}.log
"""

    Path(script_path).write_text(slurm)
    print(f"✅ Wrote SLURM script to {script_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a SLURM script for weak scaling runs.")
    parser.add_argument("--min-nodes", type=int, default=1)
    parser.add_argument("--max-nodes", type=int, default=16)
    parser.add_argument("--ranks-per-node", type=int, default=32)
    parser.add_argument("--points-per-rank", type=int, default=250000)
    parser.add_argument("--local-depth", type=int, default=4)
    parser.add_argument("--output", type=str, default="job.slurm")
    parser.add_argument("--config", action='append')

    args = parser.parse_args()

    if args.config is not None:
        for fname in args.config:
            with open(fname, 'r') as f:
                parser.set_defaults(**json.load(f))

    args = parser.parse_args()

    n_nodes, n_tasks, global_depths, max_threads = experiment_parameters(
        args.min_nodes, args.max_nodes, args.ranks_per_node, args.points_per_rank, args.local_depth
    )

    write_slurm(args.output, n_nodes, n_tasks, global_depths, max_threads, args.points_per_rank, args.local_depth)
