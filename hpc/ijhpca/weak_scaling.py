#!/usr/bin/env python3
import numpy as np
import argparse
from pathlib import Path
import json

AMD_EPYC_ROME = {
        "cores_per_node": 128, # total number of CPU cores per node
        "sockets_per_node": 2, # two sockets per node
        "cores_per_socket": 64, # A socket consists of 8 CCDs (64 cores total) common IO/memory controllers
        "cores_per_ccd": 8, # Each CCD is a processor die, consists of two CCDs which are infinity linked together and connected to io
        "cores_per_ccx": 4, # Each CCX shares an L3 cache (16MB)
        "ccx_per_ccd": 2
    }

DISTRIBUTION = {
    "uniform": 0,
    "sphere": 1
}

def parse_process_mapping(arch):
    ccd_threads_per_rank = arch["cores_per_ccd"]
    ccx_threads_per_rank = arch["cores_per_ccx"]
    ccd_max_ranks_per_node = arch["cores_per_node"] / arch["cores_per_ccd"] # Mapping each rank to a CCD
    ccx_max_ranks_per_node = arch["cores_per_node"] / arch["cores_per_ccx"] # Mapping each rank to a CCX
    socket_max_ranks_per_node = arch["cores_per_node"] / arch["cores_per_socket"] # Mapping each rank to a socket
    return {"socket": socket_max_ranks_per_node, "ccd": ccd_max_ranks_per_node, "ccx": ccx_max_ranks_per_node}


def global_depth(rank):
    """
    Return the first power of 8 >= rank, the exponent (tree level).
    """
    level = 1
    while rank > 8**level:
        level += 1
    return level

v_global_depth = np.vectorize(global_depth)

def pow2(x):
    return np.log2(x).is_integer()

def pow8(x):
    return np.emath.logn(8, x).is_integer()


def experiment_parameters(
    min_nodes,
    max_nodes,
    max_ranks_per_node,
    points_per_rank,
    local_depth=4,
    scaling_func=pow8,
    arch=AMD_EPYC_ROME,
    distribution="uniform"
    ):

    max_cpus=arch["cores_per_node"]

    # Calculate the min/max number of ranks required to scale this problem by scaling_func given
    # the resources specified by min/max nodes
    min_ranks = int(min_nodes*max_ranks_per_node)
    max_ranks = int(max_nodes*max_ranks_per_node)

    # Want to scale the resources (ranks) by scaling function between min/max ranks
    n_ranks = []
    curr = min_ranks
    while curr <= max_ranks:
        if scaling_func == pow2:
            n_ranks.append(curr)
            curr *= 2

        elif scaling_func == pow8:
            n_ranks.append(curr)
            curr *= 8
        else:
            raise ValueError("Unknown scaling func")

    n_ranks = np.array(n_ranks)

    # Can use the number of required ranks with this scaling function
    # to calculate the number of required nodes
    n_nodes = (n_ranks / max_ranks_per_node).astype(np.int32)

    # we calculate the global depth as the least power of 8 smaller than or equal to the available
    # ranks for each configuration
    global_depth = v_global_depth(n_ranks)

    local_trees_per_rank = 8**global_depth / n_ranks

    # Problem size defined by largest valid rank set
    n = points_per_rank * n_ranks[-1]

    points_per_node = max_ranks_per_node*points_per_rank

    points_per_leaf = points_per_rank / 8**local_depth

    max_threads_per_rank = int(max_cpus/max_ranks_per_node)

    print(f"Total problem size n={n/1e6}M")
    print(f"global depth {global_depth}")
    print(f"local depth {local_depth}")
    print(f"total depth {local_depth+global_depth}")
    print(f"max local trees per rank {local_trees_per_rank}")
    print(f"number of nodes {n_nodes} max ranks per node {max_ranks_per_node}")
    print(f"number of ranks {n_ranks}")
    print(f"points per rank {points_per_rank}")
    print(f"points per node {points_per_node}")
    print(f"points per leaf {points_per_leaf}")
    print(f"max threads per rank {max_threads_per_rank}")
    print(f"distribution {distribution}")

    return n_nodes.tolist(), n_ranks.tolist(), global_depth.tolist(), max_threads_per_rank, distribution


def write_slurm(script_path, n_nodes, n_tasks, global_depths, max_threads, points_per_rank, local_depth, distribution="uniform", script_name="fmm_m2l_fft_mpi_f32"):
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
"""
    slurm += f"""
module load PrgEnv-aocc
module load craype-network-ucx
module load cray-mpich-ucx

export UCX_UD_TIMEOUT=20m
export HOME="/home/e738/e738/skailasa"
export WORK="/work/e738/e738/skailasa"

script_name="{script_name}"

export SCRATCH=$WORK/weak_fft_n={max_points}_p={last_nodes}_points_per_rank={n_points}_distribution={distribution}_${{SLURM_JOBID}}
mkdir -p $SCRATCH
cd $SCRATCH

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=1

export OUTPUT=$SCRATCH/weak_fft_n={max_points}_p={last_nodes}_points_per_rank={n_points}_distribution={distribution}_${{SLURM_JOBID}}.csv
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
     --n-samples {n_samples} --block-size {block_size} --n-threads {int(max_threads)} --distribution {DISTRIBUTION[distribution]} \\
     >> $OUTPUT 2> $SCRATCH/err_run_{i}.log
"""

    Path(script_path).write_text(slurm)
    print(f"✅ Wrote SLURM script to {script_path}")


if __name__ == "__main__":

    ranks_per_node = parse_process_mapping(AMD_EPYC_ROME)

    parser = argparse.ArgumentParser(description="Generate a SLURM script for weak scaling runs.")
    parser.add_argument("--min-nodes", type=int, default=1)
    parser.add_argument("--max-nodes", type=int, default=16)
    parser.add_argument("--points-per-rank", type=int, default=250000)
    parser.add_argument("--local-depth", type=int, default=4)
    parser.add_argument("--method", type=str, default="ccx")
    parser.add_argument("--output", type=str, default="job.slurm")
    parser.add_argument("--distribution", type=str, default="uniform")
    parser.add_argument("--config", action='append')

    args = parser.parse_args()

    if args.config is not None:
        for fname in args.config:
            with open(fname, 'r') as f:
                parser.set_defaults(**json.load(f))

    args = parser.parse_args()

    if args.scaling_func == 2:
        scaling_func = pow2
    elif args.scaling_func == 8:
        scaling_func = pow8
    else:
        raise ValueError("Unknown scaling function")


    valid_methods = {"ccx", "ccd" "socket"}
    if any(args.method in m for m in valid_methods):
        n_nodes, n_tasks, global_depths, max_threads, distribution = experiment_parameters(
            args.min_nodes, args.max_nodes, ranks_per_node[args.method], args.points_per_rank, args.local_depth, distribution=args.distribution
        )

        write_slurm(args.output, n_nodes, n_tasks, global_depths, max_threads, args.points_per_rank, args.local_depth, distribution=distribution)

    else:
        raise ValueError("Unknown method")