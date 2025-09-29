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


# Restore AMD compiler env
module load PrgEnv-aocc
module load craype-network-ucx
module load cray-mpich-ucx
module unload darshan
module load perftools

# Home and work directories
export HOME="/home/e738/e738/skailasa"
export WORK="/work/e738/e738/skailasa"

# Create a scratch directory for this run
export SCRATCH=${WORK}/weak_${SLURM_JOBID}

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
script_name="fmm_pat"



