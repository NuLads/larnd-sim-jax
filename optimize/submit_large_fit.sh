#!/bin/bash

#SBATCH --partition=ampere
#SBATCH --account=neutrino:dune-ml
#SBATCH --job-name=large-track-fit
#SBATCH --output=logs/large_fit-%j.out
#SBATCH --error=logs/large_fit-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=32g
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=01:30:00

# Ensure logs directory exists
mkdir -p logs

PROJECT=/sdf/group/neutrino/pgranger/larnd-sim-jax
SIF_FILE=${PROJECT}/larndsim-jax_main.sif

echo "Submitting JAX GPU job to Slurm..."
echo "Job running on GPU partition..."

apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${SIF_FILE} /bin/bash -c "
    export PYTHONPATH=${PROJECT}/src:${PROJECT}:\$PYTHONPATH;
    python3 ${PROJECT}/optimize/run_large_track_fit.py
"
