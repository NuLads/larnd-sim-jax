#!/bin/bash

#SBATCH --partition=ampere
#SBATCH --account=neutrino:dune-ml
#SBATCH --job-name=prototype-track-fit
#SBATCH --output=logs/prototype_fit-%j.out
#SBATCH --error=logs/prototype_fit-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=32g
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=01:00:00

# Ensure logs directory exists
mkdir -p logs

PROJECT=/sdf/group/neutrino/pgranger/larnd-sim-jax
SIF_FILE=${PROJECT}/larndsim-jax_main.sif

echo "Submitting JAX GPU job to Slurm..."
echo "Job running on GPU partition..."
nvidia-smi

apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${SIF_FILE} /bin/bash -c "
    export PYTHONPATH=${PROJECT}/src:${PROJECT}:\$PYTHONPATH;
    python3 ${PROJECT}/optimize/prototype_track_fit.py
"
