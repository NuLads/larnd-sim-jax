#!/bin/bash
#SBATCH --job-name=track_pos_fit
#SBATCH --partition=ampere
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/sdf/group/neutrino/pgranger/larnd-sim-jax/scripts/track_pos_fit_results/slurm-%j.out
#SBATCH --error=/sdf/group/neutrino/pgranger/larnd-sim-jax/scripts/track_pos_fit_results/slurm-%j.err
#SBATCH --account=neutrino:dune-ml

PROJECT=/sdf/group/neutrino/pgranger/larnd-sim-jax
SIF=/sdf/group/neutrino/pgranger/larnd-sim-jax.sif

mkdir -p ${PROJECT}/scripts/track_pos_fit_results

echo "=== Job ${SLURM_JOB_ID} starting at $(date) ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"

singularity exec --nv \
  -B /sdf \
  ${SIF} \
  python3 ${PROJECT}/scripts/track_pos_fit_study.py

echo "=== Job finished at $(date) ==="
