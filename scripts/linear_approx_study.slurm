#!/bin/bash
#SBATCH --job-name=linear_approx_study
#SBATCH --partition=ampere
#SBATCH --account=neutrino:dune-ml
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/sdf/group/neutrino/pgranger/larnd-sim-jax/scripts/linear_approx_study_results/slurm-%j.out
#SBATCH --error=/sdf/group/neutrino/pgranger/larnd-sim-jax/scripts/linear_approx_study_results/slurm-%j.err

PROJECT=/sdf/group/neutrino/pgranger/larnd-sim-jax
SIF=/sdf/group/neutrino/pgranger/larnd-sim-jax.sif

mkdir -p ${PROJECT}/scripts/linear_approx_study_results

echo "=== Job ${SLURM_JOB_ID} starting at $(date) ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"

singularity exec --nv \
  -B /sdf \
  ${SIF} \
  python3 ${PROJECT}/scripts/linear_approx_study.py

echo "=== Job finished at $(date) ==="
