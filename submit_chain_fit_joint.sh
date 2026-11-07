#!/bin/bash
#SBATCH --job-name=chain_joint
#SBATCH --account=neutrino
#SBATCH --partition=turing
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/sdf/group/neutrino/pgranger/larnd-sim-jax/slurm_chain_joint_%j.log
#SBATCH --error=/sdf/group/neutrino/pgranger/larnd-sim-jax/slurm_chain_joint_%j.log

export FIT_MODE=joint
export MAX_BATCHES=6

SIF=/sdf/group/neutrino/pgranger/larnd-sim-jax/larndsim-jax_main.sif
SCRIPT=/sdf/group/neutrino/pgranger/larnd-sim-jax/run_chain_fit.py

echo "=== Job started: $(date) ==="
echo "Node: $SLURMD_NODENAME  GPU: $CUDA_VISIBLE_DEVICES"
echo "FIT_MODE=$FIT_MODE  MAX_BATCHES=$MAX_BATCHES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

singularity exec --nv \
    --bind /sdf,/lscratch \
    "$SIF" \
    python3 "$SCRIPT"

echo "=== Job finished: $(date) ==="
