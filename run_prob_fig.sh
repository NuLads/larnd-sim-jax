#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=neutrino
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --job-name=probfig
#SBATCH --output=logs/probfig-%j.out
cd /sdf/group/neutrino/pgranger/larnd-sim-jax
python make_probabilistic_figure.py --runs 10000 --events 60 --batches 12 --chunk 50
