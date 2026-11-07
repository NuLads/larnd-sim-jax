#!/bin/bash
#SBATCH --job-name=bench_pixels
#SBATCH --account=neutrino:cider-nu
#SBATCH --partition=turing
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/sdf/group/neutrino/pgranger/larnd-sim-jax/slurm_bench_pixels_%j.log
#SBATCH --error=/sdf/group/neutrino/pgranger/larnd-sim-jax/slurm_bench_pixels_%j.log

SIF=/sdf/group/neutrino/pgranger/larnd-sim-jax/larndsim-jax_main.sif

singularity exec --nv --bind /sdf,/lscratch "$SIF" bash -c '
cd /sdf/group/neutrino/pgranger/larnd-sim-jax
export PYTHONPATH=$PWD/src:$PWD:$PYTHONPATH
python3 bench_pixels.py
'
echo "=== Done ==="
