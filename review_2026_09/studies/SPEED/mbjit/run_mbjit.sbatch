#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=neutrino:dune-ml
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=32g
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=02:00:00
#SBATCH --exclude=sdfampere038,sdfampere003
#SBATCH --job-name=MBJIT
#SBATCH --output=/sdf/scratch/neutrino/pgranger/review_2026_09/SPEED/mbjit/%x-%j.out
# Jit ceiling micro-bench: (1) eager + jit + cache listing, default XLA flags; (2) jit only, GPU autotuning off.
D=/sdf/scratch/neutrino/pgranger/review_2026_09/SPEED/mbjit; C=/sdf/scratch/neutrino/pgranger/jax_job_cache/mbjit_$SLURM_JOB_ID
SIF=/sdf/group/neutrino/pgranger/larnd-sim-jax/larndsim-jax_main.sif
ENV="LARND_UNIQUE_BUCKET=1 LARND_CHAIN_FRAME=drift LARND_SPLINE_KMIN=3"
env $ENV MB_CACHE=${C}_1 apptainer exec --nv -B /sdf $SIF python3 $D/mbjit.py 4 ABC $D/mbjit_default.json
env $ENV MB_CACHE=${C}_2 XLA_FLAGS=--xla_gpu_autotune_level=0 apptainer exec --nv -B /sdf $SIF python3 $D/mbjit.py 4 B $D/mbjit_noautotune.json
rm -rf ${C}_1 ${C}_2
