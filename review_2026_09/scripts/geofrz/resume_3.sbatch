#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --job-name=GEOFRZ3r
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=32g
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=24:00:00
#SBATCH --exclude=sdfampere038,sdfampere003
#SBATCH --output=/sdf/scratch/neutrino/pgranger/review_2026_09/GEOFRZ/logs/%x-%j.out
# RESUME of sci_fullGEOFRZ3 from history_iter5200_sci_full_b100_len400_mw2500.0_refit_seed0.pkl (iter 5200), identical argv + LARND_* env as the
# original launch (read from the checkpoint provenance), with the per-job JAX cache (example_run.py).
echo "=== start $(date) host $(hostname)"
apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch /sdf/group/neutrino/pgranger/larnd-sim-jax/larndsim-jax_main.sif bash /sdf/scratch/neutrino/pgranger/review_2026_09/GEOFRZ/inner_3.sh
rc=$?; echo "=== end $(date) rc=$rc"; exit $rc
