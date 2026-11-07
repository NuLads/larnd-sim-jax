#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --job-name=DRIFTTRU4r
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=32g
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=24:00:00
#SBATCH --exclude=sdfampere038,sdfampere003
#SBATCH --output=/sdf/scratch/neutrino/pgranger/review_2026_09/DRIFTTRU_resume/logs/%x-%j.out
# RESUME of sci_full_DRIFTTRU4 from iter 2200 (crashed: /sdf/data full). --iterations 7800 so the run ENDS at 10000.
echo "=== start $(date) host $(hostname)"
apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch /sdf/group/neutrino/pgranger/larnd-sim-jax/larndsim-jax_main.sif bash /sdf/scratch/neutrino/pgranger/review_2026_09/DRIFTTRU_resume/inner_4.sh
rc=$?; echo "=== end $(date) rc=$rc"; exit $rc
