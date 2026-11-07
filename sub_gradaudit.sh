#!/bin/bash
#SBATCH --partition=milano
#SBATCH --account=neutrino:cider-nu
#SBATCH --job-name=grad_audit
#SBATCH --output=/sdf/group/neutrino/pgranger/larnd-sim-jax/logs/sci_case/gradaudit-%j.out
#SBATCH --ntasks=1 --cpus-per-task=4 --mem-per-cpu=16g --time=04:00:00
cd /sdf/group/neutrino/pgranger/larnd-sim-jax
apptainer exec -B /sdf,/fs,/lscratch larndsim-jax_main.sif python3 grad_audit.py
