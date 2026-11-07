#!/bin/bash
#SBATCH --partition=milano
#SBATCH --account=neutrino:cider-nu
#SBATCH --job-name=mk_constdedx
#SBATCH --output=/sdf/group/neutrino/pgranger/larnd-sim-jax/logs/sci_case/constdedx-%j.out
#SBATCH --ntasks=1 --cpus-per-task=2 --mem-per-cpu=32g --time=02:00:00
cd /sdf/group/neutrino/pgranger/larnd-sim-jax
apptainer exec -B /sdf,/fs,/lscratch larndsim-jax_main.sif python3 mk_const_dedx.py
