#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=neutrino:cider-nu
#SBATCH --job-name=hess_scan
#SBATCH --output=/sdf/group/neutrino/pgranger/larnd-sim-jax/logs/sci_case/hessscan-%j.out
#SBATCH --ntasks=1 --cpus-per-task=2 --mem-per-cpu=32g --gpus-per-node=a100:1 --time=08:00:00
cd /sdf/group/neutrino/pgranger/larnd-sim-jax
export PYTHONPATH=$PWD/src:$PWD:$PYTHONPATH
apptainer exec --nv -B /sdf,/fs,/lscratch larndsim-jax_main.sif python3 hessian_scan.py 5,10,20,40,60
