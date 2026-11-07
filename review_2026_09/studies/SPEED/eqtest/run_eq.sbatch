#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=neutrino:dune-ml
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=32g
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=01:30:00
#SBATCH --exclude=sdfampere038,sdfampere003
#SBATCH --job-name=EQUB
#SBATCH --output=/sdf/scratch/neutrino/pgranger/review_2026_09/SPEED/eqtest/%x-%j.out
# LARND_UNIQUE_BUCKET equivalence on GPU with deterministic XLA ops: flag 0 then 1, same inputs; compare npz bitwise.
D=/sdf/scratch/neutrino/pgranger/review_2026_09/SPEED
for m in 0 1; do
  LARND_UNIQUE_BUCKET=$m LARND_CHAIN_FRAME=drift LARND_SPLINE_KMIN=3 XLA_FLAGS=--xla_gpu_deterministic_ops=true \
  apptainer exec --nv -B /sdf /sdf/group/neutrino/pgranger/larnd-sim-jax/larndsim-jax_main.sif \
    python3 $D/eqtest/test_unique_bucket.py $D/stage $D/eqtest/ub_gpu_$m.npz
done
