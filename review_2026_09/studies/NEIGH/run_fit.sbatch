#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=neutrino:dune-ml
#SBATCH --job-name=R089fit
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=32g
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=06:00:00
#SBATCH --exclude=sdfampere038,sdfampere003
#SBATCH --output=/sdf/scratch/neutrino/pgranger/review_2026_09/NEIGH/logs/%x-%j.out
# R089: exact vs coarse induction on neighboring pixels. readout -> larnd-sim geo rerun -> first-order calibration shift
D=/sdf/scratch/neutrino/pgranger/review_2026_09/NEIGH
echo "=== start $(date) host $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch /sdf/group/neutrino/pgranger/larnd-sim-jax/larndsim-jax_main.sif /bin/bash -c "
export PYTHONPATH=/sdf/group/neutrino/pgranger/larnd-sim-jax/src:/sdf/group/neutrino/pgranger/larnd-sim-jax:/sdf/scratch/neutrino/pgranger/review_2026_09/PAPERJOBS:$D
export JAX_ENABLE_COMPILATION_CACHE=false XLA_PYTHON_CLIENT_PREALLOCATE=false
python3 -c \"import jax,sys; d=jax.devices(); print('[PREFLIGHT]',d); sys.exit(0 if d and d[0].platform=='gpu' else 1)\" || { echo '[PREFLIGHT] FATAL no GPU'; exit 1; }
cd $D



python3 run_fit.py --nb 20 --iters 12 --kinds prod,E,T,L --out $D/fit.json || echo FIT FAILED
"
echo "=== end $(date) rc=$?"
