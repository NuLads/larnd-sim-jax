#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=neutrino:dune-ml
#SBATCH --job-name=R089neigh
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
python3 run_neigh.py --part readout --out $D/readout.json || echo READOUT FAILED
python3 run_geo_exact.py $D/out_stage2_jax_geo_exact.h5 > $D/logs/geo_exact.log 2>&1 || echo GEO FAILED
grep -E '\[data\]|\[perf\]' $D/logs/geo_exact.log | tail -5
python3 run_neigh.py --part bias --nb 60 --out $D/bias.json || echo BIAS FAILED
"
echo "=== end $(date) rc=$?"
