#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --job-name=E_truthdiag
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=32g
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=08:00:00
#SBATCH --exclude=sdfampere038
#SBATCH --output=/sdf/scratch/neutrino/pgranger/review_2026_09/E_truthdiag/logs/%x-%j.out
# args: B0 B1 OUT [extra]
EXTRA="${*:4}"
echo "=== start $(date) host $(hostname) args: $@"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch /sdf/group/neutrino/pgranger/larnd-sim-jax/larndsim-jax_main.sif /bin/bash -c "
export PYTHONPATH=/sdf/group/neutrino/pgranger/larnd-sim-jax/src:/sdf/group/neutrino/pgranger/larnd-sim-jax
export JAX_ENABLE_COMPILATION_CACHE=false   # persistent cache off: entries are data-specific (see optimize/example_run.py)
export XLA_FLAGS=--xla_gpu_deterministic_ops=false
python3 -c \"import jax,sys; d=jax.devices(); print('[PREFLIGHT]',d); sys.exit(0 if d and d[0].platform=='gpu' else 1)\" || { echo '[PREFLIGHT] FATAL no GPU'; exit 1; }
python3 -u /sdf/scratch/neutrino/pgranger/review_2026_09/E_truthdiag/truthdiag2.py --b0 $1 --b1 $2 --out $3 ${EXTRA}
"
rc=$?
echo "=== end $(date) rc=$rc"
exit $rc
