#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --job-name=paper_job
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=32g
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=04:00:00
#SBATCH --exclude=sdfampere038,sdfampere003
#SBATCH --output=/sdf/scratch/neutrino/pgranger/review_2026_09/PAPERJOBS/logs/%x-%j.out
# usage: sbatch run_paper.sbatch "<shell command run inside the container, from PAPERJOBS>"
echo "=== start $(date) host $(hostname) cmd: $1"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch /sdf/group/neutrino/pgranger/larnd-sim-jax/larndsim-jax_main.sif /bin/bash -c "
export PYTHONPATH=/sdf/group/neutrino/pgranger/larnd-sim-jax/src:/sdf/group/neutrino/pgranger/larnd-sim-jax:/sdf/scratch/neutrino/pgranger/review_2026_09/PAPERJOBS
export JAX_ENABLE_COMPILATION_CACHE=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_deterministic_ops=false
python3 -c \"import jax,sys; d=jax.devices(); print('[PREFLIGHT]',d); sys.exit(0 if d and d[0].platform=='gpu' else 1)\" || { echo '[PREFLIGHT] FATAL no GPU'; exit 1; }
cd /sdf/scratch/neutrino/pgranger/review_2026_09/PAPERJOBS
$1
"
rc=$?
echo "=== end $(date) rc=$rc"
exit $rc
