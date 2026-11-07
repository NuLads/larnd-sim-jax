#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=32g
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=14:00:00
#SBATCH --exclude=sdfampere038,sdfampere003
#SBATCH --output=/sdf/scratch/neutrino/pgranger/review_2026_09/CONT/logs/%x-%j.out
# usage: sbatch --job-name=<name> run_cont.sbatch <inner.sh>   (inner.sh written by submit_cont.sh from make_cmd.py)
echo "=== start $(date) host $(hostname)"; cat "$1"
apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch /sdf/group/neutrino/pgranger/larnd-sim-jax/larndsim-jax_main.sif /bin/bash -c "
python3 -c \"import jax,sys; d=jax.devices(); print('[PREFLIGHT]',d); sys.exit(0 if d and d[0].platform=='gpu' else 1)\" || { echo '[PREFLIGHT] FATAL no GPU'; exit 1; }
bash $1
"
rc=$?; echo "=== end $(date) rc=$rc"; exit $rc
