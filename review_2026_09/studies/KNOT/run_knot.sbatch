#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=32g
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=24:00:00
#SBATCH --exclude=sdfampere038,sdfampere003
#SBATCH --output=/sdf/scratch/neutrino/pgranger/review_2026_09/KNOT/logs/%x-%j.out
# usage: sbatch --job-name=<name> run_knot.sbatch <mode> <args...>
#   mode=fit  <inner.sh>                       : run a prepared command
#   mode=cont <run_name> <cont_name>           : build the CONT continuation of <run_name> at job start, then run it
#   mode=prodcont <run_name> <cont_name>       : PRODUCTION continuation (24 Oct): CONT + chain LR re-heat + freeze
#                                                release (CONT/make_prod_cont.py = the validated PRF recipe)
#   mode=prodfast <run_name> <cont_name>       : PRODUCTION fast continuation (28 Oct) of the freeze-release end point
#                                                <NAME>FR @ 14000 -> 20000 (CONT/make_prod_fast.py = the (h) recipe)
echo "=== start $(date) host $(hostname) mode $1"
if [ "$1" = cont ] || [ "$1" = prodcont ] || [ "$1" = prodfast ]; then
  IT=10000; [ "$1" = prodfast ] && IT=14000
  CK=$(ls /sdf/scratch/neutrino/pgranger/fit_result_staging/$2/history_iter${IT}_*.pkl) || { echo "no $IT checkpoint for $2"; exit 1; }
  INNER=/sdf/scratch/neutrino/pgranger/review_2026_09/KNOT/inner_$3.sh
  BUILD=make_cmd.py; [ "$1" = prodcont ] && BUILD=make_prod_cont.py; [ "$1" = prodfast ] && BUILD=make_prod_fast.py
  apptainer exec -B /sdf /sdf/group/neutrino/pgranger/larnd-sim-jax/larndsim-jax_main.sif python3 /sdf/scratch/neutrino/pgranger/review_2026_09/CONT/$BUILD $CK $3 > $INNER || exit 1
else
  INNER=$2
fi
cat $INNER
apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch /sdf/group/neutrino/pgranger/larnd-sim-jax/larndsim-jax_main.sif /bin/bash -c "
python3 -c \"import jax,sys; d=jax.devices(); print('[PREFLIGHT]',d); sys.exit(0 if d and d[0].platform=='gpu' else 1)\" || { echo '[PREFLIGHT] FATAL no GPU'; exit 1; }
# 21 Oct: speed bundle (REVIEW.md 20-21 Oct; validated NEUTRAL by full-length replicas): ~0.35x wall, ~28.5 GB host RSS.
# Defaults only -- an inner script that exports its own value (e.g. LARND_JAX_LRU=0) wins.
export LARND_JAX_LRU=\${LARND_JAX_LRU-32768} LARND_UNIQUE_BUCKET=\${LARND_UNIQUE_BUCKET-1} LARND_POSRES_VEC=\${LARND_POSRES_VEC-1}
bash $INNER
"
rc=$?; echo "=== end $(date) rc=$rc"; exit $rc
