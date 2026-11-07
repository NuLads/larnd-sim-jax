#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=32g
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=24:00:00
#SBATCH --exclude=sdfampere038,sdfampere003
#SBATCH --output=/sdf/scratch/neutrino/pgranger/review_2026_09/KNOT/logs/%x-%j.out
# DATASZ (3 Nov): one stage of an epoch-matched pipeline (make_ds.py), built AT JOB START (the checkpoint need not exist
# at submission), written to KNOT/inner_sci_full_<NAME>.sh (as run_knot does, so UQ/run_dump.sbatch works on it), then run.
# usage: sbatch --job-name=<NAME> run_ds.sbatch fit  <NAME> <data_seed> <data_sz> <N_b> <L>
#        sbatch --job-name=<NAME>FR run_ds.sbatch cont <NAME> <NAME>FR        (from <NAME> @ 10000 r)
#        sbatch --job-name=<NAME>H  run_ds.sbatch fast <NAME>FR <NAME>H       (from <NAME>FR @ 14000 r)
K=/sdf/scratch/neutrino/pgranger/review_2026_09/KNOT; DS=/sdf/scratch/neutrino/pgranger/review_2026_09/DATASZ
S=/sdf/scratch/neutrino/pgranger/fit_result_staging; SIF=/sdf/group/neutrino/pgranger/larnd-sim-jax/larndsim-jax_main.sif
MODE=$1; shift
if [ "$MODE" = fit ]; then
  NAME=$1; IN=$K/inner_sci_full_$NAME.sh
  apptainer exec -B /sdf $SIF env JAX_PLATFORMS=cpu python3 $DS/make_ds.py fit "$@" > $IN 2>/dev/null || { echo "build failed"; exit 1; }
else
  SRC=$1; NAME=$2; IN=$K/inner_sci_full_$NAME.sh
  NB=$(grep -o -- '--max_nbatch [0-9]*' $K/inner_sci_full_${SRC%FR}.sh | awk '{print $2}')
  BASE=10000; [ "$MODE" = fast ] && BASE=14000
  IT=$((BASE * NB / 100))
  CK=$(ls $S/sci_full_$SRC/history_iter${IT}_*.pkl) || { echo "no $IT checkpoint for $SRC"; exit 1; }
  apptainer exec -B /sdf $SIF env JAX_PLATFORMS=cpu python3 $DS/make_ds.py $MODE $CK sci_full_$NAME > $IN 2>/dev/null || { echo "build failed"; exit 1; }
fi
grep -q -- "--test_name sci_full_$NAME " $IN || { echo "bad inner $IN"; exit 1; }
bash $K/run_knot.sbatch fit $IN
