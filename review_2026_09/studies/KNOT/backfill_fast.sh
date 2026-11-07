#!/bin/bash
# 28 Oct: bring finished production samples (end point <NAME>FR @ 14000) to the new production end point <NAME>H @ 20000
# (run_knot.sbatch prodfast = CONT/make_prod_fast.py), as SLURM chains: at most one job per chain runs.
# usage: backfill_fast.sh "<NAME list of chain 1>" "<NAME list of chain 2>" ...
K=/sdf/scratch/neutrino/pgranger/review_2026_09/KNOT; S=/sdf/scratch/neutrino/pgranger/fit_result_staging
FR=/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result
A="--account=neutrino:dune-ml --exclude=sdfampere038,sdfampere003"
for chain in "$@"; do
  dep=
  for NAME in $chain; do
    ls $S/sci_full_${NAME}FR/history_iter14000_*.pkl > /dev/null || { echo "no 14000 checkpoint for ${NAME}FR"; exit 1; }
    r=sci_full_${NAME}H
    if [ -e $FR/$r ] && [ ! -L $FR/$r ]; then echo "$FR/$r exists and is not a link: refusing"; exit 2; fi
    mkdir -p $S/$r; [ -L $FR/$r ] || ln -s $S/$r $FR/$r
    j=$(sbatch --parsable $A --job-name=${NAME}H ${dep:+--dependency=$dep} $K/run_knot.sbatch prodfast sci_full_${NAME}FR $r) || exit 1
    echo "$(date +%F' '%T) prodfast ${NAME}H $j${dep:+ ($dep)}" | tee -a $K/submitted.txt
    dep=afterany:$j
  done
done
