#!/bin/bash
# GNPROF GPU smoke (2 batches of KN10N1 C75H @ 20000 at P0): memory, time per column, internal consistency.
S=/sdf/scratch/neutrino/pgranger/fit_result_staging; FR=/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result
G=/sdf/scratch/neutrino/pgranger/review_2026_09/GNPROF; K=/sdf/scratch/neutrino/pgranger/review_2026_09/KNOT
A="--account=neutrino:dune-ml --exclude=sdfampere038,sdfampere003"
r=sci_full_GNSMOKE
if [ -e $FR/$r ] && [ ! -L $FR/$r ]; then echo "$FR/$r not a link"; exit 2; fi
mkdir -p $S/$r; [ -L $FR/$r ] || ln -s $S/$r $FR/$r
J=$(sbatch --parsable $A --job-name=GNSMOKE $K/run_knot.sbatch fit $G/inner/inner_GNSMOKE.sh) || exit 1
echo "$(date +%F' '%T) GNSMOKE $J tree md5 $(md5sum $G/tree/optimize/fit_params.py | cut -c1-32)" | tee -a $G/gn_jobs.txt
