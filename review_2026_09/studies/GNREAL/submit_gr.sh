#!/bin/bash
# GNREAL (6 Nov; REVIEW.md "GNREAL"): 4 independent truth-point jobs on the unchanged GNPROF tree (<= 4 running).
S=/sdf/scratch/neutrino/pgranger/fit_result_staging; FR=/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result
G=/sdf/scratch/neutrino/pgranger/review_2026_09/GNPROF; R=/sdf/scratch/neutrino/pgranger/review_2026_09/GNREAL
K=/sdf/scratch/neutrino/pgranger/review_2026_09/KNOT
A="--account=neutrino:dune-ml --exclude=sdfampere038,sdfampere003"
[ -e $R/setGR_ids.txt ] && { echo "ids exist"; exit 2; }
[ "$(md5sum $G/tree/optimize/fit_params.py | cut -c1-32)" = 8a7d4c2d2b9011b04ee0f79161d3dfca ] || { echo "tree changed"; exit 2; }
echo "# submitted $(date +%F' '%T); GNPROF/tree fit_params md5 $(md5sum $G/tree/optimize/fit_params.py | cut -c1-32)" >> $R/gr_jobs.txt
for s in 29 30 31 32; do
  r=sci_full_GR_EXP$s
  if [ -e $FR/$r ] && [ ! -L $FR/$r ]; then echo "$FR/$r not a link"; exit 2; fi
  [ -d $S/$r ] && [ -n "$(ls -A $S/$r)" ] && { echo "$S/$r not empty"; exit 2; }
  mkdir -p $S/$r; [ -L $FR/$r ] || ln -s $S/$r $FR/$r
  J=$(sbatch --parsable $A --job-name=GR_EXP$s $K/run_knot.sbatch fit $R/inner/inner_GR_EXP$s.sh) || exit 1
  echo "GR_EXP$s $J" >> $R/gr_jobs.txt; echo $J >> $R/setGR_ids.txt
done
