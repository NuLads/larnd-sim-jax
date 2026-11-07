#!/bin/bash
# DXPROF (6 Nov; REVIEW.md "DXPROF"): 8 truth-point jobs as 4 chains of 2 (with the acceptance pipeline: <= 5 running).
S=/sdf/scratch/neutrino/pgranger/fit_result_staging; FR=/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result
X=/sdf/scratch/neutrino/pgranger/review_2026_09/DXPROF; K=/sdf/scratch/neutrino/pgranger/review_2026_09/KNOT
A="--account=neutrino:dune-ml --exclude=sdfampere038,sdfampere003"
[ -e $X/setDX_ids.txt ] && { echo "ids exist"; exit 2; }
echo "# submitted $(date +%F' '%T); DXPROF/tree fit_params md5 $(md5sum $X/tree/optimize/fit_params.py | cut -c1-32)" >> $X/dx_jobs.txt
for pair in "1 5" "2 6" "3 7" "4 8"; do
  LAST=""
  for s in $pair; do
    r=sci_full_DX_KN10N$s
    if [ -e $FR/$r ] && [ ! -L $FR/$r ]; then echo "$FR/$r not a link"; exit 2; fi
    [ -d $S/$r ] && [ -n "$(ls -A $S/$r)" ] && { echo "$S/$r not empty"; exit 2; }
    mkdir -p $S/$r; [ -L $FR/$r ] || ln -s $S/$r $FR/$r
    DEP=${LAST:+--dependency=afterany:$LAST}
    J=$(sbatch --parsable $A --job-name=DX_KN10N$s $DEP $K/run_knot.sbatch fit $X/inner/inner_DX_KN10N$s.sh) || exit 1
    echo "DX_KN10N$s $J ${LAST:+after $LAST}" >> $X/dx_jobs.txt; echo $J >> $X/setDX_ids.txt; LAST=$J
  done
done
