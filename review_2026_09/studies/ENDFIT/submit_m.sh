#!/bin/bash
# ENDFIT (m): append the 27 jobs to the 6 production backfill chains (at most one running job per chain).
# usage: submit_m.sh <tail job id of chain 1> ... <tail of chain 6>
K=/sdf/scratch/neutrino/pgranger/review_2026_09/KNOT; E=/sdf/scratch/neutrino/pgranger/review_2026_09/ENDFIT
S=/sdf/scratch/neutrino/pgranger/fit_result_staging; FR=/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result
A="--account=neutrino:dune-ml --exclude=sdfampere038,sdfampere003"
PLAN=("FL:KN10N2 SK:KN10N2 LM:EXP1" "FL:KN10N4 LM:EXP2 LM:EXP4" "FL:EXP2 SK:EXP2 LM:EXS1 LM:KN10N2"
      "FL:EXP4 SK:EXP4 LM:EXS2 LM:KN10N4" "FL:KN10N1 SK:KN10N1 LM:EXS4" "FL:EXP1 SK:EXP1 SK:KN10N4 LM:KN10N1")
TAILS=("$@"); [ ${#TAILS[@]} -eq 6 ] || { echo "need 6 chain tails"; exit 2; }
for c in 0 1 2 3 4 5; do
  dep="afterany:${TAILS[$c]}"
  for item in ${PLAN[$c]}; do
    arm=${item%%:*}; base=${item#*:}
    runs="${base}${arm}"; [ $arm = FL ] && runs="${base}FL ${base}FLH"
    for r in $runs; do
      d=sci_full_$r
      if [ -e $FR/$d ] && [ ! -L $FR/$d ]; then echo "$FR/$d exists and is not a link: refusing"; exit 2; fi
      mkdir -p $S/$d; [ -L $FR/$d ] || ln -s $S/$d $FR/$d
    done
    if [ $arm = FL ]; then
      j1=$(sbatch --parsable $A --job-name=${base}FL --dependency=$dep $K/run_knot.sbatch fit $K/inner_sci_full_${base}FL.sh) || exit 1
      j2=$(sbatch --parsable $A --job-name=${base}FLH --dependency=afterok:$j1 $K/run_knot.sbatch fit $K/inner_sci_full_${base}FLH.sh) || exit 1
      echo "${base}FL $j1" >> $E/m_jobs.txt; echo "${base}FLH $j2" >> $E/m_jobs.txt
      echo "$(date +%F' '%T) endfit-m ${base}FL $j1 ($dep) -> ${base}FLH $j2" >> $K/submitted.txt
      dep="afterany:$j2?afternotok:$j1"
    else
      j=$(sbatch --parsable $A --job-name=${base}${arm} --dependency=$dep $K/run_knot.sbatch fit $K/inner_sci_full_${base}${arm}.sh) || exit 1
      echo "${base}${arm} $j" >> $E/m_jobs.txt
      echo "$(date +%F' '%T) endfit-m ${base}${arm} $j ($dep)" >> $K/submitted.txt
      dep="afterany:$j"
    fi
  done
done
