#!/bin/bash
# TRUTHC (4 Nov; REVIEW.md "TRUTHC"): per chain c = 0..5, after the NOCLIP tail: the truth-point runs sci_full_TC_<RUN>
# (pre-built inners TRUTHC/inner/inner_TC_<RUN>.sh). Chain c gets KN10N<c+1>; chains 0, 1 also KN10N7, KN10N8; chain 2
# also EXP17 (P0 only, the V2 instrument check). Then the FRESH TEST chain head c is moved behind this chain's last job.
# Ids -> TRUTHC/tc_jobs.txt, TRUTHC/setTC_ids.txt.
K=/sdf/scratch/neutrino/pgranger/review_2026_09/KNOT; T=/sdf/scratch/neutrino/pgranger/review_2026_09/TRUTHC
S=/sdf/scratch/neutrino/pgranger/fit_result_staging; FR=/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result
A="--account=neutrino:dune-ml --exclude=sdfampere038,sdfampere003"
LOG=$T/tc_jobs.txt; IDS=$T/setTC_ids.txt
TAILS=(43138624 43138629 43138634 43138639 43138643 43138647)
HEADS=(43082347 43082360 43082372 43082384 43082396 43082409)
RUNS=("KN10N1 KN10N7" "KN10N2 KN10N8" "KN10N3 EXP17" "KN10N4" "KN10N5" "KN10N6")
[ -e $IDS ] && { echo "$IDS exists: refusing"; exit 2; }
link() { if [ -e $FR/$1 ] && [ ! -L $FR/$1 ]; then echo "$FR/$1 exists and is not a link"; exit 2; fi
         [ -d $S/$1 ] && [ -n "$(ls -A $S/$1)" ] && { echo "$S/$1 not empty"; exit 2; }
         mkdir -p $S/$1; [ -L $FR/$1 ] || ln -s $S/$1 $FR/$1; }
echo "# submitted $(date +%F' '%T)" >> $LOG
for c in 0 1 2 3 4 5; do
  LAST=${TAILS[$c]}
  for r in ${RUNS[$c]}; do
    IN=$T/inner/inner_TC_$r.sh; [ -f $IN ] || { echo "no $IN"; exit 2; }
    link sci_full_TC_$r
    J=$(sbatch --parsable $A --job-name=TC_$r --dependency=afterany:$LAST $K/run_knot.sbatch fit $IN) || exit 1
    echo "chain $c tc TC_$r $J :: after $LAST" >> $LOG; echo $J >> $IDS; LAST=$J
  done
  scontrol update JobId=${HEADS[$c]} Dependency=afterany:$LAST || { echo "could not move FRESH head ${HEADS[$c]}"; exit 1; }
  echo "chain $c FRESH head ${HEADS[$c]} moved behind $LAST" >> $LOG
done
echo "done: $(grep -c . $IDS) jobs"
