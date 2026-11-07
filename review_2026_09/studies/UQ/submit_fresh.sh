#!/bin/bash
# FRESH TEST (4 Nov; REVIEW.md "FRESH TEST"): per DATASZ chain c = 0..5 (afterany on its tail, so <= 6 run at once): the
# production pipelines EXP<29+c>C75 and EXP<35+c>C75, then three H-stage-only re-runs H2 / H3 / H4 of each (prodfast from
# its FR @ 14000; afterok on that FR). Job ids -> UQ/fresh_jobs.txt, UQ/setF1_ids.txt (pipelines), UQ/setF2_ids.txt (H re-runs).
K=/sdf/scratch/neutrino/pgranger/review_2026_09/KNOT; U=/sdf/scratch/neutrino/pgranger/review_2026_09/UQ
S=/sdf/scratch/neutrino/pgranger/fit_result_staging; FR=/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result
A="--account=neutrino:dune-ml --exclude=sdfampere038,sdfampere003"
LOG=$U/fresh_jobs.txt
TAILS=(42985682 42985706 42985734 42985753 42985769 42985801)
[ -e $U/setF1_ids.txt ] || [ -e $U/setF2_ids.txt ] && { echo "id files exist: refusing"; exit 2; }
declare -A JFR
link() { if [ -e $FR/$1 ] && [ ! -L $FR/$1 ]; then echo "$FR/$1 exists and is not a link"; exit 2; fi
         [ -d $S/$1 ] && [ -n "$(ls -A $S/$1)" ] && { echo "$S/$1 not empty"; exit 2; }
         mkdir -p $S/$1; [ -L $FR/$1 ] || ln -s $S/$1 $FR/$1; }
prod() {   # NAME
  for r in sci_full_$1 sci_full_${1}FR sci_full_${1}H; do [ -d $S/$r ] && [ -n "$(ls -A $S/$r)" ] && { echo "$S/$r not empty"; exit 2; }; done
  out=$(DEP=afterany:$LAST $K/submit_prod.sh $1 $K/inner_sci_full_$1.sh | tail -1) || exit 1
  set -- $1 $(echo "$out" | sed 's/.* fit \([0-9]*\) -> cont [^ ]* \([0-9]*\) -> fast [^ ]* \([0-9]*\) ::.*/\1 \2 \3/')
  [ -n "$4" ] || { echo "parse failed: $out"; exit 1; }
  JFR[$1]=$3; echo "chain $c prod $1 $2 $3 $4 :: after $LAST" >> $LOG; echo "$2 $3 $4" >> $U/setF1_ids.txt; LAST=$4; }
hre() {    # NAME REP
  link sci_full_${1}$2
  J=$(sbatch --parsable $A --job-name=${1}$2 --dependency=afterany:$LAST,afterok:${JFR[$1]} $K/run_knot.sbatch prodfast sci_full_${1}FR sci_full_${1}$2) || exit 1
  echo "chain $c hre ${1}$2 $J :: after $LAST, afterok ${JFR[$1]}" >> $LOG; echo $J >> $U/setF2_ids.txt; LAST=$J; }
echo "# submitted $(date +%F' '%T)" >> $LOG
for c in 0 1 2 3 4 5; do
  LAST=${TAILS[$c]}; a=EXP$((29 + c))C75; b=EXP$((35 + c))C75
  prod $a; prod $b
  for r in H2 H3 H4; do hre $a $r; done
  for r in H2 H3 H4; do hre $b $r; done
done
echo "done: $(grep -c ' prod ' $LOG) pipelines, $(grep -c ' hre ' $LOG) H re-runs; tails: $LAST"
