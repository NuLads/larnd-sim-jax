#!/bin/bash
# POLISH (4 Nov; REVIEW.md "POLISH"): per chain c = 0..5, after the FRESH tail: the calibration-only E1 dumps D_F<s> of the
# FRESH end points EXP<s>C75H @ 20000, s = 29 + c and 35 + c (afterok on that H job; Set D construction). Then the B200C
# head of chain c is moved behind them. Ids -> UQ/polish_jobs.txt, UQ/setPOL_ids.txt.
U=/sdf/scratch/neutrino/pgranger/review_2026_09/UQ
S=/sdf/scratch/neutrino/pgranger/fit_result_staging; FR=/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result
A="--account=neutrino:dune-ml --exclude=sdfampere038,sdfampere003"
LOG=$U/polish_jobs.txt; IDS=$U/setPOL_ids.txt
TAILS=(43082359 43082371 43082383 43082395 43082408 43082423)
B2C=(43228743 43228750 43228758 43228766 43228774 43228782)
[ -e $IDS ] && { echo "$IDS exists: refusing"; exit 2; }
link() { if [ -e $FR/$1 ] && [ ! -L $FR/$1 ]; then echo "$FR/$1 exists and is not a link"; exit 2; fi
         [ -d $S/$1 ] && [ -n "$(ls -A $S/$1)" ] && { echo "$S/$1 not empty"; exit 2; }
         mkdir -p $S/$1; [ -L $FR/$1 ] || ln -s $S/$1 $FR/$1; }
echo "# submitted $(date +%F' '%T)" >> $LOG
for c in 0 1 2 3 4 5; do
  LAST=${TAILS[$c]}
  for s in $((29 + c)) $((35 + c)); do
    H=$(grep "^chain $c prod EXP${s}C75 " $U/fresh_jobs.txt | awk '{print $7}')
    [ -n "$H" ] || { echo "no H job for EXP${s}C75"; exit 1; }
    [ "$(sacct -j $H -X -n -o JobName%20 | tr -d ' ' | head -1)" = "EXP${s}C75H" ] || { echo "job $H is not EXP${s}C75H"; exit 1; }
    [ -e $U/dumps/D_F$s.npz ] && { echo "D_F$s exists"; exit 2; }
    link sci_full_UQ_D_F$s
    J=$(sbatch --parsable $A --job-name=UQ_D_F$s --dependency=afterany:$LAST,afterok:$H $U/run_dump.sbatch EXP${s}C75H D_F$s 20000 \
        LARND_JOINT_NEWTON_DUMP_CALIB=1) || exit 1
    echo "chain $c dump D_F$s $J :: after $LAST, afterok $H (EXP${s}C75H)" >> $LOG; echo $J >> $IDS; LAST=$J
  done
  scontrol update JobId=${B2C[$c]} Dependency=afterany:$LAST || { echo "could not move B200C head ${B2C[$c]}"; exit 1; }
  echo "chain $c B200C head ${B2C[$c]} moved behind $LAST" >> $LOG
done
echo "done: $(grep -c . $IDS) jobs"
