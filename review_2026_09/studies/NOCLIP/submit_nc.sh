#!/bin/bash
# NOCLIP (4 Nov; REVIEW.md "NOCLIP"): per chain c = 0..5, after the DATASZ tail: H-stage no-clip runs EXP<s>C75HNC for
# s = 17+2c, 18+2c (pre-built inners KNOT/inner_sci_full_EXP<s>C75HNC.sh), then their calibration-only E1 dumps D_NC<s>.
# Then the FRESH TEST chain head c is moved behind this chain's last job. Ids -> NOCLIP/nc_jobs.txt, NOCLIP/setNC_ids.txt.
K=/sdf/scratch/neutrino/pgranger/review_2026_09/KNOT; U=/sdf/scratch/neutrino/pgranger/review_2026_09/UQ
N=/sdf/scratch/neutrino/pgranger/review_2026_09/NOCLIP
S=/sdf/scratch/neutrino/pgranger/fit_result_staging; FR=/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result
A="--account=neutrino:dune-ml --exclude=sdfampere038,sdfampere003"
LOG=$N/nc_jobs.txt; IDS=$N/setNC_ids.txt
TAILS=(42985682 42985706 42985734 42985753 42985769 42985801)
[ -e $IDS ] && { echo "$IDS exists: refusing"; exit 2; }
link() { if [ -e $FR/$1 ] && [ ! -L $FR/$1 ]; then echo "$FR/$1 exists and is not a link"; exit 2; fi
         [ -d $S/$1 ] && [ -n "$(ls -A $S/$1)" ] && { echo "$S/$1 not empty"; exit 2; }
         mkdir -p $S/$1; [ -L $FR/$1 ] || ln -s $S/$1 $FR/$1; }
echo "# submitted $(date +%F' '%T)" >> $LOG
for c in 0 1 2 3 4 5; do
  LAST=${TAILS[$c]}; declare -A JH=()
  for s in $((17 + 2 * c)) $((18 + 2 * c)); do
    IN=$K/inner_sci_full_EXP${s}C75HNC.sh; [ -f $IN ] || { echo "no $IN"; exit 2; }
    link sci_full_EXP${s}C75HNC
    J=$(sbatch --parsable $A --job-name=EXP${s}C75HNC --dependency=afterany:$LAST $K/run_knot.sbatch fit $IN) || exit 1
    echo "chain $c hnc EXP${s}C75HNC $J :: after $LAST" >> $LOG; echo $J >> $IDS; JH[$s]=$J; LAST=$J
  done
  for s in $((17 + 2 * c)) $((18 + 2 * c)); do
    [ -e $U/dumps/D_NC$s.npz ] && { echo "D_NC$s exists"; exit 2; }
    link sci_full_UQ_D_NC$s
    J=$(sbatch --parsable $A --job-name=UQ_D_NC$s --dependency=afterany:$LAST,afterok:${JH[$s]} $U/run_dump.sbatch EXP${s}C75HNC D_NC$s 20000 \
        LARND_JOINT_NEWTON_DUMP_CALIB=1 LARND_CALIB_ADAM_RESET=0) || exit 1
    echo "chain $c dump D_NC$s $J :: after $LAST, afterok ${JH[$s]}" >> $LOG; echo $J >> $IDS; LAST=$J
  done
  HEAD=$(grep "^chain $c prod EXP$((29 + c))C75 " $U/fresh_jobs.txt | awk '{print $5}')
  [ -n "$HEAD" ] || { echo "no FRESH head for chain $c"; exit 1; }
  scontrol update JobId=$HEAD Dependency=afterany:$LAST || { echo "could not move FRESH head $HEAD"; exit 1; }
  echo "chain $c FRESH head $HEAD moved behind $LAST" >> $LOG
done
echo "done: $(grep -c . $IDS) jobs"
