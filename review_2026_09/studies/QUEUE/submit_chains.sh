#!/bin/bash
# Pre-paper run queue as 6 SLURM dependency chains (7 Oct). Each chain continues one running round-1 job; a queued
# job becomes eligible the moment its predecessor ends (afterany), so no slot is ever idle and at most 6 run at once.
# Nothing has to stay alive in a session. Order = priority, dealt round-robin over the chains.
cd /sdf/group/neutrino/pgranger/larnd-sim-jax
ST=/sdf/scratch/neutrino/pgranger/fit_result_staging
BASE="ALL,SCIMODE=full,SCINBATCH=100,SCILEN=400,SCIITER=10000,SCIDATASZ=600,SCIBASIS=spline,SCICLR=1e-2,SCICHDECAY=0.9997,SCILRDECAY=0.91,SCIDEDXPRIOR=1.0088,SCIMEANW=2500.0,SCIDXNUL=21.760,SCIDXNUR=4.720,SCIDXSL=0.0782,SCIDXSR=0.0801,SCIDXLOC=1.8786,SCIKMIN=3,LARND_KEEP_TARGETS=1,LARND_DRIFT_FROM_HITS=1,LARND_HITS_SEED_E1=1,LARND_CHAIN_FREEZE_DRIFT=1"
TG="SCITGT=Ab:0.834881:eField:0.521519:tran_diff:1.00276e-05:long_diff:5.81418e-06:lifetime"
HEADS=(40145686 40145687 40145688 40145689 40145690 40145691)
JOBS=()   # TAG|SAMPLE|CL|EXTRA
for T in 3200 4400; do for s in 1 3 7; do JOBS+=("STRFRZ50T${T}D$s|$s|0.50|$TG:$T"); done; done
for s in 10 11 12 13 14 15 16; do JOBS+=("STRFRZ50$s|$s|0.50|"); done
for c in 30 70; do for s in 1 2 4; do JOBS+=("STRFRZ$c$s|$s|0.$c|"); done; done
acc=(neutrino:cider-nu neutrino:dune-ml)
last=("${HEADS[@]}"); i=0
for spec in "${JOBS[@]}"; do
  IFS='|' read -r tag s cl extra <<< "$spec"
  c=$(( i % 6 ))
  n=sci_full_${tag}; mkdir -p $ST/$n; [ -e fit_result/$n ] || ln -s $ST/$n fit_result/$n
  EXP="$BASE,LARND_FREEZE_STRAIGHT_CL=$cl,SCIDATASEED=$s,SCITAG=_${tag}"; [ -n "$extra" ] && EXP="$EXP,$extra"
  out=$(sbatch --parsable --array=0 --dependency=afterany:${last[$c]} --account=${acc[$((i % 2))]} --exclude=sdfampere038,sdfampere003 --export="$EXP" optimize/scripts/start_sci_case.sh 2>&1)
  jid=$(printf '%s\n' "$out" | grep -Eo '^[0-9]+' | tail -1)
  if [ -z "$jid" ]; then echo "SUBMIT FAILED for $tag: $out"; exit 1; fi
  echo "chain $c: $tag (sample $s, CL $cl) -> job $jid after ${last[$c]}"
  last[$c]=$jid; i=$((i+1))
done
