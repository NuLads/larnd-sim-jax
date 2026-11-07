#!/bin/bash
# Schedule-invariance test on the current recipe R2 (STRFRZ50), samples 3/7/8, paired with STRFRZ50 3/7/8 (same code:
# a5d7bdc + inactive endpoint knob). Calibration LR held at peak for 1200 iterations after warm-up, then decayed at
# 0.89768 per epoch: final LR matched to R2 within 1.4%, total calibration travel 1.784x. Only the lr_kw line of the
# launcher differs (transition_begin). Appended to the tails of chains 1-3, so no extra slot.
cd /sdf/group/neutrino/pgranger/larnd-sim-jax
ST=/sdf/scratch/neutrino/pgranger/fit_result_staging
BASE="ALL,SCIMODE=full,SCINBATCH=100,SCILEN=400,SCIITER=10000,SCIDATASZ=600,SCIBASIS=spline,SCICLR=1e-2,SCICHDECAY=0.9997,SCILRDECAY=0.89768,SCILRBEGIN=1200,SCIDEDXPRIOR=1.0088,SCIMEANW=2500.0,SCIDXNUL=21.760,SCIDXNUR=4.720,SCIDXSL=0.0782,SCIDXSR=0.0801,SCIDXLOC=1.8786,SCIKMIN=3,LARND_KEEP_TARGETS=1,LARND_DRIFT_FROM_HITS=1,LARND_HITS_SEED_E1=1,LARND_CHAIN_FREEZE_DRIFT=1,LARND_FREEZE_STRAIGHT_CL=0.50"
declare -a S=(3 7 8); declare -a AFTER=(40213550 40213535 40213557); declare -a ACC=(neutrino:dune-ml neutrino:cider-nu neutrino:dune-ml)
for k in 0 1 2; do
  s=${S[$k]}; tag=SCHED$s; n=sci_full_${tag}; mkdir -p $ST/$n; [ -e fit_result/$n ] || ln -s $ST/$n fit_result/$n
  out=$(sbatch --parsable --array=0 --dependency=afterany:${AFTER[$k]} --account=${ACC[$k]} --exclude=sdfampere038,sdfampere003 \
        --export="$BASE,SCIDATASEED=$s,SCITAG=_${tag}" /sdf/scratch/neutrino/pgranger/review_2026_09/SCHED/start_sci_case_sched.sh 2>&1)
  jid=$(printf '%s\n' "$out" | grep -Eo '^[0-9]+' | tail -1)
  if [ -z "$jid" ]; then echo "SUBMIT FAILED for $tag: $out"; exit 1; fi
  echo "$tag (sample $s) -> job $jid after ${AFTER[$k]}"
done
