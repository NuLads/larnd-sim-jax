#!/bin/bash
# Round 1 of the pre-paper runs (7 Oct). Recipe = STRFRZ50 (hit seed + freeze drift of straight tracks at CL 0.5).
# ENDSM10{1,2,4}: recipe + track endpoints smeared by 1 mm per coordinate (LARND_ENDPOINT_SMEAR_MM=1.0).
#   Rule (stated before results): tolerates realistic endpoint errors if |lifetime|<=5% and |long_diff|<=8% on 3/3 AND
#   the paired lifetime shift vs STRFRZ50{1,2,4} is within +-3 points on 3/3; otherwise endpoints must be fitted.
# STRFRZ50{3,7,8}: ensemble extension (n=9); per-sample check |lifetime|<=5%.
cd /sdf/group/neutrino/pgranger/larnd-sim-jax
ST=/sdf/scratch/neutrino/pgranger/fit_result_staging
BASE="ALL,SCIMODE=full,SCINBATCH=100,SCILEN=400,SCIITER=10000,SCIDATASZ=600,SCIBASIS=spline,SCICLR=1e-2,SCICHDECAY=0.9997,SCILRDECAY=0.91,SCIDEDXPRIOR=1.0088,SCIMEANW=2500.0,SCIDXNUL=21.760,SCIDXNUR=4.720,SCIDXSL=0.0782,SCIDXSR=0.0801,SCIDXLOC=1.8786,SCIKMIN=3,LARND_KEEP_TARGETS=1,LARND_DRIFT_FROM_HITS=1,LARND_HITS_SEED_E1=1,LARND_CHAIN_FREEZE_DRIFT=1,LARND_FREEZE_STRAIGHT_CL=0.50"
acc=(neutrino:cider-nu neutrino:dune-ml); i=0
for s in 1 2 4; do
  n=sci_full_ENDSM10$s; mkdir -p $ST/$n; [ -e fit_result/$n ] || ln -s $ST/$n fit_result/$n
  sbatch --array=0 --account=${acc[$((i % 2))]} --exclude=sdfampere038,sdfampere003 --export="$BASE,LARND_ENDPOINT_SMEAR_MM=1.0,SCIDATASEED=$s,SCITAG=_ENDSM10$s" optimize/scripts/start_sci_case.sh 2>&1 | grep -v OS_VER
  i=$((i+1))
done
for s in 3 7 8; do
  n=sci_full_STRFRZ50$s; mkdir -p $ST/$n; [ -e fit_result/$n ] || ln -s $ST/$n fit_result/$n
  sbatch --array=0 --account=${acc[$((i % 2))]} --exclude=sdfampere038,sdfampere003 --export="$BASE,SCIDATASEED=$s,SCITAG=_STRFRZ50$s" optimize/scripts/start_sci_case.sh 2>&1 | grep -v OS_VER
  i=$((i+1))
done
squeue -u pgranger -h -o "%i %j %T %M %R"
