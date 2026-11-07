#!/bin/bash
# Replication of STRFRZ50 (HITSEED + freeze drift on tracks called straight at CL 0.5) -- truth-free candidate.
# (a) fresh samples 5/6/9 at tau=2406 (paired refs: DSREP5/6/9); (b) tau=1200 on samples 1/3/7 (refs: CURVE1200D*, DRIFTK6T1200D*).
# Rule (stated 6 Oct ~11:30, before results): replicates if on 5/6/9 |lifetime|<=5% AND |long_diff|<=8% on 3/3 AND the
# pooled n=6 (1,2,4,5,6,9) lifetime mean is within +-3%; robustness: at tau=1200 |lifetime|<=5% on 3/3.
cd /sdf/group/neutrino/pgranger/larnd-sim-jax
ST=/sdf/scratch/neutrino/pgranger/fit_result_staging
BASE="ALL,SCIMODE=full,SCINBATCH=100,SCILEN=400,SCIITER=10000,SCIDATASZ=600,SCIBASIS=spline,SCICLR=1e-2,SCICHDECAY=0.9997,SCILRDECAY=0.91,SCIDEDXPRIOR=1.0088,SCIMEANW=2500.0,SCIDXNUL=21.760,SCIDXNUR=4.720,SCIDXSL=0.0782,SCIDXSR=0.0801,SCIDXLOC=1.8786,SCIKMIN=3,LARND_KEEP_TARGETS=1,LARND_DRIFT_FROM_HITS=1,LARND_HITS_SEED_E1=1,LARND_CHAIN_FREEZE_DRIFT=1,LARND_FREEZE_STRAIGHT_CL=0.50"
acc=(neutrino:cider-nu neutrino:dune-ml); i=0
for s in 5 6 9; do
  n=sci_full_STRFRZ50$s; mkdir -p $ST/$n; [ -e fit_result/$n ] || ln -s $ST/$n fit_result/$n
  sbatch --array=0 --account=${acc[$((i % 2))]} --exclude=sdfampere038,sdfampere003 --export="$BASE,SCIDATASEED=$s,SCITAG=_STRFRZ50$s" optimize/scripts/start_sci_case.sh 2>&1 | grep -v OS_VER
  i=$((i+1))
done
TGT="SCITGT=Ab:0.834881:eField:0.521519:tran_diff:1.00276e-05:long_diff:5.81418e-06:lifetime:1200"
for s in 1 3 7; do
  n=sci_full_STRFRZ50T1200D$s; mkdir -p $ST/$n; [ -e fit_result/$n ] || ln -s $ST/$n fit_result/$n
  sbatch --array=0 --account=${acc[$((i % 2))]} --exclude=sdfampere038,sdfampere003 --export="$BASE,$TGT,SCIDATASEED=$s,SCITAG=_STRFRZ50T1200D$s" optimize/scripts/start_sci_case.sh 2>&1 | grep -v OS_VER
  i=$((i+1))
done
squeue -u pgranger -h -o "%i %j %T %M %R"
