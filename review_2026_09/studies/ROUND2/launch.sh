#!/bin/bash
# Round 2 of the pre-paper runs: recipe R2 (STRFRZ50) at longer true lifetimes, samples 1/3/7, paired with the
# production CURVE3200D* / CURVE4400D* fits (and the truth-drift DRIFTK6T3200D* at 3200).
# Rule (stated 7 Oct before results; spread grows ~ tau): mean |lifetime| error over the 3 samples <= 5% at 3200 us
# and <= 7% at 4400 us.
cd /sdf/group/neutrino/pgranger/larnd-sim-jax
ST=/sdf/scratch/neutrino/pgranger/fit_result_staging
BASE="ALL,SCIMODE=full,SCINBATCH=100,SCILEN=400,SCIITER=10000,SCIDATASZ=600,SCIBASIS=spline,SCICLR=1e-2,SCICHDECAY=0.9997,SCILRDECAY=0.91,SCIDEDXPRIOR=1.0088,SCIMEANW=2500.0,SCIDXNUL=21.760,SCIDXNUR=4.720,SCIDXSL=0.0782,SCIDXSR=0.0801,SCIDXLOC=1.8786,SCIKMIN=3,LARND_KEEP_TARGETS=1,LARND_DRIFT_FROM_HITS=1,LARND_HITS_SEED_E1=1,LARND_CHAIN_FREEZE_DRIFT=1,LARND_FREEZE_STRAIGHT_CL=0.50"
acc=(neutrino:cider-nu neutrino:dune-ml); i=0
for T in 3200 4400; do for s in 1 3 7; do
  n=sci_full_STRFRZ50T${T}D$s; mkdir -p $ST/$n; [ -e fit_result/$n ] || ln -s $ST/$n fit_result/$n
  TGT="SCITGT=Ab:0.834881:eField:0.521519:tran_diff:1.00276e-05:long_diff:5.81418e-06:lifetime:$T"
  sbatch --array=0 --account=${acc[$((i % 2))]} --exclude=sdfampere038,sdfampere003 --export="$BASE,$TGT,SCIDATASEED=$s,SCITAG=_STRFRZ50T${T}D$s" optimize/scripts/start_sci_case.sh 2>&1 | grep -v OS_VER
  i=$((i+1))
done; done
squeue -u pgranger -h -o "%i %j %T %M %R"
