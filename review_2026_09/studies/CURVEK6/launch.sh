#!/bin/bash
# Launch the K6 drift-from-truth configuration at tau = 1800 and 3200 us on samples 1/3/7
# (paired with the production CURVE1800D*/CURVE3200D*). Outputs on scratch staging.
cd /sdf/group/neutrino/pgranger/larnd-sim-jax
ST=/sdf/scratch/neutrino/pgranger/fit_result_staging
BASE="ALL,SCIMODE=full,SCINBATCH=100,SCILEN=400,SCIITER=10000,SCIDATASZ=600,SCIBASIS=spline,SCICLR=1e-2,SCICHDECAY=0.9997,SCILRDECAY=0.91,SCIDEDXPRIOR=1.0088,SCIMEANW=2500.0,SCIDXNUL=21.760,SCIDXNUR=4.720,SCIDXSL=0.0782,SCIDXSR=0.0801,SCIDXLOC=1.8786,SCIGEOMTRUTHINIT=drift,LARND_CHAIN_FREEZE_DRIFT=1,SCIKMIN=6"
acc=(neutrino:cider-nu neutrino:dune-ml); i=0
for T in 1800 3200; do for s in 1 3 7; do
  n=sci_full_DRIFTK6T${T}D$s; mkdir -p $ST/$n; [ -e fit_result/$n ] || ln -s $ST/$n fit_result/$n
  TGT="SCITGT=Ab:0.834881:eField:0.521519:tran_diff:1.00276e-05:long_diff:5.81418e-06:lifetime:$T"
  sbatch --array=0 --account=${acc[$((i % 2))]} --exclude=sdfampere038,sdfampere003 --export="$BASE,SCIDATASEED=$s,$TGT,SCITAG=_DRIFTK6T${T}D$s" optimize/scripts/start_sci_case.sh 2>&1 | grep -v OS_VER
  i=$((i+1))
done; done
squeue -u pgranger -h -o "%i %T %M %N %R"
