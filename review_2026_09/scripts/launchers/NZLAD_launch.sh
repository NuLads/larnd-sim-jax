#!/bin/bash
# Drift-shape noise tolerance ladder: K6 drift-from-truth + frozen, with s um of per-track noise on the
# truth drift shape (NZ50 passed: +0.35%; straight line ~250 um systematic failed: +22%). tau = 2406, samples 1/2/4.
cd /sdf/group/neutrino/pgranger/larnd-sim-jax
ST=/sdf/scratch/neutrino/pgranger/fit_result_staging
BASE="ALL,SCIMODE=full,SCINBATCH=100,SCILEN=400,SCIITER=10000,SCIDATASZ=600,SCIBASIS=spline,SCICLR=1e-2,SCICHDECAY=0.9997,SCILRDECAY=0.91,SCIDEDXPRIOR=1.0088,SCIMEANW=2500.0,SCIDXNUL=21.760,SCIDXNUR=4.720,SCIDXSL=0.0782,SCIDXSR=0.0801,SCIDXLOC=1.8786,SCIGEOMTRUTHINIT=drift,LARND_CHAIN_FREEZE_DRIFT=1,SCIKMIN=6"
acc=(neutrino:cider-nu neutrino:dune-ml); i=0
for NZ in 150; do for s in 1 2 4; do
  n=sci_full_DRIFTNZ${NZ}D$s; mkdir -p $ST/$n; [ -e fit_result/$n ] || ln -s $ST/$n fit_result/$n
  sbatch --array=0 --account=${acc[$((i % 2))]} --exclude=sdfampere038,sdfampere003 --export="$BASE,SCIDATASEED=$s,LARND_DRIFT_TRUTH_NOISE_UM=$NZ,LARND_DRIFT_TRUTH_NOISE_SEED=$s,SCITAG=_DRIFTNZ${NZ}D$s" optimize/scripts/start_sci_case.sh 2>&1 | grep -v OS_VER
  i=$((i+1))
done; done
# truth-free candidate: production config with 6 spline modes (no truth drift, no freeze)
BASEF="${BASE%%,SCIGEOMTRUTHINIT=drift*},SCIKMIN=6"
for s in 1 2 4; do
  n=sci_full_K6FREE$s; mkdir -p $ST/$n; [ -e fit_result/$n ] || ln -s $ST/$n fit_result/$n
  sbatch --array=0 --account=${acc[$((i % 2))]} --exclude=sdfampere038,sdfampere003 --export="$BASEF,SCIDATASEED=$s,SCITAG=_K6FREE$s" optimize/scripts/start_sci_case.sh 2>&1 | grep -v OS_VER
  i=$((i+1))
done
squeue -u pgranger -h -o "%i %T %M %N %R"
