#!/bin/bash
# HITSEED: production fit (K=3, free geometry) started from the gated hit-derived geometry (truth-free).
# GTRU:    production fit started from the TRUE geometry projection, then free (ceiling for any init fix).
# Samples 1/2/4, paired with DSREP1/2/4. Targets kept for offline analysis.
cd /sdf/group/neutrino/pgranger/larnd-sim-jax
ST=/sdf/scratch/neutrino/pgranger/fit_result_staging
BASE="ALL,SCIMODE=full,SCINBATCH=100,SCILEN=400,SCIITER=10000,SCIDATASZ=600,SCIBASIS=spline,SCICLR=1e-2,SCICHDECAY=0.9997,SCILRDECAY=0.91,SCIDEDXPRIOR=1.0088,SCIMEANW=2500.0,SCIDXNUL=21.760,SCIDXNUR=4.720,SCIDXSL=0.0782,SCIDXSR=0.0801,SCIDXLOC=1.8786,SCIKMIN=3,LARND_KEEP_TARGETS=1"
acc=(neutrino:cider-nu neutrino:dune-ml); i=0
for s in 1 2 4; do
  for arm in HITSEED GTRU; do
    n=sci_full_${arm}$s; mkdir -p $ST/$n; [ -e fit_result/$n ] || ln -s $ST/$n fit_result/$n
    if [ $arm = HITSEED ]; then X="LARND_DRIFT_FROM_HITS=1,LARND_HITS_SEED_E1=1"; else X="SCIGEOMTRUTHINIT=1"; fi
    sbatch --array=0 --account=${acc[$((i % 2))]} --exclude=sdfampere038,sdfampere003 --export="$BASE,$X,SCIDATASEED=$s,SCITAG=_${arm}$s" optimize/scripts/start_sci_case.sh 2>&1 | grep -v OS_VER
    i=$((i+1))
  done
done
squeue -u pgranger -h -o "%i %j %T %M %R"
