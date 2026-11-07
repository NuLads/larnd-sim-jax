#!/bin/bash
# Self-consistent targets (SCIPROBTGT=1: targets drawn from the probabilistic fit model itself -> no FEE timing
# mismatch by construction). PSTGTRU = truth start, free geometry; PSTDTRU = drift frozen at truth (K3).
# Rule (stated 5 Oct before results): mismatch drives the valley if PSTGTRU-PSTDTRU lifetime gap is within +-3
# on 3/3 (vs ~+15 with stochastic targets) AND PSTGTRU coherent drift offset < 10 um.
cd /sdf/group/neutrino/pgranger/larnd-sim-jax
ST=/sdf/scratch/neutrino/pgranger/fit_result_staging
BASE="ALL,SCIMODE=full,SCINBATCH=100,SCILEN=400,SCIITER=10000,SCIDATASZ=600,SCIBASIS=spline,SCICLR=1e-2,SCICHDECAY=0.9997,SCILRDECAY=0.91,SCIDEDXPRIOR=1.0088,SCIMEANW=2500.0,SCIDXNUL=21.760,SCIDXNUR=4.720,SCIDXSL=0.0782,SCIDXSR=0.0801,SCIDXLOC=1.8786,SCIKMIN=3,LARND_KEEP_TARGETS=1,SCIPROBTGT=1"
acc=(neutrino:cider-nu neutrino:dune-ml); i=0
for s in 1 2 4; do
  for arm in PSTGTRU PSTDTRU; do
    n=sci_full_${arm}$s; mkdir -p $ST/$n; [ -e fit_result/$n ] || ln -s $ST/$n fit_result/$n
    if [ $arm = PSTGTRU ]; then X="SCIGEOMTRUTHINIT=1"; else X="SCIGEOMTRUTHINIT=drift,LARND_CHAIN_FREEZE_DRIFT=1"; fi
    sbatch --array=0 --account=${acc[$((i % 2))]} --exclude=sdfampere038,sdfampere003 --export="$BASE,$X,SCIDATASEED=$s,SCITAG=_${arm}$s" optimize/scripts/start_sci_case.sh 2>&1 | grep -v OS_VER
    i=$((i+1))
  done
done
squeue -u pgranger -h -o "%i %j %T %M %R"
