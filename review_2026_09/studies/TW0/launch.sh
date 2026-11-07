#!/bin/bash
# Exact point-wise likelihood (--loss_time_window 0, no Gaussian time smearing) with free geometry from truth.
# PSTGTRUTW0 = self-consistent targets; GTRUTW0 = stochastic (production-type) targets. Samples 1/2/4.
# Rule (stated 5 Oct 20:20 before results): smearing is the main cause if PSTGTRUTW0 coherent drift offset < 10 um
# on 3/3 (TW=2: +24..+31); production-relevant fix if GTRUTW0 offset < 15 um on 3/3 (TW=2: +40..+42) AND
# lifetime within +-5% of truth on 3/3.
cd /sdf/group/neutrino/pgranger/larnd-sim-jax
ST=/sdf/scratch/neutrino/pgranger/fit_result_staging
BASE="ALL,SCIMODE=full,SCINBATCH=100,SCILEN=400,SCIITER=10000,SCIDATASZ=600,SCIBASIS=spline,SCICLR=1e-2,SCICHDECAY=0.9997,SCILRDECAY=0.91,SCIDEDXPRIOR=1.0088,SCIMEANW=2500.0,SCIDXNUL=21.760,SCIDXNUR=4.720,SCIDXSL=0.0782,SCIDXSR=0.0801,SCIDXLOC=1.8786,SCIKMIN=3,LARND_KEEP_TARGETS=1,SCIGEOMTRUTHINIT=1,NOISEFLAG=--loss_time_window 0"
acc=(neutrino:cider-nu neutrino:dune-ml); i=0
for s in 1 2 4; do
  for arm in PSTGTRUTW0 GTRUTW0; do
    n=sci_full_${arm}$s; mkdir -p $ST/$n; [ -e fit_result/$n ] || ln -s $ST/$n fit_result/$n
    if [ $arm = PSTGTRUTW0 ]; then X=",SCIPROBTGT=1"; else X=""; fi
    sbatch --array=0 --account=${acc[$((i % 2))]} --exclude=sdfampere038,sdfampere003 --export="$BASE$X,SCIDATASEED=$s,SCITAG=_${arm}$s" optimize/scripts/start_sci_case.sh 2>&1 | grep -v OS_VER
    i=$((i+1))
  done
done
squeue -u pgranger -h -o "%i %j %T %M %R"
