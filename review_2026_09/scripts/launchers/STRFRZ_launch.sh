#!/bin/bash
# Truth-free valley pinning: HITSEED + freeze e2 (drift) at the straight line ONLY for tracks whose hit-derived e2
# modes are not significant at CL (LARND_FREEZE_STRAIGHT_CL). STRFRZ90: CL 0.9 (~35% frozen); STRFRZ50: CL 0.5 (~15%).
# Paired with HITSEED1/2/4 (identical config without freezing). Rule (stated 5 Oct ~21:00, before results):
# valley pinned if STRFRZ90 lifetime SD over 3 samples <= 2 pts (HITSEED 5.8) AND coherent offset < 15 um on 3/3
# AND |lifetime| <= 8% on 3/3.
cd /sdf/group/neutrino/pgranger/larnd-sim-jax
ST=/sdf/scratch/neutrino/pgranger/fit_result_staging
BASE="ALL,SCIMODE=full,SCINBATCH=100,SCILEN=400,SCIITER=10000,SCIDATASZ=600,SCIBASIS=spline,SCICLR=1e-2,SCICHDECAY=0.9997,SCILRDECAY=0.91,SCIDEDXPRIOR=1.0088,SCIMEANW=2500.0,SCIDXNUL=21.760,SCIDXNUR=4.720,SCIDXSL=0.0782,SCIDXSR=0.0801,SCIDXLOC=1.8786,SCIKMIN=3,LARND_KEEP_TARGETS=1,LARND_DRIFT_FROM_HITS=1,LARND_HITS_SEED_E1=1,LARND_CHAIN_FREEZE_DRIFT=1"
acc=(neutrino:cider-nu neutrino:dune-ml); i=0
for s in 1 2 4; do
  for cl in 90 50; do
    arm=STRFRZ$cl; n=sci_full_${arm}$s; mkdir -p $ST/$n; [ -e fit_result/$n ] || ln -s $ST/$n fit_result/$n
    sbatch --array=0 --account=${acc[$((i % 2))]} --exclude=sdfampere038,sdfampere003 --export="$BASE,LARND_FREEZE_STRAIGHT_CL=0.$cl,SCIDATASEED=$s,SCITAG=_${arm}$s" optimize/scripts/start_sci_case.sh 2>&1 | grep -v OS_VER
    i=$((i+1))
  done
done
squeue -u pgranger -h -o "%i %j %T %M %R"
