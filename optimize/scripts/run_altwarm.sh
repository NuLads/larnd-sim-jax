#!/bin/bash
# ALT-WARM: the frozen-geometry second pass, but with calibration WARM-STARTED from BT2.
#
# ALT (fig 59) froze BT2's geometry and re-fitted calibration and dEdx COLD -- from the same
# nominal start BT2 itself used (A_b 0.800, lifetime 2200, ...). It reached A_b 0.147 against
# BT2's 0.340 in 3 000 iterations rather than 10 000.
#
# THE QUESTION THIS ANSWERS. Is 0.147 the optimum of the frozen-geometry objective, or is it an
# artefact of where the cold start happened to land? Approach the same objective from the OTHER
# direction -- initialise calibration at BT2's own fitted values -- and see where it goes:
#
#   converges back to ~0.15  ->  0.15 is the real optimum, reached from two directions, and BT2's
#                               0.34 is the co-adaptation penalty. Adopt the second pass.
#   stays near BT2's 0.34    ->  the frozen objective has more than one basin and the cold start
#                               found a better one by luck. A much weaker, and more worrying, story.
#
# LIMITATION, stated because it is not fixable without new code: only CALIBRATION is warm-started.
# There is no init-from for the dEdx block, so it restarts at the prior centre in both arms -- a
# controlled constant rather than a confound, but it means this is a partial warm start.
set -euo pipefail
cd /sdf/group/neutrino/pgranger/larnd-sim-jax

SEEDS=${SEEDS:-"0 1"}
ACCOUNTS=(neutrino:cider-nu neutrino:dune-ml mli:cider-ml)
CKDIR=fit_result/sci_full_BT2
BASE="SCIMODE=full,SCINBATCH=100,SCILEN=400,SCIBASIS=spline,SCIDEDXPRIOR=1.0088,SCIMEANW=2500.0"
REFIT="SCIDXNUL=21.760,SCIDXNUR=4.720,SCIDXLOC=1.8786,SCIDXSL=0.0782,SCIDXSR=0.0801"
SCHED="SCICLR=1e-2,SCICHDECAY=0.9997,SCILRDECAY=0.91"

k=0
for S in ${SEEDS}; do
  CK=$(ls ${CKDIR}/history_iter10000_*seed${S}.pkl 2>/dev/null | head -1)
  [ -z "${CK}" ] && { echo "SKIP seed ${S}"; continue; }
  # BT2's converged calibration for this seed, colon-separated (sbatch --export forbids spaces)
  INIT=$(/sdf/group/neutrino/pgranger/conda/bin/python3.12 -c "
import pickle,numpy as np,sys
h=pickle.load(open('${CK}','rb')); out=[]
for p in ['Ab','eField','tran_diff','long_diff','lifetime']:
    v=np.ravel(h[p+'_iter']); out += [p, '%.8g' % np.median(v[int(len(v)*.8):])]
print(':'.join(out))")
  ACC=${ACCOUNTS[$((k % ${#ACCOUNTS[@]}))]}; k=$((k + 1))
  echo "seed ${S} on ${ACC}: init ${INIT}"
  sbatch --account="${ACC}" --job-name="altwarm" --array="${S}-${S}" --time=10:00:00 \
    --export=ALL,${BASE},${REFIT},${SCHED},SCITAG=_ALTW,SCIITER=3000,\
SCICHAININIT=${CK},SCICHFREEZE=1,SCIINIT=${INIT} \
    optimize/scripts/start_sci_case.sh
done
echo "submitted; results in fit_result/sci_full_ALTW/"
