#!/bin/bash
# PRIORITY 2: the two-stage production recipe, which no arm has tested.
#
# The block matrix found that freezing CALIBRATION lets both nuisance blocks descend further
# (BMC: loss -3.25% vs BT2, position 115.7 um against 165.3, best dEdx of any arm). But BMC STOPS
# there, with calibration still at BT2's values -- so it is a better FIT, not a better calibration.
# G20K separately found that geometry keeps descending well past 10 000 iterations.
#
# THE RECIPE combines them, and it is genuinely untested: no arm in the matrix does it.
#
#   STAGE 1 (TS1, this script)   calibration FROZEN at BT2's values; geometry and dEdx warm-started
#                                from BT2 and re-annealed for 10 000 iterations. Let the nuisances
#                                reach the floor BMC was still descending toward at 3 000.
#   STAGE 2 (TS2, submitted once TS1 lands)  everything warm-started from TS1, calibration RELEASED,
#                                a fresh 10 000-iteration anneal. Calibration now solves against a
#                                nuisance background that is already good, instead of co-adapting
#                                with one that is still wrong.
#
# WHY TWO JOBS RATHER THAN ONE. LARND_CALIB_START_ITER would give a single-job version -- freeze
# calibration until iteration N, then release -- but the calibration LR schedule is indexed from
# iteration 0 regardless, so by the time calibration woke up it would be at ~9e-3 of peak and could
# barely move. Splitting the run gives stage 2 a FULL fresh anneal, which is the point.
#
# COST: 2 x 13.4 h = 26.8 h, deliberately matched to G20K's 26.4 h so the comparison is at equal
# compute rather than equal iteration count.
#
# EXPECTED, AND STATED IN ADVANCE so it cannot be rationalised afterwards: this should beat BT2 on
# position, dEdx, A_b, E field and transverse diffusion. It may well land WORSE on lifetime -- every
# arm that improved the loss so far has (fig 65f). That is what Priority 1 is running to explain,
# and a worse lifetime here would be confirmation, not a failure of the recipe.
set -euo pipefail
cd /sdf/group/neutrino/pgranger/larnd-sim-jax

SEEDS=${SEEDS:-"0 1 2"}
STAGE=${STAGE:-1}
ACCOUNTS=(neutrino:cider-nu neutrino:dune-ml mli:cider-ml mli:nu-ml-dev)
BASE="SCIMODE=full,SCINBATCH=100,SCILEN=400,SCIBASIS=spline,SCIDEDXPRIOR=1.0088,SCIMEANW=2500.0"
REFIT="SCIDXNUL=21.760,SCIDXNUR=4.720,SCIDXLOC=1.8786,SCIDXSL=0.0782,SCIDXSR=0.0801"
# 10 000 iterations, so the BASELINE schedules apply unchanged (they were tuned for this length)
SCHED="SCICLR=1e-2,SCICHDECAY=0.9997,SCILRDECAY=0.91"

k=0
for S in ${SEEDS}; do
  if [ "${STAGE}" = "1" ]; then
    SRC=fit_result/sci_full_BT2; PAT="history_iter10000_*seed${S}.pkl"; TAG=_TS1; CFZ=1
  else
    SRC=fit_result/sci_full_TS1; PAT="history_iter10000_*seed${S}.pkl"; TAG=_TS2; CFZ=0
  fi
  CK=$(ls ${SRC}/${PAT} 2>/dev/null | head -1)
  [ -z "${CK}" ] && { echo "SKIP stage ${STAGE} seed ${S}: no checkpoint in ${SRC}"; continue; }

  # calibration init: BT2's values in stage 1 (frozen there anyway, but it keeps the two stages
  # reading from one place), TS1's in stage 2 -- which are BT2's, since stage 1 froze them.
  INIT=$(/sdf/group/neutrino/pgranger/conda/bin/python3.12 -c "
import pickle,numpy as np
h=pickle.load(open('${CK}','rb')); out=[]
for p in ['Ab','eField','tran_diff','long_diff','lifetime']:
    v=np.ravel(h[p+'_iter']); out += [p, '%.8g' % np.median(v[int(len(v)*.8):])]
print(':'.join(out))")

  ACC=${ACCOUNTS[$((k % ${#ACCOUNTS[@]}))]}; k=$((k + 1))
  echo "TS${STAGE} seed ${S} on ${ACC}  (calib frozen=${CFZ})"
  echo "   from ${CK}"
  # sdfampere038 has twice reported a healthy GPU to the host while JAX saw none inside the
  # container; the task died in seconds. There is now a preflight AND an exit-status fix in
  # start_sci_case.sh, so such a task would at least be recorded FAILED -- but avoiding the node
  # is free. Override with SCIEXCLUDE= to allow it.
  sbatch --account="${ACC}" --job-name="TS${STAGE}" --array="${S}-${S}" --time=30:00:00 \
    --exclude="${SCIEXCLUDE-sdfampere038}" \
    --export=ALL,${BASE},${REFIT},${SCHED},SCITAG=${TAG},SCIITER=10000,\
SCIINIT=${INIT},SCICHAININIT=${CK},SCIDEDXINIT=${CK},SCICALIBFREEZE=${CFZ} \
    optimize/scripts/start_sci_case.sh
done
echo "submitted stage ${STAGE}; results in fit_result/sci_full_TS${STAGE}/"
