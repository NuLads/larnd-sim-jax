#!/bin/bash
# PRIORITY 1: why does a better fit mean a WORSE lifetime?
#
# THE OBSERVATION (report fig 63f / 65f). Every arm of the block matrix that improves the loss over
# BT2 has a worse lifetime -- 7 of 8 arm x seed points -- and the three largest loss gains (-149,
# -137, -72 NLL) are the three worst lifetime degradations. G20K shows the same sign. Better fit,
# worse physics.
#
# THE HYPOTHESIS THIS TESTS. Fig 58 scanned the calibration parameters at BT2's geometry and found
# BT2 sitting AT the objective's optimum for every parameter (<=1.3%). But BM0 and G20K have BETTER
# geometry, and there the fit lands ~3.6 points away on lifetime with a LOWER loss. So the natural
# reading is that the objective's lifetime optimum MOVES AWAY FROM TRUTH as geometry improves.
#
# This re-runs EXACTLY the fig 58 scan at the improved checkpoints instead of BT2's. Two outcomes,
# both decisive, and they point to completely different work:
#
#   optimum has MOVED to ~+5%   ->  the objective genuinely prefers it. A FORWARD-MODEL bias, not an
#                                   optimiser failure. No schedule, freeze or iteration count fixes
#                                   it; the next question becomes what in the simulation couples
#                                   lifetime to geometry.
#   optimum still near truth,   ->  a coupling / staging problem, and fixable by optimisation.
#   but the fit sits at +5%
#
# WHY IT IS CHEAP: evaluation only. No fitting, no optimiser, ~3 h per seed for 145 points.
#
# The 4 (geometry, dEdx) in {fit, truth} combinations are kept from fig 58 rather than just the
# (fit, fit) one we strictly need, because they answer a second question for free: with geometry now
# much closer to truth, the geometry->truth SHIFT of each calibration optimum should SHRINK. If it
# does not, geometry accuracy is not what drives the coupling.
set -euo pipefail
cd /sdf/group/neutrino/pgranger/larnd-sim-jax

ACCOUNTS=(neutrino:cider-nu neutrino:dune-ml mli:cider-ml mli:nu-ml-dev)
BASE="SCIMODE=full,SCINBATCH=100,SCILEN=400,SCIBASIS=spline,SCIDEDXPRIOR=1.0088,SCIMEANW=2500.0"
REFIT="SCIDXNUL=21.760,SCIDXNUR=4.720,SCIDXLOC=1.8786,SCIDXSL=0.0782,SCIDXSR=0.0801"
SCHED="SCICLR=1e-2,SCICHDECAY=0.9997,SCILRDECAY=0.91"
# identical grid to fig 58, so the two scans are directly comparable
CALIB="Ab+eField+tran_diff+long_diff+lifetime:0.8:0.9:0.95:1:1.05:1.1:1.2"

k=0
submit () {   # $1 = source arm, $2 = seed, $3 = checkpoint
  local ARM=$1 S=$2 CK=$3
  ACC=${ACCOUNTS[$((k % ${#ACCOUNTS[@]}))]}; k=$((k + 1))
  echo "scan at ${ARM} seed ${S} on ${ACC}"
  echo "   ${CK}"
  sbatch --account="${ACC}" --job-name="lscan${ARM}" --array="${S}-${S}" --time=8:00:00 \
    --export=ALL,${BASE},${REFIT},${SCHED},SCITAG=_LSCAN${ARM},SCITRUTHPOINT=${CK},SCITPCALIB=${CALIB} \
    optimize/scripts/start_sci_case.sh
}

for S in 0 1 2; do
  CK=$(ls fit_result/sci_full_G20K/history_iter20000_*seed${S}.pkl 2>/dev/null | head -1)
  [ -z "${CK}" ] && { echo "SKIP G20K seed ${S}"; continue; }
  submit G20K "${S}" "${CK}"
done
for S in 0 1; do
  CK=$(ls fit_result/sci_full_BM0/history_iter3000_*seed${S}.pkl 2>/dev/null | head -1)
  [ -z "${CK}" ] && { echo "SKIP BM0 seed ${S}"; continue; }
  submit BM0 "${S}" "${CK}"
done
echo "submitted; results in fit_result/sci_full_LSCAN{G20K,BM0}/"
