#!/bin/bash
# Three cheap overnight tests, all against BT2.
#
# A  DEGENERACY MATRIX (evaluation, ~3 h)
#    Every calibration parameter scanned at all four combinations of (geometry, dEdx) in
#    {fit, truth}. The SHIFT of each optimum when a nuisance block moves to truth IS its
#    degeneracy with that block; the curvature is how well the data determines it. We have
#    measured only dEdx<->geometry and long_diff<->geometry so far (fig 56) -- this is the rest
#    of the matrix, for ~3 GPU-h.
#
# B  DOES ALTERNATING HELP? (fitting, 3 000 iters)
#    Re-fit calibration and dEdx against BT2's geometry held FROZEN (LARND_CHAIN_INIT_FROM +
#    LARND_CHAIN_FREEZE_ITER=1). Fig 56 showed the calibration optima MOVE as geometry improves,
#    so the blocks are currently chasing each other. If calibration improves against a static
#    geometry, block-alternation is worth building properly; if not, simultaneous updates are
#    fine and the question is closed.
#
# C  IS THE THIRD SPLINE MODE WORTH FITTING? (fitting, 3 000 iters)
#    Measured against the truth projection, mode 3 recovers -61.7% -- its error EXCEEDS its
#    signal, so fitting it injects noise. K=2 drops it: fewer parameters, no noise injection, at
#    the cost of a higher representation floor. Note the floor in _spline_K is what sets the basis
#    resolution here, NOT chain_spline_knot_cm: with a 40 cm knot and a 44.4 cm median track,
#    round(L/knot)=1 for almost every track and everything is clipped up to the floor.
#
# B and C are 3 000-iteration screens and use BT2's OWN trace at iteration 3 000 as the matched
# control, which costs no job: at equal iteration count every arm indexes its schedules
# identically, on the same seeds and targets.
set -euo pipefail
cd /sdf/group/neutrino/pgranger/larnd-sim-jax

SEEDS=${SEEDS:-"0 1"}
ACCOUNTS=(neutrino:cider-nu neutrino:dune-ml mli:cider-ml mli:nu-ml-dev)
CKDIR=fit_result/sci_full_BT2
BASE="SCIMODE=full,SCINBATCH=100,SCILEN=400,SCIBASIS=spline,SCIDEDXPRIOR=1.0088,SCIMEANW=2500.0"
REFIT="SCIDXNUL=21.760,SCIDXNUR=4.720,SCIDXLOC=1.8786,SCIDXSL=0.0782,SCIDXSR=0.0801"
SCHED="SCICLR=1e-2,SCICHDECAY=0.9997,SCILRDECAY=0.91"
# 5 params x 7 multipliers x 4 (geometry, dEdx) combinations = 140 points + 5 base
CALIB="Ab+eField+tran_diff+long_diff+lifetime:0.8:0.9:0.95:1:1.05:1.1:1.2"

k=0
for S in ${SEEDS}; do
  CK=$(ls ${CKDIR}/history_iter10000_*seed${S}.pkl 2>/dev/null | head -1)
  if [ -z "${CK}" ]; then echo "SKIP seed ${S}: no BT2 checkpoint"; continue; fi

  ACC=${ACCOUNTS[$((k % ${#ACCOUNTS[@]}))]}; k=$((k + 1))
  echo "A: degeneracy matrix, seed ${S} on ${ACC}"
  sbatch --account="${ACC}" --job-name="degen" --array="${S}-${S}" --time=8:00:00 \
    --export=ALL,${BASE},${REFIT},${SCHED},SCITAG=_DEGEN,SCITRUTHPOINT=${CK},SCITPCALIB=${CALIB} \
    optimize/scripts/start_sci_case.sh

  ACC=${ACCOUNTS[$((k % ${#ACCOUNTS[@]}))]}; k=$((k + 1))
  echo "B: frozen-geometry re-fit, seed ${S} on ${ACC}"
  sbatch --account="${ACC}" --job-name="altern" --array="${S}-${S}" --time=10:00:00 \
    --export=ALL,${BASE},${REFIT},${SCHED},SCITAG=_ALT,SCIITER=3000,\
SCICHAININIT=${CK},SCICHFREEZE=1 \
    optimize/scripts/start_sci_case.sh

  ACC=${ACCOUNTS[$((k % ${#ACCOUNTS[@]}))]}; k=$((k + 1))
  echo "C: K=2 spline basis, seed ${S} on ${ACC}"
  sbatch --account="${ACC}" --job-name="k2" --array="${S}-${S}" --time=10:00:00 \
    --export=ALL,${BASE},${REFIT},${SCHED},SCITAG=_K2,SCIITER=3000,SCIKMIN=2 \
    optimize/scripts/start_sci_case.sh
done
echo "submitted; results in fit_result/sci_full_{DEGEN,ALT,K2}/"
