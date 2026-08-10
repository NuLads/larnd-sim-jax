#!/bin/bash
# Geometry: the long run, and a cheap search for faster convergence.
#
# WHY GEOMETRY, NOW. Fig 56 measured it: along the geometry axis the objective's optimum is
# alpha* = 0.988 -- it wants geometry essentially exactly at truth, with no shrinkage -- and the
# likelihood range along that axis is 192 637 nats against 37 475 for the whole dEdx dispersion
# question and 845 for long. diffusion. Geometry is optimiser-limited AND carries ~5x the
# information of anything else this campaign has tuned. Position is also still descending at
# 10 000 iterations in every arm (drift -6 to -8 um over the last 20%, fig 55).
#
# ─────────────────────────────────────────────────────────────────────────────────────────────
# MODE=long   3 seeds x 20 000 iterations.  BOTH schedules are stretched, not just extended:
#
#   SCICHDECAY 0.9997 -> 0.99985   ( = 0.9997^0.5 )
#   SCILRDECAY 0.91   -> 0.9539    ( = 0.91^0.5   )
#
#   Without this the run is a near-null by construction. At 0.9997 the geometry LR reaches 0.050
#   of peak by iteration 10 000 but 0.0025 by 20 000 -- geometry would cool 20x FURTHER and gain
#   almost no extra movement. Stretching gives the same final LR at 20 000 that it had at 10 000,
#   so the extra iterations are spent moving rather than frozen.
#
#   Stretching the CALIBRATION schedule matters for a second reason, and fig 56 is what says so:
#   the calibration optima MOVE as geometry improves (the dEdx optimum slides 0.006 -> 0.334, and
#   long. diffusion's optimum drops to <=0.80x). At 0.91 the calibration LR is ~6e-9 of peak by
#   20 000 -- frozen through the entire second half, exactly while geometry is improving and the
#   optima it should be tracking are moving. Calibration has to stay live.
#
# MODE=quick  3 arms x 2 seeds x 3 000 iterations.  A cheap ranking of geometry schedules.
#   BT2 ITSELF IS THE CONTROL: at the same iteration count both schedules are indexed identically
#   (calibration per epoch, geometry per iteration, 100 batches in every arm), so BT2's own
#   position trace at iteration 3 000 is the matched baseline and needs no job of its own.
#
#   CAVEAT worth stating: a higher LR can win early and lose late. A 3 000-iteration ranking is a
#   screen for which variants deserve a full run, not a verdict.
#
# geom_optimizer is left at adam throughout: lbfgs/gn/ggn are all measured non-viable for this
# block (per-batch recompile ~170 s, OOM, ~400x Adam).
set -euo pipefail
cd /sdf/group/neutrino/pgranger/larnd-sim-jax

MODE=${MODE:-quick}
ACCOUNTS=(neutrino:cider-nu neutrino:dune-ml mli:cider-ml mli:nu-ml-dev)
# the BT2 configuration, so geometry is the only thing that changes
BASE="SCIMODE=full,SCINBATCH=100,SCILEN=400,SCIBASIS=spline,SCIDEDXPRIOR=1.0088,SCIMEANW=2500.0"
REFIT="SCIDXNUL=21.760,SCIDXNUR=4.720,SCIDXLOC=1.8786,SCIDXSL=0.0782,SCIDXSR=0.0801"

if [ "${MODE}" = "long" ]; then
  SEEDS=${SEEDS:-"0 1 2"}
  k=0
  for S in ${SEEDS}; do
    ACC=${ACCOUNTS[$((k % ${#ACCOUNTS[@]}))]}
    echo "submitting G20K seed ${S} on ${ACC}"
    sbatch --account="${ACC}" --job-name="G20K" --array="${S}-${S}" --time=40:00:00 \
      --export=ALL,${BASE},${REFIT},SCITAG=_G20K,SCIITER=20000,SCICLR=1e-2,\
SCICHDECAY=0.99985,SCILRDECAY=0.9539 \
      optimize/scripts/start_sci_case.sh
    k=$((k + 1))
  done
  echo "submitted; results in fit_result/sci_full_G20K/"
else
  SEEDS=${SEEDS:-"0 1"}
  # name | chain_lr | chain_decay   (BT2 baseline is 1e-2 / 0.9997, supplied by BT2's own trace)
  ARMS=("GQ1|3e-2|0.9997" "GQ2|1e-1|0.9997" "GQ3|1e-2|0.99993")
  k=0
  for arm in "${ARMS[@]}"; do
    IFS='|' read -r NAME CLR CDEC <<< "${arm}"
    for S in ${SEEDS}; do
      ACC=${ACCOUNTS[$((k % ${#ACCOUNTS[@]}))]}
      echo "submitting ${NAME} seed ${S}: chain_lr=${CLR} chain_decay=${CDEC} on ${ACC}"
      sbatch --account="${ACC}" --job-name="${NAME}" --array="${S}-${S}" --time=10:00:00 \
        --export=ALL,${BASE},${REFIT},SCITAG=_${NAME},SCIITER=3000,SCICLR=${CLR},\
SCICHDECAY=${CDEC},SCILRDECAY=0.91 \
        optimize/scripts/start_sci_case.sh
      k=$((k + 1))
    done
  done
  echo "submitted; results in fit_result/sci_full_GQ{1,2,3}/"
fi
