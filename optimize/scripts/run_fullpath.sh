#!/bin/bash
# STUDY 1: is there a BARRIER between the fit and truth, or is the path downhill?
#
# WHAT WE KNOW. The objective PREFERS truth -- with calibration, dEdx and geometry all set to truth
# the loss beats the fit on 7/7 arm-seeds by 5.6% (report fig 66) -- and yet the optimiser does not
# get there. We also know that moving any SINGLE block to truth alone can make the loss WORSE
# (calibration alone +0.18%, dEdx alone +2.79%), so the one-block axes cannot answer the question.
#
# WHAT THIS MEASURES. The loss along the straight line joining the fit to truth with all three
# blocks moving TOGETHER: X(alpha) = X_fit + alpha*(X_truth - X_fit), calibration in physical units,
# dEdx in log space, geometry in spline coefficients. alpha=0 reproduces P0_fit and alpha=1
# reproduces P1_truth exactly, which is a built-in consistency check on the interpolation.
#
#   monotone decreasing  ->  nothing structural blocks us. The remaining gap is conditioning or
#                            step size, and a better-preconditioned optimiser should close it.
#                            THEN it is worth building one.
#   barrier or shelf     ->  no local method reaches truth from here. Effort should go to the
#                            OBJECTIVE instead -- the dEdx prior is the one proven defect (setting
#                            dEdx to truth makes the loss worse by 2.79% on 7/7).
#
# Sampled slightly past alpha=1 on purpose: if the minimum along the line sits BEYOND truth, then
# the objective's optimum is not at truth even jointly, which is a much stronger statement than any
# conditional scan can support.
#
# THREE ARMS, deliberately at different distances from truth. BT2 is furthest (truth beats it by
# 9.4%), G20K closest (2.5%), TS1 in between. If the path shape changes systematically as the fit
# improves -- e.g. a shelf that shortens -- that is itself the answer to "why does it stall".
#
# Evaluation only: no fitting, no optimiser. ~20 points x 100 batches, roughly 30 min per job.
set -euo pipefail
cd /sdf/group/neutrino/pgranger/larnd-sim-jax

SEEDS=${SEEDS:-"0 1"}
ACCOUNTS=(neutrino:cider-nu neutrino:dune-ml mli:cider-ml mli:nu-ml-dev)
BASE="SCIMODE=full,SCINBATCH=100,SCILEN=400,SCIBASIS=spline,SCIDEDXPRIOR=1.0088,SCIMEANW=2500.0"
REFIT="SCIDXNUL=21.760,SCIDXNUR=4.720,SCIDXLOC=1.8786,SCIDXSL=0.0782,SCIDXSR=0.0801"
SCHED="SCICLR=1e-2,SCICHDECAY=0.9997,SCILRDECAY=0.91"
# finer near alpha=0, because a barrier shows up as the loss RISING as we first leave the fit
ALPHA="0:0.05:0.1:0.15:0.2:0.3:0.4:0.5:0.6:0.7:0.8:0.9:1:1.05:1.1:1.2"

k=0
for arm in BT2 TS1 G20K; do
  case ${arm} in
    BT2)  PAT="fit_result/sci_full_BT2/history_iter10000_*seed%s.pkl" ;;
    TS1)  PAT="fit_result/sci_full_TS1/history_iter10000_*seed%s.pkl" ;;
    G20K) PAT="fit_result/sci_full_G20K/history_iter20000_*seed%s.pkl" ;;
  esac
  for S in ${SEEDS}; do
    CK=$(ls $(printf "${PAT}" "${S}") 2>/dev/null | head -1)
    [ -z "${CK}" ] && { echo "SKIP ${arm} seed ${S}: no checkpoint"; continue; }
    ACC=${ACCOUNTS[$((k % ${#ACCOUNTS[@]}))]}; k=$((k + 1))
    echo "FP${arm} seed ${S} on ${ACC}"
    sbatch --account="${ACC}" --job-name="fp${arm}" --array="${S}-${S}" --time=4:00:00 \
      --exclude="${SCIEXCLUDE-sdfampere038}" \
      --export=ALL,${BASE},${REFIT},${SCHED},SCITAG=_FP${arm},SCITRUTHPOINT=${CK},SCITPFULL=${ALPHA} \
      optimize/scripts/start_sci_case.sh
  done
done
echo "submitted; results in fit_result/sci_full_FP{BT2,TS1,G20K}/"
