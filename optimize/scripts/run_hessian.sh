#!/bin/bash
# The CALIBRATION HESSIAN at a fitted point -- the deterministic route to an uncertainty.
#
# WHY NOT PROFILE BY RE-FITTING. That was tried and it fails by a factor 2 105 (report fig 70): each
# profile point is an independent stochastic re-optimisation of ~9 000 nuisance parameters, and
# where that lands varies more in loss (41 nats/batch) than the lifetime dependence being measured
# (7.5 nats/batch across the whole grid). A Hessian has no landing-point noise -- it is evaluated at
# one point, not searched for.
#
# WHAT THIS PRODUCES. The 5x5 calibration Hessian at the fit, per batch. Summed over batches it is
# the Hessian of the summed loss -- the correct object, since the batches are independent data --
# and its inverse is a covariance:
#   * the DIAGONAL gives a 1-sigma per parameter. This is the CONDITIONAL uncertainty, geometry and
#     dEdx held fixed, so it is a LOWER BOUND on the profile uncertainty. Stated as such.
#   * the OFF-DIAGONAL gives the calibration CORRELATIONS, which no measurement in this campaign has
#     ever produced. If two parameters are strongly correlated, their individual errors are not
#     independently meaningful and the whole error budget has to be read differently.
#
# THE VALIDATION IS THE POINT OF INCLUDING BT2. The degeneracy-matrix scan (fig 58) measured the
# conditional curvature at BT2's point by finite differences of the loss along each parameter --
# lifetime 1-sigma = 0.258% of the fitted value, A_b 0.017%, and so on. The Hessian computes the
# same quantity by an entirely different route (forward-over-reverse AD). If the two agree, both are
# trustworthy; if they do not, one of them is wrong and we need to know that before quoting either.
#
# Evaluation only: 5 base points plus 5 Hessian-vector products per batch, ~25 min per job.
set -euo pipefail
cd /sdf/group/neutrino/pgranger/larnd-sim-jax

ACCOUNTS=(neutrino:cider-nu neutrino:dune-ml mli:cider-ml mli:nu-ml-dev)
BASE="SCIMODE=full,SCINBATCH=100,SCILEN=400,SCIBASIS=spline,SCIDEDXPRIOR=1.0088,SCIMEANW=2500.0"
REFIT="SCIDXNUL=21.760,SCIDXNUR=4.720,SCIDXLOC=1.8786,SCIDXSL=0.0782,SCIDXSR=0.0801"
SCHED="SCICLR=1e-2,SCICHDECAY=0.9997,SCILRDECAY=0.91"

k=0
submit () {   # $1 arm, $2 seed, $3 checkpoint
  ACC=${ACCOUNTS[$((k % ${#ACCOUNTS[@]}))]}; k=$((k + 1))
  echo "hessian at $1 seed $2 on ${ACC}"
  sbatch --account="${ACC}" --job-name="hs$1" --array="$2-$2" --time=6:00:00 \
    --exclude="${SCIEXCLUDE-sdfampere038}" \
    --export=ALL,${BASE},${REFIT},${SCHED},SCITAG=_HESS$1,SCITRUTHPOINT=$3,SCITPHESS=P0_fit \
    optimize/scripts/start_sci_case.sh
}

# BT2 seeds 0-1: the arm the fig-58 scan was run on, so the Hessian can be checked against it
for S in ${BT2SEEDS:-"0 1"}; do
  CK=$(ls fit_result/sci_full_BT2/history_iter10000_*seed${S}.pkl 2>/dev/null | head -1)
  [ -n "${CK}" ] && submit BT2 "${S}" "${CK}" || echo "SKIP BT2 seed ${S}"
done
# TS1: the best fit in the campaign, i.e. the point we would actually quote an error bar at
for S in ${TS1SEEDS:-"0 1 2"}; do
  CK=$(ls fit_result/sci_full_TS1/history_iter10000_*seed${S}.pkl 2>/dev/null | head -1)
  [ -n "${CK}" ] && submit TS1 "${S}" "${CK}" || echo "SKIP TS1 seed ${S}"
done
echo "submitted; results in fit_result/sci_full_HESS{BT2,TS1}/"
