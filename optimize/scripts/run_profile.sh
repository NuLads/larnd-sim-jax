#!/bin/bash
# STUDY 2: a PROFILE likelihood over the calibration block.
#
# THE GAP THIS FILLS. Every scan in this campaign fixes the other four calibration parameters while
# one moves -- a CONDITIONAL slice. Fig 66 was originally published with the opposite conclusion
# because a conditional minimum was read as a joint one, and even after the correction the residual
# question is still open: with geometry and dEdx at truth the lifetime slice minimum is 2.26% from
# truth, but that slice still holds the other four calibration parameters fixed.
#
# A profile answers it properly. At each point the profiled parameter is PINNED and everything else
# -- the other four calibration parameters AND both nuisance blocks -- is re-optimised. The curve
# min_{rest} L(theta_pinned) is what "how much does the data constrain this parameter" actually
# means.
#
# AND IT IS THE ONLY ROUTE TO AN ERROR BAR. Every number this campaign has quoted is a point
# estimate with seed scatter as a stand-in for uncertainty. Delta(loss) = 0.5 from the profile
# minimum gives a genuine 1-sigma interval, so we can finally say whether the ~1.6% lifetime error
# is or is not consistent with zero bias. Without this there is no calibration result to publish.
#
# SETUP. Every point starts from TS1 -- the best fit in the campaign (loss 8072 at 13.5 h) -- with
# all three blocks warm-started, so each point is a short RE-optimisation rather than a fresh fit.
# Schedules are compressed so 3 000 iterations land on BT2's terminal LR, the trap-1b discipline:
# every point is read at the same schedule phase, otherwise the profile curve is contaminated by
# how converged each point happens to be.
#
# READING THE RESULT
#   minimum at multiplier 1.0, narrow   -> the fit is the profiled ML estimate and well constrained
#   minimum away from 1.0               -> the joint optimum is NOT where the fit sits, i.e. the
#                                          other blocks were absorbing the difference
#   minimum away from TRUTH             -> a genuine residual bias, the strongest such statement
#                                          this campaign could make
#   flat                                -> the parameter is unconstrained once the others adapt,
#                                          and its apparent accuracy was an artefact of holding
#                                          them fixed
#
# CAVEAT ON THE 1-SIGMA: Delta = 0.5 assumes a proper log-likelihood. Ours sums an independent
# marked-Poisson term over 100 batches, which is legitimate, but it also carries the dEdx and
# mean-dEdx PRIORS. Quote the interval from the LIKELIHOOD part of the aux decomposition, which is
# stored at every iteration, not from the total.
set -euo pipefail
cd /sdf/group/neutrino/pgranger/larnd-sim-jax

PARAM=${PARAM:-lifetime}
SEEDS=${SEEDS:-"0 1"}
MULTS=${MULTS:-"0.90 0.95 0.975 1.00 1.025 1.05 1.10"}
ITERS=${ITERS:-3000}
ACCOUNTS=(neutrino:cider-nu neutrino:dune-ml mli:cider-ml mli:nu-ml-dev)
CKDIR=fit_result/sci_full_TS1
BASE="SCIMODE=full,SCINBATCH=100,SCILEN=400,SCIBASIS=spline,SCIDEDXPRIOR=1.0088,SCIMEANW=2500.0"
REFIT="SCIDXNUL=21.760,SCIDXNUR=4.720,SCIDXLOC=1.8786,SCIDXSL=0.0782,SCIDXSR=0.0801"
# compressed so 3 000 iterations end where BT2's 10 000 did -- see trap 1b
SCHED="SCICLR=1e-2,SCICHDECAY=0.999,SCILRDECAY=0.73"

k=0
for S in ${SEEDS}; do
  CK=$(ls ${CKDIR}/history_iter10000_*seed${S}.pkl 2>/dev/null | head -1)
  [ -z "${CK}" ] && { echo "SKIP seed ${S}: no TS1 checkpoint"; continue; }
  for M in ${MULTS}; do
    # all five calibration parameters at TS1's values, with the profiled one scaled by M
    INIT=$(PARAM="${PARAM}" M="${M}" CK="${CK}" /sdf/group/neutrino/pgranger/conda/bin/python3.12 -c "
import pickle, numpy as np, os
h = pickle.load(open(os.environ['CK'], 'rb')); out = []
for p in ['Ab','eField','tran_diff','long_diff','lifetime']:
    v = np.ravel(h[p+'_iter']); m = np.median(v[int(len(v)*.8):])
    if p == os.environ['PARAM']: m *= float(os.environ['M'])
    out += [p, '%.8g' % m]
print(':'.join(out))")
    TAG=$(echo "_PROF${PARAM}${M}" | tr -d '.')
    ACC=${ACCOUNTS[$((k % ${#ACCOUNTS[@]}))]}; k=$((k + 1))
    echo "profile ${PARAM} x${M} seed ${S} on ${ACC}   tag ${TAG}"
    sbatch --account="${ACC}" --job-name="pf${M}" --array="${S}-${S}" --time=12:00:00 \
      --exclude="${SCIEXCLUDE-sdfampere038}" \
      --export=ALL,${BASE},${REFIT},${SCHED},SCITAG=${TAG},SCIITER=${ITERS},\
SCIINIT=${INIT},SCICHAININIT=${CK},SCIDEDXINIT=${CK},SCICALIBPIN=${PARAM} \
      optimize/scripts/start_sci_case.sh
  done
done
echo "submitted; results in fit_result/sci_full_PROF${PARAM}*/"
