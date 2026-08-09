#!/bin/bash
# "Best theory" arms — everything study 2 established, applied together.
#
# WHAT CHANGED AND WHY (fig 46/47/48):
#
#  1. The split student-t prior is CORRECT FOR THE WRONG SEGMENTATION. Its 16-84% half-width
#     (0.1359) matches the true per-segment spread on the TARGET file's grid (0.1354, dx 2.32 cm)
#     to 0.4%, and refitting there gains +0.0002 nat -- it is already the MLE optimum. It is
#     APPLIED on the guess file's grid (dx 0.99 cm) where the true spread is 0.0843, 1.6x narrower.
#     Refitting there gains +0.1532 nat/segment, and split-half cross-validation says that gain is
#     out-of-sample (+0.9744 held-out vs +0.9751 in-sample).
#
#  2. THE WEIGHT IS A TEMPERING EXPONENT, NOT A NUISANCE KNOB. The prior enters as
#     -w * sum_i dx_i * log p, so the objective sees p(x)^(w*dx): beta = w*dx is a
#     pseudo-observation count. beta = 1 is the only value that uses the prior once, and with
#     dx = 0.9913 cm that means w = 1/dx = 1.0088 (NOT 5, and not 0.5 either).
#
#  3. THE MEAN CONSTRAINT AT 1e5 IS 6.3x TOO STIFF. The true length-weighted mean genuinely varies
#     0.747% batch to batch (sd 0.01409 over 900 batches) -- 3.3x more than independent sampling
#     would give, so it is real track-to-track physics. w = 1/(2 sigma^2) = 2520 is the weight that
#     permits exactly that much variation; 1e5 permits only 0.118%.
#
# ARMS. One knob moves at a time so the attribution is clean.
#
#   BT0  current prior, w=5,      mean 1e5    matched control ON THIS TREE (do not compare against
#                                             ANNEALLONG2 across a dirty tree -- that mistake has
#                                             already produced one false finding this campaign)
#   BT1  current prior, w=1/dx,   mean 2500   correct WEIGHT, wrong SHAPE
#   BT2  REFIT prior,   w=1/dx,   mean 2500   <-- the headline
#
# ORACLE CAVEAT, stated plainly: the refit parameters are an MLE of the TRUE per-segment dEdx
# distribution of this MC sample. That is a legitimate workflow -- the prior describes detector
# physics and is calibrated on simulation, not on the quantity being measured -- but the numbers
# below came from the same sample these fits use. Split-half CV says the shape is not overfitted;
# it does not prove independence from the sample. BT1 exists partly to bound this: it uses the
# ORIGINAL shape with only the weight corrected, so the gap BT1 -> BT2 is what the refit is worth.
set -euo pipefail
cd /sdf/group/neutrino/pgranger/larnd-sim-jax

SEEDS=${SEEDS:-0-5}
THROTTLE=${THROTTLE:-2}          # keep studies 1 and 3 draining alongside
ACCOUNTS=(neutrino:cider-nu neutrino:dune-ml mli:cider-ml mli:nu-ml-dev)

# refit on the PRODUCTION grid (guess file, median dx 0.9913 cm)
P_NUL=21.760; P_NUR=4.720; P_LOC=1.8786; P_SL=0.0782; P_SR=0.0801; P_W=1.0088
# refit on the CEILING grid (target file, median dx 2.3157 cm) -- reassuringly close to the
# current defaults, which is the whole point of fig 48(j)
C_NUL=4.220;  C_NUR=2.054;  C_LOC=1.8742; C_SL=0.1173; C_SR=0.1054; C_W=0.4318
MEANW=2500.0

# name | mode | prior weight | mean weight | nu_L | nu_R | loc | s_L | s_R
ARMS=(
  "BT0|full|5.0|100000.0|||||"
  "BT1|full|${P_W}|${MEANW}|||||"
  "BT2|full|${P_W}|${MEANW}|${P_NUL}|${P_NUR}|${P_LOC}|${P_SL}|${P_SR}"
)
# CUT, and why -- three arms is the minimum that answers anything, so these two went first:
#
#   BTC (ceiling, refit-for-ceiling, w=1/dx=0.4318, mean 2500)
#       Would mostly reproduce a result we already have. Fig 48(j) shows the ceiling refit
#       (nu_L 4.220, nu_R 2.054, loc 1.8742, s_L 0.1173, s_R 0.1054) is within noise of the
#       CURRENT defaults, and w = 0.5 on that grid is beta = 1.16, i.e. already ~beta = 1. So the
#       completed sci_ceiling_CEILP05 arm IS approximately "everything correct at ceiling".
#
#   BT3 (refit, w=1/dx, mean weight 0)
#       The question already has a quantitative answer: at beta = 1 the refit prior alone pins the
#       batch mean to 0.224% against a genuine 0.747% batch-to-batch spread, so it over-constrains
#       the mean 3.3x WITHOUT any help. The explicit term's remaining value is degeneracy-breaking
#       robustness (its target is externally known physics; the refit's mean is bundled with a
#       shape fitted to simulation), which a single ablation would not settle. Study 3's weight
#       arms measure the stiffness axis independently. Revisit once BT2 has landed.
#
# The retained three still bracket everything that matters:
#   BT0 -> BT1  what correcting the WEIGHT alone buys  (deployable today, no oracle)
#   BT1 -> BT2  what the REFIT buys on top             (bounds the oracle concern)

# ONLY=BT1 restricts submission to one arm; combined with SEEDS this reruns individual tasks
# (e.g. after a node failure took out specific seeds).
ONLY=${ONLY:-}
EXCLUDE=${EXCLUDE:-}

k=0
for arm in "${ARMS[@]}"; do
  IFS='|' read -r NAME MODE PW MW NUL NUR LOC SL SR <<< "${arm}"
  if [ -n "${ONLY}" ] && [ "${NAME}" != "${ONLY}" ]; then k=$((k + 1)); continue; fi
  ACC=${ACCOUNTS[$((k % ${#ACCOUNTS[@]}))]}
  # --export values must be comma-free: sbatch splits its own list on commas.
  EXP="ALL,SCIMODE=${MODE},SCITAG=_${NAME},SCINBATCH=100,SCILEN=400,SCIITER=10000"
  EXP="${EXP},SCILRDECAY=0.91,SCIDEDXPRIOR=${PW},SCIMEANW=${MW}"
  [ "${MODE}" = "full" ] && EXP="${EXP},SCICLR=1e-2,SCICHDECAY=0.9997,SCIBASIS=spline"
  if [ -n "${NUL}" ]; then
    EXP="${EXP},SCIDXNUL=${NUL},SCIDXNUR=${NUR},SCIDXLOC=${LOC},SCIDXSL=${SL},SCIDXSR=${SR}"
  fi
  echo "submitting ${NAME}: mode=${MODE} prior_w=${PW} mean_w=${MW} shape=${NUL:-current} acct=${ACC}"
  sbatch --account="${ACC}" --job-name="${NAME}" --array="${SEEDS}%${THROTTLE}" \
         ${EXCLUDE:+--exclude=${EXCLUDE}} \
         --time=26:00:00 --export="${EXP}" optimize/scripts/start_sci_case.sh
  k=$((k + 1))
done
echo "submitted ${#ARMS[@]} arms x 6 seeds (throttle ${THROTTLE}/arm)"
echo "results: fit_result/sci_full_BT0 BT1 BT2"
