#!/bin/bash
# Study 1 — learning-rate schedule variations across the three blocks.
#
# The three blocks anneal in three DIFFERENT ways, and one of them does not anneal at all:
#   calibration : warmup_exponential_decay, decay indexed in EPOCHS      (SCILRDECAY)
#   geometry    : chain_decay_rate ** iter, indexed in ITERATIONS        (SCICHDECAY)
#   dE/dx       : optax.adam(constant) -- NO schedule whatsoever
# LARND_DEDX_LR_FINAL_FRAC (SCIDEDXFF) gives the dE/dx block a schedule for the first time.
#
# 1A runs at CEILING (true geometry frozen). That is deliberate: ceiling shrinks the between-seed
# s.d. 11-70x, and three runs of an identical production config span 3.6 points on lifetime, so
# production simply cannot resolve single-knob schedule differences at any affordable n.
# Geometry decay therefore cannot be studied here -- it is 1B, in production, crossed only with
# the 1A winner (and annl_2xtravel already supplies one of its three points).
#
# Arms are reported by TOTAL OPTIMISER TRAVEL (sum of LR over the run), not by decay_rate:
# lifetime has tracked travel monotonically all campaign, so travel is the physical axis.
#
# The decisive metric is TWO numbers, not one: final error AND schedule-invariance. A parameter
# whose value moves when the schedule moves is not converged, however good it looks.
#
# nbatch is IDENTICAL across all arms on purpose: the calibration decay is indexed in epochs, so
# changing the batch count silently un-anneals it (this cost a whole 4x-data run).
set -euo pipefail
cd /sdf/group/neutrino/pgranger/larnd-sim-jax

SEEDS=${SEEDS:-0-5}
ACCOUNTS=(neutrino:cider-nu neutrino:dune-ml mli:cider-ml mli:nu-ml-dev)

# name        calib_decay  dedx_final_frac  dedx_freeze   what it tests
ARMS=(
  "S1A0|0.91||"            # baseline: production calib decay, dE/dx never cools
  "S1A1|0.80||"            # faster calibration anneal  (less travel)
  "S1A2|0.97||"            # slower calibration anneal  (more travel)
  "S1A3|0.91|0.1|"         # NEW: dE/dx annealed to 0.1x by the end
  "S1A4|0.91|0.01|"        # NEW: dE/dx annealed to 0.01x by the end
  "S1A5|0.91||5000"        # reference: hard dE/dx freeze at iters/2 (the known-good 5-D fix)
)

k=0
for arm in "${ARMS[@]}"; do
  IFS='|' read -r NAME LRD DFF DFZ <<< "${arm}"
  ACC=${ACCOUNTS[$((k % ${#ACCOUNTS[@]}))]}
  # --export values must be comma-free: sbatch splits its own list on commas.
  EXP="ALL,SCIMODE=ceiling,SCITAG=_${NAME},SCINBATCH=100,SCILEN=400,SCIITER=10000"
  EXP="${EXP},SCIDEDXPRIOR=0.5,SCIMEANW=100000.0,SCILRDECAY=${LRD}"
  [ -n "${DFF}" ] && EXP="${EXP},SCIDEDXFF=${DFF}"
  [ -n "${DFZ}" ] && EXP="${EXP},SCIDEDXFREEZE=${DFZ}"
  echo "submitting ${NAME}: calib_decay=${LRD} dedx_ff=${DFF:-none} dedx_freeze=${DFZ:-none} acct=${ACC}"
  sbatch --account="${ACC}" --job-name="${NAME}" --array="${SEEDS}%3" --time=20:00:00 \
         --export="${EXP}" optimize/scripts/start_sci_case.sh
  k=$((k + 1))
done
echo "study 1A submitted: ${#ARMS[@]} arms x 6 seeds at ceiling"
