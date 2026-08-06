#!/bin/bash
# Study 3 — deliberately bias the dE/dx mean-constraint TARGET.
#
# The loss pins the length-weighted <dE/dx> to dedx_mean_constraint_target (1.887) with weight
# dedx_mean_constraint_weight. If A_b or lifetime are absorbing a mismatch there, mis-setting the
# target on purpose must move them in a measurable, monotone way, and the slope is the lever arm.
#
# NOT ALREADY DONE. sci_full_FITDM0 / FITDMP2 / FITDMM2 look like this study but are not: they
# varied the SIM INPUT FILE's dE/dx mean (quality_ladder/dedxmean+-0.02.h5) with the target left
# at 1.887 in all three. This varies the PENALTY TARGET, which is a different axis.
#
# CROSSED WITH THE WEIGHT, on purpose. The target only matters as much as the stiffness lets it:
# at 3000 the fit can ignore a wrong target, at 1e5 it cannot. The cross gives d(param)/d(target)
# AS A FUNCTION of stiffness, which is what says whether the target is a lever or a nuisance.
#
# Expected, from the 2-D scans: A_b moves ~linearly and lifetime barely moves, because A_b is the
# cheap absorber of charge normalisation. If LIFETIME moves instead, that is a new handle on our
# weakest parameter and a new systematic to control -- the more important outcome either way.
#
# Ceiling, because production cannot resolve effects this size at any affordable n.
set -euo pipefail
cd /sdf/group/neutrino/pgranger/larnd-sim-jax

SEEDS=${SEEDS:-0-5}
ACCOUNTS=(neutrino:cider-nu neutrino:dune-ml mli:cider-ml mli:nu-ml-dev)

# 1.887 x {0.98, 0.99, 1.00, 1.01, 1.02}
ARMS=(
  "S3T098|1.84926|100000.0"
  "S3T099|1.86813|100000.0"
  "S3T100|1.887|100000.0"
  "S3T101|1.90587|100000.0"
  "S3T102|1.92474|100000.0"
  "S3W098|1.84926|3000.0"
  "S3W100|1.887|3000.0"
  "S3W102|1.92474|3000.0"
)

k=0
for arm in "${ARMS[@]}"; do
  IFS='|' read -r NAME TGT W <<< "${arm}"
  ACC=${ACCOUNTS[$((k % ${#ACCOUNTS[@]}))]}
  EXP="ALL,SCIMODE=ceiling,SCITAG=_${NAME},SCINBATCH=100,SCILEN=400,SCIITER=10000"
  EXP="${EXP},SCIDEDXPRIOR=0.5,SCILRDECAY=0.91,SCIMEANW=${W},SCIDMEANTGT=${TGT}"
  echo "submitting ${NAME}: target=${TGT} weight=${W} acct=${ACC}"
  sbatch --account="${ACC}" --job-name="${NAME}" --array="${SEEDS}%3" --time=20:00:00 \
         --export="${EXP}" optimize/scripts/start_sci_case.sh
  k=$((k + 1))
done
echo "study 3 submitted: ${#ARMS[@]} arms x 6 seeds at ceiling"
