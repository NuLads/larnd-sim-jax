#!/bin/bash
# Cheap diagnostic scans on the BT2 fit — evaluation, not fitting.
#
# The truth-point evaluator costs ~20 min for 100 batches x 5 points and roughly 30 s per extra
# point, so a single ~1 h job carries ~60 constructed points. Four questions that would otherwise
# need multi-day fits therefore collapse into ONE job per seed:
#
#  1. GEOMETRY alpha-scan (LARND_TP_GEOM_ALPHA). Walk the spline coefficients fit -> truth with
#     calibration and dEdx held at the fit. Study 2 says geometry is UNREACHED (-213 nat, almost
#     no prior involvement) while dEdx is DISPLACED. If that is right the likelihood's optimum
#     here sits at alpha ~ 1 and geometry is purely optimiser-limited -- which is what would
#     justify spending ~112 GPU-h on 20 000-iteration runs. If the optimum is well below 1,
#     geometry has its own shrinkage story and those runs would chase something the objective does
#     not want. ~1 GPU-h to test a 112 GPU-h assumption.
#
#  2. JOINT (geometry x dEdx) grid (LARND_TP_JOINT_ALPHA). Does the dEdx optimum MOVE with
#     geometry? That coupling is the standing hypothesis for the longitudinal-diffusion regression
#     seen in production (fig 54e) but NOT at ceiling, where geometry is frozen.
#
#  3. LONG. DIFFUSION profiled at geometry = fit vs geometry = truth (LARND_TP_CALIB_SCAN). If the
#     optimum moves between the two, that parameter is entangled with geometry -- the direct test
#     of the regression, at a GPU-minute per point instead of a 28 h refit.
#
#  4. dEdx alpha-scan WITH THE REFIT PRIOR (LARND_TP_DEDX_ALPHA). A different prior SHAPE cannot be
#     re-scored offline (the stored aux holds only the weighted current prior), so this is the only
#     way to confirm the refit pushes alpha* UP toward the data's 0.50 rather than down. It comes
#     free here because this job already runs the BT2 configuration.
#
# ALL VALUE LISTS USE ':' -- sbatch --export splits its own list on commas and would truncate them.
set -euo pipefail
cd /sdf/group/neutrino/pgranger/larnd-sim-jax

SEEDS=${SEEDS:-"0 1 2"}
ACCOUNTS=(neutrino:cider-nu neutrino:dune-ml mli:cider-ml)
CKDIR=fit_result/sci_full_BT2

# the BT2 configuration, so the scan is evaluated at the fit it is diagnosing
REFIT="SCIDXNUL=21.760,SCIDXNUR=4.720,SCIDXLOC=1.8786,SCIDXSL=0.0782,SCIDXSR=0.0801"
GEOMA="0:0.2:0.4:0.6:0.8:1:1.2"                 #  7 points
JOINTA="0:0.25:0.5:0.75:1"                      # 25 points (5x5)
CALIB="long_diff:0.8:0.9:0.95:1:1.05:1.1:1.2"   # 14 points (7 x 2 geometries)
DEDXA="0:0.2:0.4:0.6:0.8:1:1.2:1.5:2"           #  9 points
# 5 base + 7 + 25 + 14 + 9 = 60 points x 100 batches ~ 70 min

k=0
for S in ${SEEDS}; do
  CK=$(ls ${CKDIR}/history_iter10000_*seed${S}.pkl 2>/dev/null | head -1)
  if [ -z "${CK}" ]; then echo "SKIP seed ${S}: no completed BT2 checkpoint"; continue; fi
  ACC=${ACCOUNTS[$((k % ${#ACCOUNTS[@]}))]}
  echo "submitting seed ${S} against ${CK} on ${ACC}"
  sbatch --account="${ACC}" --job-name="diag_s${S}" --array="${S}-${S}" --time=6:00:00 \
    --export=ALL,SCIMODE=full,SCITAG=_DIAG,SCINBATCH=100,SCILEN=400,SCICLR=1e-2,SCICHDECAY=0.9997,\
SCILRDECAY=0.91,SCIBASIS=spline,SCIDEDXPRIOR=1.0088,SCIMEANW=2500.0,${REFIT},SCITRUTHPOINT=${CK},\
SCITPGEOMA=${GEOMA},SCITPJOINTA=${JOINTA},SCITPCALIB=${CALIB},SCITPDEDXA=${DEDXA} \
    optimize/scripts/start_sci_case.sh
  k=$((k + 1))
done
echo "submitted; results land in fit_result/sci_full_DIAG/"
