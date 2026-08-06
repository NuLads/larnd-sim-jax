#!/bin/bash
# Study 2 — is the loss minimum misplaced, or merely unreachable?
#
# For each seed, re-runs the PRODUCTION configuration in truth-point EVALUATION mode against
# that seed's own completed checkpoint (see _eval_truth_point in fit_params.py). No fitting
# happens: each batch is visited once and the objective is evaluated at five constructed points.
#
# The configuration below MUST match the checkpoint's run exactly -- same seed, same data_seed,
# same batching -- or the batches, targets and track ordering differ and the comparison is
# meaningless. It is therefore expressed by calling start_sci_case.sh with the production
# environment rather than by re-listing flags: hand-rolled flag lists have silently dropped
# production settings in this campaign before.
#
# Usage:  bash optimize/scripts/run_study2_truthpoint.sh          # 6 seeds, 100 batches
#         SMOKE=1 bash optimize/scripts/run_study2_truthpoint.sh  # 1 seed, 2 batches
set -euo pipefail
cd /sdf/group/neutrino/pgranger/larnd-sim-jax

CKPT_DIR=fit_result/sci_full_ANNEALLONG2
SEEDS=${SEEDS:-"0 1 2 4 5 6"}
NBATCH=100
TAG=_TP
ACCOUNTS=(neutrino:cider-nu neutrino:dune-ml mli:cider-ml mli:nu-ml-dev)

if [ "${SMOKE:-0}" = "1" ]; then
  SEEDS=0; NBATCH=2; TAG=_TPSMOKE
  echo "SMOKE MODE: 1 seed, ${NBATCH} batches"
fi

k=0
for S in ${SEEDS}; do
  CKPT="${CKPT_DIR}/history_iter10000_sci_full_b100_len400_seed${S}.pkl"
  if [ ! -f "${CKPT}" ]; then
    echo "SKIP seed ${S}: no completed checkpoint at ${CKPT}"; continue
  fi
  ACC=${ACCOUNTS[$((k % ${#ACCOUNTS[@]}))]}
  # NOTE --export values must not contain commas: sbatch's own list separator is a comma and a
  # value containing one is silently split in half. All values below are comma-free.
  sbatch --account="${ACC}" \
         --job-name="tp_s${S}" \
         --array="${S}-${S}" \
         --time=8:00:00 \
         --export=ALL,SCIMODE=full,SCITAG=${TAG},SCINBATCH=${NBATCH},SCILEN=400,\
SCICLR=1e-2,SCICHDECAY=0.9997,SCILRDECAY=0.91,SCIBASIS=spline,SCIDEDXPRIOR=5.0,\
SCIMEANW=100000.0,SCITRUTHPOINT=${CKPT} \
         optimize/scripts/start_sci_case.sh
  k=$((k + 1))
done
echo "submitted; results land in fit_result/sci_full${TAG}/"
