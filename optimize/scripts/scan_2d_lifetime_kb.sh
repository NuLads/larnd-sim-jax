#!/bin/bash
#
# scan_2d_lifetime_kb.sh — 2-D loss scan of (lifetime, kb) at fixed other
# params, for A2-B5-C2-D10.
#
# Purpose: the (Ab, τ) 2-D scan established a −0.95 valley correlation
# between Ab and lifetime along the ΣQ-preserving direction. kb is the other
# Birks parameter and also controls total collected charge (dE/dx-dependent),
# so it likely has a similar valley with lifetime. Confirm/refute here and
# check if the slope-penalty shape term breaks it.
#
# Structure mirrors scan_2d_lifetime_Ab.sh — each SLURM array task fixes kb
# to one KB_GRID value via `--set_init_params kb X` and runs the ordinary
# lifetime scan. The report generator picks up the shared `_2Dkb_` marker.
#

#SBATCH --partition=ampere
##SBATCH --account=mli:nu-ml-dev
#SBATCH --account=mli:cider-ml
##SBATCH --account=neutrino:dune-ml
#SBATCH --job-name=diffsim_2d
#SBATCH --output=logs/scan/2d_lifetime_kb-%A_%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=32g
#SBATCH --gpus-per-node=a100:1
#SBATCH --exclude=sdfampere024,sdfampere038,sdfampere002
#SBATCH --requeue
#SBATCH --time=4:00:00
#SBATCH --array=0-10

# ---------- 2-D grid definition ----------

# kb outer axis — 11 values around the nominal kb = 0.0486 (see
# src/larndsim/detector_properties/module0.yaml). Range 0.030–0.070 spans
# roughly ±40 % of nominal; matches the range used in the 1-D kb scan yaml
# so the valley (if any) is well-resolved without truncation.
KB_GRID=(0.030 0.034 0.038 0.042 0.046 0.050 0.054 0.058 0.062 0.066 0.070)

TID=${SLURM_ARRAY_TASK_ID:-4}   # default = 0.046 (index 4) — closest to nominal
if [ "$TID" -ge "${#KB_GRID[@]}" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$TID exceeds grid size ${#KB_GRID[@]}"
    exit 1
fi
KB_VAL="${KB_GRID[$TID]}"

# ---------- CONFIG (locked to A2-B5-C2-D10, sweep-best D10 hyperparams) ----------

CONFIG="A2-B5-C2-D10"
TARGET_SEED=$TID              # keep seed distinct per kb point

BATCH_SIZE=200
MAX_NBATCH=50
ITERATIONS=50
MAX_CLIP_NORM_VAL=1
DATA_SEED=1
SEED_STRATEGY=same
SAMPLING_STEP=0.01
N_NEIGH=4
MODE="lut"
LR_SCHEDULER=warmup_exponential_decay_schedule
SIGNAL_LENGTH=200
DEDX_DENSITY_MODE=flow
FLOW_EXPECTATION_MODE=sample

# D10 hyperparameters — sweep-best corner (see 6-par fit + hp sweep results).
D10_INCLUDE_Q="true"
D10_SIGMA_Q="5.0"
D10_ALPHA_MODE="per_track"
D10_ALPHA_PRIOR="0.2"
D10_METRIC="marginal_llhd"    # or "marginal_llhd_prod"

# --- Shape penalty (per-track log(Q)-vs-drift-tick slope term) ---
# Calibrated from the diagnostic dry-run (aux['std_slope_resid'] ≈ 0.0012 across
# all six params on B5). Weight = 1.0 gives ~10 % of mllhd_only at truth.
SLOPE_PENALTY_KIND="between_track"   # within_track | between_track | per_hit
# Per-kind σ calibrated from the diag dry-run (see scan_ABCD.sh header).
SLOPE_PENALTY_SIGMA_BT="3e-5"
SLOPE_PENALTY_SIGMA_PH="4e-5"
SLOPE_PENALTY_SIGMA_WT="1e-3"
case "$SLOPE_PENALTY_KIND" in
    within_track)  SLOPE_PENALTY_SIGMA="$SLOPE_PENALTY_SIGMA_WT";;
    between_track) SLOPE_PENALTY_SIGMA="$SLOPE_PENALTY_SIGMA_BT";;
    per_hit)       SLOPE_PENALTY_SIGMA="$SLOPE_PENALTY_SIGMA_PH";;
esac
SLOPE_PENALTY_WEIGHT="1.0"
SLOPE_PENALTY_DIAG="false"
SLOPE_PENALTY_TICK_MIN="200"    # drop pixel-electronics ramp-up region

# ---------- A/B/C plumbing (only A2-B5-C2 supported) ----------

IFS='-' read -r CONF_A CONF_B CONF_C CONF_D <<< "$CONFIG"

INPUT_FILE_TGT=/sdf/data/neutrino/cyifan/dunend_train_prod/prod_mod0_mpvmpr/production_884072/job_23771825_0000/output_23771825_0000-edepsim_lbl_trklen2cm_containment2cm_costheta0.966_range_0.05cm.h5
CHOP_FLAG="--no_chop"
DX_LABEL="stopp_dx0.01"

INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/dunend_train_prod/prod_mod0_mpvmpr/production_884072/job_23771825_0000/output_23771825_0000-edepsim_lbl_trklen2cm_containment2cm_costheta0.966_max_evt_37815_reco_pos_dE_seg_step_0.01cm_range_0.05cm.h5
USE_DENSITY_FLAG="--use_dedx_density"
B_LABEL="reco_posdEdensity"

NORM=sigmoid
PARAMS=optimize/scripts/param_list_A2-B5-C2-D10.yaml   # scans lifetime

# ---------- D10 label + LOSS_KW ----------

if [ "$D10_INCLUDE_Q" = "true" ]; then _Q_TAG="tQ"; else _Q_TAG="t"; fi
_SQ_TAG="${D10_SIGMA_Q/./p}"
_PRIOR_TAG="${D10_ALPHA_PRIOR/./p}"
_AT_TAG="at_${D10_ALPHA_MODE}_ls${_PRIOR_TAG}"
_PRIOR_JSON="${D10_ALPHA_PRIOR}"

if [ "$D10_METRIC" = "marginal_llhd_prod" ]; then
    _METRIC_TAG="mllhdP"
else
    _METRIC_TAG="mllhd"
fi

# kb tag: replace "." with "p" (kb is always positive so no sign prefix).
_KB_TAG="${KB_VAL/./p}"

# See scan_ABCD.sh for the tag encoding.
case "$SLOPE_PENALTY_KIND" in
    within_track)  _SP_K_TAG="wt";;
    between_track) _SP_K_TAG="bt";;
    per_hit)       _SP_K_TAG="ph";;
    *) echo "ERROR: unknown SLOPE_PENALTY_KIND=$SLOPE_PENALTY_KIND"; exit 3;;
esac
_SP_TAG=""
if awk -v w="$SLOPE_PENALTY_WEIGHT" 'BEGIN{exit !(w+0>0)}'; then
    _SP_W_TAG="${SLOPE_PENALTY_WEIGHT/./p}"
    _SP_S_TAG="${SLOPE_PENALTY_SIGMA/./p}"
    _SP_TAG="_sp${_SP_K_TAG}${_SP_W_TAG}sg${_SP_S_TAG}"
elif [ "$SLOPE_PENALTY_DIAG" = "true" ]; then
    _SP_TAG="_spD${_SP_K_TAG}"
elif [ "$SLOPE_PENALTY_KIND" != "within_track" ]; then
    _SP_TAG="_sp0${_SP_K_TAG}"
fi

PROB_FLAG="--probabilistic_sim"
D_LABEL="prob_${_METRIC_TAG}_${_Q_TAG}_sQ${_SQ_TAG}_${_AT_TAG}${_SP_TAG}_kb${_KB_TAG}"
LOSS=dqdt_radial
LOSS_KW="{\"distance_metric\": \"${D10_METRIC}\", \"marginal_llhd_include_Q\": ${D10_INCLUDE_Q}, \"marginal_llhd_sigma_Q_ke\": ${D10_SIGMA_Q}, \"marginal_llhd_eps\": 1e-10, \"per_track_scale_mode\": \"${D10_ALPHA_MODE}\", \"per_track_scale_prior_log_sigma\": ${_PRIOR_JSON}, \"slope_penalty_weight\": ${SLOPE_PENALTY_WEIGHT}, \"slope_penalty_sigma\": ${SLOPE_PENALTY_SIGMA}, \"slope_penalty_diag\": ${SLOPE_PENALTY_DIAG}, \"slope_penalty_kind\": \"${SLOPE_PENALTY_KIND}\", \"slope_penalty_tick_min\": ${SLOPE_PENALTY_TICK_MIN}}"

# Marker `_2Dkb_` lets the report generator distinguish this from the other
# 2-D scans and route through the generic outer-axis renderer.
LABEL="${CONFIG}_2Dkb_${B_LABEL}_${D_LABEL}_bt${BATCH_SIZE}nbt${MAX_NBATCH}"

SIF_FILE=/sdf/group/neutrino/pgranger/larnd-sim-jax.sif

echo "==========================================================="
echo " 2-D scan (lifetime × kb)"
echo "   hostname          = $(hostname)"
echo "   nvidia-smi first line = $(nvidia-smi -L 2>&1 | head -1)"
echo "   CONFIG            = $CONFIG"
echo "   ARRAY_TASK_ID     = $TID  (of  ${#KB_GRID[@]})"
echo "   kb value          = $KB_VAL  (tag: $_KB_TAG)"
echo "   lifetime axis     = 50 points along the range in $PARAMS"
echo "   D10 σ_Q, α_prior  = $D10_SIGMA_Q ke, $D10_ALPHA_PRIOR"
echo "   slope penalty     = w=$SLOPE_PENALTY_WEIGHT, sg=$SLOPE_PENALTY_SIGMA, diag=$SLOPE_PENALTY_DIAG"
echo "==========================================================="
echo "Label: $LABEL"

apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${SIF_FILE} /bin/bash -c "
# Preflight: verify JAX can actually initialize a CUDA device on this host.
if ! python3 -c 'import jax; assert any(d.platform==\"gpu\" for d in jax.devices()), \"no GPU\"' 2>/dev/null; then
    echo \"CUDA preflight FAILED on \$(hostname). Requesting requeue.\"
    exit 75   # EX_TEMPFAIL — SLURM requeues on non-zero + --requeue
fi
export PYTHONPATH=$PWD:$PWD/src:${PYTHONPATH}; \
if ! python3 -c 'import larndsim' >/dev/null 2>&1; then \
    (flock -x 200; python3 -c 'import larndsim' >/dev/null 2>&1 || pip3 install .) 200> /tmp/pip_larndsim.lock; \
fi; \
python3 -m optimize.example_run \
    --data_sz -1 \
    --max_nbatch ${MAX_NBATCH} \
    --params lifetime \
    --set_init_params kb ${KB_VAL} \
    --input_file_sim ${INPUT_FILE_SIM} \
    --input_file_tgt ${INPUT_FILE_TGT} \
    --non_deterministic \
    --fit_type scan \
    --track_len_sel 2 \
    --max_abs_costheta_sel 0.966 \
    --min_abs_segz_sel 15. \
    --data_seed ${DATA_SEED} \
    --out_label ${LABEL} \
    --test_name scan \
    --scan_tgt_nom \
    --seed ${TARGET_SEED} \
    --optimizer_fn Adam \
    --iterations ${ITERATIONS} \
    --max_batch_len ${BATCH_SIZE} \
    --track_z_bound 28 \
    --max_clip_norm_val ${MAX_CLIP_NORM_VAL} \
    --electron_sampling_resolution ${SAMPLING_STEP} \
    --number_pix_neighbors ${N_NEIGH} \
    --signal_length ${SIGNAL_LENGTH} \
    --mode ${MODE} \
    --lut_file src/larndsim/detector_properties/response_44_v2a_full_tick.npz \
    --loss_fn ${LOSS} \
    --loss_fn_kw '${LOSS_KW}' \
    --sim_seed_strategy ${SEED_STRATEGY} \
    --clip_from_range \
    --lr_scheduler ${LR_SCHEDULER} \
    --lr_kw '{\"decay_rate\" : 0.99, \"init_value\" : 0, \"warmup_steps\": 1000}' \
    --shuffle_bt \
    --normalization_scheme ${NORM} \
    --dedx_density_mode ${DEDX_DENSITY_MODE} \
    ${PROB_FLAG} \
    ${CHOP_FLAG} \
    ${USE_DENSITY_FLAG} \
    --print_input \
    --flow_expectation_mode ${FLOW_EXPECTATION_MODE} \
"
