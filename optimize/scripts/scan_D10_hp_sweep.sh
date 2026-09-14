#!/bin/bash
#
# scan_D10_hp_sweep.sh — hyperparameter sweep for A2-B{2,5}-C2-D10.
# Sweeps a grid of (α prior log-σ) × (σ_Q) × (physics param) via SLURM array.
# Each array task fits ONE physics-param scan at ONE (α_prior, σ_Q) point.
#
# Grid = ALPHA_PRIORS × SIGMA_QS × PARAMS_LIST, decoded from
# SLURM_ARRAY_TASK_ID = param_idx + n_param*(alpha_idx + n_alpha*sigmaq_idx).
#
# Set the array range to  0-(N_ALPHA*N_SIGMAQ*N_PARAMS - 1)  in #SBATCH below
# after editing the three arrays. With the defaults (3 * 3 * 5 = 45), use
#   #SBATCH --array=0-44
# Toggle B2/B5 via CONFIG. Keep everything else in sync with scan_ABCD.sh's D10
# behavior so pickles land alongside the existing D10 runs.
#

#SBATCH --partition=ampere
##SBATCH --account=mli:nu-ml-dev
#SBATCH --account=mli:cider-ml
##SBATCH --account=neutrino:dune-ml
#SBATCH --job-name=diffsim_D10hp
#SBATCH --output=logs/scan/D10hp-%A_%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=32g
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=4:00:00
#SBATCH --array=0-17          # = N_ALPHA*N_SIGMAQ*N_PARAMS - 1 (with defaults 3*3*5=45)

# ============================================================================
#                       USER-EDITABLE HYPERPARAMETER GRID
# ============================================================================

# Which config to sweep. Toggle B2 vs B5 here.
CONFIG="A2-B5-C2-D10"          # or "A2-B2-C2-D10"

# α prior log-σ values. Use "null" (bash string) to disable prior for that run.
# Ordered by increasing looseness. Recommended starter grid:
ALPHA_PRIORS=("0.2" "0.5" "1.0")

# σ_Q (Gaussian-kernel width in ke) values.
# 0.5 = the sharp default that OOM'd on B5; 1.0 = intermediate; 3.0 = current
# recommended; 5.0 = very wide (physics blurred).
SIGMA_QS=("1.0" "3.0" "5.0")

# Physics params to scan. Comment out lines to reduce total job count.
PARAMS_LIST=("tran_diff" "lifetime")
# ("Ab" "kb" "eField" "long_diff" "tran_diff" "lifetime" "shift_x" "shift_y" "shift_z")

# Keep α on (per_track) throughout the sweep; this script varies its *prior*,
# not the on/off state. To sweep α off vs per_track vs per_batch, run this
# script multiple times with different D10_ALPHA_MODE.
D10_ALPHA_MODE="per_track"
D10_INCLUDE_Q="true"           # tick+Q; set to "false" for tick-only

# Which marginal-LLHD variant to use. The two share the same kwargs but
# differ in how they combine tick and Q into a density:
#   "marginal_llhd"      — joint density p_sim(t, Q) with per-cell tick·Q
#                          correlation inside the pixel/slot sum (L_joint).
#   "marginal_llhd_prod" — product of tick and Q marginals, p_tick(t)·p_Q(Q)
#                          with separate marginalizations (L_prod). Tolerant
#                          of per-cell correlation mismatches under B5.
# Label prefix auto-switches to "mllhd" or "mllhdP" — no filename collisions.
D10_METRIC="marginal_llhd_prod"

# Other run knobs — mirror scan_ABCD.sh values.
BATCH_SIZE=200                 # A100-safe for D10 at Npix~5000 (see OOM logs)
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

# ============================================================================
#                           GRID INDEX DECODE
# ============================================================================

N_ALPHA=${#ALPHA_PRIORS[@]}
N_SIGMAQ=${#SIGMA_QS[@]}
N_PARAMS=${#PARAMS_LIST[@]}
N_TOTAL=$((N_ALPHA * N_SIGMAQ * N_PARAMS))

TID=${SLURM_ARRAY_TASK_ID:-0}

if [ "$TID" -ge "$N_TOTAL" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$TID exceeds grid size $N_TOTAL"
    echo "  set #SBATCH --array=0-$((N_TOTAL - 1))"
    exit 1
fi

param_idx=$(( TID % N_PARAMS ))
alpha_idx=$(( (TID / N_PARAMS) % N_ALPHA ))
sigmaq_idx=$(( TID / (N_PARAMS * N_ALPHA) ))

PARAM=${PARAMS_LIST[$param_idx]}
D10_ALPHA_PRIOR=${ALPHA_PRIORS[$alpha_idx]}
D10_SIGMA_Q=${SIGMA_QS[$sigmaq_idx]}
TARGET_SEED=$param_idx   # keep same seed per param across (α, σ_Q) points

echo "==========================================================="
echo " D10 hyperparameter sweep"
echo "   CONFIG            = $CONFIG"
echo "   ARRAY_TASK_ID     = $TID  of  $N_TOTAL"
echo "   PARAM             = $PARAM   (idx $param_idx)"
echo "   α prior log-σ     = $D10_ALPHA_PRIOR   (idx $alpha_idx)"
echo "   σ_Q               = $D10_SIGMA_Q ke    (idx $sigmaq_idx)"
echo "   include_Q         = $D10_INCLUDE_Q"
echo "   α mode            = $D10_ALPHA_MODE"
echo "   metric            = $D10_METRIC"
echo "==========================================================="

# ============================================================================
#                        A/B/C RESOLUTION (D10 only)
# ============================================================================

IFS='-' read -r CONF_A CONF_B CONF_C CONF_D <<< "$CONFIG"

# A: only A2 supported here (extend if needed).
if [ "$CONF_A" != "A2" ]; then
    echo "ERROR: this sweep script only supports A2. Extend if needed."; exit 2
fi
INPUT_FILE_TGT=/sdf/data/neutrino/cyifan/dunend_train_prod/prod_mod0_mpvmpr/production_884072/job_23771825_0000/output_23771825_0000-edepsim_lbl_trklen2cm_containment2cm_costheta0.966_range_0.05cm.h5
CHOP_FLAG="--no_chop"
DX_LABEL="stopp_dx0.01"

# B: only B2 (dE_density) and B5 (reco_posdEdensity) supported.
if [ "$CONF_B" == "B2" ]; then
    INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/dunend_train_prod/prod_mod0_mpvmpr/production_884072/job_23771825_0000/output_23771825_0000-edepsim_lbl_trklen2cm_containment2cm_costheta0.966_reco_dE_range_0.05cm.h5
    USE_DENSITY_FLAG="--use_dedx_density"
    B_LABEL="dE_density"
elif [ "$CONF_B" == "B5" ]; then
    INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/dunend_train_prod/prod_mod0_mpvmpr/production_884072/job_23771825_0000/output_23771825_0000-edepsim_lbl_trklen2cm_containment2cm_costheta0.966_max_evt_37815_reco_pos_dE_seg_step_0.01cm_range_0.05cm.h5
    USE_DENSITY_FLAG="--use_dedx_density"
    B_LABEL="reco_posdEdensity"
else
    echo "ERROR: this sweep script only supports B2 or B5, got $CONF_B"; exit 2
fi

# C: C2 (sigmoid) only.
if [ "$CONF_C" != "C2" ]; then
    echo "ERROR: this sweep script only supports C2."; exit 2
fi
NORM=sigmoid

# D10 loss configuration (built from the sweep-selected hyperparameters).
if [ "$D10_INCLUDE_Q" = "true" ]; then _Q_TAG="tQ"; else _Q_TAG="t"; fi
_SQ_TAG="${D10_SIGMA_Q/./p}"

if [ "$D10_ALPHA_MODE" = "off" ]; then
    _AT_TAG="atOff"
    _PRIOR_JSON="null"
elif [ "$D10_ALPHA_PRIOR" = "null" ]; then
    _AT_TAG="at_${D10_ALPHA_MODE}_noPr"
    _PRIOR_JSON="null"
else
    _PRIOR_TAG="${D10_ALPHA_PRIOR/./p}"
    _AT_TAG="at_${D10_ALPHA_MODE}_ls${_PRIOR_TAG}"
    _PRIOR_JSON="${D10_ALPHA_PRIOR}"
fi

# Metric-family label suffix: keeps L_joint and L_prod pickles distinct.
if [ "$D10_METRIC" = "marginal_llhd_prod" ]; then
    _METRIC_TAG="mllhdP"
else
    _METRIC_TAG="mllhd"
fi

PROB_FLAG="--probabilistic_sim"
D_LABEL="prob_${_METRIC_TAG}_${_Q_TAG}_sQ${_SQ_TAG}_${_AT_TAG}"
LOSS=dqdt_radial
LOSS_KW="{\"distance_metric\": \"${D10_METRIC}\", \"marginal_llhd_include_Q\": ${D10_INCLUDE_Q}, \"marginal_llhd_sigma_Q_ke\": ${D10_SIGMA_Q}, \"marginal_llhd_eps\": 1e-10, \"per_track_scale_mode\": \"${D10_ALPHA_MODE}\", \"per_track_scale_prior_log_sigma\": ${_PRIOR_JSON}}"

# ============================================================================
#                          LABEL + LAUNCH
# ============================================================================

LABEL="${CONFIG}_sc_${DX_LABEL}_${B_LABEL}_${D_LABEL}_${DEDX_DENSITY_MODE}_${FLOW_EXPECTATION_MODE}_tgtsimsd${SEED_STRATEGY}_nngh${N_NEIGH}_${MODE}_esamp${SAMPLING_STEP}cm_sigL${SIGNAL_LENGTH}_grclp${MAX_CLIP_NORM_VAL}_${LR_SCHEDULER}_bt${BATCH_SIZE}_nbt${MAX_NBATCH}_dtsd${DATA_SEED}_adam_${LOSS}_${NORM}"

SIF_FILE=/sdf/group/neutrino/pgranger/larnd-sim-jax.sif

echo "Running Configuration: $CONFIG"
echo "Label: $LABEL"

apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${SIF_FILE} /bin/bash -c "
pip3 install .; \
export PYTHONPATH=$PWD:$PWD/src:${PYTHONPATH}; \
python3 -m optimize.example_run \
    --data_sz -1 \
    --max_nbatch ${MAX_NBATCH} \
    --params ${PARAM} \
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
