#!/bin/bash
#
# scan_2d_lifetime_shiftz.sh — 2-D loss scan of (lifetime, shift_z) at fixed
# other params, for A2-B5-C2-D10.
#
# Approach: use the existing 1-D scan machinery but drive shift_z as an outer
# axis via SLURM_ARRAY_TASK_ID. Each array task fixes shift_z to a value from
# SHIFT_Z_GRID via --set_init_params, then runs the ordinary lifetime scan.
# Reassemble the 2-D loss surface in the notebook by concatenating pickles.
#

#SBATCH --partition=ampere
##SBATCH --account=mli:nu-ml-dev
#SBATCH --account=mli:cider-ml
##SBATCH --account=neutrino:dune-ml
#SBATCH --job-name=diffsim_2d
#SBATCH --output=logs/scan/2d_lifetime_shiftz-%A_%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=32g
#SBATCH --gpus-per-node=a100:1
#SBATCH --exclude=sdfampere024,sdfampere038
#SBATCH --requeue
#SBATCH --time=4:00:00
#SBATCH --array=6-10
##SBATCH --array=0-10

# ---------- 2-D grid definition ----------

# shift_z outer axis — 11 values centered at 0. cm.
SHIFT_Z_GRID=(-0.30 -0.24 -0.18 -0.12 -0.06 0.00 0.06 0.12 0.18 0.24 0.30)

# The 1-D lifetime scan itself is done via example_run's scan machinery,
# so lifetime is sampled at 50 points along the range in
# param_list_A2-B5-C2-D10.yaml. To use a tighter/wider range, temporarily
# swap that yaml (or point PARAMS below at a different one).

TID=${SLURM_ARRAY_TASK_ID:-5}   # default = 0.00 for interactive test
if [ "$TID" -ge "${#SHIFT_Z_GRID[@]}" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$TID exceeds grid size ${#SHIFT_Z_GRID[@]}"
    exit 1
fi
SHIFT_Z_VAL="${SHIFT_Z_GRID[$TID]}"

# ---------- CONFIG (locked to A2-B5-C2-D10, sweep-best D10 hyperparams) ----------

CONFIG="A2-B5-C2-D10"
TARGET_SEED=$TID              # keep seed distinct per shift_z point

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

# ---------- D10 label + LOSS_KW (mirrors scan_ABCD.sh) ----------

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

# Shift_z tag: replace "." with "p" and prefix with "m" for negative.
_SHZ_ABS="${SHIFT_Z_VAL#-}"
_SHZ_TAG="${_SHZ_ABS/./p}"
if [ "${SHIFT_Z_VAL:0:1}" = "-" ]; then _SHZ_TAG="m${_SHZ_TAG}"; fi

PROB_FLAG="--probabilistic_sim"
D_LABEL="prob_${_METRIC_TAG}_${_Q_TAG}_sQ${_SQ_TAG}_${_AT_TAG}_shz${_SHZ_TAG}"
LOSS=dqdt_radial
LOSS_KW="{\"distance_metric\": \"${D10_METRIC}\", \"marginal_llhd_include_Q\": ${D10_INCLUDE_Q}, \"marginal_llhd_sigma_Q_ke\": ${D10_SIGMA_Q}, \"marginal_llhd_eps\": 1e-10, \"per_track_scale_mode\": \"${D10_ALPHA_MODE}\", \"per_track_scale_prior_log_sigma\": ${_PRIOR_JSON}}"

# Shortened label — the 2-D scan filename would overflow 255 chars if all the
# constant sim-config tags were included. Keep only what distinguishes runs:
#   - CONFIG (A2-B5-C2-D10)
#   - _2Dsz_ marker (used by the notebook glob)
#   - B_LABEL (input type)
#   - D_LABEL (metric + hyperparams + shift_z axis point)
#   - bt / nbt (batch stats)
# Everything else (dedx flow mode, sigmoid norm, lut mode, warmup schedule,
# seed strategy, etc.) is constant across the 2-D grid and can be re-derived
# from the config file if needed.
LABEL="${CONFIG}_2Dsz_${B_LABEL}_${D_LABEL}_bt${BATCH_SIZE}nbt${MAX_NBATCH}"

SIF_FILE=/sdf/group/neutrino/pgranger/larnd-sim-jax.sif

echo "==========================================================="
echo " 2-D scan (lifetime × shift_z)"
echo "   hostname          = $(hostname)"
echo "   nvidia-smi first line = $(nvidia-smi -L 2>&1 | head -1)"
echo "   CONFIG            = $CONFIG"
echo "   ARRAY_TASK_ID     = $TID  (of  ${#SHIFT_Z_GRID[@]})"
echo "   shift_z value     = $SHIFT_Z_VAL cm  (tag: $_SHZ_TAG)"
echo "   lifetime axis     = 50 points along the range in $PARAMS"
echo "   D10 σ_Q, α_prior  = $D10_SIGMA_Q ke, $D10_ALPHA_PRIOR"
echo "==========================================================="
echo "Label: $LABEL"

apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${SIF_FILE} /bin/bash -c "
# Preflight: verify JAX can actually initialize a CUDA device on this host.
# On some nodes (e.g. sdfampere038 seen 2026-08-31) nvidia-smi lists the GPU
# but XLA's StreamExecutor init fails with CUDA_ERROR_UNKNOWN. Exit with a
# non-zero status so #SBATCH --requeue drops the task onto a different node
# instead of silently burning the 4h wall clock.
if ! python3 -c 'import jax; assert any(d.platform==\"gpu\" for d in jax.devices()), \"no GPU\"' 2>/dev/null; then
    echo \"CUDA preflight FAILED on \$(hostname). Requesting requeue.\"
    exit 75   # EX_TEMPFAIL — SLURM requeues on non-zero + --requeue
fi
export PYTHONPATH=$PWD:$PWD/src:${PYTHONPATH}; \
if ! python3 -c 'import larndsim' >/dev/null 2>&1; then \
    # Serialize the install with a flock so at most one array task installs
    # into user site-packages at a time. Others wait until the winner is done.
    (flock -x 200; python3 -c 'import larndsim' >/dev/null 2>&1 || pip3 install .) 200> /tmp/pip_larndsim.lock; \
fi; \
python3 -m optimize.example_run \
    --data_sz -1 \
    --max_nbatch ${MAX_NBATCH} \
    --params lifetime \
    --set_init_params shift_z ${SHIFT_Z_VAL} \
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
