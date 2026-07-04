#!/bin/bash

# Benchmark the probabilistic ROI-trimming optimization.
#
# Runs two short fits (~20-50 iterations each) with the D2 (probabilistic)
# configuration used by start_fit_ABCD.sh — one with the trimmed per-hit
# window and one with an effectively full-tick window — and prints the
# runtime + peak-memory summary at the end of each fit. Compare the two
# summaries to see how much the trim helps.
#
# Usage:
#   optimize/scripts/benchmark_roi_trim.sh                 # runs both
#   optimize/scripts/benchmark_roi_trim.sh trimmed         # trimmed only
#   optimize/scripts/benchmark_roi_trim.sh baseline        # baseline only
#
# Configure ITERATIONS below to tune how long the fit runs.

#SBATCH --partition=ampere
#SBATCH --account=neutrino:dune-ml
#SBATCH --job-name=diffsim_bench
#SBATCH --output=logs/benchmark_roi_trim/job-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=32g
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=02:00:00

CONFIG="A2-B2-C2-D2"    # probabilistic (D2) — exercises the trimmed output & ProbabilisticLossStrategy
ITERATIONS=100           # short fit; step 0 is JIT compile, so keep >= ~20
BATCH_SIZE=200
MAX_NBATCH=500
MAX_CLIP_NORM_VAL=1
DATA_SEED=1
TARGET_SEED=1
SEED_STRATEGY=different
SAMPLING_STEP=0.01
N_NEIGH=4
MODE="lut"
LR_SCHEDULER=warmup_exponential_decay_schedule
RESPONSE_ROI_LENGTH=200
LOSS=llhd

# Fixed inputs (A2-B3 combo from start_fit_ABCD.sh)
INPUT_FILE_TGT=/sdf/data/neutrino/cyifan/dunend_train_prod/prod_mod0_mpvmpr/production_884072/job_23771825_0000/output_23771825_0000-edepsim_lbl_trklen2cm_containment2cm_costheta0.966_range_0.05cm.h5
INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/dunend_train_prod/prod_mod0_mpvmpr/production_884072/job_23771825_0000/output_23771825_0000-edepsim_lbl_trklen2cm_containment2cm_costheta0.966_range_0.05cm.h5
PARAMS=optimize/scripts/param_list_${CONFIG}.yaml
CHOP_FLAG="--no_chop"

# Nticks estimate: for time_window=200 μs and t_sampling=0.1 μs, Nticks ≈ 2000.
# Baseline uses a hit-ROI window that covers the entire readout (effectively no trim).
BASELINE_HIT_ROI_LENGTH=1999
BASELINE_HIT_ROI_PAD_BEFORE=1999
TRIMMED_HIT_ROI_LENGTH=400
TRIMMED_HIT_ROI_PAD_BEFORE=300

N_HITS=6
N_BEAM=32

SIF_FILE=/sdf/group/neutrino/pgranger/larnd-sim-jax.sif

WHICH="${1:-both}"

run_fit () {
    local mode_label="$1"
    local hit_roi_length="$2"
    local hit_roi_pad_before="$3"

    local LABEL="benchmark_${CONFIG}_${mode_label}_hitroi${hit_roi_length}_pad${hit_roi_pad_before}_Nhits${N_HITS}_Nbeam${N_BEAM}_it${ITERATIONS}"
    echo ""
    echo "===================================================="
    echo "  Benchmark run: ${mode_label}"
    echo "  hit_roi_length=${hit_roi_length} hit_roi_pad_before=${hit_roi_pad_before}"
    echo "  Label: ${LABEL}"
    echo "===================================================="

    # PYTHONNOUSERSITE=1 hides ~/.local/site-packages, which shadows the
    # container's own site-packages with a jaxlib built against cuDNN 9.8
    # while the container only ships cuDNN 9.1. PYTHONPATH puts this repo's
    # src/ ahead of the container's larndsim so the local ROI-trim edits
    # take effect without needing pip install.
    apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${SIF_FILE} /bin/bash -c "
export PYTHONNOUSERSITE=1; \
export PYTHONPATH=\$(pwd)/src:\$(pwd):\${PYTHONPATH}; \
python3 -m optimize.example_run \
    --data_sz -1 \
    --max_nbatch ${MAX_NBATCH} \
    --params ${PARAMS} \
    --input_file_sim ${INPUT_FILE_SIM} \
    --input_file_tgt ${INPUT_FILE_TGT} \
    --non_deterministic \
    --fit_type chain \
    --track_len_sel 2 \
    --max_abs_costheta_sel 0.966 \
    --min_abs_segz_sel 15. \
    --data_seed ${DATA_SEED} \
    --out_label ${LABEL} \
    --test_name benchmark_roi_trim \
    --seed ${TARGET_SEED} \
    --optimizer_fn Adam \
    --iterations ${ITERATIONS} \
    --max_batch_len ${BATCH_SIZE} \
    --track_z_bound 28 \
    --max_clip_norm_val ${MAX_CLIP_NORM_VAL} \
    --electron_sampling_resolution ${SAMPLING_STEP} \
    --number_pix_neighbors ${N_NEIGH} \
    --response_roi_length ${RESPONSE_ROI_LENGTH} \
    --mode ${MODE} \
    --lut_file ../Data_selection/response_44_v2a_full_tick.npz \
    --loss_fn ${LOSS} \
    --sim_seed_strategy ${SEED_STRATEGY} \
    --clip_from_range \
    --lr_scheduler ${LR_SCHEDULER} \
    --lr_kw '{\"decay_rate\" : 0.99, \"init_value\" : 0, \"warmup_steps\": 1000}' \
    --shuffle_bt \
    --normalization_scheme sigmoid \
    --probabilistic_sim \
    --hit_roi_length ${hit_roi_length} \
    --hit_roi_pad_before ${hit_roi_pad_before} \
    --max_adc_values ${N_HITS} \
    --fee_paths_scaling ${N_BEAM} \
    ${CHOP_FLAG}
"
    echo ""
    echo "History pickle:  fit_result/benchmark_roi_trim/history_iter*_${LABEL}.pkl"
}

case "$WHICH" in
    trimmed)
        run_fit "trimmed" "$TRIMMED_HIT_ROI_LENGTH" "$TRIMMED_HIT_ROI_PAD_BEFORE"
        ;;
    baseline)
        run_fit "baseline" "$BASELINE_HIT_ROI_LENGTH" "$BASELINE_HIT_ROI_PAD_BEFORE"
        ;;
    both|*)
        run_fit "trimmed" "$TRIMMED_HIT_ROI_LENGTH" "$TRIMMED_HIT_ROI_PAD_BEFORE"
        run_fit "baseline" "$BASELINE_HIT_ROI_LENGTH" "$BASELINE_HIT_ROI_PAD_BEFORE"
        ;;
esac

echo ""
echo "Compare the two summary blocks that printed at the end of each fit."
echo "To re-inspect a history file:"
echo "  python3 -m optimize.benchmark fit_result/benchmark_roi_trim/<file>.pkl"
