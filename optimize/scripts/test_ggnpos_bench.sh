#!/bin/bash
#SBATCH --partition=turing
#SBATCH --account=neutrino:cider-nu
#SBATCH --job-name=ggnpos_bench
#SBATCH --output=/sdf/group/neutrino/pgranger/larnd-sim-jax/logs/ggnpos_bench/job-%A_%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=32g
#SBATCH --gpus-per-node=geforce_rtx_2080_ti:1
#SBATCH --time=08:00:00
#SBATCH --array=0-0

# PRIVATE COPY of start_sci_case.sh (do not touch the original; jobs run from it) for the
# position-GGN vs Adam-geometry benchmark on turing. SCIMODE=full always (chain positions
# fitted); SCIGEOMOPT selects the geometry solver ('adam' default path, or 'ggn' for the
# matrix-free Gauss-Newton/CG spline solve via _solve_chain_ggn[_shared]).
SCIMODE=full

PARAMS=${SCIPARAMS:-optimize/scripts/param_list_nd_noshift.yaml}

ITERATIONS=${SCIITER:-1500}
MAX_NBATCH=${SCINBATCH:-16}
MAX_BATCH_LEN=${SCILEN:-60}
DATA_SEED=1
SEED=${SLURM_ARRAY_TASK_ID:-0}
LOSS=llhd
N_NEIGH=4
MODE=lut
LR_SCHEDULER=warmup_exponential_decay_schedule

# dEdx settings (identical across modes)
DEDX_PRIOR_WEIGHT=${SCIDEDXPRIOR:-0.5}
DEDX_DRIFT_PROFILE_WEIGHT=${SCIDPW:-0.0}
DEDX_LR=1e-2
DEDX_START_ITER=0
DEDX_FREEZE_ITER=$((ITERATIONS+200))

# Chain position settings (only used when SCIMODE=full; CHAIN_LR/CHAIN_STEP_LEN/etc are
# irrelevant when SCIGEOMOPT=ggn since the GN solver replaces the Adam geometry update)
CHAIN_LR=${SCICLR:-1e-4}
CHAIN_START_ITER=${SCICHSTART:-0}
CHAIN_UPDATE_FREQ=1
MCS_PRIOR_WEIGHT=0.5
CHAIN_STEP_LEN=${SCISTEP:-2.0}
CHAIN_MOMENTUM_GEV=3.0

INPUT_FILE_TGT=/sdf/data/neutrino/cyifan/diffsim_input/true_through_muon_edep_10cm_vol1cm.h5
LINEAR_GUESS=/sdf/group/neutrino/pgranger/lads-data/linear_guess_segments.h5
SIF_FILE=/sdf/group/neutrino/pgranger/larnd-sim-jax/larndsim-jax_main.sif
JAX_CACHE_DIR=/sdf/group/neutrino/pgranger/.jax_cache

INPUT_FILE_SIM=${SCISIMFILE:-${LINEAR_GUESS}}
CHAIN_FLAG="--fit_chain_positions"

TEST_NAME=ggnpos_bench${SCITAG:-}
LABEL=ggnpos_${SCITAG:-adam}_b${MAX_NBATCH}_len${MAX_BATCH_LEN}_seed${SEED}

mkdir -p /sdf/group/neutrino/pgranger/larnd-sim-jax/logs/ggnpos_bench

echo "=== Job started: $(date) ==="
echo "=== SEED=${SEED} INPUT_SIM=${INPUT_FILE_SIM} CHAIN_FLAG='${CHAIN_FLAG}' SCIGEOMOPT=${SCIGEOMOPT:-adam(default)} SCIBASIS=${SCIBASIS:-angle(default)} ==="
echo "=== ITERATIONS=${ITERATIONS} MAX_NBATCH=${MAX_NBATCH} LEN=${MAX_BATCH_LEN} CHAIN_START_ITER=${CHAIN_START_ITER} CHAIN_LR=${CHAIN_LR} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${SIF_FILE} /bin/bash -c "
cd /sdf/group/neutrino/pgranger/larnd-sim-jax
export PYTHONPATH=\$PWD/src:\$PWD:\$PYTHONPATH
export JAX_COMPILATION_CACHE_DIR=${JAX_CACHE_DIR}
export LARND_CHAIN_FRAME=${LARND_CHAIN_FRAME:-drift}
export LARND_CHAIN_DRIFT_W=${LARND_CHAIN_DRIFT_W:-0}
export LARND_GN_TIMING=${LARND_GN_TIMING:-}
echo \"[CHAIN-ENV] frame=\$LARND_CHAIN_FRAME drift_w=\$LARND_CHAIN_DRIFT_W gn_timing=\$LARND_GN_TIMING\"

python3 -m optimize.example_run \
    --params ${PARAMS} \
    --input_file_sim ${INPUT_FILE_SIM} \
    --input_file_tgt ${INPUT_FILE_TGT} \
    --fit_type chain \
    --iterations ${ITERATIONS} \
    --max_nbatch ${MAX_NBATCH} \
    --max_batch_len ${MAX_BATCH_LEN} \
    --data_seed ${DATA_SEED} \
    --seed ${SEED} \
    --lr 1e-1 \
    --optimizer_fn Adam \
    --lr_scheduler ${LR_SCHEDULER} \
    --lr_kw '{\"decay_rate\" : '${SCILRDECAY:-0.999}', \"init_value\" : 0, \"warmup_steps\": 500}' \
    --max_clip_norm_val 100 \
    --electron_sampling_resolution 0.01 \
    --number_pix_neighbors ${N_NEIGH} \
    --signal_length 150 \
    --mode ${MODE} \
    --lut_file src/larndsim/detector_properties/response_44.npy \
    --loss_fn ${LOSS} ${SCISETP:+--set_params $SCISETP} ${SCIINIT:+--set_init_params $SCIINIT} ${SCITGT:+--set_target_vals $SCITGT} \
    --probabilistic_sim ${SCIPROBTGT:+--probabilistic_sampling_target} \
    --sim_seed_strategy different ${SCISHUFFLE:+--shuffle_bt} \
    --non_deterministic \
    ${NOISEFLAG:-} \
    ${DEDXFLAG:---fit_dedx} \
    --dedx_lr ${DEDX_LR} \
    --dedx_start_iter ${DEDX_START_ITER} \
    --dedx_freeze_iter ${DEDX_FREEZE_ITER} \
    --dedx_prior_weight ${DEDX_PRIOR_WEIGHT} \
    --dedx_use_split_t True \
    --dedx_student_nu_l 4.785 \
    --dedx_student_nu_r 2.073 \
    --dedx_student_scale_l 0.1204 \
    --dedx_student_scale_r 0.1058 \
    --dedx_soft_barrier_threshold 8.5 \
    --dedx_soft_barrier_weight 1.0 \
    --dedx_mean_constraint_weight 100000.0 \
    --dedx_drift_profile_weight ${DEDX_DRIFT_PROFILE_WEIGHT} \
    ${CHAIN_FLAG} ${SCIBASIS:+--chain_basis $SCIBASIS --chain_spline_knot_cm 40} ${SCIGEOMOPT:+--geom_optimizer $SCIGEOMOPT} \
    --chain_lr ${CHAIN_LR} \
    --chain_start_iter ${CHAIN_START_ITER} \
    --chain_update_freq ${CHAIN_UPDATE_FREQ} \
    --chain_decay_rate ${SCICHDECAY:-0.999} \
    --mcs_prior_weight ${MCS_PRIOR_WEIGHT} \
    --chain_step_len ${CHAIN_STEP_LEN} \
    --chain_momentum_GeV ${CHAIN_MOMENTUM_GEV} \
    --out_label ${LABEL} \
    --test_name ${TEST_NAME} \
    --save_freq ${SCISAVEFREQ:-200}
"

echo "=== Job finished: $(date) ==="
