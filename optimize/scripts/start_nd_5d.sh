#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=neutrino:cider-nu
#SBATCH --job-name=nd_5d
#SBATCH --output=/sdf/group/neutrino/pgranger/larnd-sim-jax/logs/nd_5d/job-%A.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=32g
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=16:00:00

# Full 5-D calibration challenge: 5 calib params (Ab,eField,tran_diff,long_diff,lifetime)
# + per-segment dEdx + chain positions, all fit jointly.
PARAMS=optimize/scripts/param_list_nd_noshift.yaml

ITERATIONS=${ITERATIONS:-3000}
MAX_NBATCH=${MAX_NBATCH:-30}
MAX_BATCH_LEN=${MAX_BATCH_LEN:-100}
DATA_SEED=${DATA_SEED:-1}
SEED=${SEED:-0}
BASIS=${BASIS:-spline}
KNOT=${KNOT:-40}
CHAIN_LR=${CHAIN_LR:-1e-2}
DEDX_LR=1e-2
MCS_PRIOR_WEIGHT=0.5
CHAIN_STEP_LEN=2.0
DRIFT_PROFILE=${DRIFT_PROFILE:-0.0}   # dEdx<->lifetime degeneracy breaker (needed at long scale)
DEDX_PRIOR=${DEDX_PRIOR:-0.5}         # per-segment dEdx prior weight (higher = less freedom to absorb lifetime)

INPUT_FILE_TGT=/sdf/data/neutrino/cyifan/diffsim_input/true_through_muon_edep_10cm_vol1cm.h5
INPUT_FILE_SIM=/sdf/group/neutrino/pgranger/lads-data/linear_guess_segments.h5
SIF_FILE=/sdf/group/neutrino/pgranger/larnd-sim-jax/larndsim-jax_main.sif
JAX_CACHE_DIR=/sdf/group/neutrino/pgranger/.jax_cache

BASIS_ARGS=""
[ "$BASIS" = "spline" ] && BASIS_ARGS="--chain_basis spline --chain_spline_knot_cm ${KNOT}"

LABEL=nd5d_${BASIS}_b${MAX_NBATCH}_len${MAX_BATCH_LEN}_clr${CHAIN_LR}_dpw${DRIFT_PROFILE}_ds${DATA_SEED}${TAG:-}
TEST_NAME=nd_5d
mkdir -p /sdf/group/neutrino/pgranger/larnd-sim-jax/logs/nd_5d

echo "=== 5-D fit: BASIS=${BASIS} len=${MAX_BATCH_LEN} b=${MAX_NBATCH} clr=${CHAIN_LR} dpw=${DRIFT_PROFILE} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${SIF_FILE} /bin/bash -c "
cd /sdf/group/neutrino/pgranger/larnd-sim-jax
export PYTHONPATH=\$PWD/src:\$PWD:\$PYTHONPATH
export JAX_COMPILATION_CACHE_DIR=${JAX_CACHE_DIR}

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
    --lr_scheduler warmup_exponential_decay_schedule \
    --lr_kw '{\"decay_rate\" : 0.999, \"init_value\" : 0, \"warmup_steps\": '${WARMUP:-500}'}' \
    --max_clip_norm_val 100 \
    --electron_sampling_resolution 0.01 \
    --number_pix_neighbors 4 \
    --signal_length 150 \
    --mode lut \
    --lut_file src/larndsim/detector_properties/response_44.npy \
    --loss_fn llhd \
    --probabilistic_sim \
    --sim_seed_strategy different \
    --non_deterministic \
    --no-noise \
    --fit_dedx \
    --dedx_lr ${DEDX_LR} \
    --dedx_start_iter 0 \
    --dedx_freeze_iter ${DEDX_FREEZE:-5200} \
    --dedx_prior_weight ${DEDX_PRIOR} \
    --dedx_use_split_t True \
    --dedx_student_nu_l 4.785 --dedx_student_nu_r 2.073 \
    --dedx_student_scale_l 0.1204 --dedx_student_scale_r 0.1058 \
    --dedx_soft_barrier_threshold 8.5 --dedx_soft_barrier_weight 1.0 \
    --dedx_mean_constraint_weight 100000.0 \
    --dedx_drift_profile_weight ${DRIFT_PROFILE} \
    --fit_chain_positions \
    ${BASIS_ARGS} \
    --chain_lr ${CHAIN_LR} \
    --chain_start_iter 0 \
    --chain_update_freq 1 \
    --chain_decay_rate 0.999 \
    --pos_residual_freq 25 \
    --mcs_prior_weight ${MCS_PRIOR_WEIGHT} \
    --chain_step_len ${CHAIN_STEP_LEN} \
    --chain_momentum_GeV 3.0 \
    --out_label ${LABEL} \
    --test_name ${TEST_NAME} \
    --save_freq 100
"
echo "=== Job finished: $(date) ==="
