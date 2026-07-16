#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=neutrino:cider-nu
#SBATCH --job-name=dedx_splinecheck
#SBATCH --output=/sdf/group/neutrino/pgranger/larnd-sim-jax/logs/dedx_splinecheck/job-%A.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=32g
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=03:00:00

# Head-to-head: does freezing dx in the spline warp hurt dEdx recovery?
# Same tracks, calibration FROZEN (--lr 0), fit dEdx + chain geometry.
# BASIS env selects angle (dx recomputed) vs spline (dx frozen).

BASIS=${BASIS:-spline}
ITERATIONS=2500
MAX_NBATCH=3
MAX_BATCH_LEN=50
DATA_SEED=1
SEED=0

DEDX_LR=1e-2
DEDX_START_ITER=0
DEDX_FREEZE_ITER=2600

CHAIN_LR=1e-4
CHAIN_STEP_LEN=2.0
CHAIN_MOMENTUM_GEV=3.0
MCS_PRIOR_WEIGHT=0.5
KNOT=${KNOT:-40}

INPUT_FILE_TGT=/sdf/data/neutrino/cyifan/diffsim_input/true_through_muon_edep_10cm_vol1cm.h5
INPUT_FILE_SIM=/sdf/group/neutrino/pgranger/lads-data/linear_guess_segments.h5
SIF_FILE=/sdf/group/neutrino/pgranger/larnd-sim-jax/larndsim-jax_main.sif
JAX_CACHE_DIR=/sdf/group/neutrino/pgranger/.jax_cache

LABEL=dedxcheck_${BASIS}_b${MAX_NBATCH}_len${MAX_BATCH_LEN}
TEST_NAME=dedx_splinecheck

mkdir -p /sdf/group/neutrino/pgranger/larnd-sim-jax/logs/dedx_splinecheck

echo "=== Job started: $(date) | BASIS=${BASIS} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

BASIS_ARGS=""
if [ "$BASIS" = "spline" ]; then
    BASIS_ARGS="--chain_basis spline --chain_spline_knot_cm ${KNOT}"
fi

apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${SIF_FILE} /bin/bash -c "
cd /sdf/group/neutrino/pgranger/larnd-sim-jax
export PYTHONPATH=\$PWD/src:\$PWD:\$PYTHONPATH
export JAX_COMPILATION_CACHE_DIR=${JAX_CACHE_DIR}

python3 -m optimize.example_run \
    --params Ab \
    --input_file_sim ${INPUT_FILE_SIM} \
    --input_file_tgt ${INPUT_FILE_TGT} \
    --fit_type chain \
    --iterations ${ITERATIONS} \
    --max_nbatch ${MAX_NBATCH} \
    --max_batch_len ${MAX_BATCH_LEN} \
    --data_seed ${DATA_SEED} \
    --seed ${SEED} \
    --lr ${CAL_LR:-0} \
    --optimizer_fn Adam \
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
    --dedx_start_iter ${DEDX_START_ITER} \
    --dedx_freeze_iter ${DEDX_FREEZE_ITER} \
    --dedx_prior_weight 0.5 \
    --dedx_use_split_t True \
    --dedx_student_nu_l 4.785 \
    --dedx_student_nu_r 2.073 \
    --dedx_student_scale_l 0.1204 \
    --dedx_student_scale_r 0.1058 \
    --dedx_soft_barrier_threshold 8.5 \
    --dedx_soft_barrier_weight 1.0 \
    --dedx_mean_constraint_weight 100000.0 \
    --fit_chain_positions \
    ${BASIS_ARGS} \
    --chain_lr ${CHAIN_LR} \
    --chain_start_iter 0 \
    --chain_update_freq 1 \
    --pos_residual_freq 25 \
    --mcs_prior_weight ${MCS_PRIOR_WEIGHT} \
    --chain_step_len ${CHAIN_STEP_LEN} \
    --chain_momentum_GeV ${CHAIN_MOMENTUM_GEV} \
    --out_label ${LABEL} \
    --test_name ${TEST_NAME} \
    --save_freq 200
"
echo "=== Job finished: $(date) ==="
