#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=neutrino:cider-nu
#SBATCH --job-name=joint_nd_v14
#SBATCH --output=/sdf/group/neutrino/pgranger/larnd-sim-jax/logs/joint_nd_v14/job-%A_%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=32g
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=40:00:00

PARAMS=optimize/scripts/param_list_nd_noshift.yaml

ITERATIONS=5000
MAX_NBATCH=100
MAX_BATCH_LEN=50
DATA_SEED=1
SEED=0
LOSS=llhd
N_NEIGH=4
MODE=lut
LR_SCHEDULER=warmup_exponential_decay_schedule

# dEdx settings
DEDX_PRIOR_WEIGHT=0.5
DEDX_LR=1e-2
DEDX_START_ITER=0
DEDX_FREEZE_ITER=5200

# Chain position settings: update chain gradient every 5 steps (~2.5x speedup)
CHAIN_LR=1e-4
CHAIN_START_ITER=2000
CHAIN_UPDATE_FREQ=1
MCS_PRIOR_WEIGHT=0.5
CHAIN_STEP_LEN=2.0
CHAIN_MOMENTUM_GEV=3.0

INPUT_FILE_TGT=/sdf/data/neutrino/cyifan/diffsim_input/true_through_muon_edep_10cm_vol1cm.h5
INPUT_FILE_SIM=/sdf/group/neutrino/pgranger/lads-data/linear_guess_segments.h5
SIF_FILE=/sdf/group/neutrino/pgranger/larnd-sim-jax/larndsim-jax_main.sif

JAX_CACHE_DIR=/sdf/group/neutrino/pgranger/.jax_cache

LABEL=joint_nd_${ITERATIONS}iter_b${MAX_NBATCH}_len${MAX_BATCH_LEN}_chainfreq${CHAIN_UPDATE_FREQ}_dr0p999
TEST_NAME=joint_nd_v14

mkdir -p /sdf/group/neutrino/pgranger/larnd-sim-jax/logs/joint_nd_v14

echo "=== Job started: $(date) ==="
echo "=== N-D params from ${PARAMS} ==="
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
    --lr_scheduler ${LR_SCHEDULER} \
    --lr_kw '{\"decay_rate\" : 0.999, \"init_value\" : 0, \"warmup_steps\": 500}' \
    --max_clip_norm_val 100 \
    --electron_sampling_resolution 0.01 \
    --number_pix_neighbors ${N_NEIGH} \
    --signal_length 150 \
    --mode ${MODE} \
    --lut_file src/larndsim/detector_properties/response_44.npy \
    --loss_fn ${LOSS} \
    --probabilistic_sim \
    --sim_seed_strategy different \
    --non_deterministic \
    --no-noise \
    --fit_dedx \
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
    --fit_chain_positions \
    --chain_lr ${CHAIN_LR} \
    --chain_start_iter ${CHAIN_START_ITER} \
    --chain_update_freq ${CHAIN_UPDATE_FREQ} \
    --chain_decay_rate 0.999 \
    --pos_residual_freq 25 \
    --mcs_prior_weight ${MCS_PRIOR_WEIGHT} \
    --chain_step_len ${CHAIN_STEP_LEN} \
    --chain_momentum_GeV ${CHAIN_MOMENTUM_GEV} \
    --out_label ${LABEL} \
    --test_name ${TEST_NAME} \
    --save_freq 200
"

echo "=== Job finished: $(date) ==="
