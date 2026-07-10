#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=neutrino:cider-nu
#SBATCH --job-name=nd_baseline_truth
#SBATCH --output=/sdf/group/neutrino/pgranger/larnd-sim-jax/logs/nd_baseline_truth/job-%A_%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=32g
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=20:00:00
#SBATCH --array=0-9%4

PARAMS=optimize/scripts/param_list_nd_v7.yaml

ITERATIONS=5000
MAX_NBATCH=100
MAX_BATCH_LEN=400
DATA_SEED=1
SEED=${SLURM_ARRAY_TASK_ID:-0}
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
CHAIN_START_ITER=0
CHAIN_UPDATE_FREQ=5
MCS_PRIOR_WEIGHT=0.5
CHAIN_STEP_LEN=2.0
CHAIN_MOMENTUM_GEV=3.0

INPUT_FILE_TGT=/sdf/data/neutrino/cyifan/diffsim_input/true_through_muon_edep_10cm_vol1cm.h5
INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/diffsim_input/true_through_muon_edep_10cm_vol1cm.h5
SIF_FILE=/sdf/group/neutrino/pgranger/larnd-sim-jax/larndsim-jax_main.sif

JAX_CACHE_DIR=/sdf/group/neutrino/pgranger/.jax_cache

LABEL=nd_baseline_truth_b${MAX_NBATCH}_len${MAX_BATCH_LEN}_seed${SEED}
TEST_NAME=nd_baseline_truth

mkdir -p /sdf/group/neutrino/pgranger/larnd-sim-jax/logs/nd_baseline_truth

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
    --out_label ${LABEL} \
    --test_name ${TEST_NAME} \
    --save_freq 200
"

echo "=== Job finished: $(date) ==="
