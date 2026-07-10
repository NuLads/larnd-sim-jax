#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=neutrino:cider-nu
#SBATCH --job-name=joint_1d_v6nc
#SBATCH --output=/sdf/group/neutrino/pgranger/larnd-sim-jax/logs/joint_1d_v6nc/job-%A_%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=32g
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=10:00:00
#SBATCH --array=0-8%4

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    SLURM_ARRAY_TASK_ID=0
fi

PARAMS=("Ab" "kb" "eField" "tran_diff" "long_diff" "lifetime" "shift_z" "shift_x" "shift_y")
PARAM=${PARAMS[$SLURM_ARRAY_TASK_ID]}

ITERATIONS=5000
MAX_NBATCH=100
MAX_BATCH_LEN=50
DATA_SEED=1
SEED=0
LOSS=llhd
N_NEIGH=4
MODE=lut
LR_SCHEDULER=warmup_exponential_decay_schedule

# dEdx settings (same as v6)
DEDX_PRIOR_WEIGHT=0.5
DEDX_LR=1e-2
DEDX_START_ITER=0
DEDX_FREEZE_ITER=5200

INPUT_FILE_TGT=/sdf/data/neutrino/cyifan/diffsim_input/true_through_muon_edep_10cm_vol1cm.h5
INPUT_FILE_SIM=/sdf/group/neutrino/pgranger/lads-data/linear_guess_segments.h5
SIF_FILE=/sdf/group/neutrino/pgranger/larnd-sim-jax/larndsim-jax_main.sif

JAX_CACHE_DIR=/sdf/group/neutrino/pgranger/.jax_cache

LABEL=joint_1d_${PARAM}_${ITERATIONS}iter_b${MAX_NBATCH}_len${MAX_BATCH_LEN}_nochain_dr0p999
TEST_NAME=joint_1d_v6nc

echo "=== Job started: $(date) ==="
echo "=== Parameter: ${PARAM} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${SIF_FILE} /bin/bash -c "
cd /sdf/group/neutrino/pgranger/larnd-sim-jax
export PYTHONPATH=\$PWD/src:\$PWD:\$PYTHONPATH
export JAX_COMPILATION_CACHE_DIR=${JAX_CACHE_DIR}

python3 -m optimize.example_run \
    --params ${PARAM} \
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
    --dedx_mean_constraint_weight 0.0 \
    --out_label ${LABEL} \
    --test_name ${TEST_NAME} \
    --save_freq 200
"

echo "=== Job finished: $(date) ==="
