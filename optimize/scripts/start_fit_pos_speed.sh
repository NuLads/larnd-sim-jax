#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=neutrino:cider-nu
#SBATCH --job-name=pos_speed
#SBATCH --output=/sdf/group/neutrino/pgranger/larnd-sim-jax/logs/pos_speed/job-%A_%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=32g
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=12:00:00
#SBATCH --array=0-2%3

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then SLURM_ARRAY_TASK_ID=0; fi

# Factorial: isolate chain-LR (#2) and batch-size/updates-per-track (#1)
#  idx0: 50cm, clr1e-4  -> slow baseline (few updates, low LR)
#  idx1: 50cm, clr1e-3  -> +high LR only
#  idx2: 400cm, clr1e-3 -> +high LR +big batch (~6x more updates/track)
MBL_ARR=(50 50 400)
MNB_ARR=(100 100 3)
CLR_ARR=(1e-4 1e-3 1e-3)
MAX_BATCH_LEN=${MBL_ARR[$SLURM_ARRAY_TASK_ID]}
MAX_NBATCH=${MNB_ARR[$SLURM_ARRAY_TASK_ID]}
CHAIN_LR=${CLR_ARR[$SLURM_ARRAY_TASK_ID]}

PARAM=long_diff          # frozen (main lr = 0)
MCS_PRIOR_WEIGHT=0.5     # fixed prior, so we isolate SPEED (not the accuracy floor)
ITERATIONS=3000
INPUT_FILE_TGT=/sdf/data/neutrino/cyifan/diffsim_input/true_through_muon_edep_10cm_vol1cm.h5
INPUT_FILE_SIM=/sdf/group/neutrino/pgranger/lads-data/linear_guess_segments.h5
SIF_FILE=/sdf/group/neutrino/pgranger/larnd-sim-jax/larndsim-jax_main.sif
LABEL=posspeed_len${MAX_BATCH_LEN}_clr${CHAIN_LR}
TEST_NAME=pos_speed
mkdir -p /sdf/group/neutrino/pgranger/larnd-sim-jax/logs/pos_speed

apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${SIF_FILE} /bin/bash -c "
cd /sdf/group/neutrino/pgranger/larnd-sim-jax
export PYTHONPATH=\$PWD/src:\$PWD:\$PYTHONPATH
export JAX_COMPILATION_CACHE_DIR=/sdf/group/neutrino/pgranger/.jax_cache
export LARND_SNAPSHOT=1
python3 -m optimize.example_run \
    --params ${PARAM} \
    --input_file_sim ${INPUT_FILE_SIM} --input_file_tgt ${INPUT_FILE_TGT} \
    --fit_type chain --iterations ${ITERATIONS} \
    --max_nbatch ${MAX_NBATCH} --max_batch_len ${MAX_BATCH_LEN} \
    --data_seed 1 --seed 0 --lr 0 --optimizer_fn Adam \
    --lr_scheduler warmup_exponential_decay_schedule \
    --lr_kw '{\"decay_rate\" : 0.999, \"init_value\" : 0, \"warmup_steps\": 500}' \
    --max_clip_norm_val 100 --electron_sampling_resolution 0.01 --number_pix_neighbors 4 \
    --signal_length 150 --mode lut --lut_file src/larndsim/detector_properties/response_44.npy \
    --loss_fn llhd --probabilistic_sim --sim_seed_strategy different --non_deterministic --no-noise \
    --fit_chain_positions --chain_lr ${CHAIN_LR} --chain_start_iter 0 --chain_update_freq 1 \
    --chain_decay_rate 0.999 --mcs_prior_weight ${MCS_PRIOR_WEIGHT} \
    --chain_step_len 2.0 --chain_momentum_GeV 3.0 --pos_residual_freq 1 \
    --out_label ${LABEL} --test_name ${TEST_NAME} --save_freq 1500
"
echo "=== done $(date) ==="
