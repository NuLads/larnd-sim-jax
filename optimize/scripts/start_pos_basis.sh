#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=neutrino:cider-nu
#SBATCH --job-name=pos_basis
#SBATCH --output=/sdf/group/neutrino/pgranger/larnd-sim-jax/logs/pos_basis/job-%A.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=32g
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=16:00:00

# Position-only fit (calibration frozen via lr 0, dEdx at input): isolate the geometry
# parametrization. BASIS=angle|spline, LEN=50|400. Compares chain conditioning.
BASIS=${BASIS:-angle}
LEN=${LEN:-50}
CLR=${CLR:-1e-4}
ITER=${ITER:-5000}
MCS=${MCS:-0.5}
KNOT=${KNOT:-40}
STEP=${STEP:-2.0}
GEOM=${GEOM:-adam}
GNSTEPS=${GNSTEPS:-8}

INPUT_FILE_TGT=/sdf/data/neutrino/cyifan/diffsim_input/true_through_muon_edep_10cm_vol1cm.h5
INPUT_FILE_SIM=/sdf/group/neutrino/pgranger/lads-data/linear_guess_segments.h5
SIF_FILE=/sdf/group/neutrino/pgranger/larnd-sim-jax/larndsim-jax_main.sif
JAX_CACHE_DIR=/sdf/group/neutrino/pgranger/.jax_cache

LABEL=posb_${BASIS}_len${LEN}_clr${CLR}_mcs${MCS}_knot${KNOT}_geom${GEOM}${TAG:-}
TEST_NAME=pos_basis
mkdir -p /sdf/group/neutrino/pgranger/larnd-sim-jax/logs/pos_basis

echo "=== Job started: $(date) BASIS=${BASIS} LEN=${LEN} CLR=${CLR} MCS=${MCS} KNOT=${KNOT} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${SIF_FILE} /bin/bash -c "
cd /sdf/group/neutrino/pgranger/larnd-sim-jax
export PYTHONPATH=\$PWD/src:\$PWD:\$PYTHONPATH
export JAX_COMPILATION_CACHE_DIR=${JAX_CACHE_DIR}
python3 -m optimize.example_run \
    --params long_diff \
    --input_file_sim ${INPUT_FILE_SIM} --input_file_tgt ${INPUT_FILE_TGT} \
    --fit_type chain --iterations ${ITER} --max_nbatch 100 --max_batch_len ${LEN} \
    --data_seed 1 --seed ${SEED:-0} --lr 0 --optimizer_fn Adam \
    --lr_scheduler warmup_exponential_decay_schedule \
    --lr_kw '{\"decay_rate\" : 0.999, \"init_value\" : 0, \"warmup_steps\": 500}' \
    --max_clip_norm_val 100 --electron_sampling_resolution 0.01 \
    --number_pix_neighbors 4 --signal_length 150 --mode lut \
    --lut_file src/larndsim/detector_properties/response_44.npy \
    --loss_fn llhd ${LOSS_TW:+--loss_time_window $LOSS_TW} ${LOSS_SC:+--loss_sigma_charge $LOSS_SC} ${LOSS_ST:+--loss_sigma_time $LOSS_ST} --probabilistic_sim --sim_seed_strategy different --non_deterministic --no-noise \
    --fit_chain_positions --chain_lr ${CLR} --chain_start_iter 0 --chain_update_freq 1 \
    --mcs_prior_weight ${MCS} --chain_step_len ${STEP} --chain_momentum_GeV 3.0 \
    --chain_basis ${BASIS} --chain_spline_knot_cm ${KNOT} \
    --geom_optimizer ${GEOM} --geom_lbfgs_steps ${GNSTEPS} \
    --pos_residual_freq 5 \
    --out_label ${LABEL} --test_name ${TEST_NAME} --save_freq 200
"
echo "=== Job finished: $(date) ==="
