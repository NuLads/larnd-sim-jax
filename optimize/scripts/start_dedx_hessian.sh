#!/bin/bash
#SBATCH --partition=turing
#SBATCH --job-name=dedx_hess
#SBATCH --output=/sdf/group/neutrino/pgranger/larnd-sim-jax/logs/dedx_hessian/job-%A_%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=32g
#SBATCH --gpus-per-node=geforce_rtx_2080_ti:1
#SBATCH --time=06:00:00

# dEdx-Hessian diagonal-dominance investigation. TRUE geometry (input_sim == input_tgt),
# fitted dEdx, NO position fitting, small scale (fits an 11 GiB card). Runs
# GradientDescentFitter.analyze_dedx_hessian on ONE batch via the --dedx_hessian_batch CLI hook
# (dispatches BEFORE the training loop -- no optimizer state is ever touched) and exits.
#
# Env knobs (all optional):
#   DHACCOUNT   SLURM account (pass via --account on submission; sbatch does not read this)
#   DHNCOLS     number of full Hessian columns to sample (default 12)
#   DHCHUNK     HVP columns per vmap chunk (default 8; lower if you OOM)
#   DHMAXDIAG   cap on number of diagonal entries (default: unset = all segments)
#   DHLEN       --max_batch_len (default 60)
#   DHNBATCH    --max_nbatch (default 2)
#   DHBATCHIDX  which loaded batch to analyze (default 0)
#   DHTAG       suffix for out_label / output pickle name
#   DHMEANW     --dedx_mean_constraint_weight (default 100000.0, matches start_sci_case.sh)

set -e

PARAMS=optimize/scripts/param_list_nd_noshift.yaml
INPUT_FILE_TRUE=/sdf/data/neutrino/cyifan/diffsim_input/true_through_muon_edep_10cm_vol1cm.h5
SIF_FILE=/sdf/group/neutrino/pgranger/larnd-sim-jax/larndsim-jax_main.sif
JAX_CACHE_DIR=/sdf/group/neutrino/pgranger/.jax_cache

MAX_BATCH_LEN=${DHLEN:-60}
MAX_NBATCH=${DHNBATCH:-2}
BATCH_IDX=${DHBATCHIDX:-0}
NCOLS=${DHNCOLS:-12}
CHUNK=${DHCHUNK:-8}
MAXDIAG_FLAG=${DHMAXDIAG:+--dedx_hessian_maxdiag $DHMAXDIAG}
MEANW=${DHMEANW:-100000.0}
TAG=${DHTAG:-}
OUT_PATH=/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result/dedx_hessian_b${BATCH_IDX}${TAG}.pkl

mkdir -p /sdf/group/neutrino/pgranger/larnd-sim-jax/logs/dedx_hessian
mkdir -p /sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result

echo "=== Job started: $(date) ==="
echo "=== MAX_BATCH_LEN=${MAX_BATCH_LEN} MAX_NBATCH=${MAX_NBATCH} BATCH_IDX=${BATCH_IDX} NCOLS=${NCOLS} CHUNK=${CHUNK} MAXDIAG=${DHMAXDIAG:-all} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

apptainer exec --nv -B /sdf,/fs,/lscratch ${SIF_FILE} /bin/bash -c "
cd /sdf/group/neutrino/pgranger/larnd-sim-jax
export PYTHONPATH=\$PWD/src:\$PWD:\$PYTHONPATH
export JAX_COMPILATION_CACHE_DIR=${JAX_CACHE_DIR}

python3 -m optimize.example_run \
    --params ${PARAMS} \
    --input_file_sim ${INPUT_FILE_TRUE} \
    --input_file_tgt ${INPUT_FILE_TRUE} \
    --fit_type chain \
    --iterations 50 \
    --max_nbatch ${MAX_NBATCH} \
    --max_batch_len ${MAX_BATCH_LEN} \
    --data_seed 1 \
    --seed 0 \
    --lr 1e-1 \
    --optimizer_fn Adam \
    --lr_scheduler warmup_exponential_decay_schedule \
    --lr_kw '{\"decay_rate\" : 0.999, \"init_value\" : 0, \"warmup_steps\": 500}' \
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
    --fit_dedx \
    --dedx_lr 1e-2 \
    --dedx_start_iter 0 \
    --dedx_freeze_iter 100000 \
    --dedx_prior_weight 0.5 \
    --dedx_drift_profile_weight 0.0 \
    --dedx_use_split_t True \
    --dedx_student_nu_l 4.785 \
    --dedx_student_nu_r 2.073 \
    --dedx_student_scale_l 0.1204 \
    --dedx_student_scale_r 0.1058 \
    --dedx_soft_barrier_threshold 8.5 \
    --dedx_soft_barrier_weight 1.0 \
    --dedx_mean_constraint_weight ${MEANW} \
    --dedx_hessian_batch ${BATCH_IDX} \
    --dedx_hessian_ncols ${NCOLS} \
    --dedx_hessian_chunk ${CHUNK} \
    ${MAXDIAG_FLAG} \
    --dedx_hessian_out ${OUT_PATH} \
    --out_label dedx_hess${TAG} \
    --test_name dedx_hess${TAG} \
    --save_freq 200
"

echo "=== Job finished: $(date) ==="
