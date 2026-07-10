#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=neutrino:cider-nu
#SBATCH --job-name=gn_compare
#SBATCH --output=/sdf/group/neutrino/pgranger/larnd-sim-jax/logs/gn_compare/job-%A.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=32g
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=12:00:00

# Gauss-Newton (LM) vs Adam on the calibration-only problem with GROUND-TRUTH tracks.
# INPUT_SIM = INPUT_TGT = true file  ->  positions & dEdx are correct; only the 5 global
# calibration params are unknown. Both modes share config, init, and the same batches.
#   GNMODE=gn        -> fit_type gn_calib, exact Hessian curvature (2nd-order graphs)
#   GNMODE=ggn       -> fit_type gn_calib, Fisher/GGN curvature (1st-order, PSD, cheap)
#   GNMODE=ggn_val   -> validation only: exact H vs Fisher F on batch 0 at true params
#   GNMODE=adam      -> fit_type chain    (per-batch Adam),  ITERS = Adam iterations
#   GNMODE=gn_smoke  -> tiny gn_calib (1 batch, 3 iters) — correctness smoke test
GNMODE=${GNMODE:-gn}

PARAMS=optimize/scripts/param_list_nd_noshift.yaml
MAX_BATCH_LEN=50
DATA_SEED=1
SEED=0
LOSS=llhd
N_NEIGH=4
MODE=lut
TRUE=/sdf/data/neutrino/cyifan/diffsim_input/true_through_muon_edep_10cm_vol1cm.h5
SIF_FILE=/sdf/group/neutrino/pgranger/larnd-sim-jax/larndsim-jax_main.sif
JAX_CACHE_DIR=/sdf/group/neutrino/pgranger/.jax_cache

EXTRA_ARGS=""
# NOTE: dataio only loads max_batch_len*(max_nbatch+2) cm of leading tracks and then drops
# trajectories longer than max_batch_len, so small max_nbatch values starve the dataset
# (max_nbatch=8 -> only 2 usable batches). Use 100 -> ~32 batches at 50 cm.
case ${GNMODE} in
  gn)
    FIT_TYPE=gn_calib; ITERS=40;   MAX_NBATCH=100
    LR_SCHEDULER=constant_schedule ;;
  ggn)
    FIT_TYPE=gn_calib; ITERS=40;   MAX_NBATCH=100
    LR_SCHEDULER=constant_schedule
    EXTRA_ARGS="--gn_curvature ggn" ;;
  ggn_val)
    FIT_TYPE=gn_calib; ITERS=1;    MAX_NBATCH=1
    LR_SCHEDULER=constant_schedule
    EXTRA_ARGS="--gn_curvature ggn --gn_validate --set_init_params Ab 0.8348813503927325 eField 0.521518936637242 tran_diff 1.0027633760716438e-05 long_diff 5.8141822809782784e-06 lifetime 2406.4465970250712" ;;
  gn_smoke)
    FIT_TYPE=gn_calib; ITERS=3;    MAX_NBATCH=1
    LR_SCHEDULER=constant_schedule ;;
  gn_polish)
    # GN as final polish: start from the Adam 32-batch endpoint (iter 3000). Near the
    # optimum the likelihood is a PSD bowl -> quadratic convergence + H^-1 covariance.
    FIT_TYPE=gn_calib; ITERS=15;   MAX_NBATCH=100
    LR_SCHEDULER=constant_schedule
    EXTRA_ARGS="--set_init_params Ab 0.8348568677902222 eField 0.5210036635398865 tran_diff 1.0434385330881923e-05 long_diff 6.1161390476627275e-06 lifetime 2243.3828125" ;;
  gn_hybrid)
    # Hybrid Adam->GN: 250-iter Adam warmup cycles with GN takeover attempts + automatic
    # fallback (stall/saturation guards). Self-discovers the earliest viable switch point.
    FIT_TYPE=gn_calib; ITERS=25;   MAX_NBATCH=100
    LR_SCHEDULER=constant_schedule
    EXTRA_ARGS="--gn_warmup_iters 250" ;;
  gn_degen)
    # Degeneracy-valley study: line scan truth->GN endpoint, 2D (lifetime,long_diff) grid,
    # Hessian flat-eigenvector rotation. Init at truth.
    FIT_TYPE=gn_calib; ITERS=1;    MAX_NBATCH=100
    LR_SCHEDULER=constant_schedule
    EXTRA_ARGS="--gn_degeneracy --set_init_params Ab 0.8348813503927325 eField 0.521518936637242 tran_diff 1.0027633760716438e-05 long_diff 5.8141822809782784e-06 lifetime 2406.4465970250712" ;;
  gn_cov)
    # Covariance at the polish endpoint: H, F, and H^-1 parameter errors (gn_validate mode).
    FIT_TYPE=gn_calib; ITERS=1;    MAX_NBATCH=100
    LR_SCHEDULER=constant_schedule
    EXTRA_ARGS="--gn_curvature ggn --gn_validate --set_init_params Ab 0.829012393951416 eField 0.5215992331504822 tran_diff 1.0320958608645014e-05 long_diff 6.558915629284456e-06 lifetime 3310.435546875" ;;
  adam)
    FIT_TYPE=chain;    ITERS=3000; MAX_NBATCH=100
    LR_SCHEDULER=warmup_exponential_decay_schedule ;;
  *) echo "Unknown GNMODE=${GNMODE}"; exit 1;;
esac

TEST_NAME=gn_compare
LABEL=${GNMODE}${LABEL_SUFFIX:-}_b${MAX_NBATCH}_len${MAX_BATCH_LEN}
mkdir -p /sdf/group/neutrino/pgranger/larnd-sim-jax/logs/gn_compare

echo "=== Job started: $(date) ==="
echo "=== GNMODE=${GNMODE} FIT_TYPE=${FIT_TYPE} ITERS=${ITERS} NBATCH=${MAX_NBATCH} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${SIF_FILE} /bin/bash -c "
cd /sdf/group/neutrino/pgranger/larnd-sim-jax
export PYTHONPATH=\$PWD/src:\$PWD:\$PYTHONPATH
export JAX_COMPILATION_CACHE_DIR=${JAX_CACHE_DIR}

python3 -m optimize.example_run \
    --params ${PARAMS} \
    --input_file_sim ${TRUE} --input_file_tgt ${TRUE} \
    --fit_type ${FIT_TYPE} \
    --iterations ${ITERS} \
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
    ${EXTRA_ARGS} \
    --out_label ${LABEL} \
    --test_name ${TEST_NAME} \
    --save_freq 50
"

echo "=== Job finished: $(date) ==="
