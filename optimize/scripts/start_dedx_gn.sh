#!/bin/bash
#SBATCH --partition=turing
#SBATCH --account=mli:cider-ml
#SBATCH --job-name=dedx_gn
#SBATCH --output=/sdf/group/neutrino/pgranger/larnd-sim-jax/logs/dedx_gn/job-%A_%a.out
#SBATCH --ntasks=1 --cpus-per-task=2 --mem-per-cpu=8g
#SBATCH --gpus-per-node=geforce_rtx_2080_ti:1 --time=04:00:00
#
# STRUCTURED NEWTON (Gauss-Newton) FOR THE dEdx BLOCK.
#
#   H = band(bandwidth 2) + q q^T,  q = sqrt(2 w_mean) * l_i exp(x_i)/L
#   band  <- 5 HVPs via mod-5 probing (cost INDEPENDENT of N)
#   solve <- banded LU + one Sherman-Morrison correction, LM-damped, with backtracking
#
# Calibration is held at TRUTH (init == target) so this isolates the dEdx block: the only
# thing being optimised is the 376-ish per-segment log_dedx array.
#
# Env:
#   DEDXOPT   adam | gn                (default gn)
#   GNVAL     1 -> validate the band against the DENSE Hessian on the first step (small N only)
#   MEANW     dedx_mean_constraint_weight (default 3000 = the measured-honest value)
#   ITERS, NB, LEN, FITSEED
set -e

DEDXOPT=${DEDXOPT:-gn}
MEANW=${MEANW:-3000.0}
ITERS=${ITERS:-2000}
NB=${NB:-40}
LEN=${LEN:-60}
FITSEED=${FITSEED:-0}
GNVAL=${GNVAL:-0}
GNREFRESH=${GNREFRESH:-1}
GNCHUNK=${GNCHUNK:-1}
GNLAM0=${GNLAM0:-1.0}
GNCAP=${GNCAP:-0.5}
GNBT=${GNBT:-4}
TAG="${DEDXOPT}_mw${MEANW}_r${GNREFRESH}_s${FITSEED}"

TGT="Ab 0.83488 eField 0.52152 tran_diff 1.0028e-05 long_diff 5.8142e-06 lifetime 2406.4"
TGTFILE=/sdf/data/neutrino/cyifan/diffsim_input/true_through_muon_edep_10cm_vol1cm.h5
mkdir -p /sdf/group/neutrino/pgranger/larnd-sim-jax/logs/dedx_gn

echo "=== dEdx optimiser=${DEDXOPT}  mean_w=${MEANW}  iters=${ITERS}  seed=${FITSEED}  refresh=${GNREFRESH}  validate=${GNVAL} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
GPUMEM_LOG=/tmp/gpumem.${SLURM_JOB_ID:-$$}.txt
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 5; done > "$GPUMEM_LOG" 2>/dev/null ) &
MEMSAMPLER=$!

apptainer exec --nv -B /sdf,/fs,/lscratch larndsim-jax_main.sif /bin/bash -c "
cd /sdf/group/neutrino/pgranger/larnd-sim-jax
export PYTHONPATH=\$PWD/src:\$PWD:\$PYTHONPATH
export JAX_COMPILATION_CACHE_DIR=/sdf/group/neutrino/pgranger/.jax_cache
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export LARND_DEDX_OPT=${DEDXOPT}
export LARND_DEDX_GN_VALIDATE=${GNVAL}
export LARND_DEDX_GN_LAMBDA0=${GNLAM0}
export LARND_DEDX_GN_STEP_CAP=${GNCAP}
export LARND_DEDX_GN_MAXBT=${GNBT}
export LARND_DEDX_GN_CHUNK=${GNCHUNK}
export LARND_DEDX_GN_REFRESH=${GNREFRESH}
echo \"[ENV-INSIDE] opt=\$LARND_DEDX_OPT refresh=\$LARND_DEDX_GN_REFRESH chunk=\$LARND_DEDX_GN_CHUNK lam0=\$LARND_DEDX_GN_LAMBDA0 cap=\$LARND_DEDX_GN_STEP_CAP validate=\$LARND_DEDX_GN_VALIDATE\"

python3 -m optimize.example_run \
  --params optimize/scripts/param_list_nd_noshift.yaml \
  --input_file_sim ${TGTFILE} --input_file_tgt ${TGTFILE} \
  --fit_type chain \
  --iterations ${ITERS} \
  --max_nbatch ${NB} --max_batch_len ${LEN} \
  --data_seed 1 --seed ${FITSEED} \
  --set_target_vals ${TGT} --set_init_params ${TGT} \
  --lr 1e-1 --optimizer_fn Adam \
  --lr_scheduler warmup_exponential_decay_schedule \
  --lr_kw '{\"decay_rate\":0.999,\"init_value\":0,\"warmup_steps\":100}' \
  --max_clip_norm_val 100 --electron_sampling_resolution 0.01 --number_pix_neighbors 4 \
  --signal_length 150 --mode lut --lut_file src/larndsim/detector_properties/response_44.npy \
  --loss_fn llhd --probabilistic_sim --sim_seed_strategy different --non_deterministic \
  --fit_dedx \
  --dedx_lr 1e-2 --dedx_start_iter 0 --dedx_freeze_iter 100000 \
  --dedx_prior_weight 0.5 \
  --dedx_use_split_t True \
  --dedx_student_loc 1.874 \
  --dedx_student_nu_l 4.785 --dedx_student_nu_r 2.073 \
  --dedx_student_scale_l 0.1204 --dedx_student_scale_r 0.1058 \
  --dedx_soft_barrier_threshold 8.5 --dedx_soft_barrier_weight 1.0 \
  --dedx_mean_constraint_weight ${MEANW} \
  --dedx_drift_profile_weight 0.0 \
  --out_label dedxgn_${TAG} --test_name dedx_gn --save_freq 500
" 2>&1 | grep -vE "UserWarning|warnings.warn" | tail -200

kill $MEMSAMPLER 2>/dev/null || true
echo "[GPU-PEAK-SMI] $(sort -n "$GPUMEM_LOG" 2>/dev/null | tail -1) MiB"
rm -f "$GPUMEM_LOG"
echo "=== done (${TAG}) ==="
