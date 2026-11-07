#!/bin/bash
#SBATCH --partition=turing
#SBATCH --account=neutrino:cider-nu
#SBATCH --job-name=dedx_bias
#SBATCH --output=/sdf/group/neutrino/pgranger/larnd-sim-jax/logs/dedx_bias/job-%A_%a.out
#SBATCH --ntasks=1 --cpus-per-task=2 --mem-per-cpu=32g
#SBATCH --gpus-per-node=geforce_rtx_2080_ti:1 --time=04:00:00
#SBATCH --array=0-11
#
# dEdx MODE-vs-MEAN BIAS TOY.
#
# Question: the split-t prior's MODE is 1.8740 (= DEDX_SPLIT_LOC, where MAP shrinkage pulls)
# but its MEAN is 1.8847 (~= dedx_mean_constraint_target 1.887). MAP therefore biases the
# fitted mean LOW, and dedx_mean_constraint_weight=1e5 exists to drag it back -- at a cost of
# kappa 24.6 -> 23874 and a rank-1 dense coupling across every segment.
#
# This scans the two candidate root-cause fixes against the existing patch:
#   LOC   1.874  (mode, current)      vs  1.8847 (prior's own mean)
#   JAC   0      (current)            vs  1      (proper prior in log space, +Jacobian)
#   MEANW 1e5    (current)  /  1e3  /  0
# 2 x 2 x 3 = 12 arms. TRUE geometry, no position fitting, 5 calibration params + dEdx free.
#
# Measured downstream from the checkpoint (analyze_dedx_bias_toy.py):
#   fitted dEdx mean vs TRUE mean (the bias), dispersion vs truth (the over-concentration),
#   dEdx MAE, and the 5 calibration parameter errors.
#
# NOTE loc also sets the dEdx initialisation (fit_params.py: log_dedx init = log(loc)), so the
# LOC arms start at their own prior centre. That is the natural pairing, not a confound to
# remove -- but it does mean "LOC" changes shrinkage target AND init together.
set -e

LOCS=(1.874 1.8847)
JACS=(0 1)
MEANWS=(100000.0 3000.0 0.0)   # 3000 = the statistically honest weight at 400cm/batch (measured)

I=${SLURM_ARRAY_TASK_ID:-0}
IL=$(( I / 6 ))          # 0,1
IJ=$(( (I / 3) % 2 ))    # 0,1
IM=$(( I % 3 ))          # 0,1,2
LOC=${LOCS[$IL]}
JAC=${JACS[$IJ]}
MEANW=${MEANWS[$IM]}
FITSEED=${FITSEED:-0}
TAG="loc${LOC}_jac${JAC}_mw${MEANW}_off${CALOFF:-0}_s${FITSEED}"

# Calibration target = the seed-0 truth draw. CALOFF sets the init offset:
#   CALOFF=0     init == truth. Use this for the PRIMARY bias measurement -- calibration error
#                would otherwise be absorbed into dEdx and contaminate the fitted mean.
#   CALOFF=0.02  2% offset, for asking whether the mean constraint is needed to RECOVER
#                calibration. Stay well under ~0.075 or Ab clips (see start_gn_toy.sh).
CALOFF=${CALOFF:-0}
TGT="Ab 0.83488 eField 0.52152 tran_diff 1.0028e-05 long_diff 5.8142e-06 lifetime 2406.4"
INIT=$(python3 -c "
f=float('${CALOFF}')
v={'Ab':0.83488,'eField':0.52152,'tran_diff':1.0028e-05,'long_diff':5.8142e-06,'lifetime':2406.4}
rng={'Ab':(0.75,0.9),'eField':(0.4,0.6),'tran_diff':(3e-6,15e-6),'long_diff':(1e-6,1e-5),'lifetime':(400,6000)}
print(' '.join('%s %.6g'%(k, min(max(x*(1+f), rng[k][0]), rng[k][1])) for k,x in v.items()))")

TGTFILE=/sdf/data/neutrino/cyifan/diffsim_input/true_through_muon_edep_10cm_vol1cm.h5
mkdir -p /sdf/group/neutrino/pgranger/larnd-sim-jax/logs/dedx_bias

echo "=== ARM ${I}: loc=${LOC} jacobian=${JAC} mean_w=${MEANW} ==="
echo "target: $TGT"
echo "init  : $INIT"
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
export LARND_DEDX_LOG_JACOBIAN=${JAC}
echo \"[TOY-ENV] LARND_DEDX_LOG_JACOBIAN=\$LARND_DEDX_LOG_JACOBIAN loc=${LOC} mean_w=${MEANW}\"

python3 -m optimize.example_run \
  --params optimize/scripts/param_list_nd_noshift.yaml \
  --input_file_sim ${TGTFILE} --input_file_tgt ${TGTFILE} \
  --fit_type chain \
  --iterations ${ITERS:-600} \
  --max_nbatch ${NB:-2} --max_batch_len ${LEN:-60} \
  --data_seed 1 --seed ${FITSEED} \
  --set_target_vals ${TGT} --set_init_params ${INIT} \
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
  --dedx_student_loc ${LOC} \
  --dedx_student_nu_l 4.785 --dedx_student_nu_r 2.073 \
  --dedx_student_scale_l 0.1204 --dedx_student_scale_r 0.1058 \
  --dedx_soft_barrier_threshold 8.5 --dedx_soft_barrier_weight 1.0 \
  --dedx_mean_constraint_weight ${MEANW} \
  --dedx_drift_profile_weight 0.0 \
  --out_label dedxbias_${TAG} --test_name dedx_bias --save_freq 300
" 2>&1 | grep -vE "UserWarning|warnings.warn" | tail -120

kill $MEMSAMPLER 2>/dev/null || true
echo "[GPU-PEAK-SMI] $(sort -n "$GPUMEM_LOG" 2>/dev/null | tail -1) MiB"
rm -f "$GPUMEM_LOG"
echo "=== done arm ${I} (${TAG}) ==="
