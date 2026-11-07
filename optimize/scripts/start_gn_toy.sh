#!/bin/bash
#SBATCH --partition=turing
#SBATCH --account=neutrino:cider-nu
#SBATCH --job-name=gnB_toy
#SBATCH --output=/sdf/group/neutrino/pgranger/larnd-sim-jax/logs/gn_toy/job-%A_%a.out
#SBATCH --ntasks=1 --cpus-per-task=2 --mem-per-cpu=32g
#SBATCH --gpus-per-node=geforce_rtx_2080_ti:1 --time=04:00:00
#SBATCH --array=0-3
# GN TOY: pure 5-parameter calibration (truth positions, truth dEdx, no nuisances).
# Explicit target; init placed a controlled distance away. Maps GN's basin of convergence.
set -e
# MULTI-BATCH NOTE (dataio.py:336 `valid_traj_mask = traj_len <= max_batch_len`):
# trajectories LONGER than max_batch_len are SILENTLY DISCARDED. Typical trajectories in this
# file are ~48 cm, so LEN=50 throws away most of the data (500 cm loaded -> 91.6 cm kept -> only
# 2 batches) and LEN=20/30 keeps almost nothing. Larger LEN therefore yields MORE batches, not
# fewer. For a genuine multi-batch toy use LEN >= ~60 and raise NB. Batches pack whole EVENTS,
# so actual cm/batch can EXCEED LEN. Always trust the logged "number of simulation batches".
OFF=${OFFSETS:-"0.01 0.05 0.20 0.50"}
FRAC=$(echo $OFF | cut -d' ' -f$((SLURM_ARRAY_TASK_ID+1)))
# truth (seed-0 draw). Init placement is controlled by INITMODE:
#   rel        (default, legacy) init = truth*(1+FRAC) on every parameter.
#              WARNING: only valid for FRAC <~ 0.075. Ab's range is [0.75,0.9] around truth
#              0.83488, so FRAC=0.20 and FRAC=0.50 BOTH clip Ab to exactly 0.9 and eField to
#              exactly 0.6 -> the two "offsets" share a starting point for 2 of 5 params and
#              begin with those params already pinned at a bound, which trips the saturation
#              guard at iteration 0 for reasons unrelated to GN. Kept as the default so previously
#              submitted configurations keep reproducing exactly.
#   frac2bound init = truth + FRAC*(bound - truth), bound taken in the +FRAC direction (or the
#              - direction for negative FRAC). FRAC is then a fraction of the distance to each
#              parameter's OWN bound, so every parameter is perturbed comparably, FRAC=1 sits
#              exactly at the bound, and nothing ever clips. Use this to map the basin.
TGT="Ab 0.83488 eField 0.52152 tran_diff 1.0028e-05 long_diff 5.8142e-06 lifetime 2406.4"
INIT=$(INITMODE=${INITMODE:-rel} python3 -c "
import os
f=float('$FRAC'); mode=os.environ['INITMODE']
v={'Ab':0.83488,'eField':0.52152,'tran_diff':1.0028e-05,'long_diff':5.8142e-06,'lifetime':2406.4}
rng={'Ab':(0.75,0.9),'eField':(0.4,0.6),'tran_diff':(3e-6,15e-6),
     'long_diff':(1e-6,1e-5),'lifetime':(400,6000)}
out={}
for k,x in v.items():
    lo,hi=rng[k]
    if mode=='frac2bound':
        b = hi if f>=0 else lo
        out[k]= x + f*(b-x)
    else:
        out[k]= x*(1+f)
    # never emit an out-of-range init: the optimizer would silently project it and the
    # nominal offset would be a fiction (this is exactly the 'rel' trap documented above).
    out[k]=min(max(out[k],lo),hi)
print(' '.join(f'{k} {y:.6g}' for k,y in out.items()))")
# INITOVR overrides the computed init with an explicit "name value ..." string. Used to
# evaluate the FULL-DATA loss at another optimizer's endpoint: run with GNITER=1 and read the
# "GN init: loss" line, which is by construction the full-batch loss at those parameters.
if [ -n "${INITOVR:-}" ]; then INIT="${INITOVR}"; fi
echo "[TOY-INIT] mode=${INITMODE:-rel} frac=${FRAC} override=${INITOVR:+yes}"
echo "=== offset ${FRAC} ==="; echo "target: $TGT"; echo "init  : $INIT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
mkdir -p /sdf/group/neutrino/pgranger/larnd-sim-jax/logs/gn_toy
# Allocator-independent peak-memory sampler. jax's memory_stats() returns nothing under
# XLA_PYTHON_CLIENT_ALLOCATOR=platform (that is why every earlier [GN-PROF] line read
# "peak_mem 0.00 GiB"), so poll nvidia-smi as a cross-check that works either way.
GPUMEM_LOG=/tmp/gpumem.${SLURM_JOB_ID:-$$}.txt
if [ "${MEMSAMPLE:-0}" = "1" ]; then
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 2; done \
      > "$GPUMEM_LOG" 2>/dev/null ) &
  MEMSAMPLER=$!
fi
TGTFILE=/sdf/data/neutrino/cyifan/diffsim_input/true_through_muon_edep_10cm_vol1cm.h5
apptainer exec --nv -B /sdf,/fs,/lscratch larndsim-jax_main.sif /bin/bash -c "
cd /sdf/group/neutrino/pgranger/larnd-sim-jax
export PYTHONPATH=\$PWD/src:\$PWD:\$PYTHONPATH
export JAX_COMPILATION_CACHE_DIR=/sdf/group/neutrino/pgranger/.jax_cache
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=${ALLOC:-platform}
export LARND_GN_SOFT_MODES=${LARND_GN_SOFT_MODES:-0}   # value baked by outer shell
export LARND_GN_REFRESH=${LARND_GN_REFRESH:-1}
# Bound-aware stepping (all default to previous behaviour: 0 / 0 / diag / off).
export LARND_GN_BOUND_TR=${LARND_GN_BOUND_TR:-0}
export LARND_GN_STALL_PATIENCE=${LARND_GN_STALL_PATIENCE:-1}
export LARND_GN_ACTIVE_SET=${LARND_GN_ACTIVE_SET:-0}
export LARND_GN_BOUND=${LARND_GN_BOUND:-3.5}
export LARND_GN_DAMP_MODE=${LARND_GN_DAMP_MODE:-diag}
export LARND_GN_LAMBDA0_SCALE=${LARND_GN_LAMBDA0_SCALE:-0}
export LARND_GN_STEP_CAP=${LARND_GN_STEP_CAP:-0}
export LARND_GN_PROFILE=${LARND_GN_PROFILE:-0}
echo \"[TOY-ENV] soft_modes=\$LARND_GN_SOFT_MODES refresh=\$LARND_GN_REFRESH bound_tr=\$LARND_GN_BOUND_TR active_set=\$LARND_GN_ACTIVE_SET bound=\$LARND_GN_BOUND damp=\$LARND_GN_DAMP_MODE patience=\$LARND_GN_STALL_PATIENCE lam0scale=\$LARND_GN_LAMBDA0_SCALE stepcap=\$LARND_GN_STEP_CAP alloc=\$XLA_PYTHON_CLIENT_ALLOCATOR prof=\$LARND_GN_PROFILE\"
# GNVALIDATE=1 -> append --gn_validate (dumps exact-vs-GGN eig/Frobenius/cosine at the CURRENT
# params i.e. init, then exits without running the GN loop). Default off = current behaviour.
GNVAL_FLAG=\"\"
if [ \"${GNVALIDATE:-0}\" = \"1\" ]; then GNVAL_FLAG=\"--gn_validate\"; fi
# DETPROPS optionally swaps the detector-properties YAML (e.g. to bump MAX_ADC_VALUES for the
# digitisation non-smoothness test). Default empty = example_run's own default file.
DETPROPS_FLAG=\"\"
if [ -n \"${DETPROPS:-}\" ]; then DETPROPS_FLAG=\"--detector_props ${DETPROPS}\"; fi
python3 -m optimize.example_run \
  --params optimize/scripts/param_list_nd_noshift.yaml \
  --input_file_sim ${TGTFILE} --input_file_tgt ${TGTFILE} \
  --fit_type gn_calib --gn_curvature ${GNCURV:-exact} \$GNVAL_FLAG \$DETPROPS_FLAG \
  --iterations ${GNITER:-15} --max_nbatch ${NB:-8} --max_batch_len ${LEN:-50} \
  --data_seed ${DATASEED:-1} --seed ${FITSEED:-0} ${NOISEFLAG:-} \
  --set_target_vals ${TGT} --set_init_params ${INIT} \
  --lr 1e-1 --optimizer_fn Adam --lr_scheduler constant_schedule \
  --lr_kw '{\"decay_rate\":0.999,\"init_value\":0,\"warmup_steps\":500}' \
  --max_clip_norm_val 100 --electron_sampling_resolution 0.01 --number_pix_neighbors 4 \
  --signal_length 150 --mode lut --lut_file src/larndsim/detector_properties/response_44.npy \
  --loss_fn llhd ${PROBSIM:---probabilistic_sim} --sim_seed_strategy ${SEEDSTRAT:-different} --non_deterministic \
  --out_label toy${TOYTAG:+_${TOYTAG}}_sm${LARND_GN_SOFT_MODES}_off${FRAC} --test_name gn_toy --save_freq 10000
" 2>&1 | grep -vE "UserWarning|warnings.warn" | tail -400
if [ "${MEMSAMPLE:-0}" = "1" ]; then
  kill $MEMSAMPLER 2>/dev/null || true
  echo "[GPU-PEAK-SMI] $(sort -n "$GPUMEM_LOG" 2>/dev/null | tail -1) MiB  (nsamples $(wc -l < "$GPUMEM_LOG" 2>/dev/null))"
  rm -f "$GPUMEM_LOG"
fi
echo "=== done offset ${FRAC} ==="
