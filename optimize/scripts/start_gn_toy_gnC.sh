#!/bin/bash
#SBATCH --partition=turing
#SBATCH --account=neutrino:cider-nu
#SBATCH --job-name=gnC_toy
#SBATCH --output=/sdf/group/neutrino/pgranger/larnd-sim-jax/logs/gn_toy/gnC-%A_%a.out
#SBATCH --ntasks=1 --cpus-per-task=2 --mem-per-cpu=32g
#SBATCH --gpus-per-node=geforce_rtx_2080_ti:1 --time=00:30:00
#SBATCH --array=0-0
# GN CURVATURE PERF INVESTIGATION (agent gnC_): private copy of start_gn_toy.sh so it does not
# collide with other agents editing the shared script concurrently. Same toy (pure 5-parameter
# calibration, truth positions/dE/dx). Adds LARND_GN_PROFILE (timing+memory logging) and
# LARND_GN_REMAT (jax.checkpoint around the forward sim) passthrough on top of the existing
# LARND_GN_SOFT_MODES / LARND_GN_REFRESH knobs. Single offset arm by default (OFFSETS="0.05").
set -e
OFF=${OFFSETS:-"0.05"}
FRAC=$(echo $OFF | cut -d' ' -f$((SLURM_ARRAY_TASK_ID+1)))
TGT="Ab 0.83488 eField 0.52152 tran_diff 1.0028e-05 long_diff 5.8142e-06 lifetime 2406.4"
INIT=$(python3 -c "
f=float('$FRAC')
v={'Ab':0.83488,'eField':0.52152,'tran_diff':1.0028e-05,'long_diff':5.8142e-06,'lifetime':2406.4}
print(' '.join(f'{k} {x*(1+f):.6g}' for k,x in v.items()))")
echo "=== offset ${FRAC} ==="; echo "target: $TGT"; echo "init  : $INIT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
mkdir -p /sdf/group/neutrino/pgranger/larnd-sim-jax/logs/gn_toy
TGTFILE=/sdf/data/neutrino/cyifan/diffsim_input/true_through_muon_edep_10cm_vol1cm.h5
apptainer exec --nv -B /sdf,/fs,/lscratch larndsim-jax_main.sif /bin/bash -c "
cd /sdf/group/neutrino/pgranger/larnd-sim-jax
export PYTHONPATH=\$PWD/src:\$PWD:\$PYTHONPATH
export JAX_COMPILATION_CACHE_DIR=/sdf/group/neutrino/pgranger/.jax_cache
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export LARND_GN_SOFT_MODES=${LARND_GN_SOFT_MODES:-0}
export LARND_GN_REFRESH=${LARND_GN_REFRESH:-1}
export LARND_GN_PROFILE=${GNPROFILE:-1}
export LARND_GN_REMAT=${GNREMAT:-0}
export LARND_CHUNK_SCAN_REMAT=${CHUNKREMAT:-0}
echo \"[TOY-ENV] soft_modes=\$LARND_GN_SOFT_MODES refresh=\$LARND_GN_REFRESH profile=\$LARND_GN_PROFILE remat=\$LARND_GN_REMAT chunk_remat=\$LARND_CHUNK_SCAN_REMAT nb=${NB:-8} len=${LEN:-50} curv=${GNCURV:-exact}\"
python3 -m optimize.example_run \
  --params optimize/scripts/param_list_nd_noshift.yaml \
  --input_file_sim ${TGTFILE} --input_file_tgt ${TGTFILE} \
  --fit_type gn_calib --gn_curvature ${GNCURV:-exact} \
  --iterations ${GNITER:-3} --max_nbatch ${NB:-8} --max_batch_len ${LEN:-50} \
  --data_seed 1 --seed 0 \
  --set_target_vals ${TGT} --set_init_params ${INIT} \
  --lr 1e-1 --optimizer_fn Adam --lr_scheduler constant_schedule \
  --lr_kw '{\"decay_rate\":0.999,\"init_value\":0,\"warmup_steps\":500}' \
  --max_clip_norm_val 100 --electron_sampling_resolution 0.01 --number_pix_neighbors 4 \
  --signal_length 150 --mode lut --lut_file src/larndsim/detector_properties/response_44.npy \
  --loss_fn llhd --probabilistic_sim --sim_seed_strategy different --non_deterministic \
  --out_label gnC_nb${NB:-8}_len${LEN:-50}_curv${GNCURV:-exact}_ref${LARND_GN_REFRESH:-1}_remat${GNREMAT:-0}_cremat${CHUNKREMAT:-0} \
  --test_name gn_toy_gnC --save_freq 10000
" 2>&1 | grep -vE "UserWarning|warnings.warn"
echo "=== done offset ${FRAC} ==="
