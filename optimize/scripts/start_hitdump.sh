#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=neutrino:cider-nu
#SBATCH --job-name=hitdump
#SBATCH --output=/sdf/group/neutrino/pgranger/larnd-sim-jax/logs/hitdump/job-%A_%a.out
#SBATCH --ntasks=1 --cpus-per-task=2 --mem-per-cpu=32g
#SBATCH --gpus-per-node=a100:1 --time=04:00:00
#SBATCH --array=0-0
#
# Dump SIMULATED HIT LISTS (adc, tick, pixel) so the standard lifetime measurement can be redone
# on reconstructed hits instead of truth segments.
#
# WHY: the truth-level study (S6k) recovers tau to -1.2% and is exactly scale-immune, but it
# bypasses the entire front end. The real question is what the FEE does: the discrimination
# threshold (5000 e-) preferentially kills SMALL hits, and small hits are the ones that drifted
# FURTHEST and lost the most charge -- a selection that flattens the dQ/dx-vs-t slope and
# therefore BIASES the lifetime upward. Electronics noise then adds an Eddington-style bias on
# top: near threshold, upward noise fluctuations are selected in and downward ones are lost.
#
# HOW: the fitter already simulates and caches a target hit list per batch as
#   {target_dir}/batch{i}_target.npz  (adcs, pixel_x, pixel_y, pixel_z, ticks, hit_prob, event)
# generated from `target_params` on each batch's FIRST visit. Setting LARND_KEEP_TARGETS=1 stops
# them being deleted at the end of the run. A plain `--fit_type chain` run visits every batch once
# per epoch, so ITERATIONS=NB generates exactly one target per batch and stops -- a few minutes,
# rather than the ~1h a scan would take.
#
# NOISE=on  -> readout noise simulated for the target (production default)
# NOISE=off -> --no-noise-target, the control
set -e
NB=${HITNB:-60}
LEN=${HITLEN:-400}
NOISE=${NOISE:-on}
NOISEFLAG=""
TAG="noise"
# NOTE argparse declares these with HYPHENS ("--no-noise-target", dest=no_noise_target); the
# underscore spelling is silently rejected as an unrecognised argument.
if [ "$NOISE" = "off" ]; then
  NOISEFLAG="--no-noise-target"
  TAG="nonoise"
fi

TGT=/sdf/data/neutrino/cyifan/diffsim_input/true_through_muon_edep_10cm_vol1cm.h5
mkdir -p /sdf/group/neutrino/pgranger/larnd-sim-jax/logs/hitdump
echo "=== hit dump: NOISE=${NOISE} NB=${NB} LEN=${LEN} ==="
nvidia-smi --query-gpu=name --format=csv,noheader

apptainer exec --nv -B /sdf,/fs,/lscratch larndsim-jax_main.sif /bin/bash -c "
cd /sdf/group/neutrino/pgranger/larnd-sim-jax
export PYTHONPATH=\$PWD/src:\$PWD:\$PYTHONPATH
export JAX_COMPILATION_CACHE_DIR=/sdf/group/neutrino/pgranger/.jax_cache
export LARND_KEEP_TARGETS=1
python3 -m optimize.example_run \
  --params optimize/scripts/param_list_nd_noshift.yaml \
  --input_file_sim ${TGT} --input_file_tgt ${TGT} \
  --fit_type chain --iterations ${NB} --max_nbatch ${NB} --max_batch_len ${LEN} \
  --data_seed 1 --seed 0 ${NOISEFLAG} \
  --lr 1e-6 --optimizer_fn Adam --lr_scheduler constant_schedule --lr_kw '{}' \
  --max_clip_norm_val 100 --electron_sampling_resolution 0.01 --number_pix_neighbors 4 \
  --signal_length 150 --mode lut --lut_file src/larndsim/detector_properties/response_44.npy \
  --loss_fn llhd --probabilistic_sim --sim_seed_strategy different --non_deterministic \
  --out_label hitdump_${TAG} --test_name hitdump --save_freq 1000000
" 2>&1 | grep -vE "UserWarning|warnings.warn" | tail -40
echo "=== targets kept in: target_hitdump_hitdump_${TAG}_* ==="
ls -d target_hitdump* 2>/dev/null
echo "=== done ==="
