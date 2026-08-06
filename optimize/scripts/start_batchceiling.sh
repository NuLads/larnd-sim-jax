#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=neutrino:cider-nu
#SBATCH --job-name=batchceil
#SBATCH --output=/sdf/group/neutrino/pgranger/larnd-sim-jax/logs/batchceiling/job-%A.out
#SBATCH --ntasks=1 --cpus-per-task=2 --mem-per-cpu=32g
#SBATCH --gpus-per-node=a100:1 --time=06:00:00
#
# DIRECT batch-size ceiling test on the PRODUCTION configuration.
#
# WHY: the existing scaling law (peak_MiB = 3471 + 70.1*cm/batch) was fitted on the GN path with an
# exact Hessian, on an 11 GiB turing card. Production is Adam + dEdx + chain geometry on an A100,
# and measures 23.4 GiB at 400 cm/batch where that law predicts ~30.8 -- so the law is conservative
# and the real ceiling is unknown. This measures it instead of extrapolating.
#
# ALSO: production sets no allocator env, so JAX preallocates its DEFAULT 75% of the card. That is
# why every run reports bytes_limit = 29.55 GiB on a 40 GB A100 -- ~10 GiB is simply never offered
# to the process. MEMFRAC=0.95 tests how much of that is recoverable for free.
#
# Each LEN runs in its OWN python process so an OOM kills only that point, not the sweep.
# Peak memory is sampled from nvidia-smi, which works regardless of allocator (jax's memory_stats
# returns nothing under the platform allocator -- that is the "peak_mem 0.00 GiB" trap).
set -u
LENS=${LENS:-"400 600 800 1000 1200"}
NB=${CEILNB:-25}
ITERS=${CEILITERS:-25}
MEMFRAC=${MEMFRAC:-}
SIM=/sdf/group/neutrino/pgranger/lads-data/linear_guess_segments.h5
TGT=/sdf/data/neutrino/cyifan/diffsim_input/true_through_muon_edep_10cm_vol1cm.h5
mkdir -p /sdf/group/neutrino/pgranger/larnd-sim-jax/logs/batchceiling

echo "=== batch-size ceiling sweep: LENS='${LENS}' NB=${NB} ITERS=${ITERS} MEMFRAC='${MEMFRAC:-default(0.75)}' ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for LEN in ${LENS}; do
  MEMLOG=/tmp/ceil_mem_${SLURM_JOB_ID:-$$}_${LEN}.txt
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 2; done \
      > "$MEMLOG" 2>/dev/null ) &
  SAMPLER=$!
  RUNLOG=/tmp/ceil_run_${SLURM_JOB_ID:-$$}_${LEN}.log

  apptainer exec --nv -B /sdf,/fs,/lscratch larndsim-jax_main.sif /bin/bash -c "
    cd /sdf/group/neutrino/pgranger/larnd-sim-jax
    export PYTHONPATH=\$PWD/src:\$PWD:\$PYTHONPATH
    export JAX_COMPILATION_CACHE_DIR=/sdf/group/neutrino/pgranger/.jax_cache
    ${MEMFRAC:+export XLA_PYTHON_CLIENT_MEM_FRACTION=${MEMFRAC}}
    python3 -m optimize.example_run \
      --params optimize/scripts/param_list_nd_noshift.yaml \
      --input_file_sim ${SIM} --input_file_tgt ${TGT} \
      --fit_type chain --iterations ${ITERS} --max_nbatch ${NB} --max_batch_len ${LEN} \
      --data_seed 1 --seed 0 \
      --lr 1e-1 --optimizer_fn Adam --lr_scheduler exponential_decay \
      --lr_kw '{\"decay_rate\":0.91,\"init_value\":0,\"warmup_steps\":500}' \
      --max_clip_norm_val 1 --electron_sampling_resolution 0.005 --number_pix_neighbors 4 \
      --signal_length 150 --mode lut --lut_file src/larndsim/detector_properties/response_44.npy \
      --loss_fn llhd --probabilistic_sim --sim_seed_strategy same --non_deterministic \
      --fit_dedx --dedx_prior_weight 5 --dedx_mean_constraint_weight 1e5 \
      --fit_chain_positions --chain_basis spline --chain_lr 1e-2 --chain_decay_rate 0.999 \
      --mcs_prior_weight 0.5 \
      --out_label ceil_len${LEN} --test_name batchceiling --save_freq 1000000
  " > "$RUNLOG" 2>&1
  STATUS=$?
  kill $SAMPLER 2>/dev/null || true

  PEAK=$(sort -n "$MEMLOG" 2>/dev/null | tail -1)
  # realised geometry: batches pack whole EVENTS, so actual cm/batch can differ from LEN, and
  # dataio DISCARDS trajectories longer than max_batch_len -- both make LEN a request, not a fact.
  CM=$(grep -m1 "total track length" "$RUNLOG" | grep -oE "[0-9.]+" | head -1)
  NBATCH=$(grep -m1 "number of simulation batches" "$RUNLOG" | grep -oE "[0-9]+" | tail -1)
  OOM=""
  grep -qiE "RESOURCE_EXHAUSTED|out of memory|OOM|XlaRuntimeError" "$RUNLOG" && OOM="OOM"
  ITDONE=$(grep -oE "[0-9]+/${ITERS}" "$RUNLOG" | tail -1)
  echo "[CEIL] LEN=${LEN} status=${STATUS} ${OOM:-ok} peak=${PEAK:-?} MiB total_cm=${CM:-?} nbatch=${NBATCH:-?} progress=${ITDONE:-none}"
  if [ -n "$CM" ] && [ -n "$NBATCH" ] && [ "$NBATCH" -gt 0 ] 2>/dev/null; then
    echo "[CEIL]   -> realised $(python3 -c "print(f'{${CM}/${NBATCH}:.1f}')") cm/batch"
  fi
  [ -n "$OOM" ] && echo "[CEIL]   first OOM line: $(grep -m1 -iE 'RESOURCE_EXHAUSTED|out of memory' "$RUNLOG" | cut -c1-160)"
  rm -f "$MEMLOG"
done
echo "=== sweep done ==="
