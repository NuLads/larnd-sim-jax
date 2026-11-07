#!/bin/bash
# Updated by Claude (9 Oct, 3rd pass) after SLURM run 40317652 (matched
# input, JAX now really on GPU) showed the real bug: JAX wasn't chopping
# segments into electron_sampling_resolution=0.01cm steps before
# simulate_wfs. See run_jax_direct.py / analyze_comparison.py docstrings
# for the full diagnosis of every fix below; summary:
#   1. load_batch() now passes chopped=True, pad=True (was chopped=False,
#      pad=False) -- matches paper_common.load()'s TracksDataset call
#      exactly. This was THE main bug: without it each ~3.3cm segment was
#      one point charge, touching only 66 "pixels" (one per segment) with
#      2.5-3 unrealistically-high-ADC hits each, instead of spreading
#      charge continuously and touching ~1500 pixels like the reference.
#   2. Timing: warm-up now runs on the SAME input/shape as the timed call
#      (was tiny_input.h5, a different padded shape -> the previous
#      "timed" 13.8s was mostly a fresh recompile, not steady state).
#      Each of simulate_wfs/simulate_stochastic/simulate_probabilistic is
#      now called twice back-to-back (1st discarded, 2nd timed).
#   3. Memory: now prints jax.devices()[0].memory_stats()['peak_bytes_in_use']
#      (the number to quote) in addition to nvidia-smi (which mostly shows
#      JAX's ~75% GPU preallocation, not actual use). Reference side now
#      also reports the cupy memory pool's used_bytes/total_bytes and the
#      nvidia-smi delta over the warm-up baseline.
#   4. Pixel matching transform was WRONG in the previous pass (a y-offset
#      of -21.8236cm derived from comparing bare geometry objects, not
#      real data) -- re-derived empirically on a real CPU run (see the
#      task report): x needs -pixel_pitch/2, y needs +0.1247cm (exact to
#      1e-6cm on 25 independently matched points). analyze_comparison.py
#      now uses this and restricts tick-mean comparisons to the matched-
#      pixel subset (global means mix genuinely different physical hits
#      and are not comparable -- CPU-validated: matched-pixel tick means
#      now agree to <1%, see task report).
#   5. Tick units: JAX 'ticks' is a raw bin index; convert via
#      ticks_us = ticks_raw * t_sampling(0.1) - 1.5 (empirically confirmed,
#      not fully source-traced -- see analyze_comparison.py docstring).
#
# Reference outputs (out_stage{1,2,3}_ref.h5) do NOT need rerunning --
# the active-volume cut is config-independent and matched_input.h5 is
# derived from them already; this script reruns the reference ONLY if
# RERUN_REFERENCE=1 is set in the environment, and reruns the JAX side +
# analysis unconditionally. This is the "reruns only the JAX side plus
# the analysis, and the reference only if needed" version requested.
#
# Output files (all under LARNDSIM_CMP/):
#   out_stage{1,2,3}_ref.h5         reference flat output (unchanged, kept from SLURM 40317652)
#   out_stage{1,2,3}_jax_direct.h5  run_jax_direct.py output (unique_pixels/wfs/adc/ticks/...)
#   log_stage{1,2,3}_jax_direct.txt JAX per-stage timing/memory log (stochastic + probabilistic)
#   analysis_output/stage{1,2,3}_comparison.png, analysis_output/metrics.json
#
#SBATCH --job-name=larndsim_cmp
#SBATCH --partition=ampere
#SBATCH --account=neutrino:dune-ml
#SBATCH --exclude=sdfampere038,sdfampere003
#SBATCH --gpus-per-node=a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32g
#SBATCH --time=01:30:00
#SBATCH --output=/sdf/scratch/neutrino/pgranger/review_2026_09/LARNDSIM_CMP/slurm-%j.out

set -euo pipefail

CMP_DIR=/sdf/scratch/neutrino/pgranger/review_2026_09/LARNDSIM_CMP
REF_DIR=$CMP_DIR/reference_copy
REF_SIF=/sdf/group/neutrino/images/develop.sif
JAX_SIF=/sdf/group/neutrino/pgranger/larnd-sim-jax/larndsim-jax_main.sif
JAX_REPO=/sdf/group/neutrino/pgranger/larnd-sim-jax

INPUT=$CMP_DIR/matched_input.h5            # 66 segments / 216.13 cm -- see make_matched_input.py
WARMUP=$CMP_DIR/tiny_input.h5              # 11 segments, 1 event -- reference-side JIT warm-up only
                                            # (JAX side now warms up on $INPUT itself, see above)
RERUN_REFERENCE=${RERUN_REFERENCE:-0}      # set RERUN_REFERENCE=1 to also redo the reference stages

cd $CMP_DIR

echo "### Sanity: matched_input.h5 track length ###"
/sdf/group/neutrino/pgranger/conda/bin/python -c "
import h5py, numpy as np
with h5py.File('$INPUT') as f:
    print('matched_input.h5:', len(f['segments']), 'segments,', f['segments']['dx'][:].sum(), 'cm')
"

if [ "$RERUN_REFERENCE" = "1" ]; then
  echo "### Reference stages (develop.sif container, GPU) -- RERUN_REFERENCE=1 ###"
  for stage in 1 2 3; do
    echo "--- reference stage $stage ---"
    apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch "$REF_SIF" /bin/bash -c "
      cd $REF_DIR && \
      export SKIP_CUPY_INSTALL=1 PYTHONNOUSERSITE=1 PYTHONPATH=$REF_DIR:$CMP_DIR/pydeps && \
      python3 $CMP_DIR/run_reference_stage.py \
          --reference-dir $REF_DIR \
          --stage $stage \
          --input $INPUT \
          --warmup-input $WARMUP \
          --output $CMP_DIR/out_stage${stage}_ref.h5
    " > $CMP_DIR/log_stage${stage}_ref.txt 2>&1 || echo "  [FAILED] see log_stage${stage}_ref.txt"
    grep -E "Simulated|Initial number of tracks|perf\]|mem\]" $CMP_DIR/log_stage${stage}_ref.txt || true
  done
else
  echo "### Reference stages: SKIPPED (RERUN_REFERENCE=0, existing out_stage{1,2,3}_ref.h5 are valid) ###"
  for stage in 1 2 3; do
    if [ ! -f "$CMP_DIR/out_stage${stage}_ref.h5" ]; then
      echo "  [ERROR] out_stage${stage}_ref.h5 missing -- set RERUN_REFERENCE=1"
      exit 1
    fi
  done
fi

echo "### larnd-sim-jax stages (larndsim-jax_main.sif container, GPU) ###"
for stage in 1 2 3; do
  echo "--- jax stage $stage ---"
  apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch "$JAX_SIF" /bin/bash -c "
    python3 $CMP_DIR/run_jax_direct.py \
        --stage $stage \
        --input $INPUT \
        --output $CMP_DIR/out_stage${stage}_jax_direct.h5 \
        --max-batch-len 1000
  " > $CMP_DIR/log_stage${stage}_jax_direct.txt 2>&1 || echo "  [FAILED] see log_stage${stage}_jax_direct.txt"
  grep -E "\[data\]|\[timing\]|\[perf\]|\[jax\]|\[mem\]" $CMP_DIR/log_stage${stage}_jax_direct.txt || true
done

echo "### CPU analysis (plain conda python, CPU-only, no GPU/container needed) ###"
/sdf/group/neutrino/pgranger/conda/bin/python $CMP_DIR/analyze_comparison.py \
    --cmp-dir $CMP_DIR --ref-dir $REF_DIR --stages 1 2 3 --jax-suffix jax_direct

echo "Done. See $CMP_DIR/analysis_output/"
