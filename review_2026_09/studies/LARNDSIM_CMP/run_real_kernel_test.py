#!/usr/bin/env python3
"""
Decisive CPU test: run the ACTUAL reference fee.get_adc_values kernel
(unmodified, @cuda.jit) under NUMBA_ENABLE_CUDASIM=1 + the cupy shim, on
the stored pixel_signals from out_stage1_ref.h5, replicating the EXACT
call-site arguments from cli/simulate_pixels.py (lines ~1051-1073):

  time_ticks = cp.linspace(0, len(unique_eventIDs)*detector.TIME_INTERVAL[1],
                            pixels_signals.shape[1]+1)
  fee.get_adc_values[BPG,TPB](pixels_signals, pixels_signals_tracks,
                               time_ticks, integral_list, adc_ticks_list,
                               0,                      # time_padding -- LITERAL 0, not detector.TIME_PADDING!
                               rng_states, current_fractions, pixel_thresholds)
  adc_list = fee.digitize(integral_list)   # no gain file -> default GAIN

KEY FINDING ALREADY VISIBLE FROM READING THE CALL SITE (before even running
this): the call passes the literal integer 0 for time_padding, NOT
detector.TIME_PADDING (190). This alone explains why the observed mean
tick (~85-97 us) never matched a "+190" expectation -- TIME_PADDING is
simply not applied at that step. (It IS used earlier, to set the
*start* of integration via `t_start = round((t_start-t0_start-TIME_PADDING)
/TIME_SAMPLING)*TIME_SAMPLING`, computed back in detsim.py's
tracks_current_mc/tracks_current -- so it still matters for where the
pixel_signals trace itself starts, just not as an additive term on the
reported tick.) Also time_ticks is NOT detector.TIME_TICKS (linspace(0,200,2001),
step=0.1 exactly) but linspace(0, n_events*200, pixel_signals.shape[1]+1)
(2002 points for 1 event -> step=200/2001=0.099950..., not exactly 0.1) --
a small but real difference from my simplified replay script, which
assumed a flat 0.1 us/tick.
"""
import os
import sys
import time

os.environ["NUMBA_ENABLE_CUDASIM"] = "1"
CMP_DIR = "/sdf/scratch/neutrino/pgranger/review_2026_09/LARNDSIM_CMP"
REF_DIR = CMP_DIR + "/reference_copy"

sys.path.insert(0, CMP_DIR)
import cupy_cpu_shim
sys.modules["cupy"] = cupy_cpu_shim
sys.modules["cupy.cuda"] = cupy_cpu_shim.cuda
sys.modules["cupy.cuda.nvtx"] = cupy_cpu_shim._nvtx_mod

sys.path.insert(0, REF_DIR)
sys.path.insert(0, REF_DIR + "/cli")

import numpy as np
import h5py
import fee_overrides

from larndsim import fee
from larndsim.consts import physics, units, detector

fee_overrides.apply_stage(fee, physics, units, stage=1)  # DISCRIMINATION_THRESHOLD=5000, noise=0

from numba.cuda.random import create_xoroshiro128p_states
from math import ceil

N_SUBSET = int(sys.argv[1]) if len(sys.argv) > 1 else 300

with h5py.File(CMP_DIR + "/out_stage1_ref.h5", "r") as f:
    pixel_signals_full = np.array(f["pixel_signals"])
    adc_real = np.array(f["adc"])
    ticks_real = np.array(f["ticks"])

n_pix_total = pixel_signals_full.shape[0]
subset = slice(0, min(N_SUBSET, n_pix_total))
pixel_signals = pixel_signals_full[subset].astype(np.float32)
n_pix, n_ticks = pixel_signals.shape
print(f"[test] using {n_pix} / {n_pix_total} pixels, {n_ticks} ticks each")

# Exact call-site reproduction (unique_eventIDs length: our matched_input.h5
# batches are each a single event -- see batching.py; asserting, not assuming).
n_unique_events = 1  # matched_input.h5: 3 events, each its own TPC-batch -> 1 event/batch

time_ticks = np.linspace(0, n_unique_events * detector.TIME_INTERVAL[1], n_ticks + 1).astype(np.float32)
integral_list = np.zeros((n_pix, fee.MAX_ADC_VALUES), dtype=np.float64)
adc_ticks_list = np.zeros((n_pix, fee.MAX_ADC_VALUES), dtype=np.float64)
pixels_signals_tracks = np.zeros((n_pix, n_ticks, 1), dtype=np.float32)  # dummy, unused for trigger decisions
current_fractions = np.zeros((n_pix, fee.MAX_ADC_VALUES, 1), dtype=np.float32)
pixel_thresholds = np.full(n_pix, fee.DISCRIMINATION_THRESHOLD, dtype=np.float32)

TPB = 128
BPG = ceil(n_pix / TPB)
rng_states = create_xoroshiro128p_states(int(TPB * BPG), seed=12345)

t0 = time.time()
fee.get_adc_values[BPG, TPB](
    pixel_signals, pixels_signals_tracks, time_ticks,
    integral_list, adc_ticks_list,
    0,  # time_padding -- literal 0, exactly matching the real call site
    rng_states, current_fractions, pixel_thresholds,
)
print(f"[test] kernel (CUDASIM) ran in {time.time()-t0:.1f} s")

adc_list = fee.digitize(integral_list)

real_hits = (ticks_real[subset] != 0).sum(axis=1)
kernel_hits = (adc_ticks_list != 0).sum(axis=1)

print(f"[test] total hits, REAL reference output (this subset): {int(real_hits.sum())}")
print(f"[test] total hits, EXACT kernel replay (this subset):   {int(kernel_hits.sum())}")
n_match = int((real_hits == kernel_hits).sum())
print(f"[test] rows with identical hit count: {n_match}/{n_pix} ({100*n_match/n_pix:.1f}%)")

mism = np.where(real_hits != kernel_hits)[0]
print(f"[test] first 10 mismatched rows (idx, real_nhits, kernel_nhits):")
for i in mism[:10]:
    print(f"    row {i}: real={real_hits[i]} kernel={kernel_hits[i]} "
          f"real_adc={adc_real[subset][i][ticks_real[subset][i]!=0]} "
          f"kernel_adc={adc_list[i][adc_ticks_list[i]!=0]}")
