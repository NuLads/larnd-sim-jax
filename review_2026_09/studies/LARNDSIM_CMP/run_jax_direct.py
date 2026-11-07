#!/usr/bin/env python3
"""
Direct JAX-side runner for the reference-vs-larnd-sim-jax comparison,
bypassing optimize/simulate.py's CLI (unmodified, tracked file) only where
needed:
  1. optimize/simulate.py's --save_wfs path writes 'wfs' but NOT the
     'unique_pixels' that index its rows, so a saved wfs array can't be
     mapped back to physical pixels from the output file alone. We save
     both here, needed for the stage-1/2 induced-current comparison.
  2. We need "probabilistic" (simulate_probabilistic) timing too, which
     optimize/simulate.py's CLI does not expose at all (--mode only
     switches between 'lut'=simulate_stochastic and
     'parametrized'=simulate_parametrized).
  3. We want warm-up-excluded timing + peak GPU memory for both forward
     passes, matching how run_reference_stage.py times the reference, and
     (fixed 2026-10-09, 2nd pass) the warm-up must run on the SAME input
     as the timed call -- array SHAPE determines JAX's JIT cache key, so a
     warm-up on tiny_input.h5 (different padded shape) triggers a fresh
     trace+compile during the "timed" call on matched_input.h5, which is
     exactly what the first GPU run's bogus 13.8s simulate_wfs number was:
     mostly recompilation, not steady-state execution.

FIXED (2026-10-09, 2nd pass), the actual main bug: `load_batch()` was
called with `chopped=False, pad=False`. Reference/production both chop
each track segment into electron_sampling_resolution=0.01 cm steps before
simulate_wfs (paper_common.py's `load()`: `TracksDataset(...,
chopped=True, pad=True, electron_sampling_resolution=0.01)`); without
chopping, every ~3.3 cm segment was projected as a single point charge,
so simulate_wfs only ever touched 66 "pixels" (one per raw segment) with
2.5-3 "hits" each at unrealistically high ADC, instead of spreading the
charge continuously along each track and touching ~1500 pixels including
neighbours, like the reference. `chopped=True, pad=True` now matches
paper_common.load() exactly; the chopped row count is printed and should
be close to track_length_cm / 0.01 (~21600 for the 216 cm matched input).

Otherwise this mirrors optimize/simulate.py's own logic closely (same
params_to_apply list, same --noise-equivalent RESET_NOISE_CHARGE/
UNCORRELATED_NOISE_CHARGE handling) so results should be directly
comparable to a "real" optimize/simulate.py run run with --chop, just
consolidated into one script with the additions above.

Usage (inside larndsim-jax_main.sif, with --nv, on a GPU node -- this
WILL run on CPU too for correctness-only checks if $JAX_PLATFORMS=cpu,
just very slowly, especially with chopping active: ~21600 rows instead
of 66):

  python3 run_jax_direct.py --stage 2 --input matched_input.h5 \
      --output out_stage2_jax_direct.h5
"""
import argparse
import os
import sys
import time

REPO = "/sdf/group/neutrino/pgranger/larnd-sim-jax"
CMP_DIR = "/sdf/scratch/neutrino/pgranger/review_2026_09/LARNDSIM_CMP"


def gpu_mem_used_mib():
    import subprocess
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        return [int(x) for x in out.decode().strip().splitlines()]
    except Exception as e:
        return f"nvidia-smi unavailable: {e}"


def jax_mem_stats():
    try:
        import jax
        stats = jax.devices()[0].memory_stats()
        return stats if stats else {}
    except Exception as e:
        return {"error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True, choices=[1, 2, 3])
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-batch-len", type=float, default=1000.0,
                     help="must exceed the longest single trajectory in "
                          "--input or dataio.py silently drops it (see "
                          "make_matched_input.py docstring)")
    args = ap.parse_args()

    sys.path.insert(0, REPO + "/src")
    sys.path.insert(0, REPO)
    os.chdir(REPO)

    import jax
    import jax.numpy as jnp
    import h5py
    import numpy as np
    from larndsim.consts_jax import build_params_class, load_detector_properties, load_lut
    from larndsim.sim_jax import simulate_wfs, simulate_stochastic, simulate_probabilistic, pad_size
    from optimize.dataio import TracksDataset

    print(f"[jax] backend = {jax.default_backend()}, devices = {jax.devices()}")
    print(f"[mem] JAX memory_stats at start: {jax_mem_stats()}")

    detprops = {
        1: CMP_DIR + "/configs/module0_stage1_jax.yaml",
        2: CMP_DIR + "/configs/module0_truth_jax.yaml",
        3: CMP_DIR + "/configs/module0_truth_jax.yaml",
    }[args.stage]

    Params = build_params_class([])
    ref_params = load_detector_properties(
        Params, detprops, "src/larndsim/pixel_layouts/multi_tile_layout-2.4.16_v4.yaml")
    response, ref_params = load_lut("src/larndsim/detector_properties/response_44.npy", ref_params)
    # Match paper_common.py's params() exactly: electron_sampling_resolution,
    # number_pix_neighbors, signal_length, diffusion_in_current_sim=False,
    # mc_diff=False, use_dedx_density=False, dedx_density_mode='histogram'.
    ref_params = ref_params.replace(
        diffusion_in_current_sim=False,
        mc_diff=False,
        electron_sampling_resolution=0.01,
        number_pix_neighbors=4,
        signal_length=150,
        use_dedx_density=False,
        dedx_density_mode="histogram",
        time_window=150,
    )
    if args.stage in (1, 2):
        # NOT exactly 0: simulate_probabilistic's noise-marginalization step
        # (fee_jax.py:_find_one_hit_step, z_scale = 1.0/sigma) divides by the
        # noise sigma and raises ZeroDivisionError at RESET_NOISE_CHARGE=
        # UNCORRELATED_NOISE_CHARGE=0 (verified by hand on CPU). 1.0 e- is
        # ~5-6 orders of magnitude below any real signal here (hits are
        # O(1e4-1e5) e-), so it is noise-off in every physical sense while
        # keeping the probabilistic path well-defined. simulate_stochastic
        # has no such issue and was NOT changed.
        ref_params = ref_params.replace(RESET_NOISE_CHARGE=1.0, UNCORRELATED_NOISE_CHARGE=1.0)
        print("[config] stage 1/2: RESET_NOISE_CHARGE/UNCORRELATED_NOISE_CHARGE set to 1.0 e- "
              "(not exactly 0) to avoid a ZeroDivisionError in simulate_probabilistic")
    # stage 3: leave class defaults (900, 500), matching TRUTH reset/uncorrelated noise.

    def load_batch(path):
        dataset = TracksDataset(
            filename=path, nevents=None, max_nbatch=None, swap_xz=True,
            random_nevents=False, data_seed=42, max_batch_len=args.max_batch_len,
            print_input=False,
            chopped=True, pad=True,  # see module docstring -- this was the main bug
            electron_sampling_resolution=0.01, live_selection=False,
            use_dedx_density=False, dedx_density_mode="histogram",
        )
        fields = dataset.get_track_fields()
        assert len(dataset) >= 1, f"no batches built from {path}"
        if len(dataset) > 1:
            print(f"[warn] {path} split into {len(dataset)} batches with "
                  f"max_batch_len={args.max_batch_len}; only batch 0 is used "
                  f"by this script -- raise --max-batch-len if this matters.")
        batch = dataset[0]
        size = pad_size(batch.shape[0], "batch_size", 0.5)
        batch = dataset.pad_batch(batch, size, 0)
        total_cm = float(np.asarray(batch)[:, fields.index("dx")].sum())
        n_nonzero_dx = int((np.asarray(batch)[:, fields.index("dx")] > 0).sum())
        print(f"[data] {path}: {n_nonzero_dx} chopped (non-pad) rows, "
              f"{batch.shape[0]} total (padded) rows, {total_cm:.3f} cm of track, "
              f"{len(dataset)} batch(es)")
        return jax.device_put(batch), fields, total_cm

    tracks, fields, total_cm = load_batch(args.input)

    def timed_twice(label, fn, *call_args):
        """Call fn(*call_args) once (warm-up, discarded) then once more
        (timed), both on the exact same input shapes -- required so the
        timed call is steady-state, not a fresh trace+compile."""
        out = fn(*call_args)
        jax.block_until_ready(out)
        t0 = time.time()
        out = fn(*call_args)
        jax.block_until_ready(out)
        dt = time.time() - t0
        print(f"[timing] stage {args.stage} {label} [warm-up excluded, same-shape 2nd call]: {dt:.3f} s")
        return out, dt

    print(f"[mem] GPU memory used before any simulate_* call (MiB): {gpu_mem_used_mib()}")

    (wfs, unique_pixels), t_wfs = timed_twice(
        "simulate_wfs (shared drift+current)",
        lambda: simulate_wfs(ref_params, response, tracks, fields))
    mem_after_wfs = gpu_mem_used_mib()

    (adcs, x, y, z, ticks, hit_prob, event, hit_pixels), t_stoch = timed_twice(
        "simulate_stochastic",
        lambda: simulate_stochastic(ref_params, wfs, unique_pixels, rngseed=args.stage))
    mem_after_stoch = gpu_mem_used_mib()

    (adcs_distrib, px, py, ticks_prob, event_p), t_prob = timed_twice(
        "simulate_probabilistic",
        lambda: simulate_probabilistic(ref_params, wfs, unique_pixels))
    mem_after_prob = gpu_mem_used_mib()

    print(f"[perf] input track length: {total_cm:.3f} cm")
    print(f"[perf] stage {args.stage} stochastic: {1000*(t_wfs+t_stoch)/total_cm:.3f} ms/cm "
          f"({t_wfs+t_stoch:.3f} s / {total_cm:.1f} cm, wfs+stochastic)")
    print(f"[perf] stage {args.stage} probabilistic: {1000*(t_wfs+t_prob)/total_cm:.3f} ms/cm "
          f"({t_wfs+t_prob:.3f} s / {total_cm:.1f} cm, wfs+probabilistic)")
    print(f"[mem] GPU memory used (nvidia-smi, includes JAX's ~75% preallocation) "
          f"after wfs/stochastic/probabilistic (MiB): {mem_after_wfs} / {mem_after_stoch} / {mem_after_prob}")
    mstats = jax_mem_stats()
    print(f"[mem] JAX memory_stats after all calls: {mstats}")
    if isinstance(mstats, dict) and "peak_bytes_in_use" in mstats:
        print(f"[mem] JAX peak_bytes_in_use: {mstats['peak_bytes_in_use']} bytes "
              f"({mstats['peak_bytes_in_use']/1e9:.3f} GB) -- this is the number to quote, "
              f"NOT nvidia-smi (JAX preallocates ~75% of the GPU by default so nvidia-smi "
              f"mostly reports that preallocation, not actual use).")

    # Decode unique_pixels -> physical coordinates with JAX's own id2pixel +
    # get_pixel_coordinates, for the stage-1/2 induced-current comparison.
    from larndsim.detsim_jax import id2pixel, get_pixel_coordinates
    pix_x_idx, pix_y_idx, pix_plane, pix_event = id2pixel(ref_params, unique_pixels)
    pix_xy = get_pixel_coordinates(ref_params, pix_x_idx, pix_y_idx, pix_plane)

    with h5py.File(args.output, "w") as f:
        f.create_dataset("unique_pixels", data=np.asarray(unique_pixels))
        f.create_dataset("wfs", data=np.asarray(wfs))
        f.create_dataset("pix_x_decoded", data=np.asarray(pix_xy[:, 0]))
        f.create_dataset("pix_y_decoded", data=np.asarray(pix_xy[:, 1]))
        f.create_dataset("pix_plane_decoded", data=np.asarray(pix_plane))
        f.create_dataset("pix_event_decoded", data=np.asarray(pix_event))
        # stochastic (sampled) hits -- 'ticks' here is a RAW bin index into
        # wfs's time axis, NOT yet in physical us: multiply by
        # ref_params.t_sampling (0.1 us) to convert, see analyze_comparison.py.
        f.create_dataset("adc", data=np.asarray(adcs))
        f.create_dataset("pix_x", data=np.asarray(x))
        f.create_dataset("pix_y", data=np.asarray(y))
        f.create_dataset("pix_z", data=np.asarray(z))
        f.create_dataset("ticks", data=np.asarray(ticks))
        f.create_dataset("hit_prob", data=np.asarray(hit_prob))
        f.create_dataset("event", data=np.asarray(event))
        f.create_dataset("hit_pixels", data=np.asarray(hit_pixels))
        # probabilistic (expected-value) outputs
        f.create_dataset("adcs_distrib", data=np.asarray(adcs_distrib))
        f.create_dataset("px_prob", data=np.asarray(px))
        f.create_dataset("py_prob", data=np.asarray(py))
        f.create_dataset("ticks_prob", data=np.asarray(ticks_prob))
        f.attrs["t_sampling"] = float(ref_params.t_sampling)
    print(f"Output -> {args.output}")


if __name__ == "__main__":
    main()
