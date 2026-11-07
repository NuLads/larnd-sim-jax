#!/usr/bin/env python3
"""
Run the reference (DUNE/larnd-sim) CLI for one comparison stage, on GPU,
inside the /sdf/group/neutrino/images/develop.sif container (real numba.cuda
+ cupy; this script will NOT work on CPU -- cupy raises cudaErrorInsufficientDriver
with no GPU, verified on sdfiana007/login node).

This operates on the SCRATCH COPY of the reference checkout
(/sdf/scratch/neutrino/pgranger/review_2026_09/LARNDSIM_CMP/reference_copy),
never on the tracked ~/reference checkout.

Usage (inside the container, cwd = reference_copy):
  python3 /sdf/scratch/.../LARNDSIM_CMP/run_reference_stage.py \
      --reference-dir /sdf/scratch/.../LARNDSIM_CMP/reference_copy \
      --stage 2 \
      --input /sdf/scratch/.../LARNDSIM_CMP/batch_350cm_input.h5 \
      --output /sdf/scratch/.../LARNDSIM_CMP/out_stage2_ref.h5 \
      --warmup-input /sdf/scratch/.../LARNDSIM_CMP/tiny_input.h5

If --warmup-input is given, run_simulation() is called once on that tiny
slice first (output discarded) to absorb numba's first-call JIT compilation
cost; the timed, reported run is the second call, on --input. The warm-up
timing is printed separately and tagged "[JIT/compile, excluded]".

Config files used (matched to our JAX-side TRUTH setup -- see
LARNDSIM_CMP/configs/ and the config-mapping table in the task report):
  stage 1: configs/module0_stage1_nodiff_nonoise.yaml   + fee_overrides stage=1 (noise off)
  stage 2: configs/module0_stage23_truth.yaml            + fee_overrides stage=2 (noise off)
  stage 3: configs/module0_stage23_truth.yaml            + fee_overrides stage=3 (noise on)
  all stages: configs/singles_sim_matched.yaml, pixel_layouts/multi_tile_layout-2.4.16_v4.yaml
              (copied from larnd-sim-jax, see note below), bin/response_44.npy
              (byte-identical to our src/larndsim/detector_properties/response_44.npy),
              light_simulated=False (our JAX sim has no light model -- charge-only
              comparison).

NOTE on pixel_layout: multi_tile_layout-2.4.16_v4.yaml is OUR modified copy of
the reference's own multi_tile_layout-2.4.16.yaml (130-line diff: an added
top-level `drift_length: 304.31` key the reference loader ignores, plus a
number of pixel io-channel polarity (+1/-1) sign flips whose physical
correctness for this reference code path has NOT been independently
verified here -- flagged in the task report as an open item. Pass
--pixel-layout pointing at the reference's own unmodified
larndsim/pixel_layouts/multi_tile_layout-2.4.16.yaml instead if you want to
rule that out as a source of disagreement.
"""
import argparse
import os
import subprocess
import sys
import time

CMP_DIR = "/sdf/scratch/neutrino/pgranger/review_2026_09/LARNDSIM_CMP"


def gpu_mem_used_mib():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
        )
        return [int(x) for x in out.decode().strip().splitlines()]
    except Exception as e:
        return f"nvidia-smi unavailable: {e}"


def cupy_mempool_bytes():
    """The actual cupy allocator's current/peak usage, which is a much more
    meaningful number than nvidia-smi here: nvidia-smi's reported "used"
    memory includes the whole CUDA context (driver, cupy's memory pool
    reserve, etc.), not just the live arrays. Mirrors the coordinator's ask
    to report JAX's own peak_bytes_in_use rather than nvidia-smi for the
    JAX side -- this is the reference-side equivalent."""
    try:
        import cupy as cp
        pool = cp.get_default_memory_pool()
        return dict(used_bytes=pool.used_bytes(), total_bytes=pool.total_bytes())
    except Exception as e:
        return {"error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference-dir", default=os.path.join(CMP_DIR, "reference_copy"))
    ap.add_argument("--stage", type=int, required=True, choices=[1, 2, 3])
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--warmup-input", default=None)
    ap.add_argument("--pixel-layout", default=None,
                     help="defaults to our 2.4.16_v4 copy staged into "
                          "<reference-dir>/larndsim/pixel_layouts/")
    ap.add_argument("--rand-seed", type=int, default=12345)
    ap.add_argument("--no-align-adc-constants", action="store_true")
    args = ap.parse_args()

    ref_dir = os.path.abspath(args.reference_dir)
    sys.path.insert(0, ref_dir)
    sys.path.insert(0, os.path.join(ref_dir, "cli"))
    sys.path.insert(0, CMP_DIR)

    import fee_overrides

    t_import0 = time.time()
    import simulate_pixels  # cli/simulate_pixels.py
    from larndsim import fee
    from larndsim.consts import physics, units
    print(f"[timing] import simulate_pixels + larndsim: {time.time()-t_import0:.2f} s")

    applied = fee_overrides.apply_stage(
        fee, physics, units, args.stage,
        align_adc_constants=not args.no_align_adc_constants,
    )
    print(f"[config] stage {args.stage} constants applied: {applied}")

    # FIX (9 Oct): run_simulation() calls importlib.reload(fee), which re-executes fee.py and RESETS every override
    # (threshold back to 7000 e-, noise/dead time/ADC constants to the file values). Re-apply after every reload.
    import importlib
    _orig_reload = importlib.reload
    def _reload_and_reapply(mod):
        r = _orig_reload(mod)
        if getattr(mod, '__name__', '') == 'larndsim.fee':
            a = fee_overrides.apply_stage(mod, physics, units, args.stage,
                                          align_adc_constants=not args.no_align_adc_constants)
            print(f"[config] re-applied after importlib.reload(fee): {a}", flush=True)
        return r
    importlib.reload = _reload_and_reapply

    detprops = {
        1: os.path.join(CMP_DIR, "configs", "module0_stage1_nodiff_nonoise.yaml"),
        2: os.path.join(CMP_DIR, "configs", "module0_stage23_truth.yaml"),
        3: os.path.join(CMP_DIR, "configs", "module0_stage23_truth.yaml"),
    }[args.stage]
    simprops = os.path.join(CMP_DIR, "configs", "singles_sim_matched.yaml")
    pixel_layout = args.pixel_layout or os.path.join(
        ref_dir, "larndsim", "pixel_layouts", "multi_tile_layout-2.4.16_v4.yaml")
    response_file = os.path.join(ref_dir, "larndsim", "bin", "response_44.npy")

    common_kwargs = dict(
        config="module0",
        pixel_layout=pixel_layout,
        detector_properties=detprops,
        simulation_properties=simprops,
        response_file=response_file,
        light_simulated=False,
        rand_seed=args.rand_seed,
    )

    mem_before_mib = gpu_mem_used_mib()
    print(f"[mem] GPU memory used before run (MiB): {mem_before_mib}")

    mem_after_warmup_mib = mem_before_mib
    if args.warmup_input:
        t0 = time.time()
        simulate_pixels.run_simulation(
            input_filename=args.warmup_input,
            output_filename=args.output + ".warmup.h5",
            **common_kwargs,
        )
        print(f"[timing] warm-up run (stage {args.stage}) [JIT/compile, excluded]: "
              f"{time.time()-t0:.2f} s")
        mem_after_warmup_mib = gpu_mem_used_mib()
        print(f"[mem] GPU memory used after warm-up (MiB): {mem_after_warmup_mib}")
        print(f"[mem] cupy memory pool after warm-up: {cupy_mempool_bytes()}")

    t0 = time.time()
    simulate_pixels.run_simulation(
        input_filename=args.input,
        output_filename=args.output,
        **common_kwargs,
    )
    dt = time.time() - t0
    print(f"[timing] stage {args.stage} timed run: {dt:.2f} s")
    mem_after_timed_mib = gpu_mem_used_mib()
    print(f"[mem] GPU memory used after timed run (MiB): {mem_after_timed_mib}")
    try:
        delta = [a - b for a, b in zip(mem_after_timed_mib, mem_after_warmup_mib)]
        print(f"[mem] nvidia-smi delta over warm-up baseline (MiB): {delta}")
    except Exception as e:
        print(f"[mem] could not compute nvidia-smi delta: {e}")
    cp_pool = cupy_mempool_bytes()
    print(f"[mem] cupy memory pool after timed run: {cp_pool}")
    if isinstance(cp_pool, dict) and "used_bytes" in cp_pool:
        print(f"[mem] cupy pool used_bytes: {cp_pool['used_bytes']} bytes "
              f"({cp_pool['used_bytes']/1e9:.3f} GB) -- the meaningful number, "
              f"not nvidia-smi's raw 'used' which includes the whole CUDA context.")

    # Report ms/cm using the input track length, for the perf table.
    try:
        import h5py
        import numpy as np
        with h5py.File(args.input, "r") as f:
            total_cm = float(np.array(f["segments"]["dx"]).sum())
        print(f"[perf] input track length: {total_cm:.3f} cm")
        print(f"[perf] stage {args.stage}: {1000*dt/total_cm:.3f} ms/cm "
              f"({dt:.2f} s / {total_cm:.1f} cm)")
    except Exception as e:
        print(f"[perf] could not compute ms/cm: {e}")


if __name__ == "__main__":
    main()
