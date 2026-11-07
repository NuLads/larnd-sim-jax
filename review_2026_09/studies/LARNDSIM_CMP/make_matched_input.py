#!/usr/bin/env python3
"""
Build ONE matched input file that both the reference and larnd-sim-jax will
simulate in full, with no further drops on either side.

Root cause of the stage-1..3 mismatch (reported 2026-10-09): the reference
and JAX runs were each silently dropping DIFFERENT subsets of
batch_350cm_input.h5 (127 segments, 438.0 cm, 3 events), for two completely
different reasons:

  - reference: cli/simulate_pixels.py calls
    active_volume.select_active_volume(tracks, detector.TPC_BORDERS) right
    after swap_coordinates() + the "match the jax simulation" TPC_BORDERS
    hack (TPC_BORDERS[0][2][1] = -0.15875). This keeps a segment iff its
    start OR end point falls strictly inside some TPC's (x,y,z) box -- a
    real, physical geometric cut (segments outside module0's small active
    volume truly can't produce hits). Result: 127 -> 66 segments, 3 events.

  - larnd-sim-jax: optimize/simulate.py's TracksDataset call uses
    live_selection=False, so the track_len_sel/costheta/track_z_bound cuts
    in dataio.py are NOT the cause (those only run under live_selection=True
    -- a red herring). The actual culprit is `--max_batch_len` (CLI default
    50.0 cm): dataio.py drops, upfront, any whole trajectory (grouped by
    eventID+trackID) whose summed dx exceeds max_batch_len, because a
    trajectory can't be split across batches (`valid_traj_mask = traj_len
    <= max_batch_len`, optimize/dataio.py line ~344). Our through-going
    muon trajectories are mostly longer than 50 cm, so nearly everything
    was dropped: 438 -> 91.58 cm, "2 batches" survived. NOT a geometric
    cut at all -- fixed by passing a large --max_batch_len (see
    submit_comparison.sh), so once given the matched_input.h5 below it
    drops nothing further.

Two ways to get the reference's 66-segment set are implemented:

  1. `extract_from_reference_output()` (DEFAULT, exact): read
     out_stage1_ref.h5['tracks']['segment_id'] (the 66 segment_ids the
     reference actually simulated in the real GPU run) and select those
     exact rows, by segment_id, from the pristine (pre-quench/pre-drift)
     batch_350cm_input.h5. This is bit-exact to what the reference did,
     with no need to replicate its float32 kernel arithmetic.

  2. `replicate_active_volume_filter()` (fallback / cross-check,
     reproduces reference's active_volume + swap_coordinates logic
     directly, for use when no reference output exists yet, e.g. for a
     brand-new input). Gives 63/66 segments (211.4 cm vs 216.1 cm) -- the
     3-segment gap was tracked down to 3 segments whose z_start (after
     swap) lands EXACTLY on the hardcoded -0.15875 anode-plane boundary;
     the real run's float32 kernel comparison apparently rounds that
     boundary check the other way (strict ">" at a literal float32/float64
     tie). Not worth chasing further given method 1 is exact and available.

Both methods select rows from the ORIGINAL (unswapped) array -- row order/
values are preserved (no reordering), and downstream consumers (reference's
run_simulation, JAX's TracksDataset) each apply their own equivalent swap_xz
internally (verified identical: compare swap_coordinates() in
reference_copy/cli/simulate_pixels.py with TracksDataset's swap_xz block in
larnd-sim-jax/optimize/dataio.py), so they must receive unswapped input.

Runs on CPU: active_volume.select_active_volume and swap_coordinates are
plain numpy/cupy-dispatch functions (no @cuda.jit), so the fake-cupy shim
is enough -- NUMBA_ENABLE_CUDASIM is not even needed for this script.
"""
import argparse
import sys
import numpy as np
import h5py

CMP_DIR = "/sdf/scratch/neutrino/pgranger/review_2026_09/LARNDSIM_CMP"
REF_DIR = CMP_DIR + "/reference_copy"

sys.path.insert(0, CMP_DIR)
import cupy_cpu_shim
sys.modules["cupy"] = cupy_cpu_shim
sys.modules["cupy.cuda"] = cupy_cpu_shim.cuda
sys.modules["cupy.cuda.nvtx"] = cupy_cpu_shim._nvtx_mod

sys.path.insert(0, REF_DIR)
sys.path.insert(0, REF_DIR + "/cli")

from larndsim import consts, active_volume
import simulate_pixels  # for swap_coordinates

INPUT_FILE = CMP_DIR + "/batch_350cm_input.h5"
OUTPUT_FILE = CMP_DIR + "/matched_input.h5"


def extract_from_reference_output(ref_output_file):
    with h5py.File(ref_output_file, "r") as f:
        sids = np.unique(np.array(f["tracks"])["segment_id"])
    with h5py.File(INPUT_FILE, "r") as f:
        tracks_orig = np.array(f["segments"])
    match = np.isin(tracks_orig["segment_id"], sids)
    print(f"extract_from_reference_output({ref_output_file}): "
          f"{match.sum()}/{len(sids)} requested segment_ids found, "
          f"{tracks_orig['dx'][match].sum():.3f} cm")
    return tracks_orig, match


def replicate_active_volume_filter():
    detector_properties = CMP_DIR + "/configs/module0_stage23_truth.yaml"
    pixel_layout = REF_DIR + "/larndsim/pixel_layouts/multi_tile_layout-2.4.16_v4.yaml"
    simulation_properties = CMP_DIR + "/configs/singles_sim_matched.yaml"

    consts.load_properties(detector_properties, pixel_layout, simulation_properties)
    from larndsim.consts import detector
    detector.TPC_BORDERS[0][2][1] = -0.15875  # same hack as cli/simulate_pixels.py

    with h5py.File(INPUT_FILE, "r") as f:
        tracks_orig = np.array(f["segments"])

    tracks_swapped = simulate_pixels.swap_coordinates(np.copy(tracks_orig))
    mask_idx = active_volume.select_active_volume(tracks_swapped, detector.TPC_BORDERS)
    mask = np.zeros(len(tracks_orig), dtype=bool)
    mask[mask_idx] = True
    print(f"replicate_active_volume_filter(): {mask.sum()} segments, "
          f"{tracks_orig['dx'][mask].sum():.3f} cm")
    return tracks_orig, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["from_ref_output", "replicate_filter"],
                     default="from_ref_output")
    ap.add_argument("--ref-output", default=CMP_DIR + "/out_stage1_ref.h5",
                     help="reference output h5 to pull segment_ids from "
                          "(active-volume selection is stage-independent, so "
                          "any of out_stage{1,2,3}_ref.h5 give the same set)")
    args = ap.parse_args()

    if args.method == "from_ref_output":
        tracks_orig, mask = extract_from_reference_output(args.ref_output)
    else:
        tracks_orig, mask = replicate_active_volume_filter()

    matched = tracks_orig[mask]
    print(f"input: {len(tracks_orig)} segments, {tracks_orig['dx'].sum():.3f} cm, "
          f"{len(np.unique(tracks_orig['event_id']))} events")
    print(f"matched: {len(matched)} segments, {matched['dx'].sum():.3f} cm, "
          f"{len(np.unique(matched['event_id']))} events")

    with h5py.File(INPUT_FILE, "r") as f:
        has_traj = "trajectories" in f
        if has_traj:
            traj = np.array(f["trajectories"])

    with h5py.File(OUTPUT_FILE, "w") as fout:
        fout.create_dataset("segments", data=matched)
        if has_traj:
            traj_mask = np.isin(traj["event_id"], np.unique(matched["event_id"]))
            fout.create_dataset("trajectories", data=traj[traj_mask])

    print(f"Wrote {len(matched)} segments ({matched['dx'].sum():.3f} cm) -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
