#!/usr/bin/env python3
"""
CPU-only analysis: turn the reference-vs-JAX stage1/2/3 outputs into
agreement metrics + a figure, and the timing logs into a performance table.

REWRITTEN 2026-10-09 to fix three bugs found in the first pass (see
run_log / task report for the full diagnosis):

 1. Different inputs were simulated on each side (reference 66/127
    segments via its active-volume cut; JAX 91.6/438 cm via a
    --max_batch_len trajectory-drop, unrelated to the former). Fixed
    upstream: both sides now simulate matched_input.h5 (the exact 66
    segments / 216.13 cm the reference's active-volume filter keeps,
    extracted via make_matched_input.py), and the JAX run uses a large
    --max-batch-len so it drops nothing further. This script now assumes
    both inputs match and does not try to compensate for a mismatch.

 2. Pixel-ID conventions differ completely: the reference's pixel2id is
    x + nx*(y + ny*plane) (no event component); JAX's packs eventID into
    the same integer (eventID*ntpc + plane)*ny*nx + ... + x. Raw IDs can
    never coincide. Fixed: decode BOTH sides to PHYSICAL (x, y) via each
    side's own id2pixel/get_pixel_coordinates (reference's via
    reference_copy, imported under the CPU cupy shim; JAX's already
    written out by run_jax_direct.py), then match nearest-neighbour in
    physical space.
    [SUPERSEDED 3rd pass -- see "PIXEL MATCHING TRANSFORM" below: the
    y-offset derived here (-21.8236cm, from comparing bare tpc_borders
    objects) turned out to be WRONG against real decoded hit data and is
    kept only as a cautionary note -- always validate a geometry
    transform against real matched data, not static config objects alone.]

 3. The reference's dense (n_pix, MAX_ADC_VALUES=30) adc/ticks arrays
    pad empty (never-triggered) slots by repeating a baseline value, NOT
    0 or -1. Empirically verified (out_stage1_ref.h5): the padding value
    is fee.digitize(0) with that stage's V_CM/V_REF (74.0 ADC counts for
    stage 1's aligned constants) and the matching ticks padding is
    exactly 0.0, with hits always left-aligned per row. `ticks != 0` is a
    perfect, config-independent stand-in for "real hit" (confirmed to
    agree with `adc != baseline` on every one of the 1529x30 = 45870
    slots in out_stage1_ref.h5), so that is now the validity mask -- NOT
    `adc > 0`. The previous `ref_hits_per_pixel_mean == 30.0` bug (every
    slot counted as a hit) is fixed by this.

TICK CONVENTION (added 2026-10-09, 3rd pass): JAX's 'ticks' is a raw bin
index into wfs's time axis (confirmed from sim_jax.py: `integral, ticks =
get_adc_values(...)`, used un-scaled; `get_hit_z` separately multiplies by
`params.t_sampling` to get a physical z, confirming ticks itself is NOT
yet in physical units). `run_jax_direct.py` now stores `t_sampling` as an
h5 attr (0.1 us) for this.

The reference's 'ticks' are written as
`adc_ticks_list[ip][iadc] = TIME_TICKS[crossing_time_tick] + TIME_PADDING - 2 + post_adc_ticks`
(fee.py's get_adc_values) where TIME_TICKS = linspace(0, 200, 2001) us and
TIME_PADDING=190 us (confirmed by loading the actual stage yaml on CPU) --
read literally this formula is at least 188, yet the real run's observed
mean tick is ~83-85 us. I could not fully re-derive this from the source
(most likely `ic`/`crossing_time_tick` indexes a per-pixel LOCAL buffer
already shifted by -TIME_PADDING via `t_start = round((t_start-t0_start-
TIME_PADDING)/TIME_SAMPLING)*TIME_SAMPLING`, not the global TIME_TICKS
origin, which would explain it, but I did not trace the exact index
algebra end to end within budget).

What IS established empirically, independently across all 3 stages (ratio
ref_mean_tick / jax_mean_tick_raw = 0.0981, 0.0978, 0.0983 for stages
1/2/3 -- consistent to ~0.5%, not a coincidence): `jax_tick_us =
jax_tick_raw * t_sampling (0.1)` lands within ~2% of the reference's
'ticks' convention already, with a small residual (~1.5-1.8 us) that
numerically matches ADC_HOLD_DELAY*CLOCK_CYCLE = 15*0.1 = 1.5 us (a
plausible candidate for the missing additive term, not confirmed from
source). TICK_ADDITIVE_OFFSET_US below applies that residual correction;
set it to 0.0 to use the bare multiplicative conversion instead. Given
the ~1.5-2 us residual is well under the time granularity relevant to
this comparison, this is treated as "good enough to compare", not as a
fully resolved derivation.

PIXEL MATCHING TRANSFORM (3rd pass, 2026-10-09): the -21.8236 cm y-offset
from the earlier pass was WRONG in practice -- it was derived from
comparing bare `tpc_borders` objects in isolation, not from real decoded
hit coordinates, and real data flatly contradicts it (JAX's raw pix_y for
real hits already overlaps the reference's range with no shift at all).
Re-derived empirically instead, on a real matched tiny_input.h5 run (both
sides, stage 1): scanned a (dx, dy) grid, found the count-maximizing shift,
then refined on the actual matched pairs. Result, exact to 6 decimal
places across all 25 independently-matched points: `jax_x_in_ref_frame =
jax_x - 0.2217` (= -pixel_pitch/2, cleanly explained: JAX's
get_pixel_coordinates adds +pixel_pitch/2 that the reference's doesn't,
confirmed in the 2nd-pass notes) and `jax_y_in_ref_frame = jax_y + 0.1247`
(found empirically; NOT yet traced to a specific source-code term despite
several attempts -- candidates checked and ruled out: it is not
tpc_offsets-related, not a simple multiple of pixel_pitch or
response_bin_size. Treat as a precisely-measured empirical constant, not
a derived one.). With this transform, 25 of the reference's 69 unique
hit pixels (stage 1, tiny_input.h5) match a JAX pixel to <1e-6 cm; the
other 44 simply have no JAX counterpart (and vice versa, 69 of JAX's 94) --
expected, since the two simulations don't have to touch identical pixel
sets, only mostly-overlapping ones.

Reference output format (written by reference_copy/cli/simulate_pixels.py,
the uncommitted-edit flat schema): 'pixels', 'tracks', 'adc' (n_pix, 30),
'ticks' (n_pix, 30), 'pixel_signals' (n_pix, n_time_ticks), 'sizes'.

JAX output format (written by run_jax_direct.py, NOT optimize/simulate.py
-- see that script's docstring for why): 'unique_pixels' + 'wfs' (Npix,
Nticks-1, induced current, index-aligned with unique_pixels) for the
stage-1/2 induced-current comparison; 'pix_x_decoded'/'pix_y_decoded'/
'pix_plane_decoded' (physical coordinates for every row of unique_pixels/
wfs); flat per-hit 'adc'/'pix_x'/'pix_y'/'pix_z'/'ticks'/'event'/
'hit_pixels' (stochastic, sampled) and 'adcs_distrib'/'px_prob'/'py_prob'/
'ticks_prob' (probabilistic, expected-value) for the hit/ADC/tick/
per-event comparisons.
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

CMP_DIR_DEFAULT = "/sdf/scratch/neutrino/pgranger/review_2026_09/LARNDSIM_CMP"
REF_DIR_DEFAULT = CMP_DIR_DEFAULT + "/reference_copy"

# Empirically-measured transform into the reference's frame (see "PIXEL
# MATCHING TRANSFORM" in the module docstring). 4th pass (2026-10-09):
# dy was off by exactly one pixel pitch in the 3rd pass (0.1247 - 0.4434 =
# -0.3187) -- a grid-periodicity alias from matching on ALL touched
# pixels (which includes neighbours) instead of HIT pixels only. Verified
# on hit-pixel-only matching, real GPU stage-1 output: residual < 6e-6 cm
# on all 409 matched pairs (ref 445 hit pixels, jax 505).
#   x_ref_frame = jax_x - PIX_DX_JAX_TO_REF
#   y_ref_frame = jax_y + PIX_DY_JAX_TO_REF
PIX_DX_JAX_TO_REF = 0.2217    # = pixel_pitch/2, analytically explained (JAX's
                               # get_pixel_coordinates adds +pitch/2 that the
                               # reference's doesn't -- confirmed in source)
PIX_DY_JAX_TO_REF = -0.318697  # empirical, measured to 1e-6 cm precision.
                               # NOT confirmed from source despite a direct
                               # attempt: reference's TPC_BORDERS y_border is
                               # built in larndsim/consts/detector.py (~line
                               # 281-290) purely from TILE_POSITIONS' y-column
                               # + tpc_offset[1], with NO dependence on
                               # z_border or the TPC_BORDERS[0][2][1]=-0.15875
                               # hack (that hack is applied later, by
                               # cli/simulate_pixels.py, to the z field only).
                               # The numeric closeness to -2*0.15875=-0.3175
                               # (0.4% off) may be coincidental. Treat as a
                               # precisely-measured empirical constant.
PIXEL_PITCH_CM = 0.4434
MATCH_TOL_CM = 0.01  # cm -- matches should now be near-exact (<1e-5 cm
                      # typical), not "within one pixel pitch"; a tight
                      # tolerance also avoids falsely pairing adjacent
                      # pixels now that the real transform is known.

# See "TICK CONVENTION" in the module docstring: jax_tick_us =
# jax_tick_raw * TICK_SCALE - TICK_ADDITIVE_OFFSET_US. Empirically verified
# multiplicative factor (t_sampling); additive term is a plausible-but-
# unconfirmed guess (ADC_HOLD_DELAY*CLOCK_CYCLE), included because it
# visibly improves the match -- treat it as approximate.
TICK_ADDITIVE_OFFSET_US = 1.5


def load_reference_flat(fname):
    with h5py.File(fname, "r") as f:
        pixels = np.array(f["pixels"])
        adc = np.array(f["adc"])
        ticks = np.array(f["ticks"])
        pixel_signals = np.array(f["pixel_signals"]) if "pixel_signals" in f else None
    valid = ticks != 0  # see module docstring point 3 -- NOT adc > 0
    return dict(pixels=pixels, adc=adc, ticks=ticks, pixel_signals=pixel_signals, valid=valid)


def load_jax_direct(fname):
    with h5py.File(fname, "r") as f:
        out = {k: np.array(f[k]) for k in f.keys()}
        t_sampling = f.attrs.get("t_sampling", 0.1)
    out["ticks_us"] = out["ticks"] * t_sampling - TICK_ADDITIVE_OFFSET_US
    out["t_sampling"] = t_sampling
    return out


def ref_pixel_physical_coords(pixels_1d, ref_dir=REF_DIR_DEFAULT, cmp_dir=CMP_DIR_DEFAULT):
    """Decode reference unique pixel ids -> physical (x, y, plane) using the
    reference's OWN id2pixel/get_pixel_coordinates (not reimplemented), with
    the real TPC_BORDERS (incl. the -0.15875 hack) loaded exactly as the
    real run did."""
    sys.path.insert(0, cmp_dir)
    import cupy_cpu_shim
    sys.modules.setdefault("cupy", cupy_cpu_shim)
    sys.modules.setdefault("cupy.cuda", cupy_cpu_shim.cuda)
    sys.modules.setdefault("cupy.cuda.nvtx", cupy_cpu_shim._nvtx_mod)
    sys.path.insert(0, ref_dir)
    from larndsim import consts
    consts.load_properties(
        cmp_dir + "/configs/module0_stage23_truth.yaml",
        ref_dir + "/larndsim/pixel_layouts/multi_tile_layout-2.4.16_v4.yaml",
        cmp_dir + "/configs/singles_sim_matched.yaml",
    )
    from larndsim.consts import detector
    detector.TPC_BORDERS[0][2][1] = -0.15875
    from larndsim.pixels_from_track import id2pixel

    i_x, i_y, plane = id2pixel(pixels_1d.astype(np.int64))
    i_x = np.asarray(i_x); i_y = np.asarray(i_y); plane = np.asarray(plane)
    borders = np.asarray(detector.TPC_BORDERS)
    pix_x = i_x * detector.PIXEL_PITCH + borders[plane, 0, 0]
    pix_y = i_y * detector.PIXEL_PITCH + borders[plane, 1, 0]
    return pix_x, pix_y, plane


def match_physical(ref_x, ref_y, jax_x, jax_y, tol=MATCH_TOL_CM):
    """Nearest-neighbour match in physical (x,y) space, reference frame.
    jax_x/jax_y are transformed into the reference's frame first (see
    "PIXEL MATCHING TRANSFORM" in the module docstring). Returns
    (ref_idx, jax_idx) arrays of matched pairs within tol, brute-force
    (fine at these sizes: O(1e2-1e3) pixels per side)."""
    jax_x_t = jax_x - PIX_DX_JAX_TO_REF
    jax_y_t = jax_y + PIX_DY_JAX_TO_REF
    if len(ref_x) == 0 or len(jax_x) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    d2 = (ref_x[:, None] - jax_x_t[None, :]) ** 2 + (ref_y[:, None] - jax_y_t[None, :]) ** 2
    jax_best = np.argmin(d2, axis=1)
    best_d = np.sqrt(d2[np.arange(len(ref_x)), jax_best])
    ok = best_d < tol
    ref_idx = np.where(ok)[0]
    jax_idx = jax_best[ok]
    # Guard against many-to-one: keep only the closest ref pixel per jax pixel.
    order = np.argsort(best_d[ok])
    seen_jax = set()
    keep = []
    for k in order:
        j = jax_idx[k]
        if j not in seen_jax:
            seen_jax.add(j)
            keep.append(k)
    keep = np.array(sorted(keep), dtype=int) if keep else np.array([], dtype=int)
    return ref_idx[keep], jax_idx[keep]


def compare_stage(ref_file, jax_file, stage, outdir, ref_dir, cmp_dir):
    print(f"\n=== Stage {stage}: {ref_file} vs {jax_file} ===")
    ref = load_reference_flat(ref_file)
    jax = load_jax_direct(jax_file)
    metrics = dict(stage=stage)

    # --- Physical pixel matching, HIT PIXELS ONLY (4th pass fix, see
    # "PIXEL MATCHING TRANSFORM" in the module docstring) ---
    # Matching on ALL touched pixels (as the 3rd pass did) is wrong: the
    # reference stores pixel_signals for every NEIGHBOUR pixel too (1529
    # touched vs. only ~440-450 actually hit), so a touched-pixel nearest-
    # neighbour search always finds *something* nearby regardless of the
    # transform, including one pixel-row off -- which is exactly how the
    # 3rd pass's dy=+0.1247cm survived: it is dy_correct - 1*pixel_pitch
    # (0.1247 - 0.4434 = -0.3187), a grid-periodicity alias. Matching HIT
    # pixels (ticks!=0 on the reference; present in 'hit_pixels' on the JAX
    # side) only is a much sharper test precisely because it is sparse.
    ref_hit_rows = np.where(ref["valid"].any(axis=1))[0]
    ref_hit_pix_ids = ref["pixels"][ref_hit_rows]
    rx, ry, r_plane = ref_pixel_physical_coords(ref_hit_pix_ids, ref_dir, cmp_dir)

    jax_hit_ids = np.unique(jax["hit_pixels"])
    jax_hit_ids = jax_hit_ids[jax_hit_ids >= 0]
    uniq = jax["unique_pixels"]
    idx_in_uniq = np.searchsorted(uniq, jax_hit_ids)
    idx_in_uniq = np.clip(idx_in_uniq, 0, len(uniq) - 1)
    jax_hit_ok = uniq[idx_in_uniq] == jax_hit_ids
    jax_hit_ids = jax_hit_ids[jax_hit_ok]
    idx_in_uniq = idx_in_uniq[jax_hit_ok]
    jx = jax["pix_x_decoded"][idx_in_uniq]
    jy = jax["pix_y_decoded"][idx_in_uniq]

    hm_ref, hm_jax = match_physical(rx, ry, jx, jy)
    ref_idx = ref_hit_rows[hm_ref]              # indices into the FULL ref arrays
    matched_jax_pids = jax_hit_ids[hm_jax]       # matched JAX pixel ids
    matched_jax_wfs_rows = idx_in_uniq[hm_jax]   # their row in jax['wfs']/unique_pixels
    n_common = len(ref_idx)
    metrics.update(n_ref_hit_pixels=len(rx), n_jax_hit_pixels=len(jx), n_common=n_common)
    print(f"  HIT-pixel matching (tol={MATCH_TOL_CM} cm): "
          f"ref hit pixels={len(rx)} jax hit pixels={len(jx)} common={n_common}")

    # --- Induced current (stage 1/2 only, needs wfs/pixel_signals), on the
    # correctly-matched (hit) pixels ---
    # Both sides consume the SAME response_44.npy LUT (byte-identical, see
    # task report), so there is no a-priori reason to expect a scale factor
    # -- but we check empirically (median ratio) rather than assume 1:1,
    # since JAX's wfs units were not independently confirmed from source
    # (likely electrons-equivalent current per tick, same as the LUT's own
    # convention, but not traced end-to-end).
    if stage in (1, 2) and ref["pixel_signals"] is not None and "wfs" in jax and n_common > 0:
        ref_wave = ref["pixel_signals"][ref_idx]                  # (n_common, n_ticks_ref)
        jax_wave = jax["wfs"][matched_jax_wfs_rows]                # (n_common, n_ticks_jax)
        n_ticks = min(ref_wave.shape[1], jax_wave.shape[1])
        ref_tot = ref_wave[:, :n_ticks].sum(axis=1)
        jax_tot = jax_wave[:, :n_ticks].sum(axis=1)
        nz = ref_tot != 0
        ratio = jax_tot[nz] / ref_tot[nz] if nz.any() else np.array([])
        metrics["induced_current_scale_ratio_median"] = float(np.median(ratio)) if len(ratio) else float("nan")
        reldiff = (jax_tot - ref_tot) / np.where(ref_tot != 0, ref_tot, np.nan)
        metrics["induced_current_total_mean_rel_diff_pct"] = float(np.nanmean(reldiff) * 100)
        metrics["induced_current_total_rms_rel_diff_pct"] = float(np.nanstd(reldiff) * 100)
        print(f"  induced current (time-integrated per matched HIT pixel): "
              f"median jax/ref scale ratio={metrics['induced_current_scale_ratio_median']:.4g} "
              f"(1.0 = same units, as expected from the shared response LUT) "
              f"mean rel diff={metrics['induced_current_total_mean_rel_diff_pct']:.2f}% "
              f"RMS={metrics['induced_current_total_rms_rel_diff_pct']:.2f}% "
              f"(n_ticks compared: {n_ticks}, ref has {ref_wave.shape[1]}, jax has {jax_wave.shape[1]})")

    # --- Hits + ADC per matched (hit) pixel ---
    ref_nhits_per_pix = ref["valid"].sum(axis=1)[ref_idx] if n_common else np.array([])
    ref_adc_sum_per_pix = np.where(ref["valid"], ref["adc"], 0).sum(axis=1)[ref_idx] if n_common else np.array([])

    jax_pix_all = jax["hit_pixels"]
    jax_adc_all = jax["adc"]
    jax_ticks_all = jax["ticks"]
    jax_unique_ok = matched_jax_pids  # kept name for downstream compatibility
    jax_idx = np.arange(n_common)
    jax_nhits_per_pix = np.zeros(n_common)
    jax_adc_sum_per_pix = np.zeros(n_common)
    for k, pid in enumerate(matched_jax_pids):
        m = jax_pix_all == pid
        jax_nhits_per_pix[k] = m.sum()
        jax_adc_sum_per_pix[k] = jax_adc_all[m].sum()
    print(f"  total hits on common pixels: ref={int(ref_nhits_per_pix.sum())} jax={int(jax_nhits_per_pix.sum())}")
    if n_common:
        same_count = int((ref_nhits_per_pix == jax_nhits_per_pix).sum())
        metrics["frac_common_pixels_same_hitcount"] = same_count / n_common
        print(f"  same hit count on {same_count}/{n_common} common pixels "
              f"({100*same_count/n_common:.1f}%)")

    if n_common:
        metrics["ref_hits_per_pixel_mean"] = float(ref_nhits_per_pix.mean())
        metrics["jax_hits_per_pixel_mean"] = float(jax_nhits_per_pix.mean())
        metrics["ref_adc_sum_per_pixel_mean"] = float(ref_adc_sum_per_pix.mean())
        metrics["jax_adc_sum_per_pixel_mean"] = float(jax_adc_sum_per_pix.mean())
        print(f"  hits/pixel: ref mean={metrics['ref_hits_per_pixel_mean']:.3f} "
              f"jax mean={metrics['jax_hits_per_pixel_mean']:.3f}")
        print(f"  summed ADC/pixel: ref mean={metrics['ref_adc_sum_per_pixel_mean']:.2f} "
              f"jax mean={metrics['jax_adc_sum_per_pixel_mean']:.2f}")

    # --- First-hit time difference, PAIRWISE per matched pixel (not a
    # global mean -- see "FIRST-HIT TIME DIFFERENCE" in the module
    # docstring). For each matched pixel with >=1 hit on EACH side,
    # diff = ref_first_hit_tick_us - jax_first_hit_tick_us. ---
    first_hit_diff_us = np.array([])
    if n_common:
        ref_ticks_matched = ref["ticks"][ref_idx]
        ref_valid_matched = ref["valid"][ref_idx]
        ref_first_tick = np.full(n_common, np.nan)
        for i in range(n_common):
            v = np.where(ref_valid_matched[i])[0]
            if len(v):
                ref_first_tick[i] = ref_ticks_matched[i, v[0]]
        jax_first_tick = np.full(n_common, np.nan)
        for i, pid in enumerate(jax_unique_ok[jax_idx]):
            m = jax_pix_all == pid
            if m.any():
                jax_first_tick[i] = jax["ticks_us"][m].min()
        both_have_hit = ~np.isnan(ref_first_tick) & ~np.isnan(jax_first_tick)
        first_hit_diff_us = ref_first_tick[both_have_hit] - jax_first_tick[both_have_hit]
        if len(first_hit_diff_us):
            metrics["first_hit_diff_us_median"] = float(np.median(first_hit_diff_us))
            metrics["first_hit_diff_us_mean"] = float(np.mean(first_hit_diff_us))
            metrics["first_hit_diff_us_std"] = float(np.std(first_hit_diff_us))
            metrics["n_pixels_with_first_hit_both_sides"] = int(both_have_hit.sum())
            print(f"  first-hit time (ref - jax), PAIRWISE on {both_have_hit.sum()} pixels with a "
                  f"hit on both sides: median={metrics['first_hit_diff_us_median']:.2f} us "
                  f"mean={metrics['first_hit_diff_us_mean']:.2f} us std={metrics['first_hit_diff_us_std']:.2f} us "
                  f"(NOTE: the matched-pixel GLOBAL tick means reported below mix in JAX's extra "
                  f"hits -- see module docstring -- and should not be read as a time-origin offset)")

    # --- Hit ticks (converted to absolute us, see "TICK CONVENTION" in the
    # module docstring). IMPORTANT: compared on the MATCHED-PIXEL subset
    # only, not globally -- the two sides don't touch identical pixel
    # sets, so a global mean mixes hits at genuinely different physical
    # locations/drift-times and is not a fair comparison (verified: global
    # means differ by ~45%, but on just the 25 matched pixels they agree
    # to <1%, see task report). ---
    ref_valid_ticks_us_matched = ref["ticks"][ref_idx][ref["valid"][ref_idx]] if n_common else np.array([])
    jax_mask_matched = np.isin(jax_pix_all, jax_unique_ok[jax_idx]) if n_common else np.zeros(0, dtype=bool)
    jax_valid_ticks_us_matched = jax["ticks_us"][jax_mask_matched]
    metrics["ref_mean_tick_us_matched"] = float(ref_valid_ticks_us_matched.mean()) if ref_valid_ticks_us_matched.size else float("nan")
    metrics["jax_mean_tick_us_matched"] = float(jax_valid_ticks_us_matched.mean()) if jax_valid_ticks_us_matched.size else float("nan")
    print(f"  hit ticks (us, MATCHED PIXELS ONLY -- see TICK CONVENTION note): "
          f"ref mean={metrics['ref_mean_tick_us_matched']:.2f} "
          f"jax mean={metrics['jax_mean_tick_us_matched']:.2f} "
          f"(n_ref_hits={ref_valid_ticks_us_matched.size}, n_jax_hits={jax_valid_ticks_us_matched.size})")
    # Also report the (not directly comparable) global means for reference.
    ref_valid_ticks_us = ref["ticks"][ref["valid"]]
    jax_valid_ticks_us = jax["ticks_us"]
    metrics["ref_mean_tick_us_global"] = float(ref_valid_ticks_us.mean()) if ref_valid_ticks_us.size else float("nan")
    metrics["jax_mean_tick_us_global"] = float(jax_valid_ticks_us.mean()) if jax_valid_ticks_us.size else float("nan")

    ref_total_adc = float(np.where(ref["valid"], ref["adc"], 0).sum())
    jax_total_adc = float(jax_adc_all.sum())
    metrics["ref_total_adc"] = ref_total_adc
    metrics["jax_total_adc"] = jax_total_adc
    print(f"  file-total summed ADC: ref={ref_total_adc:.1f} jax={jax_total_adc:.1f}")

    ref_adc_vals = np.array([])
    jax_adc_vals = np.array([])
    if stage == 3:
        ref_adc_vals = ref["adc"][ref["valid"]]
        jax_adc_vals = jax_adc_all
        metrics["ref_mean_adc"] = float(np.mean(ref_adc_vals)) if len(ref_adc_vals) else float("nan")
        metrics["jax_mean_adc"] = float(np.mean(jax_adc_vals)) if len(jax_adc_vals) else float("nan")
        if len(ref_adc_vals) and len(jax_adc_vals):
            allv = np.sort(np.concatenate([ref_adc_vals, jax_adc_vals]))
            cdf_ref = np.searchsorted(np.sort(ref_adc_vals), allv, side="right") / len(ref_adc_vals)
            cdf_jax = np.searchsorted(np.sort(jax_adc_vals), allv, side="right") / len(jax_adc_vals)
            metrics["ks_stat_adc_spectrum"] = float(np.max(np.abs(cdf_ref - cdf_jax)))
            print(f"  ADC spectrum (noise on): ref_mean={metrics['ref_mean_adc']:.2f} "
                  f"jax_mean={metrics['jax_mean_adc']:.2f} KS={metrics['ks_stat_adc_spectrum']:.3f}")

    if plt is not None:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        if n_common:
            axes[0].scatter(ref_adc_sum_per_pix, jax_adc_sum_per_pix, s=10, alpha=0.6)
            lim = [0, max(ref_adc_sum_per_pix.max(), jax_adc_sum_per_pix.max()) * 1.05]
            axes[0].plot(lim, lim, "k--", lw=1)
            axes[0].set_xlabel("reference summed ADC / pixel")
            axes[0].set_ylabel("larnd-sim-jax summed ADC / pixel")
            axes[0].set_title(f"Stage {stage}: per-pixel ADC agreement (physically matched)")
        else:
            axes[0].axis("off")
        if stage == 3 and len(ref_adc_vals) and len(jax_adc_vals):
            bins = np.linspace(0, max(ref_adc_vals.max(), jax_adc_vals.max()), 60)
            axes[1].hist(ref_adc_vals, bins=bins, histtype="step", label="reference", density=True)
            axes[1].hist(jax_adc_vals, bins=bins, histtype="step", label="larnd-sim-jax", density=True)
            axes[1].legend(); axes[1].set_xlabel("ADC value"); axes[1].set_title("ADC spectrum (stage 3)")
        else:
            axes[1].axis("off")
        fig.tight_layout()
        outdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(outdir / f"stage{stage}_comparison.png", dpi=130)
        plt.close(fig)
        print(f"  figure -> {outdir / f'stage{stage}_comparison.png'}")

    arrays = dict(
        ref_adc_sum_per_pix=ref_adc_sum_per_pix, jax_adc_sum_per_pix=jax_adc_sum_per_pix,
        ref_nhits_per_pix=ref_nhits_per_pix, jax_nhits_per_pix=jax_nhits_per_pix,
        first_hit_diff_us=first_hit_diff_us,
    )
    return metrics, arrays


def make_paper_summary_figure(stage, arrays, outdir):
    """3-panel summary figure for the paper (per the task report):
    (a) per-pixel ADC, ref vs JAX, matched pixels
    (b) first-hit time difference per matched pixel (pairwise, ref - jax)
    (c) hits per pixel on both sides
    """
    if plt is None:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    rq, jq = arrays["ref_adc_sum_per_pix"], arrays["jax_adc_sum_per_pix"]
    if len(rq):
        axes[0].scatter(rq, jq, s=10, alpha=0.5, color="tab:blue")
        lim = [0, max(rq.max(), jq.max()) * 1.05]
        axes[0].plot(lim, lim, "k--", lw=1, label="y=x")
        axes[0].set_xlabel("reference summed ADC / pixel")
        axes[0].set_ylabel("larnd-sim-jax summed ADC / pixel")
        axes[0].set_title("(a) per-pixel ADC, matched pixels")
        axes[0].legend(fontsize=8)

    fhd = arrays["first_hit_diff_us"]
    if len(fhd):
        bins = np.linspace(np.percentile(fhd, 1), np.percentile(fhd, 99), 40)
        axes[1].hist(fhd, bins=bins, color="tab:orange", edgecolor="k", linewidth=0.3)
        axes[1].axvline(np.median(fhd), color="k", ls="--", lw=1,
                         label=f"median={np.median(fhd):.2f} us")
        axes[1].set_xlabel("first-hit time, ref - jax (us)")
        axes[1].set_ylabel("matched pixels")
        axes[1].set_title("(b) first-hit time difference (pairwise)")
        axes[1].legend(fontsize=8)

    rn, jn = arrays["ref_nhits_per_pix"], arrays["jax_nhits_per_pix"]
    if len(rn):
        maxn = int(max(rn.max(), jn.max())) + 1
        bins = np.arange(-0.5, maxn + 0.5, 1)
        axes[2].hist(rn, bins=bins, histtype="step", label=f"reference (mean={rn.mean():.2f})",
                     color="tab:blue", linewidth=1.5)
        axes[2].hist(jn, bins=bins, histtype="step", label=f"larnd-sim-jax (mean={jn.mean():.2f})",
                     color="tab:red", linewidth=1.5)
        axes[2].set_xlabel("hits per pixel")
        axes[2].set_ylabel("matched pixels")
        axes[2].set_title("(c) hits per pixel, matched pixels")
        axes[2].legend(fontsize=8)

    fig.suptitle(f"Reference vs larnd-sim-jax, stage {stage}, matched_input.h5 (216.13 cm, {len(rq)} matched pixels)")
    fig.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"paper_summary_stage{stage}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  paper summary figure -> {out_path}")
    return out_path


def parse_timing_log(log_path):
    rows = []
    if not Path(log_path).exists():
        return rows
    for line in Path(log_path).read_text().splitlines():
        if line.startswith("[timing]") or line.startswith("[perf]") or line.startswith("[mem]") \
           or line.startswith("[config]") or line.startswith("[data]") or line.startswith("[jax]"):
            rows.append(line)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmp-dir", default=CMP_DIR_DEFAULT)
    ap.add_argument("--ref-dir", default=REF_DIR_DEFAULT)
    ap.add_argument("--stages", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--jax-suffix", default="jax_direct",
                     help="out_stage{N}_<suffix>.h5 -- use 'jax' for the old "
                          "optimize.simulate grouped format (no induced-current "
                          "comparison possible there, see module docstring)")
    ap.add_argument("--paper-summary-stage", type=int, default=1,
                     help="which stage's arrays to use for the 3-panel paper summary figure")
    args = ap.parse_args()
    cmp_dir = Path(args.cmp_dir)
    outdir = cmp_dir / "analysis_output"

    all_metrics = []
    arrays_by_stage = {}
    for stage in args.stages:
        ref_file = cmp_dir / f"out_stage{stage}_ref.h5"
        jax_file = cmp_dir / f"out_stage{stage}_{args.jax_suffix}.h5"
        if not ref_file.exists() or not jax_file.exists():
            print(f"[skip] stage {stage}: missing {ref_file if not ref_file.exists() else jax_file}")
            continue
        m, a = compare_stage(str(ref_file), str(jax_file), stage, outdir,
                              args.ref_dir, args.cmp_dir)
        all_metrics.append(m)
        arrays_by_stage[stage] = a

    if args.paper_summary_stage in arrays_by_stage:
        make_paper_summary_figure(args.paper_summary_stage, arrays_by_stage[args.paper_summary_stage], outdir)

    print("\n=== Timing / performance ===")
    for stage in args.stages:
        for tag, fname in [("reference", f"log_stage{stage}_ref.txt"),
                            ("larnd-sim-jax", f"log_stage{stage}_{args.jax_suffix}.txt")]:
            rows = parse_timing_log(cmp_dir / fname)
            if rows:
                print(f"-- stage {stage} / {tag} ({fname}) --")
                for r in rows:
                    print("  ", r)

    if all_metrics:
        outdir.mkdir(parents=True, exist_ok=True)
        import json
        with open(outdir / "metrics.json", "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\nWrote metrics -> {outdir / 'metrics.json'}")


if __name__ == "__main__":
    main()
